"""
Invite repository — production-grade CRUD over `invites`.

Handles token generation + lookup by email/normalised-email/token, and
prevents accidental duplicate invites (same email + same scope).
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from backend.db.mongodb.models.invite import (
    Invite,
    InviteCreate,
    InviteRole,
    InviteScope,
    InviteStatus,
    InviteUpdate,
)
from backend.db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)

COLLECTION_NAME = "invites"
DEFAULT_TOKEN_BYTES = 32


def _coerce_oid(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value))
    except Exception:
        return str(value)


def _normalise_email(email: str) -> str:
    return (email or "").strip().lower()


def _generate_token() -> str:
    """URL-safe random token.  Signed JWT layer added in Phase 5."""
    return secrets.token_urlsafe(DEFAULT_TOKEN_BYTES)


class InviteRepository:
    def __init__(self) -> None:
        self._collection = None

    async def _get_collection(self):
        if self._collection is None:
            self._collection = await MongoDB.get_collection(COLLECTION_NAME)
            await self._create_indexes()
        return self._collection

    async def _create_indexes(self) -> None:
        col = self._collection
        await col.create_index("email_normalized")
        await col.create_index("invited_by")
        await col.create_index("organization_id")
        await col.create_index("project_id")
        await col.create_index("status")
        await col.create_index("expires_at")
        await col.create_index("created_at")
        # The token must be unique — accept invites only if the token matches.
        await col.create_index("token", unique=True)
        # Prevent duplicate pending invites for the same scope.
        # Sparse — a None field still allows multiple rows.
        await col.create_index(
            [
                ("email_normalized", ASCENDING),
                ("scope", ASCENDING),
                ("organization_id", ASCENDING),
                ("project_id", ASCENDING),
                ("status", ASCENDING),
            ],
            name="invite_dedup_idx",
        )

    # ── CRUD ───────────────────────────────────────────────────────────

    async def create(
        self,
        payload: InviteCreate,
        invited_by: str,
        inviter_name: Optional[str] = None,
        inviter_email: Optional[str] = None,
    ) -> Invite:
        """Create a new pending invite.  Returns the saved row."""
        email_norm = _normalise_email(payload.email)
        if not email_norm:
            raise ValueError("Invite email is required")

        # Reject if there is already a PENDING invite for the exact same scope.
        col = await self._get_collection()
        dedup_query: Dict[str, Any] = {
            "email_normalized": email_norm,
            "scope": payload.scope.value,
            "status": InviteStatus.PENDING.value,
        }
        if payload.organization_id:
            dedup_query["organization_id"] = _coerce_oid(payload.organization_id)
        if payload.project_id:
            dedup_query["project_id"] = _coerce_oid(payload.project_id)

        existing = await col.find_one(dedup_query)
        if existing:
            # Touch and return — the API will resend the email if desired.
            await col.update_one(
                {"_id": existing["_id"]},
                {"$set": {"updated_at": datetime.utcnow()}},
            )
            return Invite(**existing)

        expires = datetime.utcnow() + timedelta(days=max(1, payload.expires_in_days))
        doc: Dict[str, Any] = {
            "email": payload.email,
            "email_normalized": email_norm,
            "invited_by": _coerce_oid(invited_by),
            "inviter_name": inviter_name,
            "inviter_email": inviter_email,
            "scope": payload.scope.value,
            "organization_id": _coerce_oid(payload.organization_id),
            "project_id": _coerce_oid(payload.project_id),
            "task_ids": [_coerce_oid(t) for t in (payload.task_ids or [])],
            "role": payload.role.value,
            "token": _generate_token(),
            "status": InviteStatus.PENDING.value,
            "message": payload.message,
            "expires_at": expires,
            "created_at": datetime.utcnow(),
            "reminder_count": 0,
            "metadata": {},
        }
        result = await col.insert_one(doc)
        created = await col.find_one({"_id": result.inserted_id})
        if not created:
            raise RuntimeError("Invite insert succeeded but find_one returned None")
        return Invite(**created)

    async def get_by_id(self, invite_id: str) -> Optional[Invite]:
        oid = _coerce_oid(invite_id)
        if not isinstance(oid, ObjectId):
            return None
        col = await self._get_collection()
        doc = await col.find_one({"_id": oid})
        return Invite(**doc) if doc else None

    async def get_by_token(self, token: str) -> Optional[Invite]:
        if not token:
            return None
        col = await self._get_collection()
        doc = await col.find_one({"token": token})
        return Invite(**doc) if doc else None

    async def update(
        self,
        invite_id: str,
        update: InviteUpdate | Dict[str, Any],
    ) -> Optional[Invite]:
        oid = _coerce_oid(invite_id)
        if not isinstance(oid, ObjectId):
            return None

        if hasattr(update, "model_dump"):
            patch = update.model_dump(exclude_none=True)
        else:
            patch = {k: v for k, v in (update or {}).items() if v is not None}

        # Strip immutable fields
        for forbidden in ("_id", "id", "email", "email_normalized", "token", "invited_by", "created_at"):
            patch.pop(forbidden, None)
        if not patch:
            return await self.get_by_id(invite_id)

        col = await self._get_collection()
        result = await col.find_one_and_update(
            {"_id": oid},
            {"$set": patch},
            return_document=True,
        )
        return Invite(**result) if result else None

    async def mark_accepted(
        self,
        invite_id: str,
        accepted_user_id: str,
    ) -> Optional[Invite]:
        oid = _coerce_oid(invite_id)
        if not isinstance(oid, ObjectId):
            return None
        col = await self._get_collection()
        result = await col.find_one_and_update(
            {"_id": oid, "status": InviteStatus.PENDING.value},
            {
                "$set": {
                    "status": InviteStatus.ACCEPTED.value,
                    "accepted_at": datetime.utcnow(),
                    "accepted_user_id": _coerce_oid(accepted_user_id),
                }
            },
            return_document=True,
        )
        return Invite(**result) if result else None

    async def mark_revoked(self, invite_id: str) -> bool:
        oid = _coerce_oid(invite_id)
        if not isinstance(oid, ObjectId):
            return False
        col = await self._get_collection()
        result = await col.update_one(
            {"_id": oid, "status": InviteStatus.PENDING.value},
            {"$set": {"status": InviteStatus.REVOKED.value, "revoked_at": datetime.utcnow()}},
        )
        return result.modified_count > 0

    async def mark_reminder_sent(self, invite_id: str) -> None:
        oid = _coerce_oid(invite_id)
        if not isinstance(oid, ObjectId):
            return
        col = await self._get_collection()
        await col.update_one(
            {"_id": oid},
            {
                "$set": {"last_reminder_sent_at": datetime.utcnow()},
                "$inc": {"reminder_count": 1},
            },
        )

    # ── Discovery queries ──────────────────────────────────────────────

    async def find_pending_by_email(self, email: str) -> List[Invite]:
        """All pending invites for a given email (used on signup to auto-accept)."""
        col = await self._get_collection()
        cursor = col.find({
            "email_normalized": _normalise_email(email),
            "status": InviteStatus.PENDING.value,
        }).sort("created_at", ASCENDING)
        docs = await cursor.to_list(length=200)
        return [Invite(**d) for d in docs]

    async def list_by_inviter(
        self,
        invited_by: str,
        status: Optional[InviteStatus] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Invite]:
        col = await self._get_collection()
        query: Dict[str, Any] = {"invited_by": _coerce_oid(invited_by)}
        if status:
            query["status"] = status.value
        cursor = col.find(query).sort("created_at", DESCENDING).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [Invite(**d) for d in docs]

    async def list_by_organization(
        self,
        organization_id: str,
        status: Optional[InviteStatus] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Invite]:
        col = await self._get_collection()
        query: Dict[str, Any] = {"organization_id": _coerce_oid(organization_id)}
        if status:
            query["status"] = status.value
        cursor = col.find(query).sort("created_at", DESCENDING).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [Invite(**d) for d in docs]

    async def expire_overdue(self, now: Optional[datetime] = None) -> int:
        """Bulk-mark PENDING invites as EXPIRED when expires_at < now.

        Returns the number of rows updated.  Called from a Celery beat job
        in Phase 5; safe to call ad-hoc.
        """
        now = now or datetime.utcnow()
        col = await self._get_collection()
        result = await col.update_many(
            {"status": InviteStatus.PENDING.value, "expires_at": {"$lt": now}},
            {"$set": {"status": InviteStatus.EXPIRED.value}},
        )
        return result.modified_count


invite_repository = InviteRepository()
