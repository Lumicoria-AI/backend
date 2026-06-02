"""
Phase 8 — Organizations REST API.

Endpoints:
  GET    /organizations/me                    — caller's current organization
  PATCH  /organizations/{org_id}              — admin: edit name / logo / settings
  GET    /organizations/{org_id}/members      — list members + their roles
  POST   /organizations/{org_id}/members/{user_id}/role  — admin: promote/demote
  DELETE /organizations/{org_id}/members/{user_id}       — admin: remove member
  POST   /organizations/{org_id}/leave        — caller leaves the org
  POST   /organizations/{org_id}/transfer-ownership      — owner only
  GET    /organizations/{org_id}/invites      — list pending invites for org
  POST   /organizations/{org_id}/invites      — admin: send invite (hooks Phase 5)
  POST   /organizations/{org_id}/invites/{invite_id}/resend
  DELETE /organizations/{org_id}/invites/{invite_id}     — admin: revoke
  GET    /organizations/{org_id}/stats        — basic counts (members, pending invites)

All write operations require the caller to be an admin of the target org
(except `leave`, which only requires membership).  Ownership transfer
requires the caller to be the owner.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field

import structlog

from backend.api.deps import get_current_active_user
from backend.db.mongodb.models.invite import (
    InviteCreate,
    InviteRole,
    InviteScope,
    InviteStatus,
)
from backend.db.mongodb.repositories.invite_repository import invite_repository
from backend.db.mongodb.repositories.organization_repository import (
    organization_repository,
)
from backend.db.mongodb.repositories.user_repository import (
    get_user_repository,
    user_repository,
)
from backend.models.mongodb_models import OrganizationCreate, OrganizationUpdate
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.invite_service import invite_service

logger = structlog.get_logger(__name__)
router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────────────


def _oid(value: Any) -> Optional[ObjectId]:
    if value is None:
        return None
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value))
    except Exception:
        return None


def _primary_org_id(user: Any) -> Optional[str]:
    """Pick the user's primary org id.  Prefers the singular convenience
    field when present (set during login / signup); otherwise falls back
    to the first entry of `organization_ids` from Mongo."""
    primary = getattr(user, "organization_id", None)
    if primary:
        return str(primary)
    org_ids = getattr(user, "organization_ids", None) or []
    if isinstance(org_ids, (list, tuple)) and org_ids:
        return str(org_ids[0])
    return None


def _serialize_org(org: Any) -> Dict[str, Any]:
    """Convert an Organization model to a JSON-safe dict for the API."""
    if hasattr(org, "model_dump"):
        d = org.model_dump(mode="json")
    elif hasattr(org, "dict"):
        d = org.dict()
    else:
        d = dict(org) if isinstance(org, dict) else {}

    # Coerce ObjectId-shaped fields
    if d.get("_id") is not None and "id" not in d:
        d["id"] = str(d.pop("_id"))
    if d.get("id") is not None:
        d["id"] = str(d["id"])
    for k in ("owner_id",):
        if d.get(k) is not None:
            d[k] = str(d[k])
    for k in ("member_ids", "admin_ids"):
        if isinstance(d.get(k), list):
            d[k] = [str(v) for v in d[k] if v is not None]
    return d


def _serialize_user_row(user: Any, *, role: str) -> Dict[str, Any]:
    """Compact view of a user for the org-members table."""
    if hasattr(user, "model_dump"):
        d = user.model_dump(mode="json")
    elif hasattr(user, "dict"):
        d = user.dict()
    else:
        d = dict(user) if isinstance(user, dict) else {}
    uid = d.get("_id") or d.get("id")
    return {
        "id": str(uid) if uid is not None else "",
        "email": d.get("email") or "",
        "full_name": d.get("full_name") or d.get("username") or "",
        "first_name": d.get("first_name"),
        "last_name": d.get("last_name"),
        "profile_picture": d.get("profile_picture") or d.get("avatar_url"),
        "role": role,
        "created_at": d.get("created_at"),
    }


def _serialize_invite(inv: Any) -> Dict[str, Any]:
    if hasattr(inv, "model_dump"):
        d = inv.model_dump(mode="json")
    elif hasattr(inv, "dict"):
        d = inv.dict()
    else:
        d = dict(inv) if isinstance(inv, dict) else {}
    iid = d.get("_id") or d.get("id")
    return {
        "id": str(iid) if iid is not None else "",
        "email": d.get("email"),
        "role": d.get("role"),
        "scope": d.get("scope"),
        "status": d.get("status"),
        "inviter_name": d.get("inviter_name"),
        "inviter_email": d.get("inviter_email"),
        "organization_id": str(d["organization_id"]) if d.get("organization_id") else None,
        "expires_at": d.get("expires_at"),
        "created_at": d.get("created_at"),
        "accepted_at": d.get("accepted_at"),
        "reminder_count": d.get("reminder_count", 0),
    }


async def _require_org_access(
    org_id: str, current_user: User
) -> Any:
    """Return the Organization the caller belongs to, or 404/403."""
    org_oid = _oid(org_id)
    if not org_oid:
        raise HTTPException(status_code=400, detail="Invalid organization id")
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    user_oid = _oid(current_user.id)
    member_ids = [_oid(m) for m in (org.member_ids or [])]
    if user_oid not in member_ids:
        raise HTTPException(status_code=403, detail="Not a member of this organization")
    return org


async def _require_org_admin(
    org_id: str, current_user: User
) -> Any:
    org = await _require_org_access(org_id, current_user)
    user_oid = _oid(current_user.id)
    admin_ids = [_oid(a) for a in (org.admin_ids or [])]
    if user_oid not in admin_ids:
        raise HTTPException(status_code=403, detail="Admin permission required")
    return org


def _is_owner(org: Any, user_id: Any) -> bool:
    owner = _oid(getattr(org, "owner_id", None))
    if owner is None:
        # Legacy orgs without an explicit owner — fall back to the first admin.
        admins = [_oid(a) for a in (org.admin_ids or [])]
        return bool(admins) and admins[0] == _oid(user_id)
    return owner == _oid(user_id)


# ── Create ────────────────────────────────────────────────────────────


@router.post("", response_model=None, status_code=status.HTTP_201_CREATED)
async def create_organization(
    payload: OrganizationCreate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Create a new organization with the caller as owner + admin.

    For users who signed up before the explicit Organization row was
    tracked: this is how they bootstrap their workspace.  Idempotent —
    if a user already belongs to an org, we just return that one.
    """
    # Already in an org?  Return it instead of double-creating.
    existing_id = _primary_org_id(current_user)
    if not existing_id:
        try:
            existing = await organization_repository.get_user_organizations(
                user_id=str(current_user.id), skip=0, limit=1,
            )
            if existing:
                existing_id = str(existing[0].id)
        except Exception:
            pass
    if existing_id:
        org = await organization_repository.get_by_id(existing_id)
        if org:
            return _serialize_org(org)

    name = (payload.name or "").strip()
    if not name:
        raise HTTPException(status_code=400, detail="Organization name is required")

    duplicate = await organization_repository.get_organization_by_name(name)
    if duplicate:
        raise HTTPException(status_code=409, detail="An organization with that name already exists")

    created = await organization_repository.create_organization(
        org_data=payload, creator_id=str(current_user.id)
    )
    # Stamp the owner pointer (the repo doesn't do this yet — Phase 1 orgs
    # only tracked admin_ids).
    try:
        await organization_repository.update(
            str(created.id),
            {"owner_id": _oid(str(current_user.id)), "updated_at": datetime.utcnow()},
        )
    except Exception:
        pass

    await log_activity(
        user_id=str(current_user.id),
        organization_id=str(created.id),
        activity_type="organization.created",
        details={"name": name},
        related_resource_type="ORGANIZATION",
        related_resource_id=str(created.id),
    )

    refreshed = await organization_repository.get_by_id(str(created.id))
    payload_out = _serialize_org(refreshed or created)
    payload_out["my_role"] = "owner"
    return payload_out


# ── Read endpoints ────────────────────────────────────────────────────


@router.get("/me", response_model=None)
async def get_my_organization(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Return the caller's primary organization (the one stamped on User.organization_id)."""
    org_id = _primary_org_id(current_user)
    if not org_id:
        # Last-ditch fallback — query the org collection directly in case the
        # user record was created before the org-id field was tracked.
        try:
            orgs = await organization_repository.get_user_organizations(
                user_id=str(current_user.id), skip=0, limit=1,
            )
            if orgs:
                org_id = str(orgs[0].id)
        except Exception:
            org_id = None
    if not org_id:
        raise HTTPException(status_code=404, detail="You don't belong to an organization yet")
    org = await organization_repository.get_by_id(str(org_id))
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    payload = _serialize_org(org)
    # Surface whether the caller is owner / admin / member so the UI doesn't
    # need a second round-trip to decide what to render.
    user_oid = _oid(current_user.id)
    payload["my_role"] = (
        "owner" if _is_owner(org, current_user.id)
        else "admin" if user_oid in [_oid(a) for a in (org.admin_ids or [])]
        else "member"
    )
    return payload


@router.get("/{org_id}", response_model=None)
async def get_organization(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    org = await _require_org_access(org_id, current_user)
    payload = _serialize_org(org)
    user_oid = _oid(current_user.id)
    payload["my_role"] = (
        "owner" if _is_owner(org, current_user.id)
        else "admin" if user_oid in [_oid(a) for a in (org.admin_ids or [])]
        else "member"
    )
    return payload


@router.patch("/{org_id}", response_model=None)
async def update_organization(
    org_id: str,
    payload: OrganizationUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Admin: edit org name / logo / industry / settings."""
    await _require_org_admin(org_id, current_user)
    updates = payload.dict(exclude_unset=True)
    if not updates:
        raise HTTPException(status_code=400, detail="No fields to update")

    if "name" in updates and updates["name"]:
        # Enforce name uniqueness in a friendly way.
        existing = await organization_repository.get_organization_by_name(updates["name"])
        if existing and str(existing.id) != str(org_id):
            raise HTTPException(status_code=409, detail="Organization name is already taken")

    updates["updated_at"] = datetime.utcnow()
    updated = await organization_repository.update(org_id, updates)
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to update organization")

    await log_activity(
        user_id=str(current_user.id),
        organization_id=org_id,
        activity_type="organization.updated",
        details={"fields": list(updates.keys())},
        related_resource_type="ORGANIZATION",
        related_resource_id=org_id,
    )
    return _serialize_org(updated)


# ── Members ───────────────────────────────────────────────────────────


@router.get("/{org_id}/members", response_model=None)
async def list_members(
    org_id: str,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    org = await _require_org_access(org_id, current_user)
    members = await organization_repository.get_organization_members(
        organization_id=org_id, skip=skip, limit=limit
    )
    admin_ids = {str(a) for a in (org.admin_ids or [])}
    owner_id = str(getattr(org, "owner_id", "") or "")
    rows: List[Dict[str, Any]] = []
    for u in members:
        uid = str(getattr(u, "id", "") or "")
        role = "owner" if uid == owner_id else "admin" if uid in admin_ids else "member"
        rows.append(_serialize_user_row(u, role=role))
    return {"count": len(rows), "items": rows}


class MemberRoleUpdate(BaseModel):
    role: str = Field(..., description='"admin" | "member"')


@router.post("/{org_id}/members/{user_id}/role", response_model=None)
async def update_member_role(
    org_id: str,
    user_id: str,
    body: MemberRoleUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Admin: promote a member to admin or demote an admin to member.

    The owner can never be demoted by anyone but themselves via the
    `transfer-ownership` endpoint.
    """
    org = await _require_org_admin(org_id, current_user)
    target_role = (body.role or "").lower().strip()
    if target_role not in {"admin", "member"}:
        raise HTTPException(status_code=400, detail="role must be 'admin' or 'member'")

    target_oid = _oid(user_id)
    if not target_oid:
        raise HTTPException(status_code=400, detail="Invalid user id")
    member_ids = [_oid(m) for m in (org.member_ids or [])]
    if target_oid not in member_ids:
        raise HTTPException(status_code=404, detail="User is not a member of this organization")

    if _is_owner(org, user_id) and target_role != "admin":
        raise HTTPException(
            status_code=400,
            detail="Cannot demote the organization owner. Transfer ownership first.",
        )

    if target_role == "admin":
        await organization_repository.promote_to_admin(org_id, user_id)
        action = "organization.member_promoted"
    else:
        await organization_repository.demote_from_admin(org_id, user_id)
        action = "organization.member_demoted"

    await log_activity(
        user_id=str(current_user.id),
        organization_id=org_id,
        activity_type=action,
        details={"target_user_id": user_id, "role": target_role},
        related_resource_type="ORGANIZATION",
        related_resource_id=org_id,
    )
    return {"updated": True, "user_id": user_id, "role": target_role}


@router.delete("/{org_id}/members/{user_id}", response_model=None)
async def remove_member(
    org_id: str,
    user_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Admin: remove a member from the org.  Owners cannot be removed —
    they must transfer ownership first.
    """
    org = await _require_org_admin(org_id, current_user)
    if _is_owner(org, user_id):
        raise HTTPException(
            status_code=400,
            detail="The owner can't be removed. Transfer ownership first.",
        )
    target_oid = _oid(user_id)
    if not target_oid or target_oid not in [_oid(m) for m in (org.member_ids or [])]:
        raise HTTPException(status_code=404, detail="User is not a member of this organization")

    await organization_repository.remove_member(org_id, user_id)
    await log_activity(
        user_id=str(current_user.id),
        organization_id=org_id,
        activity_type="organization.member_removed",
        details={"target_user_id": user_id},
        related_resource_type="ORGANIZATION",
        related_resource_id=org_id,
    )
    return {"removed": True, "user_id": user_id}


@router.post("/{org_id}/leave", response_model=None)
async def leave_organization(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Caller leaves the org.  Owner must transfer ownership before leaving."""
    org = await _require_org_access(org_id, current_user)
    if _is_owner(org, current_user.id):
        raise HTTPException(
            status_code=400,
            detail="You're the owner. Transfer ownership before leaving.",
        )
    await organization_repository.remove_member(org_id, str(current_user.id))
    await log_activity(
        user_id=str(current_user.id),
        organization_id=org_id,
        activity_type="organization.member_left",
        details={},
        related_resource_type="ORGANIZATION",
        related_resource_id=org_id,
    )
    return {"left": True}


class TransferOwnershipRequest(BaseModel):
    new_owner_id: str


@router.post("/{org_id}/transfer-ownership", response_model=None)
async def transfer_ownership(
    org_id: str,
    body: TransferOwnershipRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Owner only.  Promotes the new owner to admin (if not already) and
    flips the `owner_id` pointer.  The previous owner remains a member /
    admin until they explicitly leave or get demoted.
    """
    org = await _require_org_access(org_id, current_user)
    if not _is_owner(org, current_user.id):
        raise HTTPException(status_code=403, detail="Only the owner can transfer ownership")
    new_oid = _oid(body.new_owner_id)
    if not new_oid:
        raise HTTPException(status_code=400, detail="Invalid new_owner_id")
    if new_oid not in [_oid(m) for m in (org.member_ids or [])]:
        raise HTTPException(status_code=400, detail="The new owner must already be a member")

    # Ensure they're an admin too.
    await organization_repository.promote_to_admin(org_id, body.new_owner_id)
    updated = await organization_repository.update(
        org_id, {"owner_id": new_oid, "updated_at": datetime.utcnow()}
    )
    if not updated:
        raise HTTPException(status_code=500, detail="Failed to transfer ownership")

    await log_activity(
        user_id=str(current_user.id),
        organization_id=org_id,
        activity_type="organization.ownership_transferred",
        details={"new_owner_id": body.new_owner_id},
        related_resource_type="ORGANIZATION",
        related_resource_id=org_id,
    )
    return {"transferred": True, "new_owner_id": body.new_owner_id}


# ── Invites (Phase 5 hookup) ──────────────────────────────────────────


@router.get("/{org_id}/invites", response_model=None)
async def list_organization_invites(
    org_id: str,
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(100, ge=1, le=500),
    skip: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Admin: list invites for the org.  Members get a read-only view too."""
    await _require_org_access(org_id, current_user)
    parsed_status: Optional[InviteStatus] = None
    if status_filter:
        try:
            parsed_status = InviteStatus(status_filter.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status_filter}")
    invites = await invite_repository.list_by_organization(
        organization_id=org_id,
        status=parsed_status,
        limit=limit,
        skip=skip,
    )
    return {"count": len(invites), "items": [_serialize_invite(i) for i in invites]}


class OrgInviteRequest(BaseModel):
    """Body for sending one or many invites in a single call.

    Accepts either:
      - `email: "name@x.com"`                 — single invite (legacy shape)
      - `emails: ["a@x.com", "b@y.com"]`      — bulk invite (preferred)
      - `email: "a@x.com, b@y.com"`           — comma- or newline-separated string

    All three end up normalised into a list of emails.  The endpoint
    de-duplicates, validates, skips existing members, and returns a
    per-email result so the UI can show "8 sent, 1 already a member,
    1 invalid".
    """
    email: Optional[str] = None
    emails: Optional[List[str]] = None
    role: Optional[str] = "member"
    message: Optional[str] = None
    expires_in_days: int = Field(14, ge=1, le=90)


# Reused regex — same one we lean on elsewhere.
_EMAIL_RE = __import__("re").compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _split_emails(body: OrgInviteRequest) -> List[str]:
    """Normalise the request body into a clean, de-duplicated list of
    lower-cased email addresses.  Handles commas, semicolons, newlines,
    and whitespace.
    """
    parts: List[str] = []
    if body.emails:
        for e in body.emails:
            if e:
                parts.append(str(e))
    if body.email:
        # Split on `,`, `;`, or newlines so users can paste a list.
        import re as _re
        parts.extend(_re.split(r"[,\n;]+", str(body.email)))

    cleaned: List[str] = []
    seen: set = set()
    for raw in parts:
        e = (raw or "").strip().lower()
        if not e:
            continue
        if e in seen:
            continue
        seen.add(e)
        cleaned.append(e)
    return cleaned


@router.post("/{org_id}/invites", response_model=None)
async def send_organization_invite(
    org_id: str,
    body: OrgInviteRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Admin: invite one or many people to the org by email.

    Hooks Phase 5 `invite_service.issue_invite` so the same branded
    email + signup acceptance flow runs for every recipient.

    Returns:
        {
          "summary": {"sent": int, "skipped": int, "failed": int, "total": int},
          "results": [
            {"email": str, "status": "sent" | "skipped" | "failed",
             "reason": str, "invite": {...} | None},
            ...
          ]
        }

    Single-email callers still get a flat response (the first item from
    `results`) so existing clients don't break.
    """
    await _require_org_admin(org_id, current_user)

    role_raw = (body.role or "member").lower()
    try:
        role = InviteRole(role_raw)
    except ValueError:
        raise HTTPException(status_code=400, detail="role must be 'admin', 'member', or 'viewer'")

    org_oid = _oid(org_id)
    if not org_oid:
        raise HTTPException(status_code=400, detail="Invalid organization id")

    emails = _split_emails(body)
    if not emails:
        raise HTTPException(status_code=400, detail="No email addresses provided")

    # Hard cap — protect the SMTP service from a runaway paste.
    MAX_PER_CALL = 200
    if len(emails) > MAX_PER_CALL:
        raise HTTPException(
            status_code=400,
            detail=f"Too many addresses ({len(emails)}). Send in batches of {MAX_PER_CALL}.",
        )

    user_repo = await get_user_repository()

    results: List[Dict[str, Any]] = []
    sent_ids: List[str] = []
    skipped = 0
    failed = 0

    for email in emails:
        # 1. Validate email shape (catch typos before we hit Mongo).
        if not _EMAIL_RE.match(email):
            failed += 1
            results.append({
                "email": email,
                "status": "failed",
                "reason": "Invalid email format",
                "invite": None,
            })
            continue

        # 2. Skip existing org members.
        try:
            target_user = await user_repo.get_user_by_email(email)
        except Exception:
            target_user = None
        if target_user is not None:
            existing_org_id = str(getattr(target_user, "organization_id", "") or "")
            existing_org_ids = [str(x) for x in (getattr(target_user, "organization_ids", []) or [])]
            if existing_org_id == str(org_id) or str(org_id) in existing_org_ids:
                skipped += 1
                results.append({
                    "email": email,
                    "status": "skipped",
                    "reason": "Already a member of this organization",
                    "invite": None,
                })
                continue

        # 3. Issue the invite (repo is idempotent — duplicate pending invites
        # for the same email/scope are merged into one).
        try:
            payload = InviteCreate(
                email=email,
                scope=InviteScope.ORGANIZATION,
                organization_id=org_oid,
                role=role,
                message=body.message,
                expires_in_days=body.expires_in_days,
            )
            invite = await invite_service.issue_invite(
                payload=payload,
                invited_by=str(current_user.id),
                send_email=True,
            )
            sent_ids.append(str(invite.id))
            results.append({
                "email": email,
                "status": "sent",
                "reason": "Invite email sent",
                "invite": _serialize_invite(invite),
            })
        except Exception as e:  # noqa: BLE001
            failed += 1
            logger.warning("bulk_invite_failed", email=email, error=str(e))
            results.append({
                "email": email,
                "status": "failed",
                "reason": str(e)[:200],
                "invite": None,
            })

    sent_count = len(sent_ids)

    await log_activity(
        user_id=str(current_user.id),
        organization_id=org_id,
        activity_type="organization.invite_sent_bulk" if sent_count > 1 else "organization.invite_sent",
        details={
            "role": role.value,
            "total": len(emails),
            "sent": sent_count,
            "skipped": skipped,
            "failed": failed,
            "invite_ids": sent_ids,
        },
        related_resource_type="ORGANIZATION",
        related_resource_id=org_id,
    )

    # Backward-compat: single-email caller gets the legacy flat invite shape.
    if len(emails) == 1 and results and results[0]["status"] == "sent" and results[0]["invite"]:
        return results[0]["invite"]

    return {
        "summary": {
            "sent": sent_count,
            "skipped": skipped,
            "failed": failed,
            "total": len(emails),
        },
        "results": results,
    }


@router.post("/{org_id}/invites/{invite_id}/resend", response_model=None)
async def resend_organization_invite(
    org_id: str,
    invite_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    await _require_org_admin(org_id, current_user)
    invite = await invite_repository.get_by_id(invite_id)
    if not invite or str(getattr(invite, "organization_id", "")) != str(org_id):
        raise HTTPException(status_code=404, detail="Invite not found")
    if invite.status != InviteStatus.PENDING:
        raise HTTPException(status_code=400, detail=f"Cannot resend an invite in '{invite.status}' status")

    # Bypass invite_service.resend_invite's "only the original inviter" guard
    # because org admins are explicitly authorised to manage *any* invite on
    # their own org — and we already enforced that above via _require_org_admin.
    success = await invite_service.send_invite_email(invite=invite)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to resend invite")
    refreshed = await invite_repository.get_by_id(invite_id)
    await log_activity(
        user_id=str(current_user.id),
        organization_id=org_id,
        activity_type="organization.invite_resent",
        details={"invite_id": invite_id},
        related_resource_type="ORGANIZATION",
        related_resource_id=org_id,
    )
    return _serialize_invite(refreshed) if refreshed else {"resent": True}


@router.delete("/{org_id}/invites/{invite_id}", response_model=None)
async def revoke_organization_invite(
    org_id: str,
    invite_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    await _require_org_admin(org_id, current_user)
    invite = await invite_repository.get_by_id(invite_id)
    if not invite or str(getattr(invite, "organization_id", "")) != str(org_id):
        raise HTTPException(status_code=404, detail="Invite not found")
    # Same reasoning as resend — go through the repo directly because org
    # admins must be able to revoke invites issued by any teammate, not just
    # their own.
    if invite.status != InviteStatus.PENDING:
        raise HTTPException(status_code=400, detail="Only pending invites can be revoked")
    success = await invite_repository.mark_revoked(invite_id)
    if not success:
        raise HTTPException(status_code=400, detail="Could not revoke invite")
    await log_activity(
        user_id=str(current_user.id),
        organization_id=org_id,
        activity_type="organization.invite_revoked",
        details={"invite_id": invite_id},
        related_resource_type="ORGANIZATION",
        related_resource_id=org_id,
    )
    return {"revoked": True, "invite_id": invite_id}


@router.get("/{org_id}/stats", response_model=None)
async def organization_stats(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Counts surface for the Organization settings page header."""
    org = await _require_org_access(org_id, current_user)
    members_count = len(org.member_ids or [])
    admins_count = len(org.admin_ids or [])
    pending_invites = await invite_repository.list_by_organization(
        organization_id=org_id, status=InviteStatus.PENDING, limit=1000
    )
    return {
        "members": members_count,
        "admins": admins_count,
        "pending_invites": len(pending_invites),
        "created_at": getattr(org, "created_at", None),
    }
