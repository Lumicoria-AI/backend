"""
ApiTokensRepository — org-scoped personal & service-account tokens.

Tokens are issued with a `prefix` (first 8 chars, visible) and stored as
`token_hash` (SHA-256).  The plaintext token is shown to the caller once at
creation time and never persisted.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog
from bson import ObjectId
from pymongo import ASCENDING

from backend.db.mongodb.mongodb import MongoDB
from backend.models.enterprise import ApiTokenInDB

logger = structlog.get_logger(__name__)

COLLECTION = "api_tokens"
TOKEN_PREFIX = "lmc_"


def _hash(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_plaintext_token() -> Tuple[str, str]:
    """Returns (plaintext, prefix)."""
    body = secrets.token_urlsafe(32)
    plaintext = f"{TOKEN_PREFIX}{body}"
    prefix = plaintext[: len(TOKEN_PREFIX) + 8]
    return plaintext, prefix


class ApiTokensRepository:
    def __init__(self) -> None:
        self._initialised = False

    async def _ensure_indexes(self) -> None:
        if self._initialised:
            return
        col = await MongoDB.get_collection(COLLECTION)
        await col.create_index([("organization_id", ASCENDING)])
        await col.create_index("token_hash", unique=True)
        await col.create_index("user_id")
        await col.create_index("revoked_at")
        self._initialised = True

    async def create(
        self,
        *,
        organization_id: str,
        name: str,
        scopes: List[str],
        user_id: Optional[str] = None,
        expires_at: Optional[datetime] = None,
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        """Returns (plaintext_token, persisted_row)."""
        await self._ensure_indexes()
        plaintext, prefix = generate_plaintext_token()
        col = await MongoDB.get_collection(COLLECTION)
        doc = {
            "organization_id": ObjectId(organization_id),
            "user_id": ObjectId(user_id) if user_id else None,
            "name": name,
            "prefix": prefix,
            "token_hash": _hash(plaintext),
            "scopes": list(scopes or []),
            "last_used_at": None,
            "expires_at": expires_at,
            "revoked_at": None,
            "created_by": ObjectId(created_by) if created_by else None,
            "created_at": datetime.utcnow(),
            "metadata": metadata or {},
        }
        result = await col.insert_one(doc)
        doc["_id"] = result.inserted_id
        return plaintext, self._serialize(doc)

    async def list(
        self,
        *,
        organization_id: str,
        include_revoked: bool = False,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        q: Dict[str, Any] = {"organization_id": ObjectId(organization_id)}
        if not include_revoked:
            q["revoked_at"] = None
        cursor = col.find(q).sort("created_at", -1).limit(limit)
        return [self._serialize(r) for r in await cursor.to_list(length=limit)]

    async def get(self, token_id: str, organization_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        row = await col.find_one({
            "_id": ObjectId(token_id),
            "organization_id": ObjectId(organization_id),
        })
        return self._serialize(row) if row else None

    async def revoke(self, token_id: str, organization_id: str) -> bool:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        result = await col.update_one(
            {"_id": ObjectId(token_id), "organization_id": ObjectId(organization_id)},
            {"$set": {"revoked_at": datetime.utcnow()}},
        )
        return result.modified_count > 0

    async def rotate(
        self,
        token_id: str,
        organization_id: str,
    ) -> Optional[Tuple[str, Dict[str, Any]]]:
        """Issue a new secret for the same token row, mark old as rotated."""
        await self._ensure_indexes()
        existing = await self.get(token_id, organization_id)
        if not existing:
            return None
        plaintext, prefix = generate_plaintext_token()
        col = await MongoDB.get_collection(COLLECTION)
        await col.update_one(
            {"_id": ObjectId(token_id)},
            {"$set": {
                "prefix": prefix,
                "token_hash": _hash(plaintext),
                "last_used_at": None,
            }},
        )
        existing["prefix"] = prefix
        return plaintext, existing

    async def lookup_by_plaintext(self, plaintext: str) -> Optional[Dict[str, Any]]:
        """Find an active token by its plaintext.  Updates last_used_at."""
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        row = await col.find_one({"token_hash": _hash(plaintext), "revoked_at": None})
        if not row:
            return None
        if row.get("expires_at") and row["expires_at"] <= datetime.utcnow():
            return None
        await col.update_one({"_id": row["_id"]}, {"$set": {"last_used_at": datetime.utcnow()}})
        return self._serialize(row)

    @staticmethod
    def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc:
            return doc
        d = dict(doc)
        d["id"] = str(d.pop("_id"))
        # Do NOT surface token_hash in API responses.
        d.pop("token_hash", None)
        for k in ("organization_id", "user_id", "created_by"):
            if d.get(k) is not None:
                d[k] = str(d[k])
        return d


api_tokens_repository = ApiTokensRepository()
