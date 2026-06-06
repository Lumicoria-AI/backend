"""
SsoRepository + ScimTokenRepository + DomainClaimRepository + SessionPolicyRepository.

Phase E governance storage.  The SAML handshake itself is stubbed in this
pass (config is captured, metadata.xml is generated, ACS endpoint accepts
the assertion blob and returns a TODO 501 if SP verification isn't fully
wired); SCIM 2.0 user CRUD is real.
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

logger = structlog.get_logger(__name__)

SSO_CONFIGS = "sso_configurations"
SCIM_TOKENS = "scim_tokens"
DOMAIN_CLAIMS = "domain_claims"
SESSION_POLICIES = "session_policies"


def _hash(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_scim_token() -> Tuple[str, str, str]:
    """Returns (plaintext, prefix, hash)."""
    body = secrets.token_urlsafe(32)
    plaintext = f"scim_{body}"
    return plaintext, plaintext[:12], _hash(plaintext)


# ── SSO config ───────────────────────────────────────────────────────


class SsoRepository:
    def __init__(self) -> None:
        self._initialised = False

    async def _ensure_indexes(self) -> None:
        if self._initialised:
            return
        col = await MongoDB.get_collection(SSO_CONFIGS)
        await col.create_index("organization_id", unique=True)
        self._initialised = True

    async def get(self, organization_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(SSO_CONFIGS)
        row = await col.find_one({"organization_id": ObjectId(organization_id)})
        return self._serialize(row) if row else None

    async def upsert(
        self,
        organization_id: str,
        *,
        provider: str = "saml",
        patch: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(SSO_CONFIGS)
        now = datetime.utcnow()
        set_doc = dict(patch or {})
        set_doc.update({
            "organization_id": ObjectId(organization_id),
            "provider": provider,
            "updated_at": now,
        })
        await col.update_one(
            {"organization_id": ObjectId(organization_id)},
            {"$set": set_doc, "$setOnInsert": {"created_at": now, "enabled": False}},
            upsert=True,
        )
        row = await col.find_one({"organization_id": ObjectId(organization_id)})
        return self._serialize(row)

    @staticmethod
    def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc:
            return doc
        d = dict(doc)
        d["id"] = str(d.pop("_id"))
        if d.get("organization_id") is not None:
            d["organization_id"] = str(d["organization_id"])
        return d


sso_repository = SsoRepository()


# ── SCIM tokens ─────────────────────────────────────────────────────


class ScimTokensRepository:
    def __init__(self) -> None:
        self._initialised = False

    async def _ensure_indexes(self) -> None:
        if self._initialised:
            return
        col = await MongoDB.get_collection(SCIM_TOKENS)
        await col.create_index("organization_id")
        await col.create_index("token_hash", unique=True)
        self._initialised = True

    async def create(
        self,
        *,
        organization_id: str,
        name: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        await self._ensure_indexes()
        plaintext, prefix, h = generate_scim_token()
        col = await MongoDB.get_collection(SCIM_TOKENS)
        doc = {
            "organization_id": ObjectId(organization_id),
            "token_hash": h,
            "prefix": prefix,
            "name": name or "SCIM token",
            "created_by": ObjectId(created_by) if created_by else None,
            "last_used_at": None,
            "revoked_at": None,
            "created_at": datetime.utcnow(),
        }
        result = await col.insert_one(doc)
        doc["_id"] = result.inserted_id
        return plaintext, self._serialize(doc)

    async def list(self, organization_id: str) -> List[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(SCIM_TOKENS)
        cursor = col.find({"organization_id": ObjectId(organization_id)}).sort("created_at", -1)
        return [self._serialize(r) for r in await cursor.to_list(length=200)]

    async def revoke(self, token_id: str, organization_id: str) -> bool:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(SCIM_TOKENS)
        result = await col.update_one(
            {"_id": ObjectId(token_id), "organization_id": ObjectId(organization_id)},
            {"$set": {"revoked_at": datetime.utcnow()}},
        )
        return result.modified_count > 0

    async def lookup_by_plaintext(self, plaintext: str) -> Optional[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(SCIM_TOKENS)
        row = await col.find_one({"token_hash": _hash(plaintext), "revoked_at": None})
        if not row:
            return None
        await col.update_one({"_id": row["_id"]}, {"$set": {"last_used_at": datetime.utcnow()}})
        return self._serialize(row)

    @staticmethod
    def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc:
            return doc
        d = dict(doc)
        d["id"] = str(d.pop("_id"))
        d.pop("token_hash", None)
        for k in ("organization_id", "created_by"):
            if d.get(k) is not None:
                d[k] = str(d[k])
        return d


scim_tokens_repository = ScimTokensRepository()


# ── Domain claims ────────────────────────────────────────────────────


class DomainClaimsRepository:
    def __init__(self) -> None:
        self._initialised = False

    async def _ensure_indexes(self) -> None:
        if self._initialised:
            return
        col = await MongoDB.get_collection(DOMAIN_CLAIMS)
        await col.create_index("domain", unique=True)
        await col.create_index("organization_id")
        self._initialised = True

    async def create(
        self,
        *,
        organization_id: str,
        domain: str,
        auto_join_role: str = "member",
        enforced: bool = False,
    ) -> Dict[str, Any]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(DOMAIN_CLAIMS)
        token = secrets.token_urlsafe(24)
        doc = {
            "organization_id": ObjectId(organization_id),
            "domain": domain.lower().strip(),
            "verification_token": token,
            "verified_at": None,
            "auto_join_role": auto_join_role,
            "enforced": bool(enforced),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        await col.update_one(
            {"domain": doc["domain"]},
            {"$set": doc},
            upsert=True,
        )
        row = await col.find_one({"domain": doc["domain"]})
        return self._serialize(row)

    async def list_for_org(self, organization_id: str) -> List[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(DOMAIN_CLAIMS)
        cursor = col.find({"organization_id": ObjectId(organization_id)})
        return [self._serialize(r) for r in await cursor.to_list(length=200)]

    async def verify(self, domain: str) -> Optional[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(DOMAIN_CLAIMS)
        row = await col.find_one_and_update(
            {"domain": domain.lower().strip()},
            {"$set": {"verified_at": datetime.utcnow(), "updated_at": datetime.utcnow()}},
            return_document=True,
        )
        return self._serialize(row) if row else None

    async def delete(self, domain: str, organization_id: str) -> bool:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(DOMAIN_CLAIMS)
        result = await col.delete_one({
            "domain": domain.lower().strip(),
            "organization_id": ObjectId(organization_id),
        })
        return result.deleted_count > 0

    async def find_for_email(self, email: str) -> Optional[Dict[str, Any]]:
        if "@" not in email:
            return None
        domain = email.split("@", 1)[1].lower()
        await self._ensure_indexes()
        col = await MongoDB.get_collection(DOMAIN_CLAIMS)
        row = await col.find_one({"domain": domain, "verified_at": {"$ne": None}})
        return self._serialize(row) if row else None

    @staticmethod
    def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc:
            return doc
        d = dict(doc)
        d["id"] = str(d.pop("_id"))
        if d.get("organization_id") is not None:
            d["organization_id"] = str(d["organization_id"])
        return d


domain_claims_repository = DomainClaimsRepository()


# ── Session policy ──────────────────────────────────────────────────


class SessionPolicyRepository:
    def __init__(self) -> None:
        self._initialised = False

    async def _ensure_indexes(self) -> None:
        if self._initialised:
            return
        col = await MongoDB.get_collection(SESSION_POLICIES)
        await col.create_index("organization_id", unique=True)
        self._initialised = True

    async def get(self, organization_id: str) -> Dict[str, Any]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(SESSION_POLICIES)
        row = await col.find_one({"organization_id": ObjectId(organization_id)})
        if not row:
            return {
                "organization_id": str(organization_id),
                "idle_timeout_minutes": 120,
                "max_sessions_per_user": 10,
                "require_mfa": False,
                "ip_allowlist_enabled": False,
                "ip_allowlist": [],
                "data_residency": "us",
                "cmk_enabled": False,
            }
        return self._serialize(row)

    async def upsert(self, organization_id: str, *, patch: Dict[str, Any], updated_by: Optional[str] = None) -> Dict[str, Any]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(SESSION_POLICIES)
        set_doc = {k: v for k, v in (patch or {}).items() if v is not None}
        set_doc["organization_id"] = ObjectId(organization_id)
        set_doc["updated_at"] = datetime.utcnow()
        if updated_by:
            set_doc["updated_by"] = ObjectId(updated_by)
        await col.update_one(
            {"organization_id": ObjectId(organization_id)},
            {"$set": set_doc},
            upsert=True,
        )
        return await self.get(organization_id)

    @staticmethod
    def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc:
            return doc
        d = dict(doc)
        d["id"] = str(d.pop("_id"))
        for k in ("organization_id", "updated_by"):
            if d.get(k) is not None:
                d[k] = str(d[k])
        return d


session_policy_repository = SessionPolicyRepository()
