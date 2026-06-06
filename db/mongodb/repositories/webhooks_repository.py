"""
WebhooksRepository + WebhookDeliveriesRepository.

Outbound webhooks per organization.  Delivery rows are written by the
`backend/services/webhook_dispatcher.py` worker (Phase E follow-up) which
retries with exponential backoff on failure.
"""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import structlog
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from backend.db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)

WEBHOOKS = "webhooks"
DELIVERIES = "webhook_deliveries"


def _hash(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_secret() -> Tuple[str, str]:
    """Returns (plaintext_secret, secret_hash)."""
    plaintext = "whsec_" + secrets.token_urlsafe(32)
    return plaintext, _hash(plaintext)


class WebhooksRepository:
    def __init__(self) -> None:
        self._initialised = False

    async def _ensure_indexes(self) -> None:
        if self._initialised:
            return
        wh = await MongoDB.get_collection(WEBHOOKS)
        dv = await MongoDB.get_collection(DELIVERIES)
        await wh.create_index([("organization_id", ASCENDING), ("enabled", ASCENDING)])
        await wh.create_index("events")
        await dv.create_index([("webhook_id", ASCENDING), ("created_at", DESCENDING)])
        await dv.create_index([("status", ASCENDING), ("next_attempt_at", ASCENDING)])
        self._initialised = True

    async def create(
        self,
        *,
        organization_id: str,
        url: str,
        events: List[str],
        description: Optional[str] = None,
        created_by: Optional[str] = None,
    ) -> Tuple[str, Dict[str, Any]]:
        await self._ensure_indexes()
        plaintext, h = generate_secret()
        col = await MongoDB.get_collection(WEBHOOKS)
        now = datetime.utcnow()
        doc = {
            "organization_id": ObjectId(organization_id),
            "url": url,
            "events": list(events or []),
            "secret_hash": h,
            "enabled": True,
            "last_delivery_at": None,
            "failure_count": 0,
            "description": description,
            "created_by": ObjectId(created_by) if created_by else None,
            "created_at": now,
            "updated_at": now,
        }
        result = await col.insert_one(doc)
        doc["_id"] = result.inserted_id
        return plaintext, self._serialize(doc)

    async def list(
        self,
        *,
        organization_id: str,
        enabled_only: bool = False,
        limit: int = 200,
    ) -> List[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(WEBHOOKS)
        q: Dict[str, Any] = {"organization_id": ObjectId(organization_id)}
        if enabled_only:
            q["enabled"] = True
        cursor = col.find(q).sort("created_at", -1).limit(limit)
        return [self._serialize(r) for r in await cursor.to_list(length=limit)]

    async def get(self, webhook_id: str, organization_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(WEBHOOKS)
        row = await col.find_one({
            "_id": ObjectId(webhook_id),
            "organization_id": ObjectId(organization_id),
        })
        return self._serialize(row) if row else None

    async def update(
        self,
        webhook_id: str,
        organization_id: str,
        *,
        patch: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        col = await MongoDB.get_collection(WEBHOOKS)
        update = {k: v for k, v in patch.items() if v is not None}
        if not update:
            return await self.get(webhook_id, organization_id)
        update["updated_at"] = datetime.utcnow()
        row = await col.find_one_and_update(
            {"_id": ObjectId(webhook_id), "organization_id": ObjectId(organization_id)},
            {"$set": update},
            return_document=True,
        )
        return self._serialize(row) if row else None

    async def delete(self, webhook_id: str, organization_id: str) -> bool:
        col = await MongoDB.get_collection(WEBHOOKS)
        result = await col.delete_one({
            "_id": ObjectId(webhook_id),
            "organization_id": ObjectId(organization_id),
        })
        return result.deleted_count > 0

    async def rotate_secret(self, webhook_id: str, organization_id: str) -> Optional[Tuple[str, Dict[str, Any]]]:
        existing = await self.get(webhook_id, organization_id)
        if not existing:
            return None
        plaintext, h = generate_secret()
        col = await MongoDB.get_collection(WEBHOOKS)
        await col.update_one(
            {"_id": ObjectId(webhook_id)},
            {"$set": {"secret_hash": h, "updated_at": datetime.utcnow()}},
        )
        return plaintext, existing

    # ---- deliveries ----

    async def record_delivery(
        self,
        *,
        webhook_id: str,
        organization_id: str,
        event: str,
        payload: Dict[str, Any],
        status: str = "pending",
    ) -> Dict[str, Any]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(DELIVERIES)
        doc = {
            "webhook_id": ObjectId(webhook_id),
            "organization_id": ObjectId(organization_id),
            "event": event,
            "payload": payload,
            "status": status,
            "attempts": 0,
            "response_status": None,
            "response_body": None,
            "error": None,
            "next_attempt_at": datetime.utcnow(),
            "delivered_at": None,
            "created_at": datetime.utcnow(),
        }
        result = await col.insert_one(doc)
        doc["_id"] = result.inserted_id
        return self._serialize_delivery(doc)

    async def list_deliveries(
        self,
        *,
        webhook_id: str,
        organization_id: str,
        status: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(DELIVERIES)
        q: Dict[str, Any] = {
            "webhook_id": ObjectId(webhook_id),
            "organization_id": ObjectId(organization_id),
        }
        if status:
            q["status"] = status
        cursor = col.find(q).sort("created_at", -1).skip(skip).limit(limit)
        return [self._serialize_delivery(r) for r in await cursor.to_list(length=limit)]

    async def mark_delivery_complete(
        self,
        delivery_id: str,
        *,
        status: str,
        response_status: Optional[int] = None,
        response_body: Optional[str] = None,
        error: Optional[str] = None,
        next_attempt_at: Optional[datetime] = None,
    ) -> None:
        col = await MongoDB.get_collection(DELIVERIES)
        set_doc: Dict[str, Any] = {
            "status": status,
            "response_status": response_status,
            "response_body": (response_body or "")[:2000],
            "error": error,
            "next_attempt_at": next_attempt_at,
        }
        if status == "success":
            set_doc["delivered_at"] = datetime.utcnow()
        await col.update_one(
            {"_id": ObjectId(delivery_id)},
            {"$set": set_doc, "$inc": {"attempts": 1}},
        )

    @staticmethod
    def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc:
            return doc
        d = dict(doc)
        d["id"] = str(d.pop("_id"))
        d.pop("secret_hash", None)
        for k in ("organization_id", "created_by"):
            if d.get(k) is not None:
                d[k] = str(d[k])
        return d

    @staticmethod
    def _serialize_delivery(doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc:
            return doc
        d = dict(doc)
        d["id"] = str(d.pop("_id"))
        for k in ("organization_id", "webhook_id"):
            if d.get(k) is not None:
                d[k] = str(d[k])
        return d


webhooks_repository = WebhooksRepository()
