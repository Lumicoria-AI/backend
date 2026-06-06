"""
RemindersRepository — cross-resource reminders.

Reminders attach to any resource (task, project, document, custom) and fire
through one or more channels (email / push / in_app).  The dedicated worker
(see Phase C beat schedule) sweeps `state=pending AND due_at <= now`,
delivers, and flips the row to `sent`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING

from backend.db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)

COLLECTION = "reminders"


class RemindersRepository:
    def __init__(self) -> None:
        self._initialised = False

    async def _ensure_indexes(self) -> None:
        if self._initialised:
            return
        col = await MongoDB.get_collection(COLLECTION)
        await col.create_index([("user_id", ASCENDING), ("state", ASCENDING), ("due_at", ASCENDING)])
        await col.create_index([("organization_id", ASCENDING), ("resource_type", ASCENDING), ("resource_id", ASCENDING)])
        await col.create_index([("state", ASCENDING), ("due_at", ASCENDING)])
        self._initialised = True

    async def create(
        self,
        *,
        organization_id: str,
        user_id: str,
        resource_type: str,
        resource_id: str,
        due_at: datetime,
        channels: List[str],
        note: Optional[str] = None,
        recur_cron: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        doc = {
            "organization_id": ObjectId(organization_id),
            "user_id": ObjectId(user_id),
            "resource_type": resource_type,
            "resource_id": str(resource_id),
            "due_at": due_at,
            "channels": list(channels or ["in_app"]),
            "note": note,
            "recur_cron": recur_cron,
            "state": "pending",
            "last_sent_at": None,
            "send_count": 0,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        result = await col.insert_one(doc)
        doc["_id"] = result.inserted_id
        return self._serialize(doc)

    async def list_for_user(
        self,
        *,
        user_id: str,
        organization_id: Optional[str] = None,
        state: Optional[str] = None,
        upcoming_only: bool = False,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        query: Dict[str, Any] = {"user_id": ObjectId(user_id)}
        if organization_id:
            query["organization_id"] = ObjectId(organization_id)
        if state:
            query["state"] = state
        if upcoming_only:
            query["state"] = "pending"
            query["due_at"] = {"$gte": datetime.utcnow()}
        cursor = col.find(query).sort("due_at", ASCENDING).skip(skip).limit(limit)
        return [self._serialize(r) for r in await cursor.to_list(length=limit)]

    async def list_due(self, *, now: Optional[datetime] = None, limit: int = 500) -> List[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        cursor = col.find({"state": "pending", "due_at": {"$lte": now or datetime.utcnow()}}).limit(limit)
        return [self._serialize(r) for r in await cursor.to_list(length=limit)]

    async def get(self, reminder_id: str) -> Optional[Dict[str, Any]]:
        col = await MongoDB.get_collection(COLLECTION)
        row = await col.find_one({"_id": ObjectId(reminder_id)})
        return self._serialize(row) if row else None

    async def update(
        self,
        reminder_id: str,
        *,
        due_at: Optional[datetime] = None,
        channels: Optional[List[str]] = None,
        note: Optional[str] = None,
        state: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        col = await MongoDB.get_collection(COLLECTION)
        patch: Dict[str, Any] = {"updated_at": datetime.utcnow()}
        if due_at is not None:
            patch["due_at"] = due_at
        if channels is not None:
            patch["channels"] = list(channels)
        if note is not None:
            patch["note"] = note
        if state is not None:
            patch["state"] = state
        row = await col.find_one_and_update(
            {"_id": ObjectId(reminder_id)},
            {"$set": patch},
            return_document=True,
        )
        return self._serialize(row) if row else None

    async def mark_sent(self, reminder_id: str, *, next_due_at: Optional[datetime] = None) -> None:
        col = await MongoDB.get_collection(COLLECTION)
        update: Dict[str, Any] = {
            "$set": {"last_sent_at": datetime.utcnow(), "updated_at": datetime.utcnow()},
            "$inc": {"send_count": 1},
        }
        if next_due_at:
            update["$set"]["due_at"] = next_due_at
            update["$set"]["state"] = "pending"
        else:
            update["$set"]["state"] = "sent"
        await col.update_one({"_id": ObjectId(reminder_id)}, update)

    async def delete(self, reminder_id: str) -> bool:
        col = await MongoDB.get_collection(COLLECTION)
        result = await col.delete_one({"_id": ObjectId(reminder_id)})
        return result.deleted_count > 0

    @staticmethod
    def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc:
            return doc
        d = dict(doc)
        d["id"] = str(d.pop("_id"))
        for k in ("organization_id", "user_id"):
            if d.get(k) is not None:
                d[k] = str(d[k])
        return d


reminders_repository = RemindersRepository()
