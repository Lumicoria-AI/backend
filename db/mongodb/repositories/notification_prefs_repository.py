"""
NotificationPreferencesRepository — per-user, per-channel × per-category prefs.

Stores rows of the form:
    user_id, organization_id?, channel (email|push|in_app), category, enabled,
    quiet_hours { start, end, timezone }
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING

from backend.db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)

COLLECTION = "notification_preferences"

# Canonical category catalogue. Extend as new event types ship.
CATEGORIES = [
    "task.assigned",
    "task.completed",
    "task.proposal_review",
    "task.reminder",
    "task.comment",
    "task.mention",
    "project.member_added",
    "project.activity_digest",
    "team.member_added",
    "agent.run_completed",
    "agent.run_failed",
    "agent.handoff",
    "org.billing",
    "org.audit",
    "wellbeing",
    "system",
]

CHANNELS = ["email", "push", "in_app"]


class NotificationPreferencesRepository:
    def __init__(self) -> None:
        self._initialised = False

    async def _ensure_indexes(self) -> None:
        if self._initialised:
            return
        col = await MongoDB.get_collection(COLLECTION)
        await col.create_index(
            [("user_id", ASCENDING), ("channel", ASCENDING), ("category", ASCENDING)],
            unique=True,
        )
        self._initialised = True

    async def get_all(self, user_id: str) -> List[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        cursor = col.find({"user_id": ObjectId(user_id)})
        rows = await cursor.to_list(length=500)
        existing = {(r["channel"], r["category"]): r for r in rows}
        # Fill in defaults for missing combinations so the UI gets a complete grid.
        out: List[Dict[str, Any]] = []
        for ch in CHANNELS:
            for cat in CATEGORIES:
                key = (ch, cat)
                if key in existing:
                    out.append(self._serialize(existing[key]))
                else:
                    out.append({
                        "user_id": user_id, "channel": ch, "category": cat,
                        "enabled": True, "quiet_hours": None, "id": None,
                    })
        return out

    async def upsert(
        self,
        *,
        user_id: str,
        channel: str,
        category: str,
        enabled: Optional[bool] = None,
        quiet_hours: Optional[Dict[str, Any]] = None,
        organization_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        await self._ensure_indexes()
        if channel not in CHANNELS:
            raise ValueError(f"Unknown channel {channel}")
        col = await MongoDB.get_collection(COLLECTION)
        now = datetime.utcnow()
        set_doc: Dict[str, Any] = {"updated_at": now}
        if enabled is not None:
            set_doc["enabled"] = bool(enabled)
        if quiet_hours is not None:
            set_doc["quiet_hours"] = quiet_hours
        if organization_id is not None:
            set_doc["organization_id"] = ObjectId(organization_id)
        filt = {"user_id": ObjectId(user_id), "channel": channel, "category": category}
        await col.update_one(
            filt,
            {"$set": set_doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        row = await col.find_one(filt)
        return self._serialize(row)

    async def is_enabled(
        self,
        *,
        user_id: str,
        channel: str,
        category: str,
    ) -> bool:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        row = await col.find_one({"user_id": ObjectId(user_id), "channel": channel, "category": category})
        if not row:
            return True  # default-on
        return bool(row.get("enabled", True))

    @staticmethod
    def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc:
            return doc
        d = dict(doc)
        d["id"] = str(d.pop("_id"))
        if d.get("user_id") is not None:
            d["user_id"] = str(d["user_id"])
        if d.get("organization_id") is not None:
            d["organization_id"] = str(d["organization_id"])
        return d


notification_prefs_repository = NotificationPreferencesRepository()
