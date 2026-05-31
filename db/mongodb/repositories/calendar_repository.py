"""
Calendar repository — production-grade CRUD over `lumicoria_calendar_events`.

Tenancy: every read is scoped by `owner_user_id` (and optionally
`organization_id`).  ID coercion mirrors the task repository so we accept
ObjectId strings, UUIDs, and Firebase UIDs without crashing.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from backend.db.mongodb.models.calendar_event import (
    CalendarEvent,
    CalendarEventCreate,
    CalendarEventSource,
    CalendarEventStatus,
    CalendarEventUpdate,
)
from backend.db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)

COLLECTION_NAME = "lumicoria_calendar_events"


def _coerce_oid(value: Any) -> Any:
    """ObjectId if possible, else the original string."""
    if value is None:
        return None
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value))
    except Exception:
        return str(value)


class CalendarRepository:
    """Repository for `lumicoria_calendar_events`."""

    def __init__(self) -> None:
        self._collection = None

    async def _get_collection(self):
        if self._collection is None:
            self._collection = await MongoDB.get_collection(COLLECTION_NAME)
            await self._create_indexes()
        return self._collection

    async def _create_indexes(self) -> None:
        col = self._collection
        await col.create_index("owner_user_id")
        await col.create_index("organization_id")
        await col.create_index("task_id")
        await col.create_index("project_id")
        await col.create_index("gcal_event_id")
        await col.create_index([("owner_user_id", ASCENDING), ("start", ASCENDING)])
        await col.create_index([
            ("owner_user_id", ASCENDING),
            ("start", ASCENDING),
            ("end", ASCENDING),
        ])
        await col.create_index([("start", ASCENDING), ("end", ASCENDING)])
        await col.create_index("status")
        await col.create_index("source")
        await col.create_index("deleted_at")
        # Text search on title + description
        try:
            await col.create_index([("title", "text"), ("description", "text")])
        except Exception:
            # Mongo allows only one text index; ignore if it already exists
            pass

    # ── CRUD ───────────────────────────────────────────────────────────

    async def create(
        self,
        event: CalendarEventCreate,
        owner_user_id: str,
        organization_id: Optional[str] = None,
    ) -> CalendarEvent:
        """Insert a new calendar event."""
        if event.start > event.end:
            raise ValueError("Calendar event `start` must be before `end`")
        now = datetime.utcnow()
        doc: Dict[str, Any] = event.model_dump(exclude={"sync_to_google"})
        doc["owner_user_id"] = _coerce_oid(owner_user_id)
        if organization_id:
            doc["organization_id"] = _coerce_oid(organization_id)
        if doc.get("task_id"):
            doc["task_id"] = _coerce_oid(doc["task_id"])
        if doc.get("project_id"):
            doc["project_id"] = _coerce_oid(doc["project_id"])
        doc["status"] = CalendarEventStatus.SCHEDULED.value
        doc["created_at"] = now
        doc["updated_at"] = now

        col = await self._get_collection()
        result = await col.insert_one(doc)
        created = await col.find_one({"_id": result.inserted_id})
        if not created:
            raise RuntimeError("Calendar event insert succeeded but find_one returned None")
        return CalendarEvent(**created)

    async def get_by_id(
        self,
        event_id: str,
        owner_user_id: Optional[str] = None,
    ) -> Optional[CalendarEvent]:
        oid = _coerce_oid(event_id)
        if not isinstance(oid, ObjectId):
            return None
        col = await self._get_collection()
        query: Dict[str, Any] = {"_id": oid, "deleted_at": None}
        if owner_user_id:
            query["owner_user_id"] = _coerce_oid(owner_user_id)
        doc = await col.find_one(query)
        return CalendarEvent(**doc) if doc else None

    async def get_by_task_id(
        self,
        task_id: str,
        owner_user_id: Optional[str] = None,
    ) -> Optional[CalendarEvent]:
        """Find the auto-created event for a task (one event per task)."""
        col = await self._get_collection()
        query: Dict[str, Any] = {"task_id": _coerce_oid(task_id), "deleted_at": None}
        if owner_user_id:
            query["owner_user_id"] = _coerce_oid(owner_user_id)
        doc = await col.find_one(query)
        return CalendarEvent(**doc) if doc else None

    async def update(
        self,
        event_id: str,
        update: CalendarEventUpdate | Dict[str, Any],
        owner_user_id: Optional[str] = None,
    ) -> Optional[CalendarEvent]:
        oid = _coerce_oid(event_id)
        if not isinstance(oid, ObjectId):
            return None

        # Accept either a Pydantic model or a raw dict
        if hasattr(update, "model_dump"):
            patch = update.model_dump(exclude_none=True)
        else:
            patch = {k: v for k, v in (update or {}).items() if v is not None}

        # Don't allow caller to mutate identity / soft-delete fields
        for forbidden in ("_id", "id", "owner_user_id", "created_at", "deleted_at"):
            patch.pop(forbidden, None)
        if not patch:
            return await self.get_by_id(event_id, owner_user_id)

        # Sanity: start must remain before end
        new_start = patch.get("start")
        new_end = patch.get("end")
        if new_start and new_end and new_start > new_end:
            raise ValueError("Calendar event `start` must be before `end`")

        patch["updated_at"] = datetime.utcnow()

        col = await self._get_collection()
        query: Dict[str, Any] = {"_id": oid, "deleted_at": None}
        if owner_user_id:
            query["owner_user_id"] = _coerce_oid(owner_user_id)

        result = await col.find_one_and_update(query, {"$set": patch}, return_document=True)
        return CalendarEvent(**result) if result else None

    async def soft_delete(
        self,
        event_id: str,
        owner_user_id: Optional[str] = None,
    ) -> bool:
        """Soft-delete preserves history (useful for audit + Google de-sync)."""
        oid = _coerce_oid(event_id)
        if not isinstance(oid, ObjectId):
            return False
        col = await self._get_collection()
        query: Dict[str, Any] = {"_id": oid, "deleted_at": None}
        if owner_user_id:
            query["owner_user_id"] = _coerce_oid(owner_user_id)
        now = datetime.utcnow()
        result = await col.update_one(
            query,
            {"$set": {"deleted_at": now, "updated_at": now, "status": CalendarEventStatus.CANCELLED.value}},
        )
        return result.modified_count > 0

    async def hard_delete(
        self,
        event_id: str,
        owner_user_id: Optional[str] = None,
    ) -> bool:
        """Hard delete — only used for orphan cleanup / GDPR."""
        oid = _coerce_oid(event_id)
        if not isinstance(oid, ObjectId):
            return False
        col = await self._get_collection()
        query: Dict[str, Any] = {"_id": oid}
        if owner_user_id:
            query["owner_user_id"] = _coerce_oid(owner_user_id)
        result = await col.delete_one(query)
        return result.deleted_count > 0

    # ── Range queries (the heart of the calendar UI) ───────────────────

    async def list_in_range(
        self,
        owner_user_id: str,
        start: datetime,
        end: datetime,
        organization_id: Optional[str] = None,
        include_completed: bool = True,
        sources: Optional[List[str]] = None,
        limit: int = 1000,
    ) -> List[CalendarEvent]:
        """All events overlapping the [start, end) window for an owner.

        An event overlaps the window iff `event.start < end AND event.end > start`.
        """
        col = await self._get_collection()
        query: Dict[str, Any] = {
            "owner_user_id": _coerce_oid(owner_user_id),
            "deleted_at": None,
            "start": {"$lt": end},
            "end": {"$gt": start},
        }
        if organization_id:
            query["organization_id"] = _coerce_oid(organization_id)
        if not include_completed:
            query["status"] = {"$ne": CalendarEventStatus.COMPLETED.value}
        if sources:
            query["source"] = {"$in": sources}

        cursor = col.find(query).sort("start", ASCENDING).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [CalendarEvent(**d) for d in docs]

    async def list_today(
        self,
        owner_user_id: str,
        timezone: str = "UTC",  # noqa: ARG002 — interpreted at the API layer
    ) -> List[CalendarEvent]:
        now = datetime.utcnow()
        start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)
        return await self.list_in_range(owner_user_id, start, end)

    async def list_upcoming(
        self,
        owner_user_id: str,
        days: int = 7,
    ) -> List[CalendarEvent]:
        now = datetime.utcnow()
        return await self.list_in_range(owner_user_id, now, now + timedelta(days=days))

    # ── Linkage helpers (task ↔ event bridge in Phase 3) ───────────────

    async def unlink_task(self, event_id: str) -> bool:
        """Detach an event from a task without deleting either."""
        oid = _coerce_oid(event_id)
        if not isinstance(oid, ObjectId):
            return False
        col = await self._get_collection()
        result = await col.update_one(
            {"_id": oid},
            {"$set": {"task_id": None, "updated_at": datetime.utcnow()}},
        )
        return result.modified_count > 0

    async def link_to_gcal(
        self,
        event_id: str,
        gcal_event_id: str,
        gcal_calendar_id: str = "primary",
    ) -> bool:
        """Stamp the Google Calendar ids on a mirrored event."""
        oid = _coerce_oid(event_id)
        if not isinstance(oid, ObjectId):
            return False
        col = await self._get_collection()
        result = await col.update_one(
            {"_id": oid},
            {"$set": {
                "gcal_event_id": gcal_event_id,
                "gcal_calendar_id": gcal_calendar_id,
                "last_synced_at": datetime.utcnow(),
                "updated_at": datetime.utcnow(),
            }},
        )
        return result.modified_count > 0

    async def find_by_gcal_id(self, gcal_event_id: str) -> Optional[CalendarEvent]:
        """Reverse lookup — used by incoming Google webhook sync (Phase 3)."""
        col = await self._get_collection()
        doc = await col.find_one({"gcal_event_id": gcal_event_id, "deleted_at": None})
        return CalendarEvent(**doc) if doc else None


# Module-level singleton.
calendar_repository = CalendarRepository()
