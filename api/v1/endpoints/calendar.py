"""
Calendar endpoints — Lumicoria-native calendar API.

Mounted at /api/v1/calendar.

Endpoints:
    GET    /events                 — list events in [start, end) window
    GET    /events/today           — today's events (server-local TZ for now)
    GET    /events/upcoming        — next N days
    GET    /events/{id}            — fetch one event
    POST   /events                 — create a manual event
    PUT    /events/{id}            — patch an event
    DELETE /events/{id}            — soft-delete an event
    POST   /events/from-task/{task_id}     — explicit create-from-task
    POST   /events/{id}/sync/google         — sync this event to Google
    POST   /sync/google                     — bulk sync upcoming events

Tenancy: every endpoint scopes to `current_user.id` as the calendar owner.
Org-shared calendars are a Phase 8 addition (Organizations API).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.api.deps import get_current_active_user
from backend.db.mongodb.models.calendar_event import (
    CalendarEvent,
    CalendarEventCreate,
    CalendarEventSource,
    CalendarEventUpdate,
)
from backend.db.mongodb.repositories.task_repository import task_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.calendar_service import calendar_service

logger = structlog.get_logger(__name__)

router = APIRouter()


def _org_id(user: User) -> Optional[str]:
    return getattr(user, "organization_id", None)


def _serialize(event: CalendarEvent) -> Dict[str, Any]:
    """JSON-safe shape the frontend expects."""
    data = event.model_dump(by_alias=True) if hasattr(event, "model_dump") else dict(event)
    # Stringify ObjectId fields
    for field in ("_id", "id", "owner_user_id", "organization_id", "task_id", "project_id"):
        if field in data and data[field] is not None:
            data[field] = str(data[field])
    # Normalise id alias
    if "_id" in data and "id" not in data:
        data["id"] = data.pop("_id")
    elif "_id" in data:
        data.pop("_id")
    # Enums → values
    for enum_field in ("source", "status"):
        v = data.get(enum_field)
        if hasattr(v, "value"):
            data[enum_field] = v.value
    # Datetimes → ISO
    for dt_field in ("start", "end", "created_at", "updated_at", "deleted_at", "last_synced_at"):
        v = data.get(dt_field)
        if isinstance(v, datetime):
            data[dt_field] = v.isoformat()
    return data


# ── Range / discovery ──────────────────────────────────────────────────

@router.get("/events", response_model=None)
async def list_events(
    start: datetime = Query(..., description="Start of the window (inclusive)"),
    end: datetime = Query(..., description="End of the window (exclusive)"),
    include_completed: bool = Query(True),
    sources: Optional[List[str]] = Query(None, description="Filter by source"),
    current_user: User = Depends(get_current_active_user),
) -> List[Dict[str, Any]]:
    """Events overlapping [start, end) for the current user."""
    if start >= end:
        raise HTTPException(status_code=400, detail="`start` must be before `end`")
    if (end - start) > timedelta(days=370):
        raise HTTPException(status_code=400, detail="Range too large (max 370 days)")
    events = await calendar_service.list_in_range(
        owner_user_id=str(current_user.id),
        start=start,
        end=end,
        organization_id=_org_id(current_user),
        include_completed=include_completed,
        sources=sources,
    )
    return [_serialize(e) for e in events]


@router.get("/events/today", response_model=None)
async def list_today(
    current_user: User = Depends(get_current_active_user),
) -> List[Dict[str, Any]]:
    events = await calendar_service.list_today(str(current_user.id))
    return [_serialize(e) for e in events]


@router.get("/events/upcoming", response_model=None)
async def list_upcoming(
    days: int = Query(7, ge=1, le=90),
    current_user: User = Depends(get_current_active_user),
) -> List[Dict[str, Any]]:
    events = await calendar_service.list_upcoming(str(current_user.id), days=days)
    return [_serialize(e) for e in events]


@router.get("/events/{event_id}", response_model=None)
async def get_event(
    event_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    event = await calendar_service.get_event(event_id, owner_user_id=str(current_user.id))
    if not event:
        raise HTTPException(status_code=404, detail="Calendar event not found")
    return _serialize(event)


# ── Mutations ─────────────────────────────────────────────────────────

@router.post("/events", response_model=None, status_code=status.HTTP_201_CREATED)
async def create_event(
    payload: CalendarEventCreate,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    try:
        event = await calendar_service.create_event(
            payload, owner_user_id=str(current_user.id), organization_id=_org_id(current_user)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    await log_activity(
        user_id=str(current_user.id),
        organization_id=_org_id(current_user) or str(current_user.id),
        activity_type="calendar.event_created",
        details={"event_id": str(event.id), "title": event.title, "source": event.source.value if hasattr(event.source, "value") else event.source},
        related_resource_type="CALENDAR_EVENT",
        related_resource_id=str(event.id),
    )
    return _serialize(event)


@router.put("/events/{event_id}", response_model=None)
async def update_event(
    event_id: str,
    update: CalendarEventUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    try:
        event = await calendar_service.update_event(
            event_id, update, owner_user_id=str(current_user.id)
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    if not event:
        raise HTTPException(status_code=404, detail="Calendar event not found")
    await log_activity(
        user_id=str(current_user.id),
        organization_id=_org_id(current_user) or str(current_user.id),
        activity_type="calendar.event_updated",
        details={
            "event_id": event_id,
            "updated_fields": list(update.model_dump(exclude_none=True).keys()),
        },
        related_resource_type="CALENDAR_EVENT",
        related_resource_id=event_id,
    )
    return _serialize(event)


@router.delete("/events/{event_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_event(
    event_id: str,
    current_user: User = Depends(get_current_active_user),
) -> None:
    ok = await calendar_service.delete_event(event_id, owner_user_id=str(current_user.id))
    if not ok:
        raise HTTPException(status_code=404, detail="Calendar event not found")
    await log_activity(
        user_id=str(current_user.id),
        organization_id=_org_id(current_user) or str(current_user.id),
        activity_type="calendar.event_deleted",
        details={"event_id": event_id},
        related_resource_type="CALENDAR_EVENT",
        related_resource_id=event_id,
    )


# ── Task ↔ Calendar bridge endpoints ──────────────────────────────────

@router.post("/events/from-task/{task_id}", response_model=None, status_code=status.HTTP_201_CREATED)
async def create_event_from_task(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Idempotent: creates the event if missing, updates if present.

    Used when a user manually wants to push an existing task to the calendar
    (e.g., the task was created before the calendar module existed).
    """
    org_id = _org_id(current_user) or str(current_user.id)
    task = await task_repository.get_task_by_id(task_id, organization_id=org_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    if not getattr(task, "due_date", None):
        raise HTTPException(status_code=400, detail="Task has no due_date — nothing to schedule")

    event = await calendar_service.create_event_for_task(
        task, owner_user_id=str(current_user.id)
    )
    if not event:
        raise HTTPException(status_code=500, detail="Failed to create event from task")

    # Stamp the task with the link so the Tasks UI can show "On calendar"
    try:
        await task_repository.update_task(
            task_id=task_id,
            organization_id=org_id,
            update_data={"calendar_event_id": str(event.id)},
        )
    except Exception as e:
        logger.warning("calendar_event link write-back failed", task_id=task_id, error=str(e))

    return _serialize(event)


# ── Google sync (opt-in, no-op in Phase 2) ────────────────────────────

@router.post("/events/{event_id}/sync/google", response_model=None)
async def sync_event_to_google(
    event_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Mirror a single event to Google Calendar.

    Returns a structured `{ synced: bool, reason?: str }` payload.  Does not
    raise when the user hasn't connected Google — that's the expected state.
    """
    result = await calendar_service.sync_event_to_google(event_id, str(current_user.id))
    if result.get("reason") == "event_not_found":
        raise HTTPException(status_code=404, detail="Calendar event not found")
    return result


@router.post("/sync/google", response_model=None)
async def sync_all_to_google(
    days_ahead: int = Query(30, ge=1, le=90),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Bulk mirror upcoming events to Google.  Idempotent."""
    return await calendar_service.sync_all_events_to_google(
        user_id=str(current_user.id), days_ahead=days_ahead
    )
