"""
Calendar Service — business logic that sits between the Calendar REST API
and the calendar repository.

Responsibilities:
  • Create / update / delete calendar events from arbitrary callers.
  • Mirror task lifecycle into the calendar (create_event_for_task,
    update_event_for_task, mark_event_completed_for_task,
    delete_event_for_task).  Called from the /tasks endpoint hooks so the
    user never has to manually sync.
  • Provide a stable interface for Google Calendar mirroring.  The actual
    Google call is implemented in Phase 3 — this module exposes
    `sync_event_to_google(event_id, user_id)` as a no-op-friendly stub so
    the API surface is stable from Phase 2 onwards.

Design rules:
  • One Lumicoria calendar event per task (1-to-1).  Re-creating an event
    for the same task is idempotent — it updates the existing row.
  • Lumicoria-native first.  Google Calendar mirroring is opt-in and
    failure-tolerant — sync errors never break task operations.
  • Tenant isolation honoured at the repo layer (every read/write scoped
    to owner_user_id).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from backend.db.mongodb.models.calendar_event import (
    CalendarEvent,
    CalendarEventCreate,
    CalendarEventSource,
    CalendarEventStatus,
    CalendarEventUpdate,
)
from backend.db.mongodb.repositories.calendar_repository import calendar_repository

logger = structlog.get_logger(__name__)


# ── Priority → colour mapping (used when a task is the source) ────────────
# Production-tuned to the brand palette.  Keep in sync with the frontend
# Calendar.tsx defaults.
PRIORITY_COLOR: Dict[str, str] = {
    "low":      "#94A3B8",  # slate-400
    "medium":   "#6C4AB0",  # lumicoria purple (default)
    "high":     "#F59E0B",  # amber-500
    "critical": "#EF4444",  # red-500
}

# Default event duration when a task only has a due *time* and no start/end.
DEFAULT_TASK_EVENT_DURATION_MIN = 60


def _color_for_task(task: Dict[str, Any]) -> str:
    priority = (task.get("priority") or "medium").lower() if task else "medium"
    return PRIORITY_COLOR.get(priority, PRIORITY_COLOR["medium"])


def _task_id(task: Any) -> Optional[str]:
    """Extract the stringified id from either a Task model or a dict."""
    if task is None:
        return None
    if isinstance(task, dict):
        raw = task.get("id") or task.get("_id")
    else:
        raw = getattr(task, "id", None)
    return str(raw) if raw else None


def _task_field(task: Any, name: str, default: Any = None) -> Any:
    if isinstance(task, dict):
        return task.get(name, default)
    return getattr(task, name, default)


class CalendarService:
    """Application-level operations on the Lumicoria calendar."""

    # ── Manual / API-driven CRUD ───────────────────────────────────────

    async def create_event(
        self,
        payload: CalendarEventCreate,
        owner_user_id: str,
        organization_id: Optional[str] = None,
    ) -> CalendarEvent:
        """Create a manual or agent-sourced calendar event."""
        event = await calendar_repository.create(payload, owner_user_id, organization_id)
        logger.info(
            "calendar_event_created",
            event_id=str(event.id),
            owner=owner_user_id,
            source=event.source.value if hasattr(event.source, "value") else event.source,
        )
        # Opt-in Google mirror — best-effort, never raises.
        if payload.sync_to_google:
            await self.sync_event_to_google(str(event.id), owner_user_id)
        return event

    async def update_event(
        self,
        event_id: str,
        update: CalendarEventUpdate,
        owner_user_id: Optional[str] = None,
    ) -> Optional[CalendarEvent]:
        updated = await calendar_repository.update(event_id, update, owner_user_id)
        if updated and updated.gcal_event_id:
            # Bounce the change to Google if the event was previously mirrored.
            await self.sync_event_to_google(str(updated.id), str(updated.owner_user_id))
        return updated

    async def delete_event(
        self,
        event_id: str,
        owner_user_id: Optional[str] = None,
    ) -> bool:
        """Soft-delete.  Removes the mirrored Google event when present."""
        existing = await calendar_repository.get_by_id(event_id, owner_user_id)
        if not existing:
            return False
        ok = await calendar_repository.soft_delete(event_id, owner_user_id)
        if ok and existing.gcal_event_id:
            await self._delete_google_event(existing)
        return ok

    async def get_event(
        self,
        event_id: str,
        owner_user_id: Optional[str] = None,
    ) -> Optional[CalendarEvent]:
        return await calendar_repository.get_by_id(event_id, owner_user_id)

    async def list_in_range(
        self,
        owner_user_id: str,
        start: datetime,
        end: datetime,
        organization_id: Optional[str] = None,
        include_completed: bool = True,
        sources: Optional[List[str]] = None,
    ) -> List[CalendarEvent]:
        return await calendar_repository.list_in_range(
            owner_user_id=owner_user_id,
            start=start,
            end=end,
            organization_id=organization_id,
            include_completed=include_completed,
            sources=sources,
        )

    async def list_today(self, owner_user_id: str) -> List[CalendarEvent]:
        return await calendar_repository.list_today(owner_user_id)

    async def list_upcoming(
        self,
        owner_user_id: str,
        days: int = 7,
    ) -> List[CalendarEvent]:
        return await calendar_repository.list_upcoming(owner_user_id, days=days)

    # ── Task ↔ Calendar bridge (called by /tasks hooks) ────────────────

    async def create_event_for_task(
        self,
        task: Any,
        owner_user_id: Optional[str] = None,
    ) -> Optional[CalendarEvent]:
        """Create-or-update the single calendar event mirroring `task`.

        Returns the event (existing or newly created), or None when the task
        has no due_date (nothing to mirror).  Idempotent — calling twice on
        the same task does not duplicate events.
        """
        due_date: Optional[datetime] = _task_field(task, "due_date")
        if not due_date:
            return None

        task_id_str = _task_id(task)
        if not task_id_str:
            logger.warning("create_event_for_task: missing task id")
            return None

        owner = owner_user_id or str(
            _task_field(task, "assigned_to") or _task_field(task, "created_by") or ""
        )
        if not owner:
            logger.warning(
                "create_event_for_task: cannot infer owner_user_id",
                task_id=task_id_str,
            )
            return None

        org_id_raw = _task_field(task, "organization_id")
        organization_id = str(org_id_raw) if org_id_raw else None

        existing = await calendar_repository.get_by_task_id(task_id_str)
        title = _task_field(task, "title") or "Untitled task"
        description = _task_field(task, "description") or ""
        color = _color_for_task(task if isinstance(task, dict) else task.__dict__)
        start = due_date
        end = due_date + timedelta(minutes=DEFAULT_TASK_EVENT_DURATION_MIN)

        if existing:
            # Update path — only patch fields that may have changed.
            patch = CalendarEventUpdate(
                title=title,
                description=description,
                start=start,
                end=end,
                color=color,
                status=(
                    CalendarEventStatus.COMPLETED
                    if str(_task_field(task, "status") or "").lower() == "completed"
                    else CalendarEventStatus.SCHEDULED
                ),
            )
            updated = await calendar_repository.update(
                str(existing.id), patch, owner
            )
            return updated

        # Create path
        payload = CalendarEventCreate(
            title=title,
            description=description,
            start=start,
            end=end,
            all_day=False,
            color=color,
            timezone="UTC",
            task_id=task_id_str,
            project_id=_task_field(task, "project_id"),
            source=CalendarEventSource.TASK,
            metadata={
                "task_title_snapshot": title,
                "task_priority": _task_field(task, "priority"),
            },
            sync_to_google=False,
        )
        try:
            return await calendar_repository.create(payload, owner, organization_id)
        except Exception as e:  # pragma: no cover — observability only
            logger.error(
                "create_event_for_task failed",
                error=str(e),
                task_id=task_id_str,
            )
            return None

    async def update_event_for_task(
        self,
        task: Any,
        owner_user_id: Optional[str] = None,
    ) -> Optional[CalendarEvent]:
        """Mirror a task update onto its linked calendar event.

        Three legal transitions:
          • task gains a due_date     → create event
          • task loses its due_date   → soft-delete event
          • task changes due_date /
            title / priority / status → update event
        """
        task_id_str = _task_id(task)
        if not task_id_str:
            return None

        due_date: Optional[datetime] = _task_field(task, "due_date")
        existing = await calendar_repository.get_by_task_id(task_id_str)

        if due_date is None:
            if existing:
                await calendar_repository.soft_delete(str(existing.id))
            return None

        if not existing:
            return await self.create_event_for_task(task, owner_user_id=owner_user_id)

        return await self.create_event_for_task(task, owner_user_id=owner_user_id)

    async def mark_event_completed_for_task(self, task_id: str) -> Optional[CalendarEvent]:
        """Flip the linked event to COMPLETED when a task is marked done."""
        event = await calendar_repository.get_by_task_id(task_id)
        if not event:
            return None
        return await calendar_repository.update(
            str(event.id),
            CalendarEventUpdate(status=CalendarEventStatus.COMPLETED),
        )

    async def delete_event_for_task(self, task_id: str) -> bool:
        """Soft-delete the linked event when a task is removed."""
        event = await calendar_repository.get_by_task_id(task_id)
        if not event:
            return False
        return await calendar_repository.soft_delete(str(event.id))

    # ── Google Calendar mirror (Phase 3 stub — safe to call now) ───────

    async def sync_event_to_google(
        self,
        event_id: str,
        user_id: str,
    ) -> Dict[str, Any]:
        """Mirror a Lumicoria event onto Google Calendar.

        Phase 2: this method is intentionally a *graceful no-op*.  It looks
        up the user's google_workspace integration; if none, it returns
        `{"synced": False, "reason": "google_not_connected"}` without raising.
        Phase 3 plugs in the real Google call here.  Callers never have to
        feature-flag — they can always call sync_event_to_google.
        """
        event = await calendar_repository.get_by_id(event_id, owner_user_id=user_id)
        if not event:
            return {"synced": False, "reason": "event_not_found", "event_id": event_id}

        # Lazy import — keeps the integration_service dependency optional
        # in environments where Google isn't wired up at all.
        try:
            from backend.services.integration_service import integration_service
        except Exception as e:
            logger.warning("sync_event_to_google: integration_service unavailable", error=str(e))
            return {"synced": False, "reason": "integration_service_unavailable"}

        # Look for a connected google_workspace integration for this user.
        try:
            from backend.db.mongodb.repositories.integration_repository import integration_repository

            user_integrations = await integration_repository.get_user_integrations(
                user_id=user_id, integration_type="google_workspace"
            )
        except Exception as e:
            logger.debug("sync_event_to_google: integration lookup failed", error=str(e))
            user_integrations = []

        if not user_integrations:
            return {"synced": False, "reason": "google_not_connected"}

        # Phase 3 will replace this branch with the real
        # integration_service.execute_integration_action(...) call that
        # invokes GoogleWorkspaceIntegration.create_calendar_event().
        # We deliberately stop short here so the surface area is stable.
        logger.info(
            "google_calendar_sync_pending_phase3",
            event_id=event_id,
            user_id=user_id,
        )
        return {
            "synced": False,
            "reason": "google_sync_not_yet_implemented",
            "event_id": event_id,
            "note": "Lumicoria-native calendar already holds the event. Phase 3 will activate Google mirror.",
        }

    async def sync_all_events_to_google(
        self,
        user_id: str,
        days_ahead: int = 30,
    ) -> Dict[str, Any]:
        """Bulk mirror upcoming events to Google.  Returns per-event status."""
        events = await calendar_repository.list_upcoming(user_id, days=days_ahead)
        results: List[Dict[str, Any]] = []
        for event in events:
            r = await self.sync_event_to_google(str(event.id), user_id)
            r["event_id"] = str(event.id)
            results.append(r)
        synced = sum(1 for r in results if r.get("synced"))
        return {
            "synced": synced,
            "total": len(results),
            "results": results,
        }

    async def _delete_google_event(self, event: CalendarEvent) -> None:
        """Remove a previously-mirrored Google event.  No-op in Phase 2."""
        logger.info(
            "google_calendar_delete_pending_phase3",
            event_id=str(event.id),
            gcal_event_id=event.gcal_event_id,
        )


# Singleton — match the convention used elsewhere (e.g. notification_service).
calendar_service = CalendarService()
