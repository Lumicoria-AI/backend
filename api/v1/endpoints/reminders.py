"""
Phase C — Reminders REST API.

Mounted at `/api/v1/reminders`.

Cross-resource reminders.  Each reminder is owned by a user inside an org
and fires through one or more channels.  The sweeper Celery beat job
(`reminders.sweep`) delivers any row with `state=pending AND due_at<=now`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.db.mongodb.repositories.reminders_repository import reminders_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity

logger = structlog.get_logger(__name__)
router = APIRouter()


def _oid(v: Any) -> Optional[ObjectId]:
    if v is None:
        return None
    if isinstance(v, ObjectId):
        return v
    try:
        return ObjectId(str(v))
    except Exception:
        return None


def _resolve_org_id(current_user: User) -> str:
    primary = getattr(current_user, "organization_id", None)
    if primary:
        return str(primary)
    ids = getattr(current_user, "organization_ids", None) or []
    if ids:
        return str(ids[0])
    raise HTTPException(status_code=400, detail="User has no organization context")


class ReminderCreate(BaseModel):
    resource_type: str = Field(..., max_length=64)
    resource_id: str = Field(..., max_length=64)
    due_at: datetime
    channels: List[str] = Field(default_factory=lambda: ["in_app"])
    note: Optional[str] = Field(None, max_length=4000)
    recur_cron: Optional[str] = Field(None, max_length=128)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ReminderUpdate(BaseModel):
    due_at: Optional[datetime] = None
    channels: Optional[List[str]] = None
    note: Optional[str] = None


@router.get("")
async def list_reminders(
    state: Optional[str] = Query(None),
    upcoming_only: bool = Query(False),
    organization_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    skip: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    return await reminders_repository.list_for_user(
        user_id=str(current_user.id),
        organization_id=org_id,
        state=state,
        upcoming_only=upcoming_only,
        limit=limit, skip=skip,
    )


@router.get("/upcoming")
async def list_upcoming(
    current_user: User = Depends(get_current_active_user),
):
    org_id = _resolve_org_id(current_user)
    return await reminders_repository.list_for_user(
        user_id=str(current_user.id), organization_id=org_id,
        upcoming_only=True, limit=200,
    )


@router.get("/overdue")
async def list_overdue(
    current_user: User = Depends(get_current_active_user),
):
    org_id = _resolve_org_id(current_user)
    rows = await reminders_repository.list_for_user(
        user_id=str(current_user.id), organization_id=org_id,
        state="pending", limit=200,
    )
    now = datetime.utcnow()
    return [r for r in rows if r.get("due_at") and r["due_at"] <= now]


@router.post("", status_code=201)
async def create_reminder(
    payload: ReminderCreate,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    row = await reminders_repository.create(
        organization_id=org_id,
        user_id=str(current_user.id),
        resource_type=payload.resource_type,
        resource_id=payload.resource_id,
        due_at=payload.due_at,
        channels=payload.channels,
        note=payload.note,
        recur_cron=payload.recur_cron,
        metadata=payload.metadata,
    )
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="reminder.created",
        details={"reminder_id": row["id"], "due_at": payload.due_at.isoformat(),
                 "resource_type": payload.resource_type, "resource_id": payload.resource_id},
        related_resource_type=payload.resource_type, related_resource_id=payload.resource_id,
    )
    return row


@router.patch("/{reminder_id}")
async def update_reminder(
    reminder_id: str,
    payload: ReminderUpdate,
    current_user: User = Depends(get_current_active_user),
):
    row = await reminders_repository.get(reminder_id)
    if not row or row.get("user_id") != str(current_user.id):
        raise HTTPException(status_code=404, detail="Reminder not found")
    return await reminders_repository.update(
        reminder_id,
        due_at=payload.due_at,
        channels=payload.channels,
        note=payload.note,
    )


@router.post("/{reminder_id}/snooze")
async def snooze_reminder(
    reminder_id: str,
    minutes: int = Query(60, ge=1, le=10080),
    current_user: User = Depends(get_current_active_user),
):
    from datetime import timedelta
    row = await reminders_repository.get(reminder_id)
    if not row or row.get("user_id") != str(current_user.id):
        raise HTTPException(status_code=404, detail="Reminder not found")
    new_due = datetime.utcnow() + timedelta(minutes=minutes)
    return await reminders_repository.update(reminder_id, due_at=new_due, state="pending")


@router.post("/{reminder_id}/send-now")
async def send_reminder_now(
    reminder_id: str,
    current_user: User = Depends(get_current_active_user),
):
    row = await reminders_repository.get(reminder_id)
    if not row or row.get("user_id") != str(current_user.id):
        raise HTTPException(status_code=404, detail="Reminder not found")
    return await reminders_repository.update(reminder_id, due_at=datetime.utcnow(), state="pending")


@router.delete("/{reminder_id}", status_code=204)
async def delete_reminder(
    reminder_id: str,
    current_user: User = Depends(get_current_active_user),
):
    row = await reminders_repository.get(reminder_id)
    if not row or row.get("user_id") != str(current_user.id):
        raise HTTPException(status_code=404, detail="Reminder not found")
    await reminders_repository.delete(reminder_id)
    return None


class BulkSnoozePayload(BaseModel):
    reminder_ids: List[str] = Field(..., min_length=1, max_length=200)
    minutes: int = Field(60, ge=1, le=10080)


@router.post("/bulk-snooze")
async def bulk_snooze(
    payload: BulkSnoozePayload,
    current_user: User = Depends(get_current_active_user),
):
    from datetime import timedelta
    new_due = datetime.utcnow() + timedelta(minutes=payload.minutes)
    n = 0
    for rid in payload.reminder_ids:
        row = await reminders_repository.get(rid)
        if row and row.get("user_id") == str(current_user.id):
            await reminders_repository.update(rid, due_at=new_due, state="pending")
            n += 1
    return {"snoozed": n}


class BulkCreatePayload(BaseModel):
    reminders: List[ReminderCreate] = Field(..., min_length=1, max_length=100)


@router.post("/bulk-create", status_code=201)
async def bulk_create(
    payload: BulkCreatePayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    rows: List[Dict[str, Any]] = []
    for r in payload.reminders:
        rows.append(await reminders_repository.create(
            organization_id=org_id, user_id=str(current_user.id),
            resource_type=r.resource_type, resource_id=r.resource_id,
            due_at=r.due_at, channels=r.channels, note=r.note,
            recur_cron=r.recur_cron, metadata=r.metadata,
        ))
    return {"created": len(rows), "reminders": rows}
