"""
Phase C — Notification preferences & rules REST API.

Mounted at `/api/v1/notification-rules`.

Lets a user fine-tune which categories of notifications reach which channel
(email / push / in_app), and within what quiet-hours window.  Defaults are
on; opt-out is explicit.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.notification_prefs_repository import (
    notification_prefs_repository, CATEGORIES, CHANNELS,
)
from backend.models.user import User

logger = structlog.get_logger(__name__)
router = APIRouter()


@router.get("/categories")
async def list_categories():
    return {"channels": CHANNELS, "categories": CATEGORIES}


@router.get("/preferences")
async def list_preferences(
    current_user: User = Depends(get_current_active_user),
):
    return await notification_prefs_repository.get_all(str(current_user.id))


class PreferenceUpsert(BaseModel):
    channel: str
    category: str
    enabled: Optional[bool] = None
    quiet_hours: Optional[Dict[str, Any]] = None


@router.patch("/preferences")
async def upsert_preference(
    payload: PreferenceUpsert,
    current_user: User = Depends(get_current_active_user),
):
    if payload.channel not in CHANNELS:
        raise HTTPException(status_code=400, detail="Invalid channel")
    if payload.category not in CATEGORIES:
        raise HTTPException(status_code=400, detail="Invalid category")
    return await notification_prefs_repository.upsert(
        user_id=str(current_user.id),
        channel=payload.channel,
        category=payload.category,
        enabled=payload.enabled,
        quiet_hours=payload.quiet_hours,
    )


class QuietHoursPayload(BaseModel):
    start: Optional[str] = Field(None, description="HH:MM 24h")
    end: Optional[str] = Field(None, description="HH:MM 24h")
    timezone: Optional[str] = None


@router.get("/quiet-hours")
async def get_quiet_hours(
    current_user: User = Depends(get_current_active_user),
):
    rows = await notification_prefs_repository.get_all(str(current_user.id))
    for r in rows:
        if r.get("quiet_hours"):
            return r["quiet_hours"]
    return {"start": None, "end": None, "timezone": None}


@router.patch("/quiet-hours")
async def update_quiet_hours(
    payload: QuietHoursPayload,
    current_user: User = Depends(get_current_active_user),
):
    qh = payload.model_dump(exclude_none=True)
    # Apply to every channel/category combination so the user has one global QH.
    updated_count = 0
    for ch in CHANNELS:
        for cat in CATEGORIES:
            await notification_prefs_repository.upsert(
                user_id=str(current_user.id), channel=ch, category=cat, quiet_hours=qh,
            )
            updated_count += 1
    return {"updated": updated_count, "quiet_hours": qh}
