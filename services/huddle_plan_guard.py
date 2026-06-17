"""
Lumicoria Huddle — per-plan limits guard.

Enforces:
  - daily huddle-creation cap
  - max meeting duration
  - max concurrent participants
  - whether recording is allowed
  - max attached agents per call

Raises HTTPException(402, code='upgrade_required') when limits are exceeded.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

from fastapi import HTTPException, status
from sqlalchemy import func, select

from backend.db.postgres import get_async_sessionmaker
from backend.db.postgres_models import HuddleSQL


# Pricing tier caps. Mirrors the table in the plan doc.
HUDDLE_PLAN_CAPS: Dict[str, Dict[str, Any]] = {
    "free": {
        "daily_max": 1,
        "max_duration_min": 30,
        "max_participants": 4,
        "recording_allowed": False,
        "max_agents_in_call": 0,
        "scheduling_allowed": False,
    },
    "starter": {
        "daily_max": 10,
        "max_duration_min": 60,
        "max_participants": 8,
        "recording_allowed": False,
        "max_agents_in_call": 1,
        "scheduling_allowed": False,
    },
    "professional": {
        "daily_max": None,  # unlimited
        "max_duration_min": 90,
        "max_participants": 25,
        "recording_allowed": True,
        "recording_retention_days_max": 7,
        "max_agents_in_call": 3,
        "scheduling_allowed": True,
    },
    "team": {
        "daily_max": None,
        "max_duration_min": 240,
        "max_participants": 50,
        "recording_allowed": True,
        "recording_retention_days_max": 30,
        "max_agents_in_call": None,
        "scheduling_allowed": True,
    },
    "business": {
        "daily_max": None,
        "max_duration_min": 480,
        "max_participants": 200,
        "recording_allowed": True,
        "recording_retention_days_max": 90,
        "max_agents_in_call": None,
        "scheduling_allowed": True,
        "sso_gated_allowed": True,
        "custom_branding_allowed": True,
    },
    "enterprise": {
        "daily_max": None,
        "max_duration_min": None,  # unlimited
        "max_participants": None,
        "recording_allowed": True,
        "recording_retention_days_max": None,
        "max_agents_in_call": None,
        "scheduling_allowed": True,
        "sso_gated_allowed": True,
        "custom_branding_allowed": True,
        "cmk_recording_allowed": True,
        "custom_jitsi_allowed": True,
    },
}


def caps_for(plan: str) -> Dict[str, Any]:
    return HUDDLE_PLAN_CAPS.get((plan or "free").lower(), HUDDLE_PLAN_CAPS["free"])


def _upgrade_error(message: str, *, plan: str, limit_name: str) -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_402_PAYMENT_REQUIRED,
        detail={
            "code": "upgrade_required",
            "message": message,
            "current_plan": plan,
            "limit": limit_name,
        },
    )


async def enforce_can_create(
    *,
    plan: str,
    organization_id: str,
    host_user_id: str,
    recording_enabled: bool,
    recording_retention_days: int,
    agent_count: int,
    require_sso: bool,
    meeting_type: str = "instant",
) -> Dict[str, Any]:
    """Run all create-time checks. Returns the resolved caps so the
    caller can clamp values that aren't outright blocked."""
    caps = caps_for(plan)

    if meeting_type in {"scheduled", "recurring"} and not caps.get("scheduling_allowed"):
        raise _upgrade_error("Scheduled meetings require a Professional plan or higher.", plan=plan, limit_name="scheduling")

    daily_max = caps.get("daily_max")
    if daily_max is not None:
        session_factory = get_async_sessionmaker()
        async with session_factory() as session:
            day_start = datetime.utcnow() - timedelta(hours=24)
            q = (
                select(func.count())
                .select_from(HuddleSQL)
                .where(
                    HuddleSQL.host_user_id == host_user_id,
                    HuddleSQL.organization_id == organization_id,
                    HuddleSQL.created_at >= day_start,
                    HuddleSQL.deleted_at.is_(None),
                )
            )
            today = int((await session.execute(q)).scalar() or 0)
            if today >= daily_max:
                raise _upgrade_error(
                    f"Your plan allows {daily_max} huddle(s) per 24 hours.",
                    plan=plan, limit_name="daily_max",
                )

    if recording_enabled and not caps.get("recording_allowed"):
        raise _upgrade_error("Recording is available on the Professional plan and higher.", plan=plan, limit_name="recording")

    retention_max = caps.get("recording_retention_days_max")
    if recording_enabled and retention_max is not None and recording_retention_days > retention_max:
        raise _upgrade_error(
            f"Your plan caps recording retention at {retention_max} days.",
            plan=plan, limit_name="recording_retention",
        )

    max_agents = caps.get("max_agents_in_call")
    if max_agents is not None and agent_count > max_agents:
        raise _upgrade_error(
            f"Your plan allows {max_agents} AI agent(s) in a call.",
            plan=plan, limit_name="max_agents_in_call",
        )

    if require_sso and not caps.get("sso_gated_allowed"):
        raise _upgrade_error("SSO-gated meeting rooms are a Business+ feature.", plan=plan, limit_name="sso_gated")

    return caps


def enforce_can_join(*, plan: str, current_participants: int) -> None:
    caps = caps_for(plan)
    max_p = caps.get("max_participants")
    if max_p is not None and current_participants >= max_p:
        raise _upgrade_error(
            f"This huddle is at the {max_p}-participant limit for the host's plan.",
            plan=plan, limit_name="max_participants",
        )


def clamp_duration_min(plan: str) -> Optional[int]:
    return caps_for(plan).get("max_duration_min")
