"""Productivity aggregator for the Well-being module.

Builds a single dict the Coach page (and the weekly digest) consumes:

    {
      "focus_minutes_today":     int,   # sum of agent run durations
      "agent_runs_today":        int,   # count of agent.* activity logs
      "tasks_total":             int,
      "tasks_completed_today":   int,
      "tasks_completed_week":    int,
      "tasks_in_progress":       int,
      "tasks_not_started":       int,
      "completion_ratio":        float, # 0..1
      "streak_days":             int,   # consecutive days with >=1 task done
    }

Reads from existing collections — no new schema:
  - `tasks`           via `task_repository`
  - `activity_logs`   via raw aggregate (read-only)
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, Optional

import structlog
from bson import ObjectId

from ...db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)


_AGENT_ACTIVITY_PREFIXES = (
    "agent.",
    "legal.",
    "ethics_bias.",
    "kg.",
    "data_analysis.",
    "customer_service.",
    "rag.",
    "research.",
    "creative.",
    "vision.",
    "wellbeing.",
)


def _start_of_day(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.utcnow()
    return now.replace(hour=0, minute=0, second=0, microsecond=0)


def _start_of_week(now: Optional[datetime] = None) -> datetime:
    now = now or datetime.utcnow()
    start = _start_of_day(now)
    return start - timedelta(days=start.weekday())  # Monday


async def compute_productivity(
    *,
    organization_id: str,
    user_id: str,
) -> Dict[str, Any]:
    """Aggregate the user's productivity signals across both stores.

    Best-effort: if any sub-query fails, log + return zero for that
    bucket so the Coach page never crashes from a missing collection.
    """
    if not organization_id or not user_id:
        return _empty()

    now = datetime.utcnow()
    today = _start_of_day(now)
    week_start = _start_of_week(now)

    out = _empty()
    out["computed_at"] = now.isoformat() + "Z"

    # ── Task buckets ─────────────────────────────────────────────
    try:
        from ...db.mongodb.repositories.task_repository import task_repository

        stats = await task_repository.get_task_stats(
            organization_id=organization_id, user_id=user_id
        )
        for bucket in stats.get("statuses") or []:
            status = (bucket.get("status") or "").lower()
            count = int(bucket.get("count") or 0)
            if status == "completed":
                out["tasks_completed"] = count
            elif status in ("in_progress", "started"):
                out["tasks_in_progress"] = count
            elif status in ("todo", "not_started", "pending"):
                out["tasks_not_started"] = count
            elif status == "blocked":
                out["tasks_blocked"] = count
        out["tasks_total"] = int(stats.get("total") or 0)
        if out["tasks_total"] > 0:
            out["completion_ratio"] = round(
                out["tasks_completed"] / out["tasks_total"], 3
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_productivity_tasks_failed", error=str(e))

    # ── Today's completed tasks ─────────────────────────────────
    try:
        tasks_coll = await MongoDB.get_collection("tasks")
        oid_org = _as_oid(organization_id)
        oid_user = _as_oid(user_id)
        match_today: Dict[str, Any] = {
            "status": "completed",
            "updated_at": {"$gte": today},
        }
        if oid_org is not None:
            match_today["organization_id"] = oid_org
        if oid_user is not None:
            match_today["$or"] = [
                {"assigned_to": oid_user},
                {"created_by": oid_user},
            ]
        out["tasks_completed_today"] = int(
            await tasks_coll.count_documents(match_today)
        )

        match_week = dict(match_today)
        match_week["updated_at"] = {"$gte": week_start}
        out["tasks_completed_week"] = int(
            await tasks_coll.count_documents(match_week)
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_productivity_tasks_today_failed", error=str(e))

    # ── Agent runs today (from activity_logs) ───────────────────
    try:
        logs_coll = await MongoDB.get_collection("activity_logs")
        prefix_or = [
            {"activity_type": {"$regex": f"^{p}", "$options": ""}}
            for p in _AGENT_ACTIVITY_PREFIXES
        ]
        match = {
            "organization_id": _as_oid(organization_id) or organization_id,
            "user_id": _as_oid(user_id) or user_id,
            "created_at": {"$gte": today},
            "$or": prefix_or,
        }
        out["agent_runs_today"] = int(await logs_coll.count_documents(match))
        # 5-minute proxy per agent run = focus minutes (good enough
        # for v1; a real heartbeat-derived focus time can come later).
        out["focus_minutes_today"] = int(out["agent_runs_today"]) * 5
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_productivity_activity_failed", error=str(e))

    # ── Streak: days in a row with >=1 task completed ───────────
    try:
        tasks_coll = await MongoDB.get_collection("tasks")
        streak = 0
        cursor_day = today
        for _ in range(60):  # cap at 60 days
            day_start = cursor_day
            day_end = day_start + timedelta(days=1)
            match: Dict[str, Any] = {
                "organization_id": _as_oid(organization_id) or organization_id,
                "$or": [
                    {"assigned_to": _as_oid(user_id) or user_id},
                    {"created_by": _as_oid(user_id) or user_id},
                ],
                "status": "completed",
                "updated_at": {"$gte": day_start, "$lt": day_end},
            }
            if await tasks_coll.count_documents(match) > 0:
                streak += 1
                cursor_day -= timedelta(days=1)
            else:
                break
        out["streak_days"] = streak
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_productivity_streak_failed", error=str(e))

    return out


def _empty() -> Dict[str, Any]:
    return {
        "focus_minutes_today": 0,
        "agent_runs_today": 0,
        "tasks_total": 0,
        "tasks_completed": 0,
        "tasks_completed_today": 0,
        "tasks_completed_week": 0,
        "tasks_in_progress": 0,
        "tasks_not_started": 0,
        "tasks_blocked": 0,
        "completion_ratio": 0.0,
        "streak_days": 0,
    }


def _as_oid(value: Any) -> Optional[ObjectId]:
    """Best-effort ObjectId coercion; returns None for non-hex strings."""
    if value is None:
        return None
    if isinstance(value, ObjectId):
        return value
    if isinstance(value, str):
        try:
            return ObjectId(value)
        except Exception:
            return None
    return None
