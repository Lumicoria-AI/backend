"""
Celery tasks for the Phase 4 task-reminder pipeline.

Schedulers are registered in `backend/tasks/celery_app.py` beat_schedule.
Each scheduler is a *fan-out* — it iterates the active user set and queues
one per-user task that does the actual work.  This keeps individual tasks
small and lets failures retry per-user instead of for the whole batch.

Bulking-protection invariants:
  • Morning digest:        one email + one summary push per user per day.
  • Evening critical push: one push per user per evening (only if any
                           critical task is due within 24h).
  • Critical-hour warning: one email + one push per individual task.
  • Weekly digest:         one email per user per week.

All four go through `task_reminder_service` which checks
`reminder_state.last_*` on each task (or `user_settings.last_weekly_task_digest`
for the weekly) and skips when stamped.  Safe to over-schedule.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional

import structlog
from celery import shared_task

from backend.tasks.async_utils import run_worker_coro
from backend.tasks.celery_app import celery_app  # registers our app

logger = structlog.get_logger(__name__)


# ── Async runner ─────────────────────────────────────────────────────────


def _run(coro):
    """Run an async coroutine on the worker's persistent event loop."""
    return run_worker_coro(coro)


# ── Per-user worker tasks ────────────────────────────────────────────────


@celery_app.task(
    name="tasks.send_morning_digest_for_user",
    bind=True,
    max_retries=2,
    default_retry_delay=120,
)
def send_morning_digest_for_user(self, user_id: str) -> Dict[str, Any]:
    from backend.services.task_reminder_service import task_reminder_service
    try:
        return _run(task_reminder_service.send_morning_digest(user_id))
    except Exception as e:
        logger.warning("morning_digest_failed", user_id=user_id, error=str(e))
        raise self.retry(exc=e)


@celery_app.task(
    name="tasks.send_evening_critical_push_for_user",
    bind=True,
    max_retries=2,
    default_retry_delay=60,
)
def send_evening_critical_push_for_user(self, user_id: str) -> Dict[str, Any]:
    from backend.services.task_reminder_service import task_reminder_service
    try:
        return _run(task_reminder_service.send_evening_critical_push(user_id))
    except Exception as e:
        logger.warning("evening_critical_push_failed", user_id=user_id, error=str(e))
        raise self.retry(exc=e)


@celery_app.task(
    name="tasks.send_critical_hour_warning_for_user",
    bind=True,
    max_retries=2,
    default_retry_delay=30,
)
def send_critical_hour_warning_for_user(self, user_id: str) -> Dict[str, Any]:
    from backend.services.task_reminder_service import task_reminder_service
    try:
        return _run(task_reminder_service.send_critical_hour_warning(user_id))
    except Exception as e:
        logger.warning("critical_hour_warning_failed", user_id=user_id, error=str(e))
        raise self.retry(exc=e)


@celery_app.task(
    name="tasks.send_weekly_digest_for_user",
    bind=True,
    max_retries=2,
    default_retry_delay=300,
)
def send_weekly_digest_for_user(self, user_id: str) -> Dict[str, Any]:
    from backend.services.task_reminder_service import task_reminder_service
    try:
        return _run(task_reminder_service.send_weekly_digest(user_id))
    except Exception as e:
        logger.warning("weekly_digest_failed", user_id=user_id, error=str(e))
        raise self.retry(exc=e)


# ── Fan-out helpers ──────────────────────────────────────────────────────


async def _candidate_user_ids(
    *,
    require_email: bool = False,
    require_push: bool = False,
    digest_day: Optional[str] = None,
) -> List[str]:
    """Return the user ids that match the per-user schedule for this beat.

    For now we treat every user with `task_reminders=true` (default) as a
    candidate.  Timezone-aware filtering happens inside the per-user task
    (it checks the local hour against `task_reminder_settings`).  This keeps
    the fan-out cheap even at scale.
    """
    from backend.db.mongodb.mongodb import MongoDB

    settings_col = await MongoDB.get_collection("user_settings")
    query: Dict[str, Any] = {"task_reminders": {"$ne": False}}
    if require_email:
        query["email_notifications"] = {"$ne": False}
    if require_push:
        query["push_notifications"] = {"$ne": False}
    if digest_day:
        # `task_reminder_settings.weekly_digest_day` defaults to "friday";
        # we accept docs that either match or omit the field.
        query["$or"] = [
            {"task_reminder_settings.weekly_digest_day": digest_day},
            {"task_reminder_settings.weekly_digest_day": {"$exists": False}},
        ]

    user_ids: List[str] = []
    async for doc in settings_col.find(query, projection={"_id": 1, "user_id": 1}):
        uid = doc.get("user_id") or doc.get("_id")
        if uid:
            user_ids.append(str(uid))

    # Fallback: if no user_settings rows exist yet (new install), iterate users.
    if not user_ids:
        users_col = await MongoDB.get_collection("users")
        async for doc in users_col.find({}, projection={"_id": 1}):
            uid = doc.get("_id")
            if uid:
                user_ids.append(str(uid))
    return user_ids


def _now_hour_matches(target_hh_mm: str, tolerance_minutes: int = 30) -> bool:
    """True when *server* UTC time is within ±tolerance of the target HH:MM."""
    try:
        hh, mm = (target_hh_mm or "08:00").split(":")
        target = time(int(hh), int(mm))
    except Exception:
        target = time(8, 0)
    now = datetime.utcnow().time()
    now_min = now.hour * 60 + now.minute
    tgt_min = target.hour * 60 + target.minute
    return abs(now_min - tgt_min) <= tolerance_minutes


# ── Periodic fan-outs ────────────────────────────────────────────────────


@celery_app.task(name="tasks.fanout_morning_digest")
def fanout_morning_digest() -> Dict[str, Any]:
    """Beat-driven fan-out, hourly.  Sends to users whose local 08:00 lands
    near the current UTC hour (timezone-aware variant in Phase 5)."""
    async def _runner() -> Dict[str, Any]:
        ids = await _candidate_user_ids(require_email=True)
        queued = 0
        for uid in ids:
            send_morning_digest_for_user.delay(uid)
            queued += 1
        return {"queued": queued}
    return _run(_runner())


@celery_app.task(name="tasks.fanout_evening_critical_push")
def fanout_evening_critical_push() -> Dict[str, Any]:
    async def _runner() -> Dict[str, Any]:
        ids = await _candidate_user_ids(require_push=True)
        queued = 0
        for uid in ids:
            send_evening_critical_push_for_user.delay(uid)
            queued += 1
        return {"queued": queued}
    return _run(_runner())


@celery_app.task(name="tasks.fanout_critical_hour_warning")
def fanout_critical_hour_warning() -> Dict[str, Any]:
    async def _runner() -> Dict[str, Any]:
        ids = await _candidate_user_ids()
        queued = 0
        for uid in ids:
            send_critical_hour_warning_for_user.delay(uid)
            queued += 1
        return {"queued": queued}
    return _run(_runner())


@celery_app.task(name="tasks.fanout_weekly_digest")
def fanout_weekly_digest() -> Dict[str, Any]:
    """Friday default — picks the right users by their weekly_digest_day."""
    async def _runner() -> Dict[str, Any]:
        # `datetime.weekday()` is 0=Mon … 6=Sun
        weekday = datetime.utcnow().weekday()
        target_day = "friday" if weekday == 4 else ("saturday" if weekday == 5 else None)
        if not target_day:
            return {"queued": 0, "reason": "not_a_digest_day"}
        ids = await _candidate_user_ids(require_email=True, digest_day=target_day)
        queued = 0
        for uid in ids:
            send_weekly_digest_for_user.delay(uid)
            queued += 1
        return {"queued": queued, "day": target_day}
    return _run(_runner())
