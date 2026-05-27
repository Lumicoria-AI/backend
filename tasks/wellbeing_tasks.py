"""Periodic Celery tasks for the Well-being module.

Three tasks, each registered in `celery_app.conf.beat_schedule`:

  - `wellbeing.check_break_reminders`   every 5 min
  - `wellbeing.schedule_mood_prompts`   every 20 min
  - `wellbeing.send_weekly_digest`      every Monday 09:00 UTC

Production: run alongside the existing worker with
`celery -A backend.tasks.celery_app beat`.

Dev: with `CELERY_TASK_ALWAYS_EAGER=true`, callers can invoke these
tasks directly and they run synchronously inside the API process.
"""

from __future__ import annotations

import asyncio
import random
import time
from datetime import datetime, timedelta
from typing import Any, Dict, List

import structlog

from backend.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)


def _run_async(coro):
    """Sync entry to drive async code from inside a Celery task."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


# ── Helper: iterate users with wellbeing enabled ──────────────────


async def _iter_active_users() -> List[Dict[str, Any]]:
    """Return the users for whom we should run periodic checks.

    Tries the existing user repository.  Falls back to whatever
    `users` collection holds — multi-tenant platforms may have
    `is_active` / `deleted_at` semantics differently set; we read
    defensively.
    """
    try:
        from backend.db.mongodb.mongodb import MongoDB

        coll = await MongoDB.get_collection("users")
        cursor = coll.find({"is_active": {"$ne": False}}, projection={
            "_id": 1, "email": 1, "full_name": 1, "organization_id": 1,
        }).limit(5000)
        return [u async for u in cursor]
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_iter_users_failed", error=str(e))
        return []


async def _user_settings_for(user_id: str):
    try:
        from backend.db.mongodb.repositories.user_repository import (
            UserRepository,
        )
        return await UserRepository().get_user_settings(user_id)
    except Exception:  # noqa: BLE001
        return None


# ── Task 1: check break reminders ─────────────────────────────────


@celery_app.task(name="wellbeing.check_break_reminders")
def check_break_reminders() -> Dict[str, Any]:
    """For every user with `break_reminders=True`, push an in-app
    notification when their break interval has been exceeded since
    last activity."""
    return _run_async(_check_break_reminders_async())


async def _check_break_reminders_async() -> Dict[str, Any]:
    from backend.services.wellbeing import session_tracker
    from backend.services.notification_service import notification_service
    from backend.services.activity_logger import log_activity

    users = await _iter_active_users()
    sent = 0
    skipped = 0

    for user in users:
        user_id = str(user.get("_id") or user.get("id") or "")
        if not user_id:
            continue
        settings = await _user_settings_for(user_id)
        if not settings or not getattr(settings, "break_reminders", True):
            skipped += 1
            continue
        interval = int(getattr(settings, "break_interval_minutes", 60) or 60)

        # No heartbeat in the last 15 min → user isn't active right now.
        last_active = session_tracker.seconds_since_activity(user_id)
        if last_active is None or last_active > 900:
            skipped += 1
            continue

        # Already nudged them in the last 30 min?
        if session_tracker.break_reminder_in_cooldown(user_id):
            skipped += 1
            continue

        seconds_since_break = session_tracker.seconds_since_break(user_id)
        if seconds_since_break is None:
            # Treat "no break recorded" as still within the first window.
            session_tracker.mark_break(user_id)
            continue

        if seconds_since_break < interval * 60:
            continue

        try:
            await notification_service.send_wellbeing_reminder(
                user_id=user_id,
                reminder_type="break_due",
                message=(
                    f"You've been working for {int(seconds_since_break // 60)} "
                    "minutes straight — a short break will help your focus."
                ),
            )
            session_tracker.mark_break_reminder_sent(user_id)
            await log_activity(
                user_id=user_id,
                organization_id=str(user.get("organization_id") or user_id),
                activity_type="wellbeing.break_reminder_sent",
                details={"minutes_since_break": int(seconds_since_break // 60)},
                agent_name="Wellbeing Coach",
            )
            sent += 1
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "wellbeing_break_reminder_send_failed",
                user_id=user_id,
                error=str(e),
            )

    return {"sent": sent, "skipped": skipped, "users": len(users)}


# ── Task 2: schedule mood prompts ─────────────────────────────────


@celery_app.task(name="wellbeing.schedule_mood_prompts")
def schedule_mood_prompts() -> Dict[str, Any]:
    """Queue a mood-log prompt for some fraction of currently-active
    users.  The frontend polls `/wellbeing/mood-prompts/poll` and
    shows the modal when a prompt is pending.  Cooldown of 90 min
    per user is enforced inside the session_tracker."""
    return _run_async(_schedule_mood_prompts_async())


async def _schedule_mood_prompts_async() -> Dict[str, Any]:
    from backend.services.wellbeing import session_tracker

    users = await _iter_active_users()
    queued = 0
    for user in users:
        user_id = str(user.get("_id") or user.get("id") or "")
        if not user_id:
            continue
        # Only target users we've seen in the last 5 minutes.
        seconds = session_tracker.seconds_since_activity(user_id)
        if seconds is None or seconds > 300:
            continue
        # Roughly 1 in 2 chance the user gets prompted on any given
        # tick (so on average one prompt per 40 min, modulo cooldown).
        if random.random() < 0.5:
            if session_tracker.queue_mood_prompt(user_id):
                queued += 1
    return {"queued": queued, "users": len(users)}


# ── Task 3: weekly digest ─────────────────────────────────────────


@celery_app.task(name="wellbeing.send_weekly_digest")
def send_weekly_digest() -> Dict[str, Any]:
    """Build and send the per-user weekly digest.  Runs Monday 09:00
    UTC for all users with `email_notifications=True`."""
    return _run_async(_send_weekly_digest_async())


async def _send_weekly_digest_async() -> Dict[str, Any]:
    from backend.services.wellbeing import digest as digest_service
    from backend.services.activity_logger import log_activity

    users = await _iter_active_users()
    sent = 0
    failed = 0

    for user in users:
        user_id = str(user.get("_id") or user.get("id") or "")
        email = str(user.get("email") or "")
        if not user_id or not email:
            continue
        settings = await _user_settings_for(user_id)
        if settings and not getattr(settings, "email_notifications", True):
            continue
        org_id = str(user.get("organization_id") or user_id)

        try:
            payload = await digest_service.build_user_digest(
                organization_id=org_id,
                user_id=user_id,
                email=email,
                name=user.get("full_name"),
            )
            success = await digest_service.send_user_digest(payload)
            if success:
                sent += 1
                try:
                    await log_activity(
                        user_id=user_id,
                        organization_id=org_id,
                        activity_type="wellbeing.digest_sent",
                        details={"week_start": payload.get("week_start")},
                        agent_name="Wellbeing Coach",
                    )
                except Exception:  # noqa: BLE001
                    pass
            else:
                failed += 1
        except Exception as e:  # noqa: BLE001
            failed += 1
            logger.warning(
                "wellbeing_digest_user_failed",
                user_id=user_id,
                error=str(e),
            )

    return {"sent": sent, "failed": failed, "users": len(users)}
