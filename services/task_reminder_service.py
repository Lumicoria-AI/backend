"""
Task reminder service (Phase 4).

Single point of coordination for the four reminder paths:

  1. Daily morning digest (one bundled email per user, ~08:00 user-tz).
  2. Evening critical push (one push per user with critical tasks due ≤24h, ~17:00 user-tz).
  3. Critical-hour warning (per-task email + push, due within ~1h).
  4. Weekly digest (Friday/Saturday roll-up).

Bulking-protection: every send path stamps `reminder_state.*` on each task it
touches, and re-checks the stamp before sending again.  No silent retries.
The cron tasks in `backend/tasks/notification_tasks.py` schedule us; this
module owns the actual composition + dispatch.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import structlog
from bson import ObjectId

from backend.core.config import settings
from backend.db.mongodb.models.notification import NotificationPriority, NotificationType
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.task_repository import task_repository
from backend.models.task import TaskStatus
from backend.services.notification_service import notification_service
from backend.services.push_notification_service import push_notification_service
from backend.services.task_action_tokens import TaskAction, action_url

logger = structlog.get_logger(__name__)

# Re-send guards (in addition to time-of-day windows).
MORNING_RESEND_GUARD = timedelta(hours=20)
EVENING_RESEND_GUARD = timedelta(hours=8)
CRITICAL_RESEND_GUARD = timedelta(hours=2)
WEEKLY_RESEND_GUARD = timedelta(days=5)


# ── Helpers ─────────────────────────────────────────────────────────────


def _frontend_base() -> str:
    base = (
        getattr(settings, "PRODUCTION_DOMAIN", None)
        or getattr(settings, "FRONTEND_URL", None)
        or "https://lumicoria.ai"
    )
    return str(base).rstrip("/")


def _api_base() -> str:
    """Where the email-button action endpoint lives.

    Defaults to the same origin as the frontend; in dev this resolves to
    `http://localhost:8000` if the user sets API_URL.  The frontend's
    Vite dev server proxies /api/* to backend, but emails open in a
    real browser so we point at the backend host directly.
    """
    return str(
        getattr(settings, "API_BASE_URL", None)
        or getattr(settings, "BACKEND_URL", None)
        or _frontend_base()
    ).rstrip("/")


def _due_label(due: datetime, now: Optional[datetime] = None) -> str:
    now = now or datetime.utcnow()
    delta = due - now
    minutes = int(delta.total_seconds() / 60)
    if delta.total_seconds() < 0:
        return "Overdue"
    if minutes < 60:
        return f"Due in {minutes} min"
    hours = minutes // 60
    if hours < 24:
        return f"Due in {hours} h"
    days = hours // 24
    if days < 7:
        return f"Due {due.strftime('%a %H:%M')}"
    return due.strftime("%a, %b %d %H:%M")


def _task_to_row(
    task: Any,
    user_id: str,
    *,
    api_base: str,
    web_base: str,
    include_snooze: bool = False,
    now: Optional[datetime] = None,
) -> Dict[str, Any]:
    """Render a task into a dict suitable for the Jinja templates."""
    task_id = str(task.id)
    due = getattr(task, "due_date", None)
    priority = getattr(task, "priority", None)
    priority_val = priority.value if hasattr(priority, "value") else str(priority or "medium")
    inferred = bool((getattr(task, "metadata", {}) or {}).get("inferred_due_date"))
    assignee = (getattr(task, "metadata", {}) or {}).get("assigned_to_name")

    complete_url = action_url(
        base_url=api_base, user_id=user_id, task_id=task_id, action=TaskAction.COMPLETE,
    )
    start_url = action_url(
        base_url=api_base, user_id=user_id, task_id=task_id, action=TaskAction.START,
    )
    snooze_url = (
        action_url(base_url=api_base, user_id=user_id, task_id=task_id, action=TaskAction.SNOOZE)
        if include_snooze else None
    )

    return {
        "id": task_id,
        "title": getattr(task, "title", "Untitled task"),
        "description": getattr(task, "description", "") or "",
        "priority": priority_val,
        "due_label": _due_label(due, now=now) if due else "No due date",
        "due_date_iso": due.isoformat() + "Z" if due else None,
        "inferred_due_date": inferred,
        "assignee": assignee,
        "complete_url": complete_url,
        "start_url": start_url,
        "snooze_url": snooze_url,
        "view_url": f"{web_base}/tasks",
    }


async def _get_user(user_id: str) -> Optional[Dict[str, Any]]:
    """Fetch the user doc (raw dict) so we have email + name available."""
    try:
        col = await MongoDB.get_collection("users")
        try:
            uid: Any = ObjectId(user_id)
        except Exception:
            uid = user_id
        doc = await col.find_one({"_id": uid}) or await col.find_one({"id": uid})
        return doc
    except Exception as e:
        logger.debug("task_reminder: user lookup failed", error=str(e), user_id=user_id)
        return None


async def _get_user_settings(user_id: str) -> Dict[str, Any]:
    """Best-effort fetch of the per-user reminder settings."""
    try:
        col = await MongoDB.get_collection("user_settings")
        try:
            uid: Any = ObjectId(user_id)
        except Exception:
            uid = user_id
        doc = await col.find_one({"_id": uid}) or await col.find_one({"user_id": uid})
        return doc or {}
    except Exception:
        return {}


def _reminder_enabled(settings_doc: Dict[str, Any]) -> bool:
    return bool(settings_doc.get("task_reminders", True))


def _email_enabled(settings_doc: Dict[str, Any]) -> bool:
    return bool(settings_doc.get("email_notifications", True))


def _push_enabled(settings_doc: Dict[str, Any]) -> bool:
    return bool(settings_doc.get("push_notifications", True))


async def _stamp_reminder_state(
    task_ids: List[str], field: str, now: Optional[datetime] = None
) -> None:
    """Set `reminder_state.<field>` on a batch of tasks for idempotency."""
    if not task_ids:
        return
    now = now or datetime.utcnow()
    try:
        col = await MongoDB.get_collection("tasks")
        oids: List[ObjectId] = []
        for tid in task_ids:
            try:
                oids.append(ObjectId(tid))
            except Exception:
                continue
        if not oids:
            return
        await col.update_many(
            {"_id": {"$in": oids}},
            {"$set": {f"reminder_state.{field}": now}},
        )
    except Exception as e:
        logger.warning("task_reminder: stamp failed", field=field, error=str(e))


# ── Public dispatchers ──────────────────────────────────────────────────


class TaskReminderService:
    """Composes + sends the four reminder kinds for a single user."""

    # 1. Daily morning digest ────────────────────────────────────────────
    async def send_morning_digest(self, user_id: str) -> Dict[str, Any]:
        """One bundled email + one summary push per user per morning."""
        settings_doc = await _get_user_settings(user_id)
        if not _reminder_enabled(settings_doc):
            return {"sent": False, "reason": "reminders_disabled"}

        now = datetime.utcnow()
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        today_end = today_start + timedelta(days=1)
        tomorrow_end = today_start + timedelta(days=2)

        col = await MongoDB.get_collection("tasks")
        try:
            uid_oid: Any = ObjectId(user_id)
        except Exception:
            uid_oid = user_id

        # Pull tasks due today (00:00–24:00) and tomorrow.
        query_base = {
            "$or": [{"assigned_to": uid_oid}, {"created_by": uid_oid}],
            "status": {"$nin": [TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value]},
        }
        today_cursor = col.find({**query_base, "due_date": {"$gte": today_start, "$lt": today_end}}).limit(50)
        tomorrow_cursor = col.find({**query_base, "due_date": {"$gte": today_end, "$lt": tomorrow_end}}).limit(50)

        from backend.models.task import Task
        today_docs = [Task(**d) for d in await today_cursor.to_list(length=50)]
        tomorrow_docs = [Task(**d) for d in await tomorrow_cursor.to_list(length=50)]

        # Idempotency — drop any task whose morning stamp is recent.
        guard = now - MORNING_RESEND_GUARD
        def _fresh(t: Any) -> bool:
            rs = (getattr(t, "reminder_state", None) or {})
            if isinstance(rs, dict):
                last = rs.get("last_morning_sent")
            else:
                last = getattr(rs, "last_morning_sent", None)
            return not last or (isinstance(last, datetime) and last < guard)

        today_docs = [t for t in today_docs if _fresh(t)]
        tomorrow_docs = [t for t in tomorrow_docs if _fresh(t)]

        if not today_docs and not tomorrow_docs:
            return {"sent": False, "reason": "no_unstamped_tasks"}

        api_base = _api_base()
        web_base = _frontend_base()
        today_rows = [_task_to_row(t, user_id, api_base=api_base, web_base=web_base, now=now) for t in today_docs]
        tomorrow_rows = [_task_to_row(t, user_id, api_base=api_base, web_base=web_base, now=now) for t in tomorrow_docs]

        user = await _get_user(user_id) or {}
        user_email = user.get("email")
        user_name = (user.get("full_name") or (user.get("email") or "").split("@")[0]).strip()

        sent_email = False
        if user_email and _email_enabled(settings_doc):
            try:
                sent_email = await notification_service.send_email_notification(
                    to_email=user_email,
                    template_name="task_daily_digest",
                    template_data={
                        "subject": f"{len(today_rows)} task{'s' if len(today_rows) != 1 else ''} due today",
                        "user_name": user_name,
                        "today_count": len(today_rows),
                        "tomorrow_count": len(tomorrow_rows),
                        "today_tasks": today_rows,
                        "tomorrow_tasks": tomorrow_rows,
                        "dashboard_url": f"{web_base}/tasks",
                        "settings_url": f"{web_base}/settings",
                    },
                    priority=NotificationPriority.MEDIUM,
                )
            except Exception as e:
                logger.warning("morning_digest_email_failed", error=str(e))

        # One push summarising the count, not one per task.
        sent_push = False
        if _push_enabled(settings_doc) and (today_rows or tomorrow_rows):
            try:
                sent_push = await push_notification_service.send_to_user(
                    user_id=user_id,
                    title="Today's tasks",
                    body=(
                        f"{len(today_rows)} task{'s' if len(today_rows) != 1 else ''} due today"
                        + (f", {len(tomorrow_rows)} tomorrow" if tomorrow_rows else "")
                    ),
                    data={"type": "task_daily_digest", "today": len(today_rows), "tomorrow": len(tomorrow_rows)},
                )
            except Exception as e:
                logger.debug("morning_digest_push_failed", error=str(e))

        # Stamp every task we touched so we don't double-send.
        await _stamp_reminder_state(
            [t["id"] for t in today_rows + tomorrow_rows], "last_morning_sent", now=now
        )

        return {"sent": True, "email": sent_email, "push": sent_push,
                "today": len(today_rows), "tomorrow": len(tomorrow_rows)}

    # 2. Evening critical push ───────────────────────────────────────────
    async def send_evening_critical_push(self, user_id: str) -> Dict[str, Any]:
        """One push per user listing critical tasks due within 24h."""
        settings_doc = await _get_user_settings(user_id)
        if not _reminder_enabled(settings_doc):
            return {"sent": False, "reason": "reminders_disabled"}
        if not _push_enabled(settings_doc):
            return {"sent": False, "reason": "push_disabled"}

        now = datetime.utcnow()
        guard = now - EVENING_RESEND_GUARD
        window_end = now + timedelta(hours=24)

        col = await MongoDB.get_collection("tasks")
        try:
            uid_oid: Any = ObjectId(user_id)
        except Exception:
            uid_oid = user_id

        cursor = col.find({
            "$or": [{"assigned_to": uid_oid}, {"created_by": uid_oid}],
            "status": {"$nin": [TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value]},
            "priority": "critical",
            "due_date": {"$gte": now, "$lte": window_end},
        }).limit(50)
        docs = await cursor.to_list(length=50)

        from backend.models.task import Task
        tasks = [Task(**d) for d in docs]
        # Idempotency filter
        def _fresh(t: Any) -> bool:
            rs = (getattr(t, "reminder_state", None) or {})
            last = (rs.get("last_evening_sent") if isinstance(rs, dict) else getattr(rs, "last_evening_sent", None))
            return not last or (isinstance(last, datetime) and last < guard)
        tasks = [t for t in tasks if _fresh(t)]
        if not tasks:
            return {"sent": False, "reason": "no_unstamped_critical_tasks"}

        title = f"{len(tasks)} critical task{'s' if len(tasks) != 1 else ''} due soon"
        body = ", ".join(getattr(t, "title", "Untitled")[:50] for t in tasks[:3])
        if len(tasks) > 3:
            body += f" (+{len(tasks) - 3} more)"

        ok = False
        try:
            ok = await push_notification_service.send_to_user(
                user_id=user_id,
                title=title,
                body=body,
                data={"type": "task_evening_critical", "task_ids": [str(t.id) for t in tasks]},
            )
        except Exception as e:
            logger.debug("evening_critical_push_failed", error=str(e))

        # Also in-app
        try:
            await notification_service.create_in_app_notification(
                user_id=user_id,
                title=title,
                content=body,
                notification_type=NotificationType.TASK,
                priority=NotificationPriority.HIGH,
                metadata={"task_ids": [str(t.id) for t in tasks]},
            )
        except Exception:
            pass

        await _stamp_reminder_state([str(t.id) for t in tasks], "last_evening_sent", now=now)
        return {"sent": ok, "count": len(tasks)}

    # 3. Critical-hour warning (per-task email + push) ──────────────────
    async def send_critical_hour_warning(self, user_id: str) -> Dict[str, Any]:
        """For each critical task whose due_date is within the next ~1 hour,
        fire ONE individual email + push.  Idempotent via reminder_state.
        """
        settings_doc = await _get_user_settings(user_id)
        if not _reminder_enabled(settings_doc):
            return {"sent": 0, "reason": "reminders_disabled"}

        now = datetime.utcnow()
        guard = now - CRITICAL_RESEND_GUARD
        window_end = now + timedelta(minutes=70)  # slight overshoot to catch round-trip drift

        col = await MongoDB.get_collection("tasks")
        try:
            uid_oid: Any = ObjectId(user_id)
        except Exception:
            uid_oid = user_id

        cursor = col.find({
            "$or": [{"assigned_to": uid_oid}, {"created_by": uid_oid}],
            "status": {"$nin": [TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value]},
            "priority": "critical",
            "due_date": {"$gte": now - timedelta(minutes=5), "$lte": window_end},
        }).limit(20)
        docs = await cursor.to_list(length=20)

        from backend.models.task import Task
        tasks = [Task(**d) for d in docs]

        def _fresh(t: Any) -> bool:
            rs = (getattr(t, "reminder_state", None) or {})
            last = (rs.get("last_critical_push") if isinstance(rs, dict) else getattr(rs, "last_critical_push", None))
            return not last or (isinstance(last, datetime) and last < guard)
        tasks = [t for t in tasks if _fresh(t)]
        if not tasks:
            return {"sent": 0, "reason": "no_unstamped_critical_tasks"}

        user = await _get_user(user_id) or {}
        user_email = user.get("email")
        sent_count = 0
        api_base = _api_base()
        web_base = _frontend_base()

        for task in tasks:
            row = _task_to_row(
                task, user_id, api_base=api_base, web_base=web_base, include_snooze=True, now=now
            )
            # Email
            if user_email and _email_enabled(settings_doc):
                try:
                    await notification_service.send_email_notification(
                        to_email=user_email,
                        template_name="task_critical_alert",
                        template_data={
                            "subject": f"⚠ Critical: {row['title'][:60]}",
                            "task": row,
                            "dashboard_url": f"{web_base}/tasks",
                        },
                        priority=NotificationPriority.HIGH,
                    )
                except Exception as e:
                    logger.debug("critical_alert_email_failed", error=str(e))
            # Push
            if _push_enabled(settings_doc):
                try:
                    await push_notification_service.send_to_user(
                        user_id=user_id,
                        title=f"⚠ {row['title'][:50]}",
                        body=row["due_label"],
                        data={"type": "task_critical_hour", "task_id": row["id"]},
                    )
                except Exception:
                    pass
            # In-app
            try:
                await notification_service.create_in_app_notification(
                    user_id=user_id,
                    title=f"Critical task: {row['title'][:60]}",
                    content=row["due_label"],
                    notification_type=NotificationType.TASK,
                    priority=NotificationPriority.HIGH,
                    metadata={"task_id": row["id"], "action": "critical_hour"},
                )
            except Exception:
                pass
            sent_count += 1

        await _stamp_reminder_state([str(t.id) for t in tasks], "last_critical_push", now=now)
        return {"sent": sent_count}

    # 4. Weekly digest (Friday or Saturday, per setting) ────────────────
    async def send_weekly_digest(self, user_id: str) -> Dict[str, Any]:
        settings_doc = await _get_user_settings(user_id)
        if not _reminder_enabled(settings_doc):
            return {"sent": False, "reason": "reminders_disabled"}
        if not _email_enabled(settings_doc):
            return {"sent": False, "reason": "email_disabled"}

        now = datetime.utcnow()
        guard = now - WEEKLY_RESEND_GUARD
        week_start = now - timedelta(days=now.weekday())  # Monday
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        next_week_end = week_start + timedelta(days=14)

        col = await MongoDB.get_collection("tasks")
        try:
            uid_oid: Any = ObjectId(user_id)
        except Exception:
            uid_oid = user_id

        from backend.models.task import Task

        # Completed this week
        completed_docs = await col.find({
            "$or": [{"assigned_to": uid_oid}, {"created_by": uid_oid}],
            "status": TaskStatus.COMPLETED.value,
            "completed_at": {"$gte": week_start},
        }).sort("completed_at", -1).limit(50).to_list(length=50)
        # Overdue (still not done, past due)
        overdue_docs = await col.find({
            "$or": [{"assigned_to": uid_oid}, {"created_by": uid_oid}],
            "status": {"$nin": [TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value]},
            "due_date": {"$lt": now},
        }).sort("due_date", 1).limit(20).to_list(length=20)
        # Upcoming next 7 days
        upcoming_docs = await col.find({
            "$or": [{"assigned_to": uid_oid}, {"created_by": uid_oid}],
            "status": {"$nin": [TaskStatus.COMPLETED.value, TaskStatus.CANCELLED.value]},
            "due_date": {"$gte": now, "$lte": next_week_end},
        }).sort("due_date", 1).limit(30).to_list(length=30)

        # Idempotency on the user, not the task — weekly digest is one email
        # per user per ~7 days.  Stamp via user_settings.last_weekly_task_digest.
        last_sent = settings_doc.get("last_weekly_task_digest")
        if isinstance(last_sent, datetime) and last_sent > guard:
            return {"sent": False, "reason": "already_sent_this_week"}

        completed = [_task_to_row(Task(**d), user_id, api_base=_api_base(), web_base=_frontend_base(), now=now) for d in completed_docs]
        overdue = [_task_to_row(Task(**d), user_id, api_base=_api_base(), web_base=_frontend_base(), now=now) for d in overdue_docs]
        upcoming = [_task_to_row(Task(**d), user_id, api_base=_api_base(), web_base=_frontend_base(), now=now) for d in upcoming_docs]

        if not (completed or overdue or upcoming):
            return {"sent": False, "reason": "nothing_to_report"}

        user = await _get_user(user_id) or {}
        user_email = user.get("email")
        user_name = (user.get("full_name") or (user.get("email") or "").split("@")[0]).strip()
        if not user_email:
            return {"sent": False, "reason": "no_email"}

        web_base = _frontend_base()
        try:
            sent = await notification_service.send_email_notification(
                to_email=user_email,
                template_name="task_weekly_digest",
                template_data={
                    "subject": "Your week in tasks",
                    "user_name": user_name,
                    "week_label": f"{week_start.strftime('%b %d')} – {(week_start + timedelta(days=6)).strftime('%b %d')}",
                    "completed_count": len(completed),
                    "overdue_count": len(overdue),
                    "upcoming_count": len(upcoming),
                    "completed_tasks": completed,
                    "overdue_tasks": overdue,
                    "upcoming_tasks": upcoming,
                    "dashboard_url": f"{web_base}/tasks",
                    "settings_url": f"{web_base}/settings",
                },
                priority=NotificationPriority.MEDIUM,
            )
        except Exception as e:
            logger.warning("weekly_digest_email_failed", error=str(e))
            return {"sent": False, "reason": "send_failed", "error": str(e)}

        # Stamp on user_settings
        try:
            col_us = await MongoDB.get_collection("user_settings")
            try:
                uid_us: Any = ObjectId(user_id)
            except Exception:
                uid_us = user_id
            await col_us.update_one(
                {"_id": uid_us},
                {
                    "$set": {"last_weekly_task_digest": now, "updated_at": now},
                    "$setOnInsert": {"user_id": uid_us, "created_at": now},
                },
                upsert=True,
            )
        except Exception:
            pass

        return {"sent": bool(sent), "completed": len(completed), "overdue": len(overdue), "upcoming": len(upcoming)}


task_reminder_service = TaskReminderService()
