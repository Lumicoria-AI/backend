"""Central Celery application for Lumicoria background work.

Production path: `celery -A backend.tasks.celery_app worker --loglevel=info`.

Dev path: set `CELERY_TASK_ALWAYS_EAGER=true` in `.env` so `.delay(...)` runs
synchronously inside the API process — no worker required.

Tasks register themselves via `@celery_app.task` imports in document_tasks.py;
`autodiscover_tasks` is called so future modules in `backend/tasks/` get
picked up automatically.
"""

from __future__ import annotations

from typing import Optional

from celery import Celery

from backend.core.config import settings


def _build_broker_url() -> str:
    if settings.CELERY_BROKER_URL:
        return settings.CELERY_BROKER_URL
    host = settings.db.REDIS_HOST
    port = settings.db.REDIS_PORT
    db = settings.db.REDIS_DB
    password = settings.db.REDIS_PASSWORD
    auth = f":{password}@" if password else ""
    return f"redis://{auth}{host}:{port}/{db}"


def _build_result_backend() -> str:
    return settings.CELERY_RESULT_BACKEND or _build_broker_url()


def _weekly_monday_9am():
    """Monday 09:00 UTC, using celery.schedules.crontab.  Defined as a
    helper so the import only fires when beat actually loads."""
    from celery.schedules import crontab
    return crontab(hour=9, minute=0, day_of_week="monday")


celery_app = Celery(
    "lumicoria",
    broker=_build_broker_url(),
    backend=_build_result_backend(),
    include=[
        "backend.tasks.document_tasks",
        "backend.tasks.wellbeing_tasks",
    ],
)


celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    # Ingestion can take minutes for large PDFs.
    task_time_limit=30 * 60,
    task_soft_time_limit=25 * 60,
    # One heavy task per worker so we don't oversubscribe CPU on the
    # ProcessPoolExecutor used by the PDF parser.
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    # Retries (tasks override these per-task where needed).
    task_default_retry_delay=30,
    task_max_retries=3,
    result_expires=6 * 3600,
    worker_concurrency=settings.CELERY_WORKER_CONCURRENCY,
    task_always_eager=settings.CELERY_TASK_ALWAYS_EAGER,
    task_eager_propagates=settings.CELERY_TASK_ALWAYS_EAGER,
)


# Let future modules drop into backend/tasks/ without a config edit.
celery_app.autodiscover_tasks(["backend.tasks"])


# ── Periodic schedule (Celery beat) ────────────────────────────────
#
# Run with `celery -A backend.tasks.celery_app beat --loglevel=info`
# alongside a worker.  In dev with `CELERY_TASK_ALWAYS_EAGER=true`
# the API process can drive these on its own via `crontab` ticks.
celery_app.conf.beat_schedule = {
    # Periodic break-reminder check.  Every 5 minutes.
    "wellbeing-check-break-reminders": {
        "task": "wellbeing.check_break_reminders",
        "schedule": 300.0,  # seconds
    },
    # Random mood-prompt scheduler.  Every 20 minutes.
    "wellbeing-schedule-mood-prompts": {
        "task": "wellbeing.schedule_mood_prompts",
        "schedule": 1200.0,
    },
    # Weekly digest — Monday 09:00 UTC.
    "wellbeing-send-weekly-digest": {
        "task": "wellbeing.send_weekly_digest",
        "schedule": _weekly_monday_9am(),
    },
}
