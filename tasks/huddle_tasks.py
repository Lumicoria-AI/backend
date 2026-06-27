"""
Celery tasks for Lumicoria Huddle.

  - delete_expired_recording: invoked by Celery Beat to sweep
    recordings past their retention window.
  - generate_post_meeting_summary: re-runs the MeetingAgent on a
    huddle's accumulated transcript (used when end_huddle's inline
    background task fails — gives us a retry surface).
"""

from __future__ import annotations

import structlog

from backend.tasks.async_utils import run_worker_coro
from backend.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)


@celery_app.task(name="huddle.delete_expired_recording", bind=True, max_retries=3)
def delete_expired_recording(self, huddle_id: str) -> dict:
    """Delete recording chunks + manifest from object storage."""
    try:
        from backend.services.huddle_recording_service import expire_recording
        result = run_worker_coro(expire_recording(huddle_id))
        logger.info("huddle_recording_expired", huddle_id=huddle_id, result=result)
        return result
    except Exception as e:
        logger.warning("huddle_recording_expire_failed", huddle_id=huddle_id, error=str(e))
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


@celery_app.task(name="huddle.generate_post_meeting_summary", bind=True, max_retries=2)
def generate_post_meeting_summary(self, huddle_id: str) -> dict:
    """Re-run MeetingAgent post-call summary (retry surface)."""
    try:
        from backend.services.huddle_service import _run_post_call_summary, get_huddle
        huddle = run_worker_coro(get_huddle(huddle_id, requesting_user_id=None))
        if huddle:
            run_worker_coro(_run_post_call_summary(huddle_id, huddle))
        return {"ok": True}
    except Exception as e:
        logger.warning("huddle_summary_failed", huddle_id=huddle_id, error=str(e))
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


@celery_app.task(name="huddle.sweep_expired_recordings")
def sweep_expired_recordings() -> dict:
    """Beat-scheduled hourly sweep — finds rows past `recording_expires_at`
    and enqueues per-huddle delete jobs."""
    try:
        from datetime import datetime
        from sqlalchemy import select
        from backend.db.postgres import get_async_sessionmaker
        from backend.db.postgres_models import HuddleSQL

        async def _find():
            factory = get_async_sessionmaker()
            async with factory() as session:
                q = select(HuddleSQL.id).where(
                    HuddleSQL.recording_expires_at < datetime.utcnow(),
                    HuddleSQL.recording_object_key.is_not(None),
                    HuddleSQL.deleted_at.is_(None),
                )
                return [r[0] for r in (await session.execute(q)).all()]

        ids = run_worker_coro(_find())
        for h_id in ids:
            delete_expired_recording.delay(h_id)
        return {"swept": len(ids)}
    except Exception as e:
        logger.warning("huddle_sweep_failed", error=str(e))
        return {"ok": False, "error": str(e)}
