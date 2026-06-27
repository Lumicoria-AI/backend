"""Celery shim for the autonomous brain.

Two beat-scheduled fan-outs:

  ``brain.fanout_morning``  — runs every hour at :00. For every user whose
                              brain preferences enable morning + whose
                              local TZ shows a 06:00 ± 15-min window
                              right now, queues ``brain.run_morning_for_user``.

  ``brain.fanout_evening``  — same shape but for the evening 22:00 window.

Plus two per-user tasks:

  ``brain.run_morning_for_user(user_id)``  — wraps `run_brain_for_user(mode="morning")`.
  ``brain.run_evening_for_user(user_id)``  — wraps `run_brain_for_user(mode="evening")`.

Plus a manual-trigger task used by ``POST /brain/trigger``:

  ``brain.run_on_demand(user_id, mode)``  — fires regardless of TZ + prefs,
                                            for dev / debug.

Idempotency lives inside the runner (per-run UUID + gate's dedup check
against ``last_brain_*_sent``), so re-queuing a fan-out is safe.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from celery import shared_task  # noqa: F401  (kept for parity with siblings)

from backend.tasks.async_utils import run_worker_coro
from backend.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Async runner — one persistent loop per Celery worker process. Keeping
# all background async work on the same loop prevents stale Motor clients
# from pointing at loops closed by asyncio.run().
# ─────────────────────────────────────────────────────────────────────


def _run(coro):
    """Run an async coroutine on the worker's persistent event loop."""
    return run_worker_coro(coro)


# ─────────────────────────────────────────────────────────────────────
# Per-user worker tasks
# ─────────────────────────────────────────────────────────────────────


@celery_app.task(
    name="brain.run_morning_for_user",
    bind=True,
    max_retries=2,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
)
def run_morning_for_user(self, user_id: str) -> Dict[str, Any]:
    """Run one morning brain pass for ``user_id``."""
    from backend.services.brain.runner import run_brain_for_user

    summary = _run(
        run_brain_for_user(
            user_id=user_id, mode="morning", initiated_by="celery_beat",
        )
    )
    return summary.model_dump()


@celery_app.task(
    name="brain.run_evening_for_user",
    bind=True,
    max_retries=2,
    autoretry_for=(Exception,),
    retry_backoff=True,
    retry_backoff_max=120,
    retry_jitter=True,
)
def run_evening_for_user(self, user_id: str) -> Dict[str, Any]:
    """Run one evening brain pass for ``user_id``."""
    from backend.services.brain.runner import run_brain_for_user

    summary = _run(
        run_brain_for_user(
            user_id=user_id, mode="evening", initiated_by="celery_beat",
        )
    )
    return summary.model_dump()


@celery_app.task(
    name="brain.run_on_demand",
    bind=True,
    max_retries=0,
)
def run_on_demand(self, user_id: str, mode: str = "morning") -> Dict[str, Any]:
    """Trigger a brain run on demand — used by ``POST /brain/trigger``
    and by dev scripts. ``force=True`` skips the TZ gate so a developer
    can fire a run at any time."""
    from backend.services.brain.runner import run_brain_for_user

    summary = _run(
        run_brain_for_user(
            user_id=user_id, mode=mode, force=True, initiated_by="api",
        )
    )
    return summary.model_dump()


# ─────────────────────────────────────────────────────────────────────
# Fan-out tasks (beat-scheduled)
# ─────────────────────────────────────────────────────────────────────


@celery_app.task(name="brain.fanout_morning", bind=True, max_retries=1)
def fanout_morning(self) -> Dict[str, Any]:
    """Hourly beat tick: queue a morning brain task for every user
    whose preference window matches the current local 06:00 ± 15 min."""
    return _fanout_for_mode(mode="morning")


@celery_app.task(name="brain.fanout_evening", bind=True, max_retries=1)
def fanout_evening(self) -> Dict[str, Any]:
    """Hourly beat tick: queue an evening brain task for every user
    whose preference window matches the current local 22:00 ± 15 min."""
    return _fanout_for_mode(mode="evening")


def _fanout_for_mode(mode: str) -> Dict[str, Any]:
    """Shared fan-out body. Discovers candidate user_ids, dispatches one
    per-user task each. Per-user gating (TZ + last_sent dedupe) happens
    inside the runner's gate node so we don't re-implement it here."""
    summary = _run(_collect_candidates(mode=mode))
    user_ids: List[str] = summary["user_ids"]
    queued = 0

    for uid in user_ids:
        try:
            if mode == "morning":
                run_morning_for_user.delay(uid)
            else:
                run_evening_for_user.delay(uid)
            queued += 1
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "brain.fanout_enqueue_failed",
                mode=mode, user_id=uid, error=str(exc),
            )

    logger.info(
        "brain.fanout_complete",
        mode=mode, candidates=len(user_ids), queued=queued,
    )
    return {"mode": mode, "candidates": len(user_ids), "queued": queued}


async def _collect_candidates(mode: str) -> Dict[str, Any]:
    """Return the set of user_ids whose local TZ window matches `mode`
    *and* whose brain preferences are enabled.

    Phase 2 keeps this conservative: only users with
    ``preferences.brain.enabled == True`` AND a Google Workspace
    integration. The gate node re-validates per user, so this is an
    upper bound — bad picks here become skipped runs, not crashes.
    """
    from backend.db.mongodb.mongodb import MongoDB
    from backend.services.brain._time import now_hour_matches_tz

    db = await MongoDB.get_database()
    users_col = db.users

    # Pull only the fields we need for gating. Pre-filtering on
    # `preferences.brain.enabled=True` shrinks the working set on a
    # large org dramatically.
    cursor = users_col.find(
        {"preferences.brain.enabled": True},
        projection={"_id": 1, "preferences": 1},
    )

    matched: List[str] = []
    async for u in cursor:
        prefs = (u.get("preferences") or {}).get("brain") or {}
        if not prefs.get("enabled", False):
            continue
        tz = (u.get("preferences") or {}).get("timezone") or "UTC"
        target_hour = int(
            prefs.get("morning_hour_local", 6)
            if mode == "morning"
            else prefs.get("evening_hour_local", 22)
        )
        if now_hour_matches_tz(tz, target_hour, tolerance_minutes=15):
            matched.append(str(u["_id"]))

    return {"user_ids": matched}
