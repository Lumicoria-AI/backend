"""
Phase B follow-up — Agent metrics materialiser.

Celery beat task that rebuilds `agent_metrics` from raw `agent_runs` for
the standard windows (day/week/month).  Once this is running on the
schedule, the workspace leaderboard + analytics drill-downs read from a
pre-aggregated table instead of computing on every request.
"""

from __future__ import annotations

from typing import Any, Dict

import structlog

from backend.tasks.async_utils import run_worker_coro
from backend.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)


async def _rebuild_all_windows() -> Dict[str, int]:
    """Rebuild metrics for every standard window across all orgs."""
    from backend.db.mongodb.repositories.agent_metrics_repository import (
        agent_metrics_repository,
    )

    out: Dict[str, int] = {}
    for window in ("day", "week", "month"):
        try:
            written = await agent_metrics_repository.rebuild(window=window)
            out[window] = written
        except Exception as exc:  # noqa: BLE001
            logger.exception("agent_metrics.window_rebuild_failed", window=window, error=str(exc))
            out[window] = -1
    return out


@celery_app.task(name="agent_metrics.materialise", bind=True, max_retries=3)
def materialise_agent_metrics(self) -> Dict[str, Any]:
    """Rebuild agent_metrics for day/week/month windows."""
    try:
        result = run_worker_coro(_rebuild_all_windows())
        logger.info("agent_metrics.materialised", **result)
        return result
    except Exception as exc:  # noqa: BLE001
        logger.exception("agent_metrics.materialise_failed", error=str(exc))
        raise self.retry(exc=exc, countdown=60)
