"""
Celery tasks that drive the Phase 6 autonomous task executor.

A single periodic task (`tasks.run_pending_agent_proposals`) is enough:
on each tick it scans for tasks whose `assigned_to_agent` is set and
no approved proposal yet, runs the agent, and stores the proposal.

Beat schedule lives in `backend/tasks/celery_app.py`.
"""

from __future__ import annotations

from typing import Any, Dict

import structlog

from backend.tasks.async_utils import run_worker_coro
from backend.tasks.celery_app import celery_app

logger = structlog.get_logger(__name__)


def _run(coro):
    """Run an async coroutine on the worker's persistent event loop.

    Same pattern as task_reminder_tasks: one loop per worker so Motor's
    connection pool stays valid across invocations.
    """
    return run_worker_coro(coro)


@celery_app.task(name="tasks.run_pending_agent_proposals", bind=True, max_retries=1)
def run_pending_agent_proposals(self, limit: int = 25) -> Dict[str, Any]:
    """Scan + execute the queue of tasks needing an agent draft."""
    from backend.services.task_executor import run_pending_proposals
    try:
        return _run(run_pending_proposals(limit=limit))
    except Exception as e:  # noqa: BLE001
        logger.warning("run_pending_agent_proposals_failed", error=str(e))
        return {"picked": 0, "ok": 0, "errors": 1, "error": str(e)[:200]}


@celery_app.task(name="tasks.run_single_agent_proposal", bind=True, max_retries=1)
def run_single_agent_proposal(self, task_id: str, organization_id: str) -> Dict[str, Any]:
    """Trigger an immediate draft for one task (used by manual 'Run agent now')."""
    from backend.db.mongodb.repositories.task_repository import task_repository
    from backend.services.task_executor import execute_task

    async def _go():
        task = await task_repository.get_task_by_id(task_id, organization_id=organization_id)
        if not task:
            return {"status": "not_found"}
        return await execute_task(task)

    try:
        return _run(_go())
    except Exception as e:  # noqa: BLE001
        logger.warning("run_single_agent_proposal_failed", task_id=task_id, error=str(e))
        return {"status": "error", "error": str(e)[:200]}
