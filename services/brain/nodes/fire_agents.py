"""Fire the autonomous executor for each created task.

For every task we just created that has ``assigned_to_agent`` set on
the underlying record, queue ``brain.run_single_agent_proposal(task_id)``.
Celery picks it up, the task_executor instantiates the right
specialist agent, runs ``process_async``, persists ``agent_proposal``
on the task, and pings the owner with the existing notification
pipeline. By the time ``wait_proposals`` polls a few seconds later,
the top-priority items typically have ``status="pending_review"``
ready to render in the digest's review buttons.

We always set ``proposal_status_by_task[task_id] = "queued"`` so the
poll node knows what to watch. Tasks without an agent assignment
remain in state but are skipped here — they'll appear in the digest
without a proposal section, and the user can assign an agent later.
"""

from __future__ import annotations

from typing import Any, Dict

import structlog

from ..state import BrainState
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


@traced_node("fire_agents")
async def fire_agents(state: BrainState) -> Dict[str, Any]:
    if not state.created_task_ids:
        return {
            "proposal_status_by_task": {},
            "__payload_summary": {"tasks": 0, "fired": 0},
            "__eval_score": 1.0,
        }

    # Load the freshly-created tasks so we can read assigned_to_agent.
    try:
        from bson import ObjectId
        from backend.db.mongodb.mongodb import MongoDB
        db = await MongoDB.get_database()
        oids = []
        for tid in state.created_task_ids:
            try:
                oids.append(ObjectId(tid))
            except Exception:
                oids.append(tid)
        cursor = db.tasks.find(
            {"_id": {"$in": oids}},
            projection={"_id": 1, "assigned_to_agent": 1, "organization_id": 1},
        )
        tasks = [t async for t in cursor]
    except Exception as exc:  # noqa: BLE001
        logger.warning("fire_agents.task_lookup_failed", error=str(exc))
        return {
            "proposal_status_by_task": {tid: "queued" for tid in state.created_task_ids},
            "__payload_summary": {"tasks": len(state.created_task_ids), "fired": 0,
                                  "error": "lookup_failed"},
            "__eval_score": 0.0,
            "__status": "fallback",
        }

    status_map: Dict[str, str] = {}
    fired = 0

    try:
        from backend.tasks.task_executor_tasks import run_single_agent_proposal
    except Exception as exc:  # noqa: BLE001
        logger.warning("fire_agents.celery_import_failed", error=str(exc))
        run_single_agent_proposal = None

    for t in tasks:
        tid = str(t.get("_id"))
        if not t.get("assigned_to_agent"):
            status_map[tid] = "unassigned"
            continue

        status_map[tid] = "queued"
        org_id = str(t.get("organization_id") or state.organization_id or state.user_id)
        if run_single_agent_proposal is None:
            continue
        try:
            run_single_agent_proposal.delay(tid, org_id)
            fired += 1
        except Exception as exc:  # noqa: BLE001
            logger.debug(
                "fire_agents.celery_dispatch_failed",
                task_id=tid, error=str(exc),
            )
            status_map[tid] = "dispatch_failed"

    return {
        "proposal_status_by_task": status_map,
        "__payload_summary": {
            "tasks": len(tasks),
            "fired": fired,
            "unassigned": sum(1 for s in status_map.values() if s == "unassigned"),
        },
        "__eval_score": 1.0 if fired or not state.created_task_ids else 0.5,
    }
