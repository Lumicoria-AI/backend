"""Kick off the autonomous executor for each newly-created task.

Phase 2 stub. Phase 3 will:
  - For each task_id in state.created_task_ids:
    `run_single_agent_proposal.delay(task_id, organization_id)` —
    fire-and-forget (Celery handles retry).
  - Skip tasks where assigned_to_agent is None (the user has to assign).
  - Record `proposal_status_by_task[task_id] = "queued"` so the next
    node knows what to wait for.
  - No retry inside this node — Celery does it.
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState
from ..tracing import traced_node


@traced_node("fire_agents")
async def fire_agents(state: BrainState) -> Dict[str, Any]:
    return {
        "proposal_status_by_task": {
            tid: "queued" for tid in state.created_task_ids
        },
        "__payload_summary": {
            "tasks": len(state.created_task_ids),
            "fired": 0,
        },
        "__eval_score": 1.0,
    }
