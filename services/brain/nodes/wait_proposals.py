"""Poll Mongo for agent proposals until the top-5 high-priority tasks
have one or we hit a 90-second timeout.

Phase 2 stub. Phase 3 will:
  - Identify top-5 high-priority created tasks (priority=critical/high).
  - Poll `task_repository.get_task` every 3s with asyncio.sleep.
  - Break early when all top-5 have agent_proposal.status != "queued".
  - Hard timeout 90s — past that, the digest goes out with whatever's
    ready and a "more agents are still drafting; check back in a few
    minutes" line.
  - Updates state.proposal_status_by_task with the live statuses
    ("pending_review", "approved", "error", "draft").
  - This node is the only one in the graph that intentionally blocks
    for a long time — Celery's task_acks_late=True keeps the run
    alive even if the worker restarts.
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState
from ..tracing import traced_node


@traced_node("wait_proposals")
async def wait_proposals(state: BrainState) -> Dict[str, Any]:
    return {
        # With no tasks created, nothing to wait for.
        "__payload_summary": {
            "watched": len(state.created_task_ids),
            "ready": 0,
            "timed_out": 0,
        },
        "__eval_score": 1.0,
    }
