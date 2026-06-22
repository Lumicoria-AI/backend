"""Persist ranked actions as Lumicoria tasks.

Phase 2 stub. Phase 3 will:
  - For each RankedAction call `task_repository.create_task` directly —
    bypasses the API endpoint so we don't double-trip CORS / FastAPI.
  - Metadata: `source="brain"`, `run_id`, `evidence_message_ids`,
    `evidence_event_ids`, `evidence_file_ids`, `digest_run_id`.
  - assigned_to_agent populated → the create endpoint already fires
    `run_single_agent_proposal.delay`, but we bypass that and fire it
    ourselves in the next node (`fire_agents`) so we control ordering.
  - Idempotency: if a task with metadata.run_id == this run_id already
    exists for this title (re-trigger of the same morning), reuse it.
  - Per-task retry x2. Failures don't abort — log and continue.
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState
from ..tracing import traced_node


@traced_node("create_tasks")
async def create_tasks(state: BrainState) -> Dict[str, Any]:
    return {
        "created_task_ids": [],
        "__payload_summary": {
            "ranked_actions": len(state.ranked_actions),
            "created": 0,
            "skipped_duplicates": 0,
            "failed": 0,
        },
        "__eval_score": 1.0,
    }
