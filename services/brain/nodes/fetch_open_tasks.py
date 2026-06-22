"""Pull the user's currently-open Lumicoria tasks.

Phase 2 stub. Phase 3 will:
  - Mongo `tasks` find: assigned_to OR created_by = user_id AND
    status NOT IN ('completed','cancelled') AND deleted_at IS NULL.
  - Limit 50, sort by priority desc + due_date asc.
  - Used by the evening compose to render "still open" + by morning
    prioritise so we don't propose tasks that already exist.
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState
from ..tracing import traced_node


@traced_node("fetch_open_tasks")
async def fetch_open_tasks(state: BrainState) -> Dict[str, Any]:
    return {
        "open_tasks": [],
        "__payload_summary": {"count": 0, "mode": state.mode},
        "__eval_score": 1.0,
    }
