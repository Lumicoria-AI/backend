"""Close out the run — write final activity log entries.

The Postgres BrainRun + BrainTrace rows are written by the runner
*after* this node returns, in a single batched commit. This node just
records app-level activity events (so the org's audit log shows
"brain ran for user X at hh:mm — Y tasks created").

Phase 2 stub. Phase 3 will:
  - log_activity(activity_type="brain.run_completed", organization_id,
    actor_id=user_id, details={mode, tasks_created, digest_sent}).
  - If state.fallback_count > 0: also log "brain.degraded".
  - These rows are what the user sees in /workspace/admin/audit.
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState
from ..tracing import traced_node


@traced_node("audit")
async def audit(state: BrainState) -> Dict[str, Any]:
    return {
        "__payload_summary": {
            "fallback_count": state.fallback_count,
            "delivery_channels": state.delivery_channels,
            "trace_event_count": len(state.trace_events),
        },
        "__eval_score": 1.0,
    }
