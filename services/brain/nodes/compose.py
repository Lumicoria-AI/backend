"""Build the DigestPayload from everything the graph collected.

Phase 2 stub. Phase 3 will:
  - Top actions = the 5 highest-priority ranked_actions.
  - Secondary actions = the rest.
  - Calendar today = events with start.date() == today (user TZ).
  - Open tasks = state.open_tasks, capped at 5.
  - Completed today (evening) = closed tasks where completed_at is today.
  - Summary line = one-sentence "Today: 3 priorities, 2 meetings,
    1 proposal awaits approval."
  - Counts dict = whole-run telemetry: emails_processed, drive_changes,
    tasks_created, proposals_drafted.
  - Each top_action carries signed action tokens for the email buttons
    (via `services/task_action_tokens.make_action_token`).
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState, DigestPayload
from ..tracing import traced_node


@traced_node("compose")
async def compose(state: BrainState) -> Dict[str, Any]:
    # Phase 2 — synthesize the simplest valid payload so send + audit
    # have something to render.
    payload = DigestPayload(
        mode=state.mode,
        user_name=None,
        summary_line=(
            "Brain is online but no data sources are wired yet "
            f"({state.mode} run — Phase 2 stub)."
        ),
        top_actions=[],
        secondary_actions=[],
        calendar_today=state.events,
        completed_today=[],
        open_tasks=state.open_tasks,
        counts={
            "emails": len(state.emails),
            "drive_changes": len(state.drive_changes),
            "tasks_created": len(state.created_task_ids),
        },
    )

    return {
        "digest_payload": payload,
        "__payload_summary": {
            "top_actions": 0,
            "secondary_actions": 0,
            "calendar_items": len(state.events),
        },
        "__eval_score": 1.0,
    }
