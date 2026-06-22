"""Pull recent huddle transcripts (last 24h) for evening recap.

Phase 2 stub. Phase 3 will:
  - SELECT * FROM huddles WHERE ended_at > now() - 24h AND
    host_user_id = :user_id (or in participant table).
  - Pull the summary + key decisions from `meetings` (linked via
    huddle.processed_meeting_id).
  - Feed into prioritise so "what your team agreed yesterday → today's
    follow-ups" works.
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState
from ..tracing import traced_node


@traced_node("fetch_huddle")
async def fetch_huddle(state: BrainState) -> Dict[str, Any]:
    return {
        "huddle_recents": [],
        "__payload_summary": {"count": 0},
        "__eval_score": 1.0,
    }
