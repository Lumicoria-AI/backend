"""Fetch calendar events around the run window.

Phase 2 stub: empty list. Phase 3 will:
  - Morning: events for *today* + first half of tomorrow.
  - Evening: events from *today* + tomorrow (for "tomorrow's focus").
  - Uses `google_workspace_client.get_events` (already wired).
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState
from ..tracing import traced_node


@traced_node("fetch_calendar")
async def fetch_calendar(state: BrainState) -> Dict[str, Any]:
    return {
        "events": [],
        "__payload_summary": {"count": 0, "mode": state.mode},
        "__eval_score": 1.0,
    }
