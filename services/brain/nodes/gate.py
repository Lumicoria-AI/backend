"""Gate node — decides whether the run should proceed.

Phase 2 stub: always passes (sets no skip_reason). Phase 3+ will check:
  - user.preferences["brain"]["enabled"]
  - now_hour_matches_tz(timezone, target_hour) — except when triggered
    manually (state.meta["force"] == True)
  - user has an active Google integration with the gmail.readonly scope
  - last_brain_*_sent dedupe — don't re-run if already sent today
  - per-user Redis lock so two workers can't double-fire
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState
from ..tracing import traced_node


@traced_node("gate")
async def gate(state: BrainState) -> Dict[str, Any]:
    # Phase 2 — always proceed. The gate runs first so its trace row
    # shows up even when the run is skipped, giving us a record that
    # the system at least *checked* whether to run.
    return {
        "__payload_summary": {
            "mode": state.mode,
            "timezone": state.timezone,
            "force": bool(state.meta.get("force")),
        },
        "__eval_score": 1.0,
    }
