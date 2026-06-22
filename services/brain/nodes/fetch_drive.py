"""Fetch Drive files changed since the last run.

Phase 2 stub: empty list. Phase 3 will:
  - Call `google_workspace_client.list_drive_changes(start_page_token)`.
  - First run for a user: just get the bootstrap page token (no changes
    yet) and store it on user.preferences["brain"]["drive_page_token"].
  - Subsequent runs: pull changes since that token, update the token.
  - Filters: skip trashed, skip files >50MB (cap from preferences).
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState
from ..tracing import traced_node


@traced_node("fetch_drive")
async def fetch_drive(state: BrainState) -> Dict[str, Any]:
    return {
        "drive_changes": [],
        "__payload_summary": {"count": 0, "mode": state.mode},
        "__eval_score": 1.0,
    }
