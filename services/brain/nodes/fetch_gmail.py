"""Fetch the user's Gmail messages since the last brain run.

Phase 2 stub: returns an empty list. Phase 3 will:
  1. Resolve the user's Google integration via
     `services.integration_service.get_user_integration` — this auto-
     refreshes the OAuth token if it's about to expire.
  2. Build a `GoogleWorkspaceClient(integ["credentials"])`.
  3. Call `client.list_message_ids_since(after_epoch_seconds=...)` —
     respects user.preferences["brain"]["mailbox_labels_exclude"].
  4. Batch-fetch each id via `client.get_message(...)` with
     `asyncio.Semaphore(3)` to stay under Gmail's per-user 250/100s
     quota.
  5. Build GmailMessageRef objects (subject, from, snippet, attachment
     ids) — bodies do not enter state; they're fetched again by the
     ingest node, hashed, and either inserted or skipped if we've seen
     the message_id before.
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState
from ..tracing import traced_node


@traced_node("fetch_gmail")
async def fetch_gmail(state: BrainState) -> Dict[str, Any]:
    # Phase 2 — no API call. We seed an empty list with the right shape
    # so downstream nodes still get a well-formed BrainState.
    return {
        "emails": [],
        "__payload_summary": {"count": 0, "mode": state.mode},
        "__eval_score": 1.0,
    }
