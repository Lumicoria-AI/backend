"""Deliver the digest via email, in-app, and (optionally) push.

Phase 2 stub. Phase 3 will:
  - Email: `notification_service.send_email_notification(template_name=
    "morning_brain_digest" or "evening_brain_review", template_data=
    payload.dict())`. Channels gated by user.preferences["brain"]["send_email"].
  - In-app: `create_in_app_notification(category="brain.morning"/"brain.evening")`.
  - Push: only when payload has at least one critical action and
    user.preferences["brain"]["send_push"] is true.
  - Each channel wrapped in try/except — one failing channel doesn't
    block the others.
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState
from ..tracing import traced_node


@traced_node("send")
async def send(state: BrainState) -> Dict[str, Any]:
    # Phase 2 — we don't actually send anything. We record the *intent*
    # so the audit row shows which channels would have fired.
    intended_channels = []
    if state.user_email:
        intended_channels.append("email")
    intended_channels.append("in_app")

    return {
        "delivery_channels": intended_channels,
        "__payload_summary": {
            "channels": intended_channels,
            "sent": 0,
        },
        "__eval_score": 1.0,
    }
