"""
Lumicoria Huddle — outbound event helpers.

  - emit_webhook(org_id, event, payload): record one delivery per
    subscribed org webhook. The existing Celery worker
    (tasks.webhook_dispatcher.deliver_due_webhooks) flushes deliveries.
  - emit_slack_huddle_started(org_id, huddle): post "🔴 Huddle live"
    into the org's Slack workspace (best-effort).

Both helpers are fire-and-forget — failures must NEVER block the
HTTP response in huddle_service / huddle.py.
"""

from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


# Events Huddle emits. Add to the catalogue of webhook-subscribable events.
HUDDLE_EVENTS = [
    "huddle.created",
    "huddle.started",
    "huddle.ended",
    "huddle.recording_ready",
    "huddle.participant_joined",
    "huddle.participant_left",
]


async def emit_webhook(
    organization_id: str,
    event: str,
    payload: Dict[str, Any],
) -> int:
    """Find every org webhook subscribed to `event` and queue a delivery.
    Returns the number of deliveries queued."""
    if not organization_id or not event:
        return 0
    try:
        from backend.db.mongodb.repositories.webhooks_repository import webhooks_repository
        hooks = await webhooks_repository.list(organization_id=organization_id, enabled_only=True)
    except Exception as e:
        logger.warning("huddle_webhook_lookup_failed", org=organization_id, error=str(e))
        return 0

    queued = 0
    enriched = {
        **payload,
        "event": event,
        "delivered_at": datetime.utcnow().isoformat() + "Z",
    }
    for h in hooks:
        events: List[str] = list(h.get("events") or [])
        if events and event not in events and "huddle.*" not in events and "*" not in events:
            continue
        try:
            await webhooks_repository.record_delivery(
                webhook_id=h["id"], organization_id=organization_id,
                event=event, payload=enriched, status="pending",
            )
            queued += 1
        except Exception as e:
            logger.warning("huddle_webhook_queue_failed", webhook_id=h.get("id"), error=str(e))
    return queued


async def emit_slack_huddle_started(
    organization_id: str,
    huddle: Dict[str, Any],
) -> bool:
    """If the org has a Slack integration configured, post a brief
    "🔴 Huddle live" message with the share URL. Best-effort."""
    try:
        from backend.integrations.slack import SlackIntegration  # type: ignore
    except Exception:
        return False

    base = "https://lumicoria.ai"
    try:
        from backend.core.config import settings as _settings
        base = getattr(_settings, "FRONTEND_URL", base) or base
    except Exception:
        pass
    share_url = f"{base}/huddles/join/{huddle.get('share_token','')}"

    title = huddle.get("title") or "A Lumicoria Huddle"
    text = (
        f":red_circle: *{title}* is live now\n"
        f"<{share_url}|Join the meeting>"
    )
    blocks = [
        {
            "type": "section",
            "text": {"type": "mrkdwn", "text": text},
            "accessory": {
                "type": "button",
                "text": {"type": "plain_text", "text": "Join"},
                "url": share_url,
                "style": "primary",
            },
        }
    ]

    # Channel resolution strategy:
    #   1. metadata.slack_channel_id if explicitly set on the huddle
    #   2. integration default channel
    channel_id: Optional[str] = (huddle.get("metadata") or {}).get("slack_channel_id")

    try:
        integ = SlackIntegration(organization_id=organization_id)  # type: ignore[arg-type]
        ok = await integ.post_message(
            channel=channel_id,
            text=text,
            blocks=blocks,
        )
        return bool(ok)
    except TypeError:
        # SlackIntegration may take (client=...) form — try the client method directly.
        try:
            from backend.integrations.slack_client import SlackClient  # type: ignore
            client = SlackClient(organization_id=organization_id)
            await client.post_message(channel=channel_id, text=text, blocks=blocks)
            return True
        except Exception as e:
            logger.warning("huddle_slack_post_failed", org=organization_id, error=str(e))
            return False
    except Exception as e:
        logger.warning("huddle_slack_post_failed", org=organization_id, error=str(e))
        return False


def fire_and_forget(coro) -> None:
    """Schedule a coroutine without blocking the caller. Discards exceptions."""
    try:
        asyncio.create_task(coro)
    except Exception as e:
        logger.warning("huddle_fire_and_forget_failed", error=str(e))
