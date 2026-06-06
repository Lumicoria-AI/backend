"""
Lumicoria AI — Event Bus

In-process publish-subscribe + Redis pub/sub for cross-worker fanout.

Every state mutation that downstream systems (automations, notifications,
real-time WS, analytics) might care about emits an event here:

    await event_bus.publish(
        "task.created",
        organization_id=str(org_id),
        actor_id=str(user_id),
        payload={"task_id": str(tid), "title": t.title},
    )

Local handlers fire synchronously inside the calling worker (good for the
notification engine + automation matcher), and the event is also forwarded
to Redis so a WS worker on a different process can broadcast it.

Failures in any single handler are logged and swallowed so the producer is
never blocked.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, List, Optional

import structlog

from backend.db.redis.redis import RedisClient

logger = structlog.get_logger(__name__)

REDIS_CHANNEL_PREFIX = "lumi:events"

Handler = Callable[["Event"], Awaitable[None]]


@dataclass
class Event:
    """An event flowing through the bus."""
    type: str                                      # "task.created", "member.added", etc.
    organization_id: Optional[str] = None
    actor_id: Optional[str] = None                 # user_id who triggered it (None for system)
    team_id: Optional[str] = None
    project_id: Optional[str] = None
    resource_type: Optional[str] = None            # "task" | "project" | "member" | ...
    resource_id: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    severity: str = "info"                         # info | warning | error
    source: str = "api"                            # api | ui | automation | agent | system
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")

    def to_json(self) -> str:
        return json.dumps(asdict(self), default=str)

    @classmethod
    def from_json(cls, blob: str) -> "Event":
        data = json.loads(blob)
        return cls(**data)


class EventBus:
    """Process-local pub/sub + Redis pub/sub forwarder."""

    def __init__(self) -> None:
        self._handlers: Dict[str, List[Handler]] = {}
        self._wildcard_handlers: List[Handler] = []
        self._redis_forward_enabled: bool = True

    # ---------------------------------------------------------------- API

    def subscribe(self, event_type: str, handler: Handler) -> None:
        """Subscribe to a specific event type (or '*' for everything)."""
        if event_type == "*":
            self._wildcard_handlers.append(handler)
            return
        self._handlers.setdefault(event_type, []).append(handler)

    def unsubscribe(self, event_type: str, handler: Handler) -> None:
        if event_type == "*" and handler in self._wildcard_handlers:
            self._wildcard_handlers.remove(handler)
            return
        if event_type in self._handlers and handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    async def publish(
        self,
        event_type: str,
        *,
        organization_id: Optional[str] = None,
        actor_id: Optional[str] = None,
        team_id: Optional[str] = None,
        project_id: Optional[str] = None,
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
        severity: str = "info",
        source: str = "api",
    ) -> None:
        evt = Event(
            type=event_type,
            organization_id=organization_id,
            actor_id=actor_id,
            team_id=team_id,
            project_id=project_id,
            resource_type=resource_type,
            resource_id=resource_id,
            payload=payload or {},
            severity=severity,
            source=source,
        )

        # 1) Fire local handlers (best-effort)
        handlers = list(self._handlers.get(event_type, [])) + list(self._wildcard_handlers)
        if handlers:
            await asyncio.gather(
                *(self._safe_call(h, evt) for h in handlers),
                return_exceptions=False,
            )

        # 2) Forward to Redis so other processes can react.
        if self._redis_forward_enabled:
            try:
                client = await RedisClient.get_client()
                await client.publish(self._channel_for(evt), evt.to_json())
            except Exception as exc:  # noqa: BLE001
                logger.warning("event_bus.redis_publish_failed", event=event_type, error=str(exc))

    # ---------------------------------------------------------------- internals

    @staticmethod
    async def _safe_call(handler: Handler, evt: Event) -> None:
        try:
            await handler(evt)
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "event_bus.handler_failed",
                event=evt.type,
                handler=getattr(handler, "__qualname__", repr(handler)),
                error=str(exc),
            )

    @staticmethod
    def _channel_for(evt: Event) -> str:
        # We publish to a per-org topic so subscribers can filter cheaply.
        if evt.organization_id:
            return f"{REDIS_CHANNEL_PREFIX}:org:{evt.organization_id}"
        return f"{REDIS_CHANNEL_PREFIX}:global"


event_bus = EventBus()


# ---------------------------------------------------------------------- helpers

async def emit(
    event_type: str,
    *,
    organization_id: Optional[str] = None,
    actor_id: Optional[str] = None,
    team_id: Optional[str] = None,
    project_id: Optional[str] = None,
    resource_type: Optional[str] = None,
    resource_id: Optional[str] = None,
    payload: Optional[Dict[str, Any]] = None,
    severity: str = "info",
    source: str = "api",
) -> None:
    """Convenience wrapper — never raises."""
    try:
        await event_bus.publish(
            event_type,
            organization_id=organization_id,
            actor_id=actor_id,
            team_id=team_id,
            project_id=project_id,
            resource_type=resource_type,
            resource_id=resource_id,
            payload=payload,
            severity=severity,
            source=source,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("event_bus.publish_failed", event=event_type, error=str(exc))
