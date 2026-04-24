"""Ingest progress pub/sub.

Events flow:
    Celery worker → publish(document_id, event)  — sync Redis
                                        │
                                        ▼
                       channel: ingest:progress:{document_id}
                                        │
                                        ▼
    SSE endpoint  ← subscribe(document_id)       — async Redis

The latest event is also stored at key `ingest:state:{document_id}` with a
1-hour TTL so a client that connects after the task has already emitted a
couple of events still sees current state instead of a blank stream.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any, AsyncIterator, Dict, Optional

import structlog

from ...core.config import settings

logger = structlog.get_logger(__name__)

_STATE_TTL_SECONDS = 3600


def _channel(document_id: str) -> str:
    return f"ingest:progress:{document_id}"


def _state_key(document_id: str) -> str:
    return f"ingest:state:{document_id}"


# ── Publisher (sync — used from Celery worker) ─────────────────────────

_sync_redis = None


def _get_sync_redis():
    """Lazy sync Redis client — `redis` package ships a sync flavour."""
    global _sync_redis
    if _sync_redis is None:
        import redis  # sync client
        _sync_redis = redis.Redis(
            host=settings.db.REDIS_HOST,
            port=settings.db.REDIS_PORT,
            password=settings.db.REDIS_PASSWORD,
            db=settings.db.REDIS_DB,
            decode_responses=True,
        )
    return _sync_redis


def publish(document_id: str, event: Dict[str, Any]) -> None:
    """Publish a progress event. Safe to call from a sync Celery task."""
    try:
        client = _get_sync_redis()
        payload = json.dumps({"document_id": document_id, **event})
        client.publish(_channel(document_id), payload)
        client.set(_state_key(document_id), payload, ex=_STATE_TTL_SECONDS)
    except Exception as e:
        # Progress is best-effort — never fail the ingest on a pub/sub hiccup.
        logger.warning("progress_publish_failed", error=str(e), document_id=document_id)


def stage(document_id: str, stage: str, **extra: Any) -> None:
    """Convenience wrapper: publish({'stage': stage, ...})."""
    publish(document_id, {"stage": stage, **extra})


# ── Subscriber (async — used from SSE endpoint) ────────────────────────


async def get_last_state(document_id: str) -> Optional[Dict[str, Any]]:
    from redis.asyncio import Redis

    client = Redis(
        host=settings.db.REDIS_HOST,
        port=settings.db.REDIS_PORT,
        password=settings.db.REDIS_PASSWORD,
        db=settings.db.REDIS_DB,
        decode_responses=True,
    )
    try:
        raw = await client.get(_state_key(document_id))
        return json.loads(raw) if raw else None
    finally:
        await client.close()


async def subscribe(
    document_id: str,
    *,
    idle_heartbeat_seconds: float = 15.0,
) -> AsyncIterator[Dict[str, Any]]:
    """Yield progress events for a document until a terminal stage arrives.

    Terminal stages: 'ready', 'error', 'cancelled'. A heartbeat event is
    yielded every `idle_heartbeat_seconds` so SSE clients don't time out.
    """
    from redis.asyncio import Redis

    client = Redis(
        host=settings.db.REDIS_HOST,
        port=settings.db.REDIS_PORT,
        password=settings.db.REDIS_PASSWORD,
        db=settings.db.REDIS_DB,
        decode_responses=True,
    )
    pubsub = client.pubsub()
    try:
        await pubsub.subscribe(_channel(document_id))

        # Replay last cached state so clients connecting late see progress.
        last = await get_last_state(document_id)
        if last:
            yield last
            if last.get("stage") in {"ready", "error", "cancelled"}:
                return

        while True:
            try:
                msg = await asyncio.wait_for(
                    pubsub.get_message(ignore_subscribe_messages=True),
                    timeout=idle_heartbeat_seconds,
                )
            except asyncio.TimeoutError:
                yield {"stage": "heartbeat", "document_id": document_id}
                continue

            if msg is None:
                continue
            data = msg.get("data")
            if not data:
                continue
            try:
                event = json.loads(data)
            except json.JSONDecodeError:
                continue

            yield event
            if event.get("stage") in {"ready", "error", "cancelled"}:
                return
    finally:
        try:
            await pubsub.unsubscribe(_channel(document_id))
            await pubsub.close()
        except Exception:
            pass
        await client.close()
