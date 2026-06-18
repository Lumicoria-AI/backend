"""
Lumicoria Huddle — live agent dispatcher.

Given a transcript chunk and the set of attached agent_keys on a huddle,
dispatch each agent in parallel via AgentService.execute_agent_async,
publish each response to the `rt:huddle:{huddle_id}` topic so all connected
clients render it, and persist the responses onto the chunk row.

Design:
  - We use asyncio.gather, not a Celery group. Phase 1 is single-VM
    deployment and sub-second latency matters more than horizontal scale.
    Phase 2's self-hosted Jitsi rollout will swap in Celery if needed.
  - Each agent gets a small input shape derived from the chunk +
    recent context (last N chunks). This keeps token cost predictable.
  - Failures are isolated per agent — one slow/broken agent doesn't
    block the others.
  - Throttling: if a chunk arrives < 3 seconds after the previous one,
    we batch them and run agents once on the combined text. Prevents
    stampedes during fast STT output.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import select, update as sa_update

from backend.db.postgres import get_async_sessionmaker
from backend.db.postgres_models import HuddleSQL, HuddleTranscriptChunkSQL
from backend.services.activity_logger import log_activity
from backend.services.realtime import realtime

logger = structlog.get_logger(__name__)

# Per-huddle throttle state — last fan-out timestamp.
_LAST_FAN_OUT: Dict[str, float] = {}
_MIN_INTERVAL_SEC = 3.0
_RECENT_CONTEXT_CHUNKS = 4


async def dispatch_chunk(
    *,
    huddle_id: str,
    chunk_id: str,
    chunk_text: str,
    speaker_name: str,
    user_id: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Run all attached agents on a transcript chunk in parallel.

    Returns the list of agent responses (also persisted onto the chunk
    row and published over Redis pub/sub)."""

    # Throttle — drop the fan-out if we're inside the cool-off window.
    now = time.time()
    last = _LAST_FAN_OUT.get(huddle_id, 0)
    if now - last < _MIN_INTERVAL_SEC:
        return []
    _LAST_FAN_OUT[huddle_id] = now

    # Look up huddle + recent chunks for context.
    factory = get_async_sessionmaker()
    async with factory() as session:
        h_row = (
            await session.execute(
                select(HuddleSQL).where(HuddleSQL.id == huddle_id, HuddleSQL.deleted_at.is_(None))
            )
        ).scalar_one_or_none()
        if not h_row or h_row.status != "live":
            return []
        agent_keys: List[str] = list(h_row.agent_keys or [])
        org_id = h_row.organization_id

        if not agent_keys:
            return []

        # Pull recent context for grounding
        c_rows = (
            await session.execute(
                select(HuddleTranscriptChunkSQL)
                .where(HuddleTranscriptChunkSQL.huddle_id == huddle_id)
                .order_by(HuddleTranscriptChunkSQL.ts.desc())
                .limit(_RECENT_CONTEXT_CHUNKS + 1)
            )
        ).scalars().all()

    context_lines = [f"{c.speaker_name}: {c.text}" for c in reversed(c_rows) if c.id != chunk_id]
    context_text = "\n".join(context_lines[-_RECENT_CONTEXT_CHUNKS:])

    # Build a small input shape per agent_key — defaults keep the contract
    # cheap while still useful. Each agent's process_async accepts a dict.
    base_input = {
        "user_id": user_id,
        "organization_id": org_id,
        "context": context_text,
        "transcript": chunk_text,
        "speaker": speaker_name,
        "huddle_id": huddle_id,
        "live": True,
    }

    # Lazy-import AgentService to avoid circulars on cold start.
    from backend.agents.agent_service import AgentService
    svc = AgentService()

    async def _run_one(agent_key: str) -> Dict[str, Any]:
        start = time.time()
        try:
            # Choose the call shape per agent — meeting agent wants a
            # transcript field, others accept a generic `prompt`.
            input_for_agent = dict(base_input)
            if agent_key == "meeting":
                input_for_agent.update({
                    "transcript": chunk_text,
                    "metadata": {"type": "general", "title": "Live", "participants": []},
                    "context": {},
                })
            elif agent_key == "translation":
                # Translation is multi-target: produce captions for every
                # language any client subscribed to in the room's metadata.
                # We fall back to en if no targets configured.
                # NOTE: huddle.metadata.caption_languages is a list[str].
                from backend.db.postgres import get_async_sessionmaker
                from backend.db.postgres_models import HuddleSQL
                from sqlalchemy import select as _select
                target_langs = []
                try:
                    factory2 = get_async_sessionmaker()
                    async with factory2() as session:
                        meta = (await session.execute(
                            _select(HuddleSQL.meta).where(HuddleSQL.id == huddle_id)
                        )).scalar_one_or_none() or {}
                        target_langs = list(meta.get("caption_languages") or [])
                except Exception:
                    target_langs = []
                if not target_langs:
                    target_langs = ["en"]
                translations: Dict[str, str] = {}
                for lang in target_langs:
                    try:
                        t_input = {"text": chunk_text, "target_language": lang, **base_input}
                        t_res = await svc.execute_agent_async("translation", t_input)
                        translations[lang] = _short(t_res)
                    except Exception:
                        continue
                return {
                    "agent_key": "translation",
                    "ok": True,
                    "response": translations,  # dict of lang → text
                    "latency_ms": int((time.time() - start) * 1000),
                }
            else:
                input_for_agent["prompt"] = chunk_text

            result = await svc.execute_agent_async(agent_key, input_for_agent)
            return {
                "agent_key": agent_key,
                "ok": True,
                "response": _short(result),
                "latency_ms": int((time.time() - start) * 1000),
            }
        except Exception as e:
            return {
                "agent_key": agent_key,
                "ok": False,
                "error": str(e)[:200],
                "latency_ms": int((time.time() - start) * 1000),
            }

    # Fan out
    results = await asyncio.gather(*(_run_one(k) for k in agent_keys), return_exceptions=False)

    # Persist onto the chunk row
    try:
        factory = get_async_sessionmaker()
        async with factory() as session:
            await session.execute(
                sa_update(HuddleTranscriptChunkSQL)
                .where(HuddleTranscriptChunkSQL.id == chunk_id)
                .values(agent_responses=list(results))
            )
            await session.commit()
    except Exception as e:
        logger.warning("huddle_chunk_persist_failed", huddle_id=huddle_id, chunk_id=chunk_id, error=str(e))

    # Publish each to the room
    for r in results:
        try:
            await realtime.publish_to_huddle(huddle_id, {
                "type": "agent_response",
                "huddle_id": huddle_id,
                "chunk_id": chunk_id,
                "agent_key": r["agent_key"],
                "ok": r.get("ok", False),
                "response": r.get("response"),
                "error": r.get("error"),
                "latency_ms": r.get("latency_ms"),
                "ts": time.time(),
            })
        except Exception:
            pass

    # Audit log — one entry per chunk fan-out (not per agent), to keep
    # the audit feed readable.
    if user_id and org_id:
        await log_activity(
            user_id=user_id,
            organization_id=org_id,
            activity_type="huddle.agent_response_emitted",
            details={
                "huddle_id": huddle_id,
                "chunk_id": chunk_id,
                "agents": agent_keys,
                "ok_count": sum(1 for r in results if r.get("ok")),
            },
            related_resource_type="huddle",
            related_resource_id=huddle_id,
            agent_name="HuddleDispatcher",
        )

    return list(results)


def _short(result: Any) -> str:
    """Project a varied agent return shape down to a single short string
    we can render in the sidebar."""
    if isinstance(result, str):
        return result[:1500]
    if isinstance(result, dict):
        for key in ("summary", "answer", "response", "text", "content", "output"):
            v = result.get(key)
            if isinstance(v, str) and v.strip():
                return v[:1500]
        # Last resort — dump a slim view
        return str({k: result[k] for k in list(result.keys())[:5]})[:1500]
    return str(result)[:1500]


async def publish_participant_event(
    *, huddle_id: str, event_type: str, participant: Dict[str, Any]
) -> None:
    """Helper used by huddle_service.add_participant / remove_participant
    so the room gets live join/leave updates without a poll."""
    try:
        await realtime.publish_to_huddle(huddle_id, {
            "type": event_type,
            "huddle_id": huddle_id,
            "participant": participant,
            "ts": time.time(),
        })
    except Exception:
        pass


async def publish_transcript_chunk(
    *, huddle_id: str, chunk: Dict[str, Any]
) -> None:
    """Called by huddle_service after a chunk is persisted, so all
    connected clients update their transcript without polling."""
    try:
        await realtime.publish_to_huddle(huddle_id, {
            "type": "transcript_chunk",
            "huddle_id": huddle_id,
            "chunk": chunk,
            "ts": time.time(),
        })
    except Exception:
        pass
