"""Trace decoration + persistence for brain nodes.

Every node in the graph wraps its implementation with ``@traced_node``.
The decorator:

  1. Records started_at when the node begins.
  2. Awaits the inner async function.
  3. Records ended_at + duration_ms when it returns or raises.
  4. Persists one row to ``brain_traces`` (Postgres) with:
     - status: "ok" | "fallback" | "fail"
     - eval_score: optional 0–1 (node sets via the ``__eval_score`` key
       in its return dict)
     - payload_summary: optional JSON dict (counts + IDs, never PII)
  5. Appends a `TraceEvent` to ``state.trace_events`` so the runner can
     emit a Langfuse trace span after the run finishes.

Node contract — the inner function returns a state-update dict (the
LangGraph convention). Two reserved keys are stripped by the decorator
and never reach the LangGraph state:

  ``__eval_score``:       float 0–1 — recorded on the trace row.
  ``__payload_summary``:  dict — counts + IDs, recorded on the trace row.

If the node raises, the trace row is written with ``status="fail"`` and
``error=str(exc)``. The exception is *not* re-raised — failed nodes
return an empty update dict so LangGraph keeps the run alive and the
audit + send nodes still get a chance to report what happened. Routing
to fallback paths is the graph's responsibility, not the decorator's.
"""

from __future__ import annotations

import functools
import os
from datetime import datetime
from typing import Any, Awaitable, Callable, Dict, Optional

import structlog

from .metrics import record_node
from .state import BrainState, TraceEvent

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────


NodeFn = Callable[[BrainState], Awaitable[Dict[str, Any]]]


def traced_node(name: str) -> Callable[[NodeFn], NodeFn]:
    """Decorate a node so every invocation writes a BrainTrace row.

    Usage:
        @traced_node("fetch_gmail")
        async def fetch_gmail(state: BrainState) -> dict:
            ...
            return {"emails": [...], "__eval_score": 0.95,
                    "__payload_summary": {"count": 12}}
    """
    def decorator(fn: NodeFn) -> NodeFn:
        @functools.wraps(fn)
        async def wrapper(state: BrainState) -> Dict[str, Any]:
            started = datetime.utcnow()
            status: str = "ok"
            error_str: Optional[str] = None
            eval_score: Optional[float] = None
            payload_summary: Dict[str, Any] = {}
            update: Dict[str, Any] = {}

            try:
                update = await fn(state) or {}
                # Extract trace metadata before returning the rest to LangGraph.
                eval_score = update.pop("__eval_score", None)
                payload_summary = update.pop("__payload_summary", {}) or {}
                # A node can self-report a fallback by returning
                # __status="fallback" — bumps state.fallback_count.
                self_status = update.pop("__status", None)
                if self_status in ("fallback", "retry"):
                    status = self_status
            except Exception as exc:  # noqa: BLE001
                status = "fail"
                error_str = str(exc)
                logger.exception("brain.node_failed", node=name, run_id=state.run_id)
                # Don't re-raise — let the graph reach `audit` so the run
                # ends with a row + a digest variant that says "something
                # broke this morning."
                update = {}

            ended = datetime.utcnow()
            duration_ms = int((ended - started).total_seconds() * 1000)

            event = TraceEvent(
                run_id=state.run_id,
                node=name,
                started_at=started,
                ended_at=ended,
                duration_ms=duration_ms,
                status=status,  # type: ignore[arg-type]
                eval_score=eval_score,
                payload_summary=payload_summary,
                error=error_str,
            )

            # Append for in-graph telemetry. The runner mirrors these to
            # Postgres + Langfuse after the run finishes.
            update.setdefault("trace_events", []).append(event)
            if status == "fallback":
                update["fallback_count"] = state.fallback_count + 1

            # Mirror to Prometheus (best-effort; never raises).
            record_node(
                node=name,
                duration_ms=duration_ms,
                status=status,
                eval_score=eval_score,
            )

            logger.info(
                "brain.node_completed",
                run_id=state.run_id,
                node=name,
                status=status,
                duration_ms=duration_ms,
                eval_score=eval_score,
            )

            return update

        return wrapper

    return decorator


# ─────────────────────────────────────────────────────────────────────
# Persistence — called by the runner once the graph finishes
# ─────────────────────────────────────────────────────────────────────


async def persist_traces(events: list[TraceEvent]) -> None:
    """Bulk-insert TraceEvents into Postgres ``brain_traces``.

    Called once per run — chunkier than per-node writes (one round-trip
    instead of 16), and works even if Postgres was briefly unavailable
    during a node (we still have the events in state). Best-effort —
    a Postgres failure here never breaks the digest delivery.
    """
    if not events:
        return
    try:
        from backend.db.postgres import get_async_sessionmaker
        from backend.db.postgres_models import BrainTrace

        session_factory = get_async_sessionmaker()
        async with session_factory() as session:
            rows = [
                BrainTrace(
                    run_id=ev.run_id,
                    node=ev.node,
                    started_at=ev.started_at,
                    ended_at=ev.ended_at,
                    duration_ms=ev.duration_ms,
                    status=ev.status,
                    eval_score=ev.eval_score,
                    payload_summary=dict(ev.payload_summary) | (
                        {"error": ev.error} if ev.error else {}
                    ),
                )
                for ev in events
            ]
            session.add_all(rows)
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("brain.persist_traces_failed", error=str(exc))


# ─────────────────────────────────────────────────────────────────────
# Langfuse — optional. Off unless LANGFUSE_PUBLIC_KEY + SECRET_KEY set.
# ─────────────────────────────────────────────────────────────────────


_langfuse_singleton: Any = None


def langfuse_client() -> Any | None:
    """Return a cached Langfuse client if env is configured, else None.

    Importing lazily avoids the dep when env is bare. Langfuse is great
    for production trace inspection but adds latency on every node, so
    we keep it opt-in via env vars.
    """
    global _langfuse_singleton
    if _langfuse_singleton is not None:
        return _langfuse_singleton

    public = os.environ.get("LANGFUSE_PUBLIC_KEY")
    secret = os.environ.get("LANGFUSE_SECRET_KEY")
    if not public or not secret:
        return None

    try:
        from langfuse import Langfuse  # type: ignore
        _langfuse_singleton = Langfuse(
            public_key=public,
            secret_key=secret,
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        return _langfuse_singleton
    except Exception as exc:  # noqa: BLE001
        logger.warning("brain.langfuse_init_failed", error=str(exc))
        return None


async def emit_to_langfuse(run_id: str, mode: str, events: list[TraceEvent]) -> None:
    """Best-effort mirror of one BrainRun's trace events to Langfuse."""
    client = langfuse_client()
    if client is None:
        return

    try:
        trace = client.trace(id=run_id, name=f"brain.{mode}", metadata={"mode": mode})
        for ev in events:
            trace.span(
                name=ev.node,
                start_time=ev.started_at,
                end_time=ev.ended_at,
                metadata={
                    "status": ev.status,
                    "eval_score": ev.eval_score,
                    "duration_ms": ev.duration_ms,
                    **ev.payload_summary,
                },
                level="ERROR" if ev.status == "fail" else "DEFAULT",
                status_message=ev.error,
            )
        # Force a flush so traces show up immediately in the UI.
        if hasattr(client, "flush"):
            client.flush()
    except Exception as exc:  # noqa: BLE001
        logger.warning("brain.langfuse_emit_failed", error=str(exc), run_id=run_id)
