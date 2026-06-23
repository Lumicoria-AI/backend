"""Drives a single brain run end-to-end.

The runner is the bridge between the LangGraph state machine and the
rest of Lumicoria:

  1. Looks up the user (Mongo) + their org_id + timezone.
  2. Mints a BrainRun row (Postgres) so the run has a stable ID before
     anything else writes telemetry.
  3. Builds the initial BrainState.
  4. Invokes `brain_graph.ainvoke(initial_state)`.
  5. Bulk-writes every TraceEvent the graph collected into the
     `brain_traces` Postgres table.
  6. Updates the BrainRun row with final status, counts, duration.
  7. (Optional) mirrors the trace events to Langfuse if env is set.
  8. Returns a `BrainRunSummary` for the API endpoint + Celery task.

This is the only file in `services/brain/` that touches Postgres
directly outside of the @traced_node decorator — keeping the graph
itself store-agnostic so it stays unit-testable with no DB.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Dict, Optional

import structlog

from .graph import brain_graph
from .metrics import record_run
from .state import BrainRunSummary, BrainState
from .tracing import emit_to_langfuse, persist_traces

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


async def run_brain_for_user(
    *,
    user_id: str,
    mode: str = "morning",
    force: bool = False,
    initiated_by: str = "scheduler",
) -> BrainRunSummary:
    """Run the brain once for ``user_id``.

    Args:
        user_id: Mongo user id (stringified).
        mode: "morning" | "evening".
        force: Skip the gate's TZ check. Used by the manual
            ``POST /brain/trigger`` endpoint so devs can fire a run any
            time without waiting for the schedule.
        initiated_by: "scheduler" | "api" | "celery_beat" — recorded on
            BrainRun.metadata for audit.

    Returns:
        BrainRunSummary — a compact dict the API + Celery task can
        return to the caller.
    """
    if mode not in ("morning", "evening"):
        raise ValueError(f"invalid mode: {mode!r} — must be 'morning' or 'evening'")

    run_id = str(uuid.uuid4())
    started_at = datetime.utcnow()

    # 1. Look up the user so we know their org_id + email + TZ.
    user_doc = await _load_user_doc(user_id)
    if not user_doc:
        return BrainRunSummary(
            run_id=run_id, user_id=user_id, mode=mode, status="failed",
            duration_ms=0, emails_processed=0, attachments_processed=0,
            tasks_created=0, proposals_drafted=0, digest_sent=False,
            error="user_not_found",
        )

    organization_id = _resolve_org_id(user_doc)
    user_email = user_doc.get("email")
    timezone = _resolve_timezone(user_doc)

    # 2. Write the BrainRun row up front (status=ok placeholder). This
    # gives the trace decorator a stable foreign key and lets the
    # GET /brain/runs endpoint show a row even for an in-flight run.
    await _create_brain_run_row(
        run_id=run_id,
        user_id=user_id,
        organization_id=organization_id,
        mode=mode,
        started_at=started_at,
        initiated_by=initiated_by,
    )

    # 3. Initial state.
    initial = BrainState(
        run_id=run_id,
        user_id=user_id,
        organization_id=organization_id,
        user_email=user_email,
        timezone=timezone,
        mode=mode,  # type: ignore[arg-type]
        started_at=started_at,
        meta={"force": bool(force), "initiated_by": initiated_by},
    )

    # 4. Invoke the graph.
    final_state: Optional[BrainState] = None
    error_str: Optional[str] = None
    try:
        # LangGraph returns the merged state as a dict; rehydrate.
        result = await brain_graph.ainvoke(initial)
        final_state = (
            result if isinstance(result, BrainState) else BrainState(**result)
        )
    except Exception as exc:  # noqa: BLE001
        error_str = str(exc)
        logger.exception("brain.graph_invoke_failed", run_id=run_id)
        # Even on failure, persist what we can so the trace row trail
        # survives. Build a half-state from the initial seed so the
        # finaliser has something to write counts from.
        final_state = initial

    # 5. Persist trace events (best-effort).
    if final_state and final_state.trace_events:
        await persist_traces(final_state.trace_events)
        # Mirror to Langfuse if configured.
        await emit_to_langfuse(run_id, mode, final_state.trace_events)

    # 6. Finalise the BrainRun row.
    ended_at = datetime.utcnow()
    duration_ms = int((ended_at - started_at).total_seconds() * 1000)
    status = _derive_status(final_state, error_str)
    skip_reason = final_state.skip_reason if final_state else None
    emails_processed = len(final_state.emails) if final_state else 0
    attachments_processed = (
        sum(len(m.attachment_ids) for m in final_state.emails)
        if final_state
        else 0
    )
    tasks_created = len(final_state.created_task_ids) if final_state else 0
    proposals_drafted = sum(
        1
        for s in (final_state.proposal_status_by_task.values() if final_state else [])
        if s in ("pending_review", "approved")
    )
    digest_sent = bool(
        final_state and "email" in final_state.delivery_channels
    )

    await _finalise_brain_run_row(
        run_id=run_id,
        ended_at=ended_at,
        duration_ms=duration_ms,
        status=status,
        emails_processed=emails_processed,
        attachments_processed=attachments_processed,
        tasks_created=tasks_created,
        proposals_drafted=proposals_drafted,
        digest_sent=digest_sent,
        skip_reason=skip_reason,
        error=error_str,
    )

    summary = BrainRunSummary(
        run_id=run_id,
        user_id=user_id,
        mode=mode,
        status=status,
        duration_ms=duration_ms,
        emails_processed=emails_processed,
        attachments_processed=attachments_processed,
        tasks_created=tasks_created,
        proposals_drafted=proposals_drafted,
        digest_sent=digest_sent,
        skip_reason=skip_reason,
        error=error_str,
    )

    # Mirror to Prometheus (best-effort).
    record_run(
        mode=mode,
        status=status,
        duration_ms=duration_ms,
        tasks_created=tasks_created,
        emails_processed=emails_processed,
    )

    logger.info(
        "brain.run_complete",
        run_id=run_id,
        user_id=user_id,
        mode=mode,
        status=status,
        duration_ms=duration_ms,
        tasks_created=tasks_created,
    )

    return summary


# ─────────────────────────────────────────────────────────────────────
# Helpers — kept module-private so the public API is just `run_brain_for_user`
# ─────────────────────────────────────────────────────────────────────


async def _load_user_doc(user_id: str) -> Optional[Dict[str, Any]]:
    """Fetch the raw user document from Mongo. Returns None if missing."""
    try:
        from bson import ObjectId
        from backend.db.mongodb.mongodb import MongoDB

        db = await MongoDB.get_database()
        try:
            oid: Any = ObjectId(user_id)
        except Exception:
            oid = user_id
        return await db.users.find_one({"_id": oid})
    except Exception as exc:  # noqa: BLE001
        logger.warning("brain.user_lookup_failed", user_id=user_id, error=str(exc))
        return None


def _resolve_org_id(user_doc: Dict[str, Any]) -> Optional[str]:
    """Best-effort: try active_organization_id, then organization_id, then
    first of organization_ids[], else None (personal-mode brain run)."""
    for key in ("active_organization_id", "organization_id"):
        val = user_doc.get(key)
        if val:
            return str(val)
    org_ids = user_doc.get("organization_ids") or []
    if org_ids:
        return str(org_ids[0])
    return None


def _resolve_timezone(user_doc: Dict[str, Any]) -> str:
    """Pull TZ from user.preferences.timezone, falling back to UTC."""
    prefs = user_doc.get("preferences") or {}
    tz = prefs.get("timezone") or user_doc.get("timezone")
    return str(tz) if tz else "UTC"


def _derive_status(
    final_state: Optional[BrainState],
    error_str: Optional[str],
) -> str:
    if error_str:
        return "failed"
    if final_state is None:
        return "failed"
    if final_state.skip_reason:
        return "skipped"
    if final_state.fallback_count >= 3:
        return "degraded"
    return "ok"


async def _create_brain_run_row(
    *,
    run_id: str,
    user_id: str,
    organization_id: Optional[str],
    mode: str,
    started_at: datetime,
    initiated_by: str,
) -> None:
    try:
        from backend.db.postgres import get_async_sessionmaker
        from backend.db.postgres_models import BrainRun

        session_factory = get_async_sessionmaker()
        async with session_factory() as session:
            row = BrainRun(
                id=run_id,
                user_id=user_id,
                organization_id=organization_id,
                mode=mode,
                status="ok",  # placeholder; finaliser updates it
                started_at=started_at,
                meta={"initiated_by": initiated_by},
            )
            session.add(row)
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning("brain.create_run_row_failed", run_id=run_id, error=str(exc))


async def _finalise_brain_run_row(
    *,
    run_id: str,
    ended_at: datetime,
    duration_ms: int,
    status: str,
    emails_processed: int,
    attachments_processed: int,
    tasks_created: int,
    proposals_drafted: int,
    digest_sent: bool,
    skip_reason: Optional[str],
    error: Optional[str],
) -> None:
    try:
        from sqlalchemy import update

        from backend.db.postgres import get_async_sessionmaker
        from backend.db.postgres_models import BrainRun

        session_factory = get_async_sessionmaker()
        async with session_factory() as session:
            await session.execute(
                update(BrainRun)
                .where(BrainRun.id == run_id)
                .values(
                    ended_at=ended_at,
                    duration_ms=duration_ms,
                    status=status,
                    emails_processed=emails_processed,
                    attachments_processed=attachments_processed,
                    tasks_created=tasks_created,
                    proposals_drafted=proposals_drafted,
                    digest_sent=digest_sent,
                    skip_reason=skip_reason,
                    error=error,
                )
            )
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "brain.finalise_run_row_failed", run_id=run_id, error=str(exc),
        )
