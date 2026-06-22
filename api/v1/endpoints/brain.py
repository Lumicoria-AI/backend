"""HTTP endpoints for the autonomous brain.

Phase 2 surface — enough to test the graph end-to-end locally:

    GET  /api/v1/brain/preferences          — read current user's brain prefs
    PUT  /api/v1/brain/preferences          — update brain prefs (mongo merge)
    POST /api/v1/brain/trigger              — fire a run now (inline, returns the summary)
    GET  /api/v1/brain/runs                 — list recent runs for the current user
    GET  /api/v1/brain/runs/{run_id}        — one run + the full trace timeline
    GET  /api/v1/brain/runs/{run_id}/traces — same payload as above, traces-only

The trigger endpoint runs inline (no Celery dispatch) so local
development can exercise the whole pipeline without a broker running.
The Celery shim (`brain.run_on_demand.delay(...)`) is still available
for production / scheduled invocation — the runner code path is the
same either way.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Path, Query
from pydantic import BaseModel, Field
from sqlalchemy import desc, select

from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.postgres import get_async_sessionmaker
from backend.db.postgres_models import BrainRun, BrainTrace
from backend.models.user import User
from backend.services.brain.runner import run_brain_for_user

logger = structlog.get_logger(__name__)
router = APIRouter()


# ─────────────────────────────────────────────────────────────────────
# Schemas
# ─────────────────────────────────────────────────────────────────────


DEFAULT_PREFS: Dict[str, Any] = {
    "enabled": False,
    "morning_hour_local": 6,
    "evening_hour_local": 22,
    "max_emails_per_run": 100,
    "max_attachments_per_run": 20,
    "mailbox_labels_include": ["INBOX"],
    "mailbox_labels_exclude": ["CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL"],
    "auto_assign_agents": True,
    "send_email": True,
    "send_push": True,
    "send_in_app": True,
    "needs_reauth": False,
}


class BrainPreferences(BaseModel):
    enabled: bool = False
    morning_hour_local: int = Field(default=6, ge=0, le=23)
    evening_hour_local: int = Field(default=22, ge=0, le=23)
    max_emails_per_run: int = Field(default=100, ge=1, le=1000)
    max_attachments_per_run: int = Field(default=20, ge=0, le=200)
    mailbox_labels_include: List[str] = Field(default_factory=lambda: ["INBOX"])
    mailbox_labels_exclude: List[str] = Field(
        default_factory=lambda: ["CATEGORY_PROMOTIONS", "CATEGORY_SOCIAL"],
    )
    auto_assign_agents: bool = True
    send_email: bool = True
    send_push: bool = True
    send_in_app: bool = True
    needs_reauth: bool = False


class BrainTriggerRequest(BaseModel):
    mode: str = Field(default="morning", pattern="^(morning|evening)$")
    # Phase 2: always inline. Once Celery beat is wired in prod, the
    # frontend can pass async=True to dispatch via Celery instead.
    async_: bool = Field(default=False, alias="async")


class BrainRunOut(BaseModel):
    id: str
    user_id: str
    organization_id: Optional[str] = None
    mode: str
    status: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    emails_processed: int = 0
    attachments_processed: int = 0
    tasks_created: int = 0
    proposals_drafted: int = 0
    digest_sent: bool = False
    skip_reason: Optional[str] = None
    error: Optional[str] = None


class BrainTraceOut(BaseModel):
    node: str
    started_at: datetime
    ended_at: Optional[datetime] = None
    duration_ms: Optional[int] = None
    status: str
    eval_score: Optional[float] = None
    payload_summary: Dict[str, Any] = Field(default_factory=dict)


class BrainRunDetail(BrainRunOut):
    traces: List[BrainTraceOut] = Field(default_factory=list)


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _user_oid(user_id: str) -> Any:
    try:
        return ObjectId(user_id)
    except Exception:
        return user_id


# ─────────────────────────────────────────────────────────────────────
# Preferences
# ─────────────────────────────────────────────────────────────────────


@router.get("/preferences", response_model=BrainPreferences)
async def get_preferences(current_user: User = Depends(get_current_active_user)) -> Any:
    """Return the calling user's brain prefs (merged with defaults)."""
    db = await MongoDB.get_database()
    user_doc = await db.users.find_one(
        {"_id": _user_oid(str(current_user.id))},
        projection={"preferences": 1},
    )
    prefs = ((user_doc or {}).get("preferences") or {}).get("brain") or {}
    merged = {**DEFAULT_PREFS, **prefs}
    return BrainPreferences(**merged)


@router.put("/preferences", response_model=BrainPreferences)
async def update_preferences(
    payload: BrainPreferences,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Replace the calling user's brain prefs."""
    db = await MongoDB.get_database()
    await db.users.update_one(
        {"_id": _user_oid(str(current_user.id))},
        {"$set": {"preferences.brain": payload.model_dump(),
                  "updated_at": datetime.utcnow()}},
        upsert=False,
    )
    return payload


# ─────────────────────────────────────────────────────────────────────
# Trigger
# ─────────────────────────────────────────────────────────────────────


@router.post("/trigger")
async def trigger_run(
    payload: BrainTriggerRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Fire one brain run for the current user.

    Phase 2: synchronous — the request blocks until the graph finishes
    (typically <10s with all stub nodes). Phase 3 stays synchronous for
    POST /trigger because the user clicked a button and expects to see
    a result; the *scheduled* path goes through Celery beat instead.
    """
    if payload.async_:
        # Optional Celery dispatch — for callers who can't wait. Phase
        # 2 ships the inline path; this is a no-op forward-compat hook.
        try:
            from backend.tasks.brain_tasks import run_on_demand
            ar = run_on_demand.delay(str(current_user.id), payload.mode)
            return {"queued": True, "task_id": ar.id, "mode": payload.mode}
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "brain.trigger_celery_dispatch_failed", error=str(exc),
            )
            # Fall through to inline.

    try:
        summary = await run_brain_for_user(
            user_id=str(current_user.id),
            mode=payload.mode,
            force=True,
            initiated_by="api",
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    return summary.model_dump()


# ─────────────────────────────────────────────────────────────────────
# Run history
# ─────────────────────────────────────────────────────────────────────


@router.get("/runs", response_model=List[BrainRunOut])
async def list_runs(
    limit: int = Query(20, ge=1, le=200),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Most recent brain runs for the calling user."""
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        q = (
            select(BrainRun)
            .where(BrainRun.user_id == str(current_user.id))
            .order_by(desc(BrainRun.started_at))
            .limit(limit)
        )
        rows = (await session.execute(q)).scalars().all()

    return [BrainRunOut.model_validate(_serialize_run(r)) for r in rows]


@router.get("/runs/{run_id}", response_model=BrainRunDetail)
async def get_run(
    run_id: str = Path(..., min_length=8),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """One run + its trace timeline."""
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        run = (
            await session.execute(
                select(BrainRun).where(BrainRun.id == run_id).limit(1),
            )
        ).scalars().first()
        if not run:
            raise HTTPException(status_code=404, detail="Run not found")
        if run.user_id != str(current_user.id):
            raise HTTPException(status_code=403, detail="Not your run")

        traces = (
            await session.execute(
                select(BrainTrace)
                .where(BrainTrace.run_id == run_id)
                .order_by(BrainTrace.started_at),
            )
        ).scalars().all()

    return BrainRunDetail.model_validate({
        **_serialize_run(run),
        "traces": [_serialize_trace(t) for t in traces],
    })


@router.get("/runs/{run_id}/traces", response_model=List[BrainTraceOut])
async def get_run_traces(
    run_id: str = Path(..., min_length=8),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Just the traces for a run — useful for the timeline component."""
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        # Auth check via the run row.
        run = (
            await session.execute(
                select(BrainRun.user_id).where(BrainRun.id == run_id).limit(1),
            )
        ).scalars().first()
        if run is None:
            raise HTTPException(status_code=404, detail="Run not found")
        if run != str(current_user.id):
            raise HTTPException(status_code=403, detail="Not your run")

        traces = (
            await session.execute(
                select(BrainTrace)
                .where(BrainTrace.run_id == run_id)
                .order_by(BrainTrace.started_at),
            )
        ).scalars().all()

    return [BrainTraceOut.model_validate(_serialize_trace(t)) for t in traces]


# ─────────────────────────────────────────────────────────────────────
# Serialisers — keep ORM rows from leaking into Pydantic v2
# ─────────────────────────────────────────────────────────────────────


def _serialize_run(r: BrainRun) -> Dict[str, Any]:
    return {
        "id": r.id,
        "user_id": r.user_id,
        "organization_id": r.organization_id,
        "mode": r.mode,
        "status": r.status,
        "started_at": r.started_at,
        "ended_at": r.ended_at,
        "duration_ms": r.duration_ms,
        "emails_processed": r.emails_processed or 0,
        "attachments_processed": r.attachments_processed or 0,
        "tasks_created": r.tasks_created or 0,
        "proposals_drafted": r.proposals_drafted or 0,
        "digest_sent": bool(r.digest_sent),
        "skip_reason": r.skip_reason,
        "error": r.error,
    }


def _serialize_trace(t: BrainTrace) -> Dict[str, Any]:
    return {
        "node": t.node,
        "started_at": t.started_at,
        "ended_at": t.ended_at,
        "duration_ms": t.duration_ms,
        "status": t.status,
        "eval_score": t.eval_score,
        "payload_summary": dict(t.payload_summary or {}),
    }
