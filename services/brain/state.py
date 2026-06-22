"""Pydantic state shared by every brain node.

`BrainState` is the LangGraph state object — it's threaded through
every node. Nodes return dicts of updates; LangGraph merges them per
the field reducer (default = "last write wins"; lists tagged with
``operator.add`` get concatenated across parallel branches).

The state intentionally does NOT carry full email bodies past the
fetch nodes — only what the next node needs. Bodies and attachment
bytes live in MinIO + Weaviate. This keeps the LangGraph checkpointer
size bounded and the per-trace payload safe to display in the admin
UI without leaking PII.
"""

from __future__ import annotations

from datetime import datetime
from operator import add
from typing import Annotated, Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


# ─────────────────────────────────────────────────────────────────────
# Telemetry shapes (mirror the BrainTrace / BrainEval Postgres rows)
# ─────────────────────────────────────────────────────────────────────


class EvalResult(BaseModel):
    """One node's quality check. ``score`` 0–1, ``passed`` controls
    routing — failed evals trigger fallback edges."""
    model_config = ConfigDict(extra="forbid")

    score: float = 1.0
    passed: bool = True
    reason: str = ""
    checked_fields: List[str] = Field(default_factory=list)


class TraceEvent(BaseModel):
    """One row destined for the ``brain_traces`` table. Counts only —
    no PII, ever."""
    model_config = ConfigDict(extra="forbid")

    run_id: str
    node: str
    started_at: datetime
    ended_at: datetime
    duration_ms: int
    status: Literal["ok", "retry", "fallback", "fail"] = "ok"
    eval_score: Optional[float] = None
    payload_summary: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────
# Domain shapes (lightweight — bodies + bytes stay out of the state)
# ─────────────────────────────────────────────────────────────────────


class GmailMessageRef(BaseModel):
    """A pointer to a Gmail message in the user's inbox. We pass refs
    around the graph; the body is fetched lazily by the ingest node
    via the existing google client + cached in Weaviate."""
    model_config = ConfigDict(extra="forbid")

    message_id: str
    thread_id: Optional[str] = None
    subject: Optional[str] = None
    from_addr: Optional[str] = None
    received_at: Optional[datetime] = None
    label_ids: List[str] = Field(default_factory=list)
    has_attachments: bool = False
    attachment_ids: List[str] = Field(default_factory=list)
    snippet: Optional[str] = None  # Gmail's preview text — short, OK to keep


class CalendarEventRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    event_id: str
    summary: Optional[str] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    attendees: List[str] = Field(default_factory=list)
    location: Optional[str] = None


class DriveFileRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    file_id: str
    name: Optional[str] = None
    mime_type: Optional[str] = None
    modified_at: Optional[datetime] = None
    removed: bool = False


class HuddleSummaryRef(BaseModel):
    """Lightweight handle to a recent huddle for evening recap."""
    model_config = ConfigDict(extra="forbid")

    huddle_id: str
    title: Optional[str] = None
    ended_at: Optional[datetime] = None
    summary: Optional[str] = None


class OpenTaskRef(BaseModel):
    """Existing Lumicoria task that's still open — used for evening
    recap ("you finished X, here's what's still open")."""
    model_config = ConfigDict(extra="forbid")

    task_id: str
    title: str
    priority: Optional[str] = None
    due_date: Optional[datetime] = None
    status: Optional[str] = None


# ─────────────────────────────────────────────────────────────────────
# Outputs of the reasoning nodes
# ─────────────────────────────────────────────────────────────────────


class ClassifiedEmail(BaseModel):
    """Output of the classify node. The prioritise node consumes this
    list + retrieved RAG context to produce ranked actions."""
    model_config = ConfigDict(extra="forbid")

    message_id: str
    category: Literal[
        "action_required",
        "scheduling",
        "informational",
        "promotional",
        "spam",
        "unknown",
    ] = "unknown"
    urgency: Literal["critical", "high", "medium", "low"] = "low"
    confidence: float = 0.0
    summary: Optional[str] = None
    suggested_agent: Optional[str] = None


class RankedAction(BaseModel):
    """The Brain Agent's output — one per task to create."""
    model_config = ConfigDict(extra="forbid")

    title: str
    description: str
    priority: Literal["critical", "high", "medium", "low"] = "medium"
    due_date: Optional[datetime] = None
    assigned_to_agent: Optional[str] = None
    confidence: float = 0.0
    evidence_message_ids: List[str] = Field(default_factory=list)
    evidence_event_ids: List[str] = Field(default_factory=list)
    evidence_file_ids: List[str] = Field(default_factory=list)


class DigestPayload(BaseModel):
    """The composed digest — fed to the email template + in-app card."""
    model_config = ConfigDict(extra="forbid")

    mode: Literal["morning", "evening"]
    user_name: Optional[str] = None
    summary_line: str = ""
    top_actions: List[RankedAction] = Field(default_factory=list)
    secondary_actions: List[RankedAction] = Field(default_factory=list)
    calendar_today: List[CalendarEventRef] = Field(default_factory=list)
    completed_today: List[OpenTaskRef] = Field(default_factory=list)
    open_tasks: List[OpenTaskRef] = Field(default_factory=list)
    counts: Dict[str, int] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# The graph state — annotated for LangGraph reducers
# ─────────────────────────────────────────────────────────────────────


class BrainState(BaseModel):
    """State threaded through every node.

    Fields tagged ``Annotated[..., add]`` get list-concat behaviour when
    multiple parallel branches return updates (LangGraph reducer
    pattern). Untagged fields are last-write-wins.
    """
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    # ── Identity ────────────────────────────────────────────────────
    run_id: str
    user_id: str
    organization_id: Optional[str] = None
    user_email: Optional[str] = None
    timezone: str = "UTC"
    mode: Literal["morning", "evening"] = "morning"
    started_at: datetime = Field(default_factory=datetime.utcnow)

    # ── Gate ────────────────────────────────────────────────────────
    skip_reason: Optional[str] = None  # gate sets to abort the run early

    # ── Fetched (parallel) ──────────────────────────────────────────
    emails: Annotated[List[GmailMessageRef], add] = Field(default_factory=list)
    events: Annotated[List[CalendarEventRef], add] = Field(default_factory=list)
    drive_changes: Annotated[List[DriveFileRef], add] = Field(default_factory=list)
    huddle_recents: Annotated[List[HuddleSummaryRef], add] = Field(default_factory=list)
    open_tasks: Annotated[List[OpenTaskRef], add] = Field(default_factory=list)

    # ── Reasoning outputs ───────────────────────────────────────────
    ingested_doc_ids: Annotated[List[str], add] = Field(default_factory=list)
    classified: Annotated[List[ClassifiedEmail], add] = Field(default_factory=list)
    ranked_actions: List[RankedAction] = Field(default_factory=list)

    # ── Persistence outputs ─────────────────────────────────────────
    created_task_ids: Annotated[List[str], add] = Field(default_factory=list)
    proposal_status_by_task: Dict[str, str] = Field(default_factory=dict)

    # ── Composition + delivery ──────────────────────────────────────
    digest_payload: Optional[DigestPayload] = None
    delivery_channels: Annotated[List[str], add] = Field(default_factory=list)

    # ── Telemetry — pure observability, never used for routing ──────
    trace_events: Annotated[List[TraceEvent], add] = Field(default_factory=list)
    evals: Annotated[List[EvalResult], add] = Field(default_factory=list)
    fallback_count: int = 0

    # ── Free-form metadata bucket for nodes to stash anything ───────
    meta: Dict[str, Any] = Field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────
# Top-level summary returned by `runner.run_brain_for_user`
# ─────────────────────────────────────────────────────────────────────


class BrainRunSummary(BaseModel):
    """Compact summary returned to the API endpoint + Celery task."""
    model_config = ConfigDict(extra="forbid")

    run_id: str
    user_id: str
    mode: str
    status: Literal["ok", "degraded", "failed", "skipped"]
    duration_ms: int
    emails_processed: int
    attachments_processed: int
    tasks_created: int
    proposals_drafted: int
    digest_sent: bool
    skip_reason: Optional[str] = None
    error: Optional[str] = None
