"""
PostgreSQL models for Lumicoria.ai

These models are used for relational data that benefits from SQL semantics:
tasks, workflows, and agent execution logs.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Optional, List
import uuid

from sqlalchemy import (
    Column,
    String,
    Text,
    DateTime,
    Integer,
    Boolean,
    Float,
    ForeignKey,
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

from backend.db.base_class import Base
from backend.models.mongodb_models import TaskStatus, TaskPriority, AgentStatus


def _uuid_str() -> str:
    return str(uuid.uuid4())


def _short_ticket_id() -> str:
    """Public ticket id — `TK-` + 8 hex chars (~4 billion namespace per
    org, more than enough for any tenant; uniqueness enforced by PK)."""
    return f"TK-{uuid.uuid4().hex[:8]}"


class TaskSQL(Base):
    __tablename__ = "tasks"

    id = Column(String(36), primary_key=True, default=_uuid_str)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    status = Column(SAEnum(TaskStatus), nullable=False, default=TaskStatus.TODO)
    priority = Column(SAEnum(TaskPriority), nullable=False, default=TaskPriority.MEDIUM)
    due_date = Column(DateTime, nullable=True)

    assigned_to = Column(String(64), nullable=True)
    created_by = Column(String(64), nullable=True)
    organization_id = Column(String(64), nullable=True)
    project_id = Column(String(64), nullable=True)
    parent_task_id = Column(String(64), nullable=True)
    agent_id = Column(String(64), nullable=True)

    tags = Column(ARRAY(String), nullable=False, default=list)
    meta = Column("metadata", JSONB, nullable=False, default=dict)
    progress = Column(Integer, nullable=False, default=0)
    completed_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)

    @property
    def name(self) -> str:
        return self.title


class WorkflowSQL(Base):
    __tablename__ = "workflows"

    id = Column(String(36), primary_key=True, default=_uuid_str)
    name = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)

    components = Column(JSONB, nullable=False, default=list)
    nodes = Column(JSONB, nullable=False, default=list)
    connections = Column(JSONB, nullable=False, default=list)

    organization_id = Column(String(64), nullable=True)
    created_by = Column(String(64), nullable=True)

    version = Column(String(50), nullable=False, default="1.0.0")
    is_public = Column(Boolean, nullable=False, default=False)
    tags = Column(ARRAY(String), nullable=False, default=list)
    status = Column(String(50), nullable=False, default=AgentStatus.DRAFT.value)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# ── Meeting Library ──────────────────────────────────────────────────

class MeetingSQL(Base):
    """Processed meeting stored in Postgres for persistent history."""
    __tablename__ = "meetings"

    id = Column(String(36), primary_key=True, default=_uuid_str)
    user_id = Column(String(64), nullable=False, index=True)
    organization_id = Column(String(64), nullable=True, index=True)

    # Core meeting data
    title = Column(String(500), nullable=True)
    meeting_type = Column(String(50), nullable=False, default="general")
    transcript = Column(Text, nullable=False)
    summary = Column(Text, nullable=True)
    raw_response = Column(Text, nullable=True)
    model_used = Column(String(100), nullable=True)

    # Structured results stored as JSONB
    action_items = Column(JSONB, nullable=False, default=list)
    decisions = Column(JSONB, nullable=False, default=list)
    key_points = Column(JSONB, nullable=False, default=list)
    follow_ups = Column(JSONB, nullable=False, default=list)
    questions = Column(JSONB, nullable=False, default=list)
    concerns = Column(JSONB, nullable=False, default=list)

    # Metadata
    meeting_date = Column(String(50), nullable=True)
    duration = Column(String(50), nullable=True)
    participants = Column(JSONB, nullable=False, default=list)
    context = Column(JSONB, nullable=False, default=dict)
    tags = Column(ARRAY(String), nullable=False, default=list)

    # Source tracking
    source = Column(String(50), nullable=False, default="manual")  # manual, file_upload, audio_upload, stt

    processed_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)


class MeetingDraftSQL(Base):
    """Draft transcript saved while user is typing or recording — one per user."""
    __tablename__ = "meeting_drafts"

    id = Column(String(36), primary_key=True, default=_uuid_str)
    user_id = Column(String(64), nullable=False, unique=True, index=True)
    transcript = Column(Text, nullable=False, default="")
    meeting_type = Column(String(50), nullable=True, default="general")
    title = Column(String(500), nullable=True)
    participants = Column(JSONB, nullable=False, default=list)
    context = Column(JSONB, nullable=False, default=dict)

    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


# ── Fact-Checker Sessions & Claims ──────────────────────────────

class FactCheckSessionSQL(Base):
    """A fact-checking session stored in Postgres."""
    __tablename__ = "fact_check_sessions"

    id = Column(String(36), primary_key=True, default=_uuid_str)
    user_id = Column(String(64), nullable=False, index=True)
    organization_id = Column(String(64), nullable=True, index=True)

    title = Column(String(500), nullable=False)
    participants = Column(JSONB, nullable=False, default=list)
    summary = Column(Text, nullable=True)
    verification_stats = Column(JSONB, nullable=False, default=dict)

    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class FactCheckClaimSQL(Base):
    """An individual verified claim within a fact-check session."""
    __tablename__ = "fact_check_claims"

    id = Column(String(36), primary_key=True, default=_uuid_str)
    session_id = Column(String(36), nullable=False, index=True)
    user_id = Column(String(64), nullable=False, index=True)

    content = Column(Text, nullable=False)
    speaker = Column(String(255), nullable=False, default="Unknown")
    claim_type = Column(String(50), nullable=False, default="assertion")

    verification_status = Column(String(50), nullable=False, default="pending")
    confidence = Column(Integer, nullable=False, default=0)  # 0-100
    severity = Column(String(50), nullable=False, default="medium")

    citations = Column(JSONB, nullable=False, default=list)
    corrections = Column(JSONB, nullable=False, default=list)
    summary = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)


class AgentExecutionSQL(Base):
    __tablename__ = "agent_executions"

    id = Column(String(36), primary_key=True, default=_uuid_str)
    agent_name = Column(String(255), nullable=True)
    agent_type = Column(String(100), nullable=True)
    agent_id = Column(String(64), nullable=True)
    workflow_id = Column(String(64), nullable=True)
    user_id = Column(String(64), nullable=True)
    organization_id = Column(String(64), nullable=True)

    status = Column(String(50), nullable=False, default="success")
    error_message = Column(Text, nullable=True)
    async_execution = Column(Boolean, nullable=False, default=False)

    started_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    ended_at = Column(DateTime, nullable=True)
    duration_ms = Column(Integer, nullable=True)

    input_payload = Column(JSONB, nullable=True)
    output_payload = Column(JSONB, nullable=True)
    meta = Column("metadata", JSONB, nullable=False, default=dict)


# ── Blog ────────────────────────────────────────────────────────────

class BlogPostStatus(str, enum.Enum):
    DRAFT = "draft"
    PUBLISHED = "published"
    ARCHIVED = "archived"


class AuthorType(str, enum.Enum):
    TEAM = "team"
    INDIVIDUAL = "individual"
    AI_AGENT = "ai_agent"


class BlogPostSQL(Base):
    """Blog post stored in Postgres for persistent, public-facing content."""
    __tablename__ = "blog_posts"

    id = Column(String(36), primary_key=True, default=_uuid_str)
    slug = Column(String(500), unique=True, nullable=False, index=True)
    title = Column(String(500), nullable=False)
    subtitle = Column(String(500), nullable=True)
    content = Column(Text, nullable=False)  # TipTap HTML
    excerpt = Column(Text, nullable=True)

    # Author (denormalized for fast public reads — no cross-DB joins)
    author_id = Column(String(64), nullable=False, index=True)
    author_type = Column(SAEnum(AuthorType), nullable=False, default=AuthorType.INDIVIDUAL)
    author_name = Column(String(255), nullable=False)
    author_avatar_url = Column(String(1000), nullable=True)
    author_title = Column(String(255), nullable=True)

    cover_image_url = Column(String(1000), nullable=True)
    category = Column(String(100), nullable=True, index=True)
    tags = Column(ARRAY(String), nullable=False, default=list)
    status = Column(SAEnum(BlogPostStatus), nullable=False, default=BlogPostStatus.DRAFT)

    collaborator_ids = Column(ARRAY(String), nullable=False, default=list)
    featured = Column(Boolean, nullable=False, default=False)
    view_count = Column(Integer, nullable=False, default=0)

    published_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)


# ── RAG Document Registry ──────────────────────────────────────────
#
# Durable source-of-truth for documents ingested via the RAG / chat pipeline.
# Chunks + embeddings still live in Weaviate; the *file* itself lives in
# MinIO (+ R2 backup).  This table keeps the mapping (document_id ↔ s3_key)
# so the vector store can be rebuilt without losing file references.


class RAGDocumentSQL(Base):
    """A document ingested by the RAG pipeline.

    One row per logical document (not per chunk).  Rows are created *before*
    chunking starts so the frontend can poll / preview the file immediately
    while embeddings are still processing in the background.
    """
    __tablename__ = "rag_documents"

    # Room for prefixed IDs like "chat_{uuid}" (41 chars), not just bare UUIDs.
    id = Column(String(64), primary_key=True, default=_uuid_str)  # == document_id in Weaviate
    user_id = Column(String(64), nullable=False, index=True)
    organization_id = Column(String(64), nullable=True, index=True)

    # Storage
    s3_key = Column(String(500), nullable=False)         # MinIO / R2 object key
    filename = Column(String(255), nullable=False)       # {document_id}.{ext} stored in bucket
    original_filename = Column(String(500), nullable=True)  # user-supplied name
    title = Column(String(500), nullable=True)
    mime_type = Column(String(100), nullable=True)

    # Provenance
    source = Column(String(50), nullable=False, default="upload")  # upload | web | manual_entry | drive | chat_history
    source_url = Column(String(2000), nullable=True)     # original URL for web source
    conversation_id = Column(String(64), nullable=True, index=True)  # stable key for chat_history upserts

    # Stats
    size_bytes = Column(Integer, nullable=False, default=0)
    chunk_count = Column(Integer, nullable=False, default=0)

    # Dedup.  `content_sha256` is the hex SHA256 of the raw payload (file
    # bytes, URL bytes, or text bytes).  When a user re-uploads identical
    # content we insert a fresh row whose `aliased_document_id` points at
    # the original and skip re-processing.
    content_sha256 = Column(String(64), nullable=True, index=True)
    aliased_document_id = Column(String(64), nullable=True, index=True)

    # Lifecycle
    status = Column(String(50), nullable=False, default="processing")  # processing | ready | error
    error_message = Column(Text, nullable=True)

    tags = Column(ARRAY(String), nullable=False, default=list)
    meta = Column("metadata", JSONB, nullable=False, default=dict)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)


class BlogCommentSQL(Base):
    """Comment on a blog post. Supports @mentions for users and agents."""
    __tablename__ = "blog_comments"

    id = Column(String(36), primary_key=True, default=_uuid_str)
    post_id = Column(String(36), nullable=False, index=True)
    user_id = Column(String(64), nullable=False, index=True)
    user_name = Column(String(255), nullable=False)
    user_avatar_url = Column(String(1000), nullable=True)

    content = Column(Text, nullable=False)
    mentions = Column(JSONB, nullable=False, default=list)  # [{"type": "user"|"agent", "id": "...", "name": "..."}]

    parent_id = Column(String(36), nullable=True, index=True)  # For threaded replies

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)


# ── Customer Service: tickets, replies, templates, branding ─────────
#
# Multi-tenant. Every row scoped by `organization_id`.  Public IDs for
# tickets are short (TK-xxxxxxxx) so they can be shared in URLs and
# emails without leaking row counts.


class SupportTicketSQL(Base):
    """A support inquiry from an end-user of a tenant.

    Created by:
      - the public portal (channel=portal)
      - the embeddable widget (channel=widget)  [future]
      - inbound email   (channel=email)         [future]
      - operator manual entry (channel=manual)
      - external API   (channel=api)            [future]
    """
    __tablename__ = "support_tickets"

    # Public-facing id, shareable in URLs.  IS the primary key — no
    # separate internal/public split, less plumbing, harder to confuse.
    id = Column(String(64), primary_key=True, default=_short_ticket_id)

    organization_id = Column(String(64), nullable=False, index=True)

    customer_email = Column(String(320), nullable=False, index=True)
    customer_name = Column(String(200), nullable=True)

    subject = Column(String(500), nullable=False)
    body = Column(Text, nullable=False)

    priority = Column(String(16), nullable=False, default="Medium")        # High|Medium|Low
    status = Column(String(32), nullable=False, default="Open")            # Open|In Progress|Resolved|Closed|Cancelled
    category = Column(String(64), nullable=True, index=True)
    channel = Column(String(32), nullable=False, default="portal")         # portal|widget|email|api|manual

    sentiment_score = Column(Integer, nullable=True)                       # -100..100 (×100 of -1..1)
    assigned_user_id = Column(String(64), nullable=True, index=True)
    submitter_user_id = Column(String(64), nullable=True)                  # if authenticated submission

    meta = Column("metadata", JSONB, nullable=False, default=dict)         # IP, user-agent, referrer, etc.

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    resolved_at = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True, index=True)


class TicketReplySQL(Base):
    """One message in a ticket conversation.

    `author_type` in {operator, customer, agent_ai}.  Operator and
    agent_ai replies are visible on the public status check; customer
    replies show in the operator inbox.
    """
    __tablename__ = "ticket_replies"

    id = Column(String(64), primary_key=True, default=_uuid_str)
    ticket_id = Column(String(64), ForeignKey("support_tickets.id"), nullable=False, index=True)
    organization_id = Column(String(64), nullable=False, index=True)

    author_type = Column(String(16), nullable=False)                       # operator|customer|agent_ai
    author_user_id = Column(String(64), nullable=True)
    author_display_name = Column(String(200), nullable=True)

    body = Column(Text, nullable=False)
    template_id = Column(String(64), nullable=True, index=True)            # response_templates.id when used

    # AI-generated drafts that the operator sent: stash citations + model
    # info here so the status page can render "Powered by AI" badges.
    ai_draft_meta = Column(JSONB, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    deleted_at = Column(DateTime, nullable=True)


class ResponseTemplateSQL(Base):
    """Reusable reply template, tenant-scoped.

    Five `is_default=True` rows are seeded per org on first read of
    /customer-service/templates.  Operators can also create their own.
    """
    __tablename__ = "response_templates"

    id = Column(String(64), primary_key=True, default=_uuid_str)
    organization_id = Column(String(64), nullable=False, index=True)

    name = Column(String(200), nullable=False)
    category = Column(String(64), nullable=False, index=True)
    tone = Column(String(32), nullable=True)                               # professional_friendly|formal|empathetic|...

    body = Column(Text, nullable=False)
    description = Column(Text, nullable=True)                              # quick-reply hover hint
    variables = Column(JSONB, nullable=False, default=list)                # ["customer_name", "ticket_id", ...]

    usage_count = Column(Integer, nullable=False, default=0)
    is_default = Column(Boolean, nullable=False, default=False)

    created_by_user_id = Column(String(64), nullable=True)
    created_by_agent = Column(Boolean, nullable=False, default=False)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True)


class OrgBrandingSQL(Base):
    """Per-tenant branding for the public support portal + widget.

    `slug` is the URL segment in `/portal/{slug}` and is enforced unique
    across all orgs.  Lower-cased ASCII; validated at the API layer.
    """
    __tablename__ = "org_branding"

    organization_id = Column(String(64), primary_key=True)
    slug = Column(String(64), nullable=False, unique=True, index=True)

    display_name = Column(String(200), nullable=False)
    logo_url = Column(String(1000), nullable=True)
    primary_color = Column(String(16), nullable=False, default="#4f46e5")
    accent_color = Column(String(16), nullable=False, default="#6366f1")
    hero_copy = Column(Text, nullable=True)

    support_email = Column(String(320), nullable=True)
    sla_response_minutes = Column(Integer, nullable=False, default=60)
    captcha_enabled = Column(Boolean, nullable=False, default=False)

    public_categories = Column(JSONB, nullable=False, default=list)        # ["technical_support", "billing", ...]

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)


class SupportArticleSQL(Base):
    """Tenant help-center article surfaced on the public portal at
    `/portal/{slug}/help`.

    Operators draft + publish here.  Anonymous visitors only see rows
    where `published=True AND deleted_at IS NULL`.  Slug is unique per
    org (so `agripro/help/getting-started` and
    `queska/help/getting-started` can coexist).
    """
    __tablename__ = "support_articles"

    id = Column(String(64), primary_key=True, default=_uuid_str)
    organization_id = Column(String(64), nullable=False, index=True)

    slug = Column(String(120), nullable=False)                             # unique within an org
    title = Column(String(300), nullable=False)
    summary = Column(String(500), nullable=True)                           # short blurb used on the list page
    body = Column(Text, nullable=False)                                    # markdown / sanitized HTML

    category = Column(String(64), nullable=True, index=True)
    tags = Column(JSONB, nullable=False, default=list)                     # ["getting_started", "billing"]

    published = Column(Boolean, nullable=False, default=False)
    featured = Column(Boolean, nullable=False, default=False)              # pin on the help home

    view_count = Column(Integer, nullable=False, default=0)
    helpful_count = Column(Integer, nullable=False, default=0)
    not_helpful_count = Column(Integer, nullable=False, default=0)

    # Optional: id of the RAG document this article was also pushed to,
    # so we can keep them in sync when the article body changes.
    rag_document_id = Column(String(64), nullable=True, index=True)

    created_by_user_id = Column(String(64), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    published_at = Column(DateTime, nullable=True)
    deleted_at = Column(DateTime, nullable=True, index=True)


# ── Data Analysis: persisted analysis runs ──────────────────────────
#
# One row per run.  Created at upload time with status=pending, then
# transitions to processing → ready (or error) as the pipeline runs.
# The original file lives in MinIO at `s3_key`; results land in the
# JSONB columns so a run can be reloaded into the UI without re running
# the heavy compute.


class DataAnalysisRunSQL(Base):
    """A persisted run of the Data Analysis Agent on an uploaded file.

    Multi tenant: every row carries `organization_id` and queries filter
    on it before any other clause.  Soft delete via `deleted_at`.
    """
    __tablename__ = "data_analysis_runs"

    id = Column(String(64), primary_key=True, default=_uuid_str)
    organization_id = Column(String(64), nullable=False, index=True)
    user_id = Column(String(64), nullable=False, index=True)

    mode = Column(String(32), nullable=False, default="exploratory")        # exploratory|statistical|visualization|anomaly|trend|report
    status = Column(String(32), nullable=False, default="pending")          # pending|processing|ready|error

    # Source file (lives in MinIO).
    s3_key = Column(String(500), nullable=False)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(500), nullable=True)
    content_type = Column(String(100), nullable=True)
    size_bytes = Column(Integer, nullable=False, default=0)

    # Parse stats.
    row_count = Column(Integer, nullable=True)
    column_count = Column(Integer, nullable=True)
    columns = Column(JSONB, nullable=False, default=list)                   # [{name, dtype}, ...]
    preview_rows = Column(JSONB, nullable=False, default=list)              # first N rows

    # Analysis results (JSONB lets each mode populate the fields it cares
    # about without rigid column constraints).
    summary_stats = Column(JSONB, nullable=True)
    visualizations = Column(JSONB, nullable=False, default=list)
    anomalies = Column(JSONB, nullable=True)
    trends = Column(JSONB, nullable=True)
    statistical_results = Column(JSONB, nullable=True)                      # hypothesis tests, regressions
    insights = Column(JSONB, nullable=False, default=list)                  # [{text, type, ...}]
    ai_summary = Column(Text, nullable=True)

    # NL Q&A history accumulates across calls to /runs/{id}/ask.
    question_history = Column(JSONB, nullable=False, default=list)          # [{question, answer, model_used, asked_at}]

    processing_time_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True, index=True)


# ── Knowledge Graph: nodes, edges, extraction audit ────────────────
#
# Multi tenant. Every row scoped by `organization_id`. The graph itself
# lives across `kg_nodes` and `kg_edges`; `kg_extractions` is a separate
# audit log so the operator UI can show recent activity without scanning
# every node and edge.


class KGNodeSQL(Base):
    """A node in a tenant's knowledge graph."""
    __tablename__ = "kg_nodes"

    id = Column(String(64), primary_key=True, default=_uuid_str)
    organization_id = Column(String(64), nullable=False, index=True)
    type = Column(String(32), nullable=False)
    label = Column(String(500), nullable=False)
    properties = Column(JSONB, nullable=False, default=dict)
    confidence = Column(Float, nullable=False, default=1.0)
    source_extraction_id = Column(String(64), nullable=True, index=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True, index=True)


class KGEdgeSQL(Base):
    """A directed relation between two nodes."""
    __tablename__ = "kg_edges"

    id = Column(String(64), primary_key=True, default=_uuid_str)
    organization_id = Column(String(64), nullable=False, index=True)
    source_id = Column(String(64), nullable=False, index=True)
    target_id = Column(String(64), nullable=False, index=True)
    type = Column(String(32), nullable=False)
    properties = Column(JSONB, nullable=False, default=dict)
    confidence = Column(Float, nullable=False, default=1.0)
    source_extraction_id = Column(String(64), nullable=True, index=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True, index=True)


class KGExtractionSQL(Base):
    """Audit + Recent Extractions feed. One row per Extract / Discover /
    Fill Gaps call."""
    __tablename__ = "kg_extractions"

    id = Column(String(64), primary_key=True, default=_uuid_str)
    organization_id = Column(String(64), nullable=False, index=True)
    user_id = Column(String(64), nullable=False, index=True)

    action = Column(String(32), nullable=False)        # extract|discover|fill_gaps
    status = Column(String(32), nullable=False, default="ready")  # ready|error
    title = Column(String(500), nullable=True)
    source_kind = Column(String(32), nullable=True)    # inline|rag_document
    source_ref = Column(String(500), nullable=True)
    content_preview = Column(String(500), nullable=True)

    node_ids = Column(JSONB, nullable=False, default=list)
    edge_ids = Column(JSONB, nullable=False, default=list)
    node_count = Column(Integer, nullable=False, default=0)
    edge_count = Column(Integer, nullable=False, default=0)

    processing_time_ms = Column(Integer, nullable=True)
    error_message = Column(Text, nullable=True)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    deleted_at = Column(DateTime, nullable=True, index=True)


# ── Huddle (live meeting rooms) ───────────────────────────────────────

def _huddle_room_name() -> str:
    """Unguessable Jitsi room slug — namespaced for Lumicoria."""
    return f"lumi-{uuid.uuid4().hex[:16]}"


def _huddle_share_token() -> str:
    """URL-safe share token, separate from room_name so we can revoke
    a public link without disrupting an in-flight call."""
    return uuid.uuid4().hex


class HuddleSQL(Base):
    """A live or scheduled meeting room. Linked to MeetingSQL via
    `processed_meeting_id` once the post-call summary runs."""
    __tablename__ = "huddles"

    id = Column(String(36), primary_key=True, default=_uuid_str)

    # Jitsi + sharing
    room_name = Column(String(128), nullable=False, unique=True, index=True, default=_huddle_room_name)
    share_token = Column(String(64), nullable=False, unique=True, index=True, default=_huddle_share_token)

    # Hierarchy
    host_user_id = Column(String(64), nullable=False, index=True)
    organization_id = Column(String(64), nullable=False, index=True)
    team_id = Column(String(64), nullable=True, index=True)
    project_id = Column(String(64), nullable=True, index=True)
    channel_id = Column(String(64), nullable=True)

    # Identity
    title = Column(String(500), nullable=False, default="Untitled meeting")
    meeting_type = Column(String(32), nullable=False, default="instant")  # instant | scheduled | recurring

    # Lifecycle
    status = Column(String(32), nullable=False, default="live", index=True)  # scheduled | live | ended | cancelled
    scheduled_start = Column(DateTime, nullable=True)
    scheduled_end = Column(DateTime, nullable=True)
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    duration_sec = Column(Integer, nullable=True)
    participant_count_peak = Column(Integer, nullable=False, default=0)

    # Agents in the call
    agent_keys = Column(JSONB, nullable=False, default=list)
    custom_agent_ids = Column(JSONB, nullable=False, default=list)

    # Recording
    recording_enabled = Column(Boolean, nullable=False, default=False)
    recording_url = Column(String(1024), nullable=True)
    recording_object_key = Column(String(1024), nullable=True)
    recording_size_bytes = Column(Integer, nullable=True)
    recording_mime = Column(String(128), nullable=True)
    recording_retention_days = Column(Integer, nullable=False, default=30)
    recording_expires_at = Column(DateTime, nullable=True, index=True)

    # Post-call processing
    transcript_text = Column(Text, nullable=True)
    processed_meeting_id = Column(String(36), ForeignKey("meetings.id"), nullable=True, index=True)

    # Security / compliance
    lobby_enabled = Column(Boolean, nullable=False, default=False)
    require_sso = Column(Boolean, nullable=False, default=False)
    e2ee_enabled = Column(Boolean, nullable=False, default=False)
    data_residency = Column(String(8), nullable=False, default="us")  # us | eu | in

    # Flexible payload — breakout rooms, polls, etc. (Phase 2)
    meta = Column("metadata", JSONB, nullable=False, default=dict)

    created_at = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.utcnow, onupdate=datetime.utcnow)
    deleted_at = Column(DateTime, nullable=True, index=True)


class HuddleParticipantSQL(Base):
    """One row per join — captures both human users and AI agents
    that joined a huddle."""
    __tablename__ = "huddle_participants"

    id = Column(String(36), primary_key=True, default=_uuid_str)
    huddle_id = Column(String(36), ForeignKey("huddles.id"), nullable=False, index=True)

    # Human participant
    user_id = Column(String(64), nullable=True, index=True)
    guest_name = Column(String(255), nullable=True)
    guest_email = Column(String(255), nullable=True)

    # Agent participant (mutually exclusive with user_id)
    agent_key = Column(String(64), nullable=True)
    custom_agent_id = Column(String(64), nullable=True)

    role = Column(String(32), nullable=False, default="participant")  # host | cohost | participant | guest | agent

    joined_at = Column(DateTime, nullable=False, default=datetime.utcnow)
    left_at = Column(DateTime, nullable=True)

    meta = Column("metadata", JSONB, nullable=False, default=dict)


class HuddleTranscriptChunkSQL(Base):
    """Streaming transcript chunks emitted during a live huddle.
    `agent_responses` is populated by the live-agent dispatcher
    (Phase 1.5) — empty list in Phase 1."""
    __tablename__ = "huddle_transcript_chunks"

    id = Column(String(36), primary_key=True, default=_uuid_str)
    huddle_id = Column(String(36), ForeignKey("huddles.id"), nullable=False, index=True)

    speaker_user_id = Column(String(64), nullable=True)
    speaker_name = Column(String(255), nullable=False, default="Speaker")
    text = Column(Text, nullable=False)
    ts = Column(DateTime, nullable=False, default=datetime.utcnow, index=True)

    agent_responses = Column(JSONB, nullable=False, default=list)

