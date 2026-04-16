"""
PostgreSQL models for Lumicoria.ai

These models are used for relational data that benefits from SQL semantics:
tasks, workflows, and agent execution logs.
"""

from __future__ import annotations

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
    ForeignKey,
    Enum as SAEnum,
)
from sqlalchemy.dialects.postgresql import JSONB, ARRAY

from backend.db.base_class import Base
from backend.models.mongodb_models import TaskStatus, TaskPriority, AgentStatus


def _uuid_str() -> str:
    return str(uuid.uuid4())


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
