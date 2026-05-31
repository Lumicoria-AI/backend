"""
AgentRun model — one row per invocation of an agent, whether autonomous
(task executor in Phase 6) or part of a multi-agent step graph (Phase 7).

Collection: `agent_runs`

This is what powers the dashboard analytics (Phase 9): runs per agent,
success rate, p50/p95 latency, parent→child fan-out, token usage.
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field

from backend.models.mongodb_models import PyObjectId


class AgentRunStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"            # gated out (e.g. low confidence)


class AgentRunTrigger(str, Enum):
    """How this run was kicked off."""
    CHAT = "chat"                  # via the main /chat endpoint
    TASK_EXECUTOR = "task_executor"  # autonomous task agent (Phase 6)
    STEP_GRAPH = "step_graph"      # mother-orchestrator calling a sub-agent
    AGENT_STUDIO = "agent_studio"  # workflow execution
    MANUAL = "manual"              # direct API call to /agents/{key}/...


class AgentRun(BaseModel):
    """One agent invocation.  Supports parent/child runs for orchestration."""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    agent_key: str                              # e.g. "rag", "meeting", "document"
    agent_name: Optional[str] = None            # human-readable name snapshot

    user_id: PyObjectId
    organization_id: Optional[PyObjectId] = None

    # Linkage
    task_id: Optional[PyObjectId] = None        # tasks._id when triggered by a task
    conversation_id: Optional[str] = None       # chat conversation id
    parent_run_id: Optional[PyObjectId] = None  # for step-graph children
    step_index: int = 0                          # order within parent

    # Execution
    trigger: AgentRunTrigger = AgentRunTrigger.MANUAL
    status: AgentRunStatus = AgentRunStatus.RUNNING
    input: Dict[str, Any] = Field(default_factory=dict)
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    # Performance
    duration_ms: Optional[int] = None
    model_used: Optional[str] = None
    provider: Optional[str] = None              # openai | anthropic | gemini | etc.
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    cost_usd: Optional[float] = None
    confidence: Optional[float] = None          # router confidence when applicable

    # Timestamps
    started_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    # Free-form (sources, citations, partial trace)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str, datetime: lambda v: v.isoformat()},
    )


class AgentRunCreate(BaseModel):
    """Payload for opening a new agent run.  `id` is server-generated."""
    agent_key: str
    agent_name: Optional[str] = None
    user_id: PyObjectId
    organization_id: Optional[PyObjectId] = None
    task_id: Optional[PyObjectId] = None
    conversation_id: Optional[str] = None
    parent_run_id: Optional[PyObjectId] = None
    step_index: int = 0
    trigger: AgentRunTrigger = AgentRunTrigger.MANUAL
    input: Dict[str, Any] = Field(default_factory=dict)
    model_used: Optional[str] = None
    provider: Optional[str] = None
    confidence: Optional[float] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(arbitrary_types_allowed=True)
