# Import task models from mongodb_models — single source of truth.
from backend.models.mongodb_models import (
    Task,
    TaskCreate,
    TaskUpdate,
    TaskStatus,
    TaskPriority,
    # Phase 1 additions
    AssigneeKind,
    AgentProposal,
    AgentProposalStatus,
    ReminderState,
)

# Re-export everything
__all__ = [
    "Task",
    "TaskCreate",
    "TaskUpdate",
    "TaskStatus",
    "TaskPriority",
    # Phase 1
    "AssigneeKind",
    "AgentProposal",
    "AgentProposalStatus",
    "ReminderState",
]
