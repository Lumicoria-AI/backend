"""
Activity Logger Service — fire-and-forget activity logging.

Provides a simple `log_activity()` helper that other endpoints call
after performing an action. Failures are logged but never propagated
to the caller, so activity logging can never break the primary flow.

Each log entry records: who did what, on which resource, which agent
(if any), and arbitrary details for debugging.
"""

from typing import Optional, Dict, Any
from datetime import datetime
import structlog

from backend.db.mongodb.repositories.activity_repository import activity_repository

logger = structlog.get_logger(__name__)


async def log_activity(
    user_id: str,
    organization_id: str,
    activity_type: str,
    details: Optional[Dict[str, Any]] = None,
    related_resource_type: Optional[str] = None,
    related_resource_id: Optional[str] = None,
    agent_id: Optional[str] = None,
    agent_name: Optional[str] = None,
    severity: str = "info",
) -> None:
    """
    Fire-and-forget activity log creation.

    Parameters
    ----------
    user_id : str
        The user who performed the action.
    organization_id : str
        The organization context.
    activity_type : str
        A dot-separated event name, e.g. "document.uploaded",
        "task.created", "agent.executed", "chat.message_sent",
        "research.query", "creative.content_generated".
    details : dict, optional
        Arbitrary JSON-safe payload (names, queries, counts, etc.).
    related_resource_type : str, optional
        E.g. "DOCUMENT", "TASK", "AGENT", "CONVERSATION", "PROJECT".
    related_resource_id : str, optional
        The ID of the related resource.
    agent_id : str, optional
        If the action involves a specific agent, pass its ID here.
        This enables per-agent activity filtering.
    agent_name : str, optional
        Human-readable agent name (stored in details for display).
    severity : str
        "info" | "warning" | "error".  Default "info".
    """
    try:
        enriched_details = dict(details or {})
        if agent_name:
            enriched_details["agent_name"] = agent_name
        if agent_id:
            enriched_details["agent_id"] = agent_id

        await activity_repository.create_log_entry(
            organization_id=organization_id,
            user_id=user_id,
            activity_type=activity_type,
            details=enriched_details,
            related_resource_type=related_resource_type,
            related_resource_id=related_resource_id,
        )

        logger.debug(
            "Activity logged",
            activity_type=activity_type,
            user_id=user_id,
            agent_id=agent_id,
        )
    except Exception as exc:
        # Never let logging failures propagate — just warn
        logger.warning(
            "Failed to log activity (non-fatal)",
            activity_type=activity_type,
            error=str(exc),
        )
