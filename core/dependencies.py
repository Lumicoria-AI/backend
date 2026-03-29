"""
Dependencies for FastAPI dependency injection system.
"""

from ..agents.agent_service import get_agent_service as _get_agent_service, AgentService


def get_agent_service() -> AgentService:
    """
    Get the AgentService instance.

    Returns the same singleton that is initialized at startup via
    init_agent_service() (which loads config.yaml with all agents).
    """
    return _get_agent_service()
