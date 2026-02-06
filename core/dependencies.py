"""
Dependencies for FastAPI dependency injection system.
"""

from typing import Optional
from fastapi import Depends, HTTPException, status
from ..agents.agent_service import AgentService

# Store singleton instances
_agent_service: Optional[AgentService] = None

def get_agent_service() -> AgentService:
    """
    Get the AgentService instance.
    
    Returns:
        The agent service instance
    """
    global _agent_service
    
    if _agent_service is None:
        from ..core.config import settings
        # Initialize with settings
        _agent_service = AgentService(settings.dict())
        
    return _agent_service
