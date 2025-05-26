from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from enum import Enum

from api.deps import get_current_active_user
from db.mongodb.repositories.agent_universe_repository import agent_universe_repository
from db.mongodb.repositories.component_repository import component_repository
from db.mongodb.repositories.permission_repository import permission_repository
from models.user import User
from models.agent import (
    Agent,
    AgentCreate,
    AgentUpdate,
    AgentCapability,
    AgentType,
    AgentStatus
)

router = APIRouter()

class AgentResponse(BaseModel):
    id: str
    name: str
    description: str
    agent_type: AgentType
    status: AgentStatus
    capabilities: List[AgentCapability]
    created_by: str
    organization_id: str
    created_at: datetime
    updated_at: Optional[datetime]
    metadata: Optional[Dict[str, Any]]
    success_rate: Optional[float]
    error_rate: Optional[float]
    avg_response_time: Optional[float]

class AgentDiscoveryFilters(BaseModel):
    capabilities: Optional[List[str]] = None
    agent_type: Optional[AgentType] = None
    min_success_rate: Optional[float] = None
    max_error_rate: Optional[float] = None
    tags: Optional[List[str]] = None
    search_query: Optional[str] = None

class AgentSummaryResponse(BaseModel):
    total_agents: int
    active_agents: int
    avg_success_rate: float
    avg_error_rate: float
    avg_response_time: float
    total_usage: int
    capability_stats: Dict[str, Any]

@router.get("/discover", response_model=List[AgentResponse])
async def discover_agents(
    filters: AgentDiscoveryFilters = Depends(),
    skip: int = Query(0, ge=0),
    limit: int = Query(10, ge=1, le=100),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Discover available agents based on filters.
    """
    agents = await agent_universe_repository.discover_agents(
        organization_id=current_user.organization_id,
        capabilities=filters.capabilities,
        agent_type=filters.agent_type,
        min_success_rate=filters.min_success_rate,
        max_error_rate=filters.max_error_rate,
        tags=filters.tags,
        search_query=filters.search_query,
        skip=skip,
        limit=limit
    )
    return agents

@router.post("", response_model=AgentResponse)
async def create_agent(
    agent_in: AgentCreate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Create a new agent.
    """
    # Check if user has permission to create agents
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="AGENT",
        resource_id="*",
        permission_type="CREATE"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to create agents"
        )

    agent = await agent_universe_repository.create_agent(
        name=agent_in.name,
        description=agent_in.description,
        agent_type=agent_in.agent_type,
        capabilities=agent_in.capabilities,
        organization_id=current_user.organization_id,
        created_by=current_user.id,
        configuration=agent_in.configuration,
        metadata=agent_in.metadata
    )
    return agent

@router.get("/{agent_id}", response_model=AgentResponse)
async def get_agent(
    agent_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get agent by ID.
    """
    agent = await agent_universe_repository.get_agent_by_id(
        agent_id=agent_id,
        organization_id=current_user.organization_id
    )
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )

    # Basic permission check: user must be in the same organization or agent is public
    if str(agent.organization_id) != current_user.organization_id and not agent.is_public:
        # More granular permission check if needed (e.g., view permission)
        has_permission = await permission_repository.check_permission(
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            resource_type="AGENT",
            resource_id=agent_id,
            permission_type="VIEW"
        )
        if not has_permission:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this agent")

    return agent

@router.put("/{agent_id}", response_model=AgentResponse)
async def update_agent(
    agent_id: str,
    agent_in: AgentUpdate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Update an agent.
    """
    # Check if user has permission to update agents
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="AGENT",
        resource_id=agent_id,
        permission_type="EDIT"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to update this agent"
        )

    agent = await agent_universe_repository.update_agent(
        agent_id=agent_id,
        organization_id=current_user.organization_id,
        update_data=agent_in.dict(exclude_unset=True)
    )
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return agent

@router.post("/{agent_id}/execute")
async def execute_agent(
    agent_id: str,
    input_data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Execute an agent with input data.
    """
    # Check if user has permission to execute agents
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="AGENT",
        resource_id=agent_id,
        permission_type="EXECUTE"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to execute this agent"
        )

    try:
        result = await agent_universe_repository.execute_agent(
            agent_id=agent_id,
            organization_id=current_user.organization_id,
            input_data=input_data,
            user_id=current_user.id
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{agent_id}/performance", response_model=Dict[str, Any])
async def get_agent_performance(
    agent_id: str,
    time_range: str = Query("7d", regex="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get agent performance metrics.
    """
    # Check if user has permission to view agent performance
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="AGENT",
        resource_id=agent_id,
        permission_type="VIEW_PERFORMANCE"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to view agent performance"
        )

    # Convert time_range string to timedelta
    time_delta = None
    if time_range:
        amount, unit = int(time_range[:-1]), time_range[-1]
        if unit == 'd':
            time_delta = timedelta(days=amount)
        elif unit == 'y':
            time_delta = timedelta(days=amount * 365) # Approximation

    # The repository method might need to be updated to filter performance by agent_id
    # For now, calling the organization-wide stats, which is not quite right for single agent performance.
    # TODO: Add get_agent_performance_by_id method to AgentUniverseRepository
    stats = await agent_universe_repository.get_agent_performance_stats(
        organization_id=current_user.organization_id,
        time_range=time_delta # This will filter by last_active, not performance history
    )

    # For now, let's just return basic info from the agent object itself
    agent_obj = await agent_universe_repository.get_by_id(agent_id)
    if not agent_obj:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Agent not found")

    return {
        "success_rate": agent_obj.success_rate,
        "error_rate": agent_obj.error_rate,
        "avg_response_time": agent_obj.average_response_time,
        "usage_count": agent_obj.usage_count,
        "status": agent_obj.status.value
        # TODO: Include error log summary or recent errors
    }

@router.get("/capabilities", response_model=List[Dict[str, Any]])
async def get_agent_capabilities(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get all available agent capabilities with usage statistics.
    """
    # Check if user has permission to view capabilities
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="CAPABILITY", # Assuming CAPABILITY is a resource type
        resource_id="*",
        permission_type="VIEW"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to view capabilities"
        )

    capabilities_stats = await agent_universe_repository.get_agent_capabilities(
        organization_id=current_user.organization_id,
        include_public=True # Include public capabilities
    )
    return capabilities_stats

@router.get("/analytics", response_model=Dict[str, Any])
async def get_agent_analytics(
    time_range: str = Query("7d", regex="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get organization-wide agent analytics.
    """
    # Check if user has permission to view organization analytics
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="ORGANIZATION", # Assuming ORGANIZATION is a resource type
        resource_id=current_user.organization_id,
        permission_type="VIEW_ANALYTICS"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to view organization analytics"
        )

    # Convert time_range string to timedelta
    time_delta = None
    if time_range:
        amount, unit = int(time_range[:-1]), time_range[-1]
        if unit == 'd':
            time_delta = timedelta(days=amount)
        elif unit == 'y':
            time_delta = timedelta(days=amount * 365) # Approximation

    analytics = await agent_universe_repository.get_agent_analytics(
        organization_id=current_user.organization_id,
        time_range=time_delta
    )
    return analytics

@router.get("/summary", response_model=AgentSummaryResponse)
async def get_agent_summary(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get summary statistics for agents in the organization.
    """
    # Check if user has permission to view organization statistics
    # Reusing VIEW_ANALYTICS permission for now, or could define a specific VIEW_SUMMARY
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="ORGANIZATION",
        resource_id=current_user.organization_id,
        permission_type="VIEW_ANALYTICS" # Or "VIEW_SUMMARY"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to view agent summary"
        )

    summary = await agent_universe_repository.get_agent_performance_stats(
        organization_id=current_user.organization_id
        # No time_range needed for a simple summary of current state
    )
    return summary 