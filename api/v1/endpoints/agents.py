from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from enum import Enum
from bson import ObjectId

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.agent_universe_repository import agent_universe_repository
from backend.db.mongodb.repositories.component_repository import component_repository
from backend.db.mongodb.repositories.permission_repository import permission_repository
from backend.models.user import User
from backend.db.postgres import get_optional_async_db
from backend.db.postgres_repositories.agent_execution_repository import PostgresAgentExecutionRepository
from backend.models.mongodb_models import (
    Agent,
    AgentCreate,
    AgentUpdate,
    AgentCapability,
    AgentType,
    AgentStatus
)
from backend.core.security import rate_limit
from backend.core.billing import enforce_agent_limit, require_subscription, BillingCheck
from backend.services.activity_logger import log_activity
from sqlalchemy.ext.asyncio import AsyncSession

# Import the AgentService and agent implementations
from backend.agents.agent_service import AgentService
from backend.agents.document_agent import DocumentAgent
# Import configuration loading (assuming it's in backend.core.config)
from backend.core.config import settings
import yaml
import structlog

logger = structlog.get_logger()
router = APIRouter()

# Load configuration and initialize AgentService (simple approach, consider dependency injection for production)
def get_agent_service() -> AgentService:
    import os
    from pathlib import Path
    
    # Determine the base directory for the project
    # First try to use settings if it has a path attribute
    if hasattr(settings, 'BASE_DIR'):
        base_dir = settings.BASE_DIR
    else:
        # If not available, construct path based on the current file location
        current_file = Path(__file__)
        base_dir = current_file.parent.parent.parent.parent.parent  # Go up to the project root
    
    config_path = os.path.join(base_dir, "backend", "config", "config.yaml")
    
    # Use a default config if file doesn't exist
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found at {config_path}, using default configuration")
        config = {
            "ai_models": {
                # Provider-neutral defaults — actual provider is selected
                # by DEFAULT_LLM_PROVIDER env var at runtime
            }
        }
    else:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    
    return AgentService(config)

# Create a singleton instance for use throughout this file
agent_service = get_agent_service()

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

def _serialize_agent(agent) -> dict:
    """Convert an Agent model or dict to a JSON-safe dict."""
    if hasattr(agent, "model_dump"):
        result = agent.model_dump(mode="json")
    elif hasattr(agent, "dict"):
        result = agent.dict()
    else:
        result = dict(agent) if not isinstance(agent, dict) else agent
    for key in ("id", "_id", "organization_id", "created_by", "workflow_id"):
        if key in result and result[key] is not None:
            result[key] = str(result[key])
    # Flatten status from state for frontend convenience
    if "state" in result and isinstance(result["state"], dict):
        result["status"] = result["state"].get("status", "active")
    return result


@router.get("", response_model=List[dict])
async def list_agents(current_user: User = Depends(get_current_active_user)) -> Any:
    """List all available agents."""
    try:
        agents = await agent_universe_repository.discover_agents(
            organization_id=getattr(current_user, "organization_id", None),
            created_by=str(current_user.id),
            capabilities=None,
            agent_type=None,
            min_success_rate=None,
            max_error_rate=None,
            tags=None,
            search_query=None,
            skip=0,
            limit=100
        )
        return [_serialize_agent(a) for a in agents]
    except Exception as e:
        logger.error(f"Error listing agents: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error listing agents"
        )

@router.get("/{agent_id}", response_model=dict)
async def get_agent(
    agent_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get agent details."""
    try:
        agent = await agent_universe_repository.get_agent_by_id(
            agent_id=agent_id,
            organization_id=getattr(current_user, "organization_id", None)
        )
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
        return _serialize_agent(agent)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error getting agent details"
        )

@router.post("/{agent_id}/chat", response_model=dict)
@rate_limit()
async def chat_with_agent(
    agent_id: str,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    billing: BillingCheck = Depends(enforce_agent_limit)
) -> Any:
    """Chat with an agent with billing enforcement."""
    try:
        data = await request.json()
        message = data.get("message")
        if not message:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message is required"
            )

        response = await agent_service.chat_with_agent(
            agent_id=agent_id,
            user_id=str(current_user.id),
            message=message
        )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="agent.chat",
            details={"agent_id": agent_id, "message_preview": message[:100]},
            related_resource_type="AGENT",
            related_resource_id=agent_id,
            agent_id=agent_id,
        )

        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error chatting with agent: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing chat request"
        )

@router.post("/{agent_id}/execute", response_model=dict)
@rate_limit()
async def execute_agent_task(
    agent_id: str,
    request: Request,
    current_user: User = Depends(get_current_active_user),
    billing: BillingCheck = Depends(enforce_agent_limit)
) -> Any:
    """Execute an agent task with billing enforcement."""
    try:
        data = await request.json()
        task = data.get("task")
        if not task:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Task is required"
            )

        result = await agent_service.execute_task(
            agent_id=agent_id,
            user_id=str(current_user.id),
            task=task
        )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="agent.executed",
            details={"agent_id": agent_id, "task_preview": str(task)[:100]},
            related_resource_type="AGENT",
            related_resource_id=agent_id,
            agent_id=agent_id,
        )

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error executing agent task: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error executing task"
        )

@router.get("/{agent_id}/status", response_model=dict)
async def get_agent_status(
    agent_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get agent status."""
    try:
        status = await agent_service.get_agent_status(agent_id)
        if not status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Agent not found"
            )
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting agent status: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error getting agent status"
        )

@router.post("")
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
        organization_id=getattr(current_user, "organization_id", None),
        resource_type="AGENT",
        resource_id="*",
        permission_type="CREATE"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to create agents"
        )

    org_id = getattr(current_user, "organization_id", None)
    user_id = current_user.id

    agent = await agent_universe_repository.create_agent(
        name=agent_in.name,
        description=agent_in.description,
        agent_type=agent_in.agent_type,
        capabilities=agent_in.capabilities,
        organization_id=org_id,
        created_by=user_id,
        configuration=agent_in.configuration,
        metadata=agent_in.metadata,
        is_public=agent_in.is_public,
        tags=agent_in.tags,
        agent_model_config=agent_in.agent_model_config.model_dump() if agent_in.agent_model_config else None,
    )

    try:
        await log_activity(
            user_id=str(user_id),
            organization_id=str(org_id) if org_id else None,
            activity_type="agent.created",
            details={"name": agent_in.name, "agent_type": agent_in.agent_type.value if hasattr(agent_in.agent_type, "value") else str(agent_in.agent_type)},
            related_resource_type="AGENT",
            related_resource_id=str(getattr(agent, "id", "")),
            agent_name=agent_in.name,
        )
    except Exception as log_err:
        logger.warning("Failed to log agent creation activity", error=str(log_err))

    return _serialize_agent(agent)

@router.put("/{agent_id}")
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
        organization_id=getattr(current_user, "organization_id", None),
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
        organization_id=getattr(current_user, "organization_id", None),
        update_data=agent_in.dict(exclude_unset=True)
    )
    if not agent:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return _serialize_agent(agent)


@router.delete("/{agent_id}")
async def delete_agent(
    agent_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Delete an agent."""
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=getattr(current_user, "organization_id", None),
        resource_type="AGENT",
        resource_id=agent_id,
        permission_type="DELETE"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to delete this agent"
        )

    deleted = await agent_universe_repository.delete_agent(
        agent_id=agent_id,
        organization_id=getattr(current_user, "organization_id", None),
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Agent not found"
        )
    return {"detail": "Agent deleted successfully"}


@router.post("/{agent_id}/execute")
async def execute_agent(
    agent_id: str,
    input_data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service) # Inject AgentService
) -> Any:
    """
    Execute an agent with input data.
    """
    # Check if user has permission to execute agents
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=getattr(current_user, "organization_id", None),
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
        # Retrieve the agent instance
        agent_instance = agent_service.get_agent(agent_id)
        
        # Process the input data using the agent
        result = agent_instance.process(input_data)
        
        return {"result": result}
        
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except NotImplementedError:
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"Agent '{agent_id}' has not implemented the process method."
        )
    except Exception as e:
        # Log the error for debugging
        logger.error(f"Error executing agent {agent_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during agent execution."
        )

# Add new endpoints for specific actions like document upload and camera scan
# These might be in other endpoint files like documents.py or live_interaction.py
# depending on the overall API design.
# For now, I will assume these actions will internally call the appropriate agents
# via the AgentService.

# Example placeholder for a document processing endpoint (could be in documents.py)
# @router.post("/documents/process")
# async def process_document_endpoint(
#     file: UploadFile = File(...),
#     current_user: User = Depends(get_current_active_user),
#     agent_service: AgentService = Depends(get_agent_service)
# ):
#     # ... file processing logic ...
#     document_agent = agent_service.get_agent("document_agent") # Assuming "document_agent" is the configured name
#     processing_results = document_agent.process(processed_document_data)
#     return {"results": processing_results}

# Example placeholder for a camera scan endpoint (could be in live_interaction.py)
# @router.post("/live-interaction/scan")
# async def scan_camera_endpoint(
#     image_data: bytes = File(...),
#     current_user: User = Depends(get_current_active_user),
#     agent_service: AgentService = Depends(get_agent_service)
# ):
#     # ... image processing logic ...
#     vision_agent = agent_service.get_agent("vision_agent") # Assuming "vision_agent" is the configured name
#     scan_results = vision_agent.process(processed_image_data)
#     return {"results": scan_results}

# Example endpoint for creating a custom agent (might already be covered by POST /agents)
# If the POST /agents endpoint handles creating different agent types based on input,
# this might not be necessary as a separate endpoint.

# Add agent customization models and endpoints
class AgentCustomizationRequest(BaseModel):
    agent_type: str
    agent_name: str
    agent_description: str
    capabilities: List[str]
    features: Optional[List[str]] = None
    integrations: Optional[List[str]] = None
    preferred_model: Optional[str] = Field(None, description="AI model to use, e.g., 'perplexity', 'gemini', 'mistral'")
    
class AgentCustomizationResponse(BaseModel):
    system_prompt: str
    configuration: Dict[str, Any]
    capabilities: List[str]
    estimated_cost: Optional[str] = None

class AgentPromptRequest(BaseModel):
    agent_type: str
    user_query: str
    context: Optional[Dict[str, Any]] = None
    model: Optional[str] = Field(None, description="AI model to use for prompt generation")

@router.post("/customize", response_model=AgentCustomizationResponse)
async def create_custom_agent_prompt(
    customization_request: AgentCustomizationRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Generate a custom agent configuration using Perplexity API.
    
    This endpoint allows users to create specialized agents based on their specific requirements.
    """
    # Check if user has permission to customize agents
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=getattr(current_user, "organization_id", None),
        resource_type="AGENT",
        resource_id="*",
        permission_type="CREATE"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to customize agents"
        )
    
    try:
        # Use the LLM abstraction layer instead of direct Perplexity access
        from backend.ai_models import get_llm_client, LLMConfig
        
        llm_client = get_llm_client()
        
        # Format features and integrations
        features_str = ""
        if customization_request.features:
            features_str = "Features:\n" + "\n".join([f"- {feature}" for feature in customization_request.features])
            
        integrations_str = ""
        if customization_request.integrations:
            integrations_str = "Integrations:\n" + "\n".join([f"- {integration}" for integration in customization_request.integrations])
            
        # Generate agent system prompt via LLM
        capability_str = "\n".join([f"- {cap}" for cap in customization_request.capabilities])
        prompt = (
            f"Please create a detailed system prompt for a new AI agent with these specifications:\n\n"
            f"Agent Type: {customization_request.agent_type}\n"
            f"Agent Name: {customization_request.agent_name}\n"
            f"Description: {customization_request.agent_description}\n"
            f"Capabilities:\n{capability_str}\n"
            f"{features_str}\n{integrations_str}\n\n"
            f"The prompt should be comprehensive, clear, and effective at guiding the AI to perform "
            f"this specific role. Include appropriate tone, constraints, and response formats."
        )
        
        response = await llm_client.generate(
            messages=[
                {"role": "system", "content": "You are an AI system architect specializing in creating effective prompts for specialized AI agents."},
                {"role": "user", "content": prompt}
            ],
            config=LLMConfig(max_tokens=2048),
        )
        
        system_prompt = response.content
        
        # Generate basic configuration
        agent_configuration = {
            "system_prompt": system_prompt,
            "model": customization_request.preferred_model or "perplexity",
            "capabilities": customization_request.capabilities,
            "features": customization_request.features or [],
            "integrations": customization_request.integrations or [],
            "created_by": current_user.id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        return AgentCustomizationResponse(
            system_prompt=system_prompt,
            configuration=agent_configuration,
            capabilities=customization_request.capabilities,
            estimated_cost="Free" if len(customization_request.capabilities) < 5 else "Pro Plan"
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating custom agent: {str(e)}"
        )

@router.post("/generate-prompt", response_model=Dict[str, Any])
async def generate_agent_prompt(
    prompt_request: AgentPromptRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Generate a specialized prompt for an agent based on user query.
    """
    try:
        # Use the LLM abstraction layer
        from backend.ai_models import get_llm_client, LLMConfig
        
        llm_client = get_llm_client()
        
        # Format context as string
        context_str = ""
        if prompt_request.context:
            context_str = "\n\nContext:\n" + "\n".join([f"{k}: {v}" for k, v in prompt_request.context.items()])
        
        # Prepare prompt
        prompt = f"""Generate a specialized AI prompt for a {prompt_request.agent_type} agent.
        
User query: {prompt_request.user_query}
{context_str}

The prompt should be detailed, specific, and tailored to the user's needs as a {prompt_request.agent_type} agent.
"""
        
        response = await llm_client.generate(
            messages=[{"role": "user", "content": prompt}],
            config=LLMConfig(model=prompt_request.model),
        )

        return {
            "generated_prompt": response.content,
            "agent_type": prompt_request.agent_type,
            "model_used": response.model
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating agent prompt: {str(e)}"
        )

# Example endpoint to get agent capabilities (already exists as GET /agents/capabilities)

@router.get("/{agent_id}/performance", response_model=Dict[str, Any])
async def get_agent_performance(
    agent_id: str,
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get agent performance metrics.
    """
    # Placeholder for fetching agent performance metrics
    performance_data = {
        "agent_id": agent_id,
        "time_range": time_range,
        "metrics": {
            "success_rate": 0.95, # Example data
            "error_rate": 0.05,
            "avg_response_time": 0.5 # seconds
        }
    }
    return performance_data

@router.get("/capabilities", response_model=List[Dict[str, Any]])
async def get_agent_capabilities(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get available agent capabilities.
    """
    # Placeholder for fetching available agent capabilities
    capabilities = [
        {"name": "document_processing", "description": "Processes documents and extracts information"},
        {"name": "wellbeing_coaching", "description": "Provides personalized well-being suggestions"},
        {"name": "vision_analysis", "description": "Analyzes image and video data"},
        {"name": "meeting_summarization", "description": "Summarizes meetings and extracts action items"},
        {"name": "creative_generation", "description": "Generates creative content"},
        {"name": "student_assistance", "description": "Assists students with study tasks"},
        # Add other capabilities
    ]
    return capabilities

@router.get("/analytics", response_model=Dict[str, Any])
async def get_agent_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    Get agent analytics.
    """
    if settings.POSTGRES_ENABLED and db is not None:
        repo = PostgresAgentExecutionRepository(db)
        return await repo.get_execution_stats(
            organization_id=getattr(current_user, "organization_id", None),
            time_range=time_range
        )
    return {
        "time_range": time_range,
        "total_executions": 0,
        "executions_by_agent_type": {},
        "successful_executions": 0,
        "failed_executions": 0
    }

@router.get("/summary", response_model=AgentSummaryResponse)
async def get_agent_summary(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Dict[str, Any]:
    """
    Get agent summary statistics.
    """
    if settings.POSTGRES_ENABLED and db is not None:
        repo = PostgresAgentExecutionRepository(db)
        stats = await repo.get_execution_stats(
            organization_id=getattr(current_user, "organization_id", None),
            time_range="30d"
        )
        return {
            "total_agents": 0,
            "active_agents": 0,
            "avg_success_rate": stats.get("success_rate", 0.0),
            "avg_error_rate": stats.get("error_rate", 0.0),
            "avg_response_time": stats.get("avg_duration_ms", 0.0) / 1000.0 if stats.get("avg_duration_ms") else 0.0,
            "total_usage": stats.get("total_executions", 0),
            "capability_stats": stats.get("executions_by_agent_type", {})
        }
    return {
        "total_agents": 0,
        "active_agents": 0,
        "avg_success_rate": 0.0,
        "avg_error_rate": 0.0,
        "avg_response_time": 0.0,
        "total_usage": 0,
        "capability_stats": {}
    }
