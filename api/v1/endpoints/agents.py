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

# Import the AgentService and agent implementations
from agents.agent_service import AgentService
from agents.document_agent import DocumentAgent
# Import configuration loading (assuming it's in backend.core.config)
from core.config import settings
import yaml

router = APIRouter()

# Load configuration and initialize AgentService (simple approach, consider dependency injection for production)
def get_agent_service() -> AgentService:
    config_path = settings.BASE_DIR / "backend" / "config" / "config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return AgentService(config)

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
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service) # Inject AgentService
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
        import logging
        logging.error(f"Error executing agent {agent_id}: {e}")
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
            organization_id=current_user.organization_id,
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
        # Initialize any needed agents
        perplexity_config = {
            "model": "sonar-large-online",
            "model_config": {
                "model": "sonar-large-online",
                "temperature": 0.7,
                "max_tokens": 2048
            }
        }
        
        # Use document agent as it has perplexity client already set up
        document_agent = agent_service.get_agent("document")
        
        if not document_agent or not hasattr(document_agent, "perplexity_client") or not document_agent.perplexity_client:
            document_agent = DocumentAgent(perplexity_config)
        
        # Ensure Perplexity client is initialized
        if not document_agent.perplexity_client:
            document_agent.initialize_models()
            
        if not document_agent.perplexity_client:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Failed to initialize Perplexity client"
            )
            
        # Format features and integrations
        features_str = ""
        if customization_request.features:
            features_str = "Features:\n" + "\n".join([f"- {feature}" for feature in customization_request.features])
            
        integrations_str = ""
        if customization_request.integrations:
            integrations_str = "Integrations:\n" + "\n".join([f"- {integration}" for integration in customization_request.integrations])
            
        # Use Perplexity to generate agent system prompt
        response = await document_agent.perplexity_client.create_agent_prompt(
            agent_type=customization_request.agent_type,
            agent_name=customization_request.agent_name,
            agent_description=customization_request.agent_description,
            capabilities=customization_request.capabilities
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
        # Get document agent for Perplexity access
        document_agent = agent_service.get_agent("document")
        
        if not document_agent or not hasattr(document_agent, "perplexity_client") or not document_agent.perplexity_client:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Perplexity client not available"
            )
        
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
        
        # Use Perplexity to generate the prompt
        model = prompt_request.model or "sonar-large-online"
        messages = [{"role": "user", "content": prompt}]
        
        response = await document_agent.perplexity_client.chat_completion(
            messages=messages,
            model=model
        )
        
        return {
            "generated_prompt": response.content,
            "agent_type": prompt_request.agent_type,
            "model_used": model
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
    time_range: str = Query("7d", regex="^(1d|7d|30d|90d|1y)$"),
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
    time_range: str = Query("7d", regex="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get agent analytics.
    """
    # Placeholder for fetching agent analytics
    analytics_data = {
        "time_range": time_range,
        "total_executions": 1000, # Example data
        "executions_by_agent_type": {
            "document": 500,
            "wellbeing": 200,
            "vision": 150,
            "meeting": 100,
            "creative": 30,
            "student": 20
        },
        "successful_executions": 950,
        "failed_executions": 50
    }
    return analytics_data

@router.get("/summary", response_model=AgentSummaryResponse)
async def get_agent_summary(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get agent summary statistics.
    """
    # Placeholder for fetching agent summary statistics
    summary_data = {
        "total_agents": 6, # Example data
        "active_agents": 5,
        "avg_success_rate": 0.93,
        "avg_error_rate": 0.07,
        "avg_response_time": 0.6,
        "total_usage": 1200,
        "capability_stats": {
            "document_processing": 500,
            "wellbeing_coaching": 200,
            "vision_analysis": 150,
            "meeting_summarization": 100,
            "creative_generation": 30,
            "student_assistance": 20
        }
    }
    return summary_data