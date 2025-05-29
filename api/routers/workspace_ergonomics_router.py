from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, File, UploadFile
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from backend.agents.workspace_ergonomics_agent import (
    WorkspaceErgonomicsAgent,
    ErgonomicCategory,
    IssueSeverity
)
from backend.api.dependencies import get_agent_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/workspace-ergonomics",
    tags=["workspace-ergonomics"],
    responses={404: {"description": "Not found"}},
)

# Request/Response Models
class WorkspaceAnalysisRequest(BaseModel):
    """Request model for workspace analysis."""
    image_data: Optional[bytes] = Field(None, description="Workspace image data")
    current_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current workspace conditions"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis context"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Analysis parameters"
    )

class WorkspaceMonitoringRequest(BaseModel):
    """Request model for workspace monitoring."""
    session_id: Optional[str] = Field(None, description="Monitoring session ID")
    image_data: Optional[bytes] = Field(None, description="Workspace image data")
    current_conditions: Dict[str, Any] = Field(
        default_factory=dict,
        description="Current workspace conditions"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Monitoring context"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Monitoring parameters"
    )

class RecommendationRequest(BaseModel):
    """Request model for getting recommendations."""
    user_profile: Dict[str, Any] = Field(..., description="User profile data")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Recommendation context"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Recommendation parameters"
    )

class GuidelineRequest(BaseModel):
    """Request model for getting guidelines."""
    categories: List[str] = Field(
        default_factory=lambda: [cat.value for cat in ErgonomicCategory],
        description="Categories to get guidelines for"
    )

# API Endpoints
@router.post("/analyze")
async def analyze_workspace(
    request: WorkspaceAnalysisRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Analyze workspace conditions from image data."""
    try:
        agent = agent_service.get_agent("workspace_ergonomics")
        if not isinstance(agent, WorkspaceErgonomicsAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "analyze",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing workspace: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-image")
async def analyze_workspace_image(
    file: UploadFile = File(...),
    context: Dict[str, Any] = Field(default_factory=dict),
    parameters: Dict[str, Any] = Field(default_factory=dict),
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Analyze workspace conditions from uploaded image."""
    try:
        # Read image data
        image_data = await file.read()
        
        agent = agent_service.get_agent("workspace_ergonomics")
        if not isinstance(agent, WorkspaceErgonomicsAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "analyze",
            "data": {
                "image_data": image_data,
                "current_conditions": {},
                "context": context,
                "parameters": parameters
            }
        })
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing workspace image: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/monitor")
async def monitor_workspace(
    request: WorkspaceMonitoringRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Monitor workspace conditions in real-time."""
    try:
        agent = agent_service.get_agent("workspace_ergonomics")
        if not isinstance(agent, WorkspaceErgonomicsAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "monitor",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error monitoring workspace: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get-recommendations")
async def get_recommendations(
    request: RecommendationRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Get personalized ergonomic recommendations."""
    try:
        agent = agent_service.get_agent("workspace_ergonomics")
        if not isinstance(agent, WorkspaceErgonomicsAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "get_recommendations",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error getting recommendations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get-guidelines")
async def get_guidelines(
    request: GuidelineRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Get ergonomic guidelines for specific categories."""
    try:
        agent = agent_service.get_agent("workspace_ergonomics")
        if not isinstance(agent, WorkspaceErgonomicsAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "get_guidelines",
            "data": request.dict(),
            "context": {},
            "parameters": {}
        })
        
        return result
    except Exception as e:
        logger.error(f"Error getting guidelines: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ergonomic-categories")
async def get_ergonomic_categories() -> List[str]:
    """Get available ergonomic categories."""
    return [category.value for category in ErgonomicCategory]

@router.get("/issue-severity-levels")
async def get_issue_severity_levels() -> List[str]:
    """Get available issue severity levels."""
    return [severity.value for severity in IssueSeverity] 