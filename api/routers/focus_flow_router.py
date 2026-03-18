from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from backend.agents.focus_flow_agent import FocusFlowAgent, FocusState, DistractionType, ProductivityTechnique
from backend.core.dependencies import get_agent_service

logger = logging.getLogger(__name__)

router = APIRouter(
    responses={404: {"description": "Not found"}},
)

# Request/Response Models
class FocusMonitoringRequest(BaseModel):
    """Request model for focus monitoring."""
    current_state: Dict[str, Any] = Field(..., description="Current state data")
    session_data: Dict[str, Any] = Field(default_factory=dict, description="Session data")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for monitoring")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Monitoring parameters")

class PatternAnalysisRequest(BaseModel):
    """Request model for pattern analysis."""
    session_history: List[Dict[str, Any]] = Field(default_factory=list, description="Session history")
    distraction_history: List[Dict[str, Any]] = Field(default_factory=list, description="Distraction history")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for analysis")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")

class RecommendationRequest(BaseModel):
    """Request model for getting recommendations."""
    user_profile: Dict[str, Any] = Field(..., description="User profile data")
    work_patterns: Dict[str, Any] = Field(..., description="Work pattern data")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for recommendations")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Recommendation parameters")

class DistractionTrackingRequest(BaseModel):
    """Request model for tracking distractions."""
    type: str = Field(..., description="Type of distraction")
    source: str = Field(..., description="Source of distraction")
    duration_seconds: int = Field(default=0, description="Duration in seconds")
    impact_score: float = Field(default=0.0, description="Impact score")
    session_id: Optional[str] = Field(None, description="Associated session ID")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for tracking")

class SessionEndRequest(BaseModel):
    """Request model for ending a session."""
    session_id: str = Field(..., description="Session ID to end")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for session end")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="End session parameters")

# API Endpoints
@router.post("/monitor")
async def monitor_focus(
    request: FocusMonitoringRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Monitor current focus state and session."""
    try:
        agent = agent_service.get_agent("focus_flow")
        if not isinstance(agent, FocusFlowAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "monitor",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error monitoring focus: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/analyze-patterns")
async def analyze_patterns(
    request: PatternAnalysisRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Analyze focus and productivity patterns."""
    try:
        agent = agent_service.get_agent("focus_flow")
        if not isinstance(agent, FocusFlowAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "analyze_patterns",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing patterns: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get-recommendations")
async def get_recommendations(
    request: RecommendationRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Get personalized productivity recommendations."""
    try:
        agent = agent_service.get_agent("focus_flow")
        if not isinstance(agent, FocusFlowAgent):
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

@router.post("/track-distraction")
async def track_distraction(
    request: DistractionTrackingRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Track a distraction event."""
    try:
        agent = agent_service.get_agent("focus_flow")
        if not isinstance(agent, FocusFlowAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "track_distraction",
            "data": request.dict(),
            "context": request.context,
            "parameters": {}
        })
        
        return result
    except Exception as e:
        logger.error(f"Error tracking distraction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/end-session")
async def end_session(
    request: SessionEndRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """End a focus session and generate summary."""
    try:
        agent = agent_service.get_agent("focus_flow")
        if not isinstance(agent, FocusFlowAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "end_session",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error ending session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/focus-states")
async def get_focus_states() -> List[str]:
    """Get available focus states."""
    return [state.value for state in FocusState]

@router.get("/distraction-types")
async def get_distraction_types() -> List[str]:
    """Get available distraction types."""
    return [distraction_type.value for distraction_type in DistractionType]

@router.get("/productivity-techniques")
async def get_productivity_techniques() -> List[str]:
    """Get available productivity techniques."""
    return [technique.value for technique in ProductivityTechnique] 