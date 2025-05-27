from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from agents.ethics_bias_agent import EthicsBiasAgent, EthicsCategory, BiasType, IssueSeverity
from api.dependencies import get_agent_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ethics-bias",
    tags=["ethics-bias"],
    responses={404: {"description": "Not found"}},
)

# Request/Response Models
class ContentAnalysisRequest(BaseModel):
    """Request model for content analysis."""
    content: str = Field(..., description="Content to analyze")
    content_type: str = Field(default="text", description="Type of content (text, document, code, etc.)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for analysis")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Analysis parameters")

class GuidelineCheckRequest(BaseModel):
    """Request model for guideline checking."""
    content: str = Field(..., description="Content to check")
    guidelines_focus: List[str] = Field(default_factory=list, description="Specific guidelines to check against")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for checking")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Checking parameters")

class SuggestionRequest(BaseModel):
    """Request model for generating suggestions."""
    issues: List[Dict[str, Any]] = Field(..., description="Issues to generate suggestions for")
    content: str = Field(..., description="Original content")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for suggestions")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Suggestion parameters")

class CitationRequest(BaseModel):
    """Request model for getting citations."""
    topic: str = Field(..., description="Topic to get citations for")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for citations")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Citation parameters")

# API Endpoints
@router.post("/analyze")
async def analyze_content(
    request: ContentAnalysisRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Analyze content for ethical issues and bias."""
    try:
        agent = agent_service.get_agent("ethics_bias")
        if not isinstance(agent, EthicsBiasAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "analyze",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error analyzing content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/check-guidelines")
async def check_guidelines(
    request: GuidelineCheckRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Check content against ethical guidelines."""
    try:
        agent = agent_service.get_agent("ethics_bias")
        if not isinstance(agent, EthicsBiasAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "check_guidelines",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error checking guidelines: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/generate-suggestions")
async def generate_suggestions(
    request: SuggestionRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Generate suggestions for addressing ethical issues and bias."""
    try:
        agent = agent_service.get_agent("ethics_bias")
        if not isinstance(agent, EthicsBiasAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "generate_suggestions",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error generating suggestions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get-citations")
async def get_citations(
    request: CitationRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Get relevant citations for ethical guidelines and best practices."""
    try:
        agent = agent_service.get_agent("ethics_bias")
        if not isinstance(agent, EthicsBiasAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "get_citations",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error getting citations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/ethics-categories")
async def get_ethics_categories() -> List[str]:
    """Get available ethics categories."""
    return [category.value for category in EthicsCategory]

@router.get("/bias-types")
async def get_bias_types() -> List[str]:
    """Get available bias types."""
    return [bias_type.value for bias_type in BiasType]

@router.get("/severity-levels")
async def get_severity_levels() -> List[str]:
    """Get available severity levels."""
    return [severity.value for severity in IssueSeverity] 