from typing import Any, List, Optional, Dict, Union
from fastapi import APIRouter, Depends, HTTPException, status, Body
from pydantic import BaseModel, Field
from datetime import datetime
import structlog

from backend.api.deps import get_current_active_user
from backend.agents.agent_service import AgentService
from backend.agents.research_agent import ResearchAgent
from backend.models.user import User
from backend.services.activity_logger import log_activity

# Configure logger
logger = structlog.get_logger(__name__)

router = APIRouter()

class ResearchContext(BaseModel):
    """Context information for research queries."""
    domain: Optional[str] = Field(None, description="Domain or field of knowledge (e.g., medicine, computer science)")
    purpose: Optional[str] = Field(None, description="Purpose of the research (e.g., academic paper, decision making)")
    background_knowledge: Optional[str] = Field(None, description="User's background knowledge level on the topic")
    time_scope: Optional[str] = Field(None, description="Relevant time period (e.g., recent 5 years, historical)")
    geographic_scope: Optional[str] = Field(None, description="Geographic focus if applicable")
    previous_findings: Optional[str] = Field(None, description="Previous research findings on this topic")

class ResearchRequest(BaseModel):
    """Base request model for research agent."""
    query: str = Field(..., description="The research query or topic to investigate")
    context: Optional[ResearchContext] = Field(None, description="Additional context for the research")
    research_type: str = Field("general", description="Type of research: topic_research, literature_review, fact_checking, source_comparison, question_answering, citation_analysis")
    depth: str = Field("comprehensive", description="Depth of research: brief, focused, comprehensive, or deep")
    focus_areas: Optional[List[str]] = Field(None, description="Specific aspects or subtopics to focus on")
    model: Optional[str] = Field(None, description="AI model to use for research")
    max_sources: Optional[int] = Field(None, description="Maximum number of sources to include")

class ResearchResponse(BaseModel):
    """Response model for research agent."""
    findings: Dict[str, Any]
    raw_response: Optional[str] = None
    processed_at: str
    model_used: str
    research_type: str
    query: str
    sub_questions: Optional[List[str]] = None
    citations: Optional[List[Dict[str, Any]]] = None

@router.post("/query", response_model=ResearchResponse)
async def research_query(
    request: ResearchRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Conduct comprehensive research on a specific query using Perplexity AI.
    
    This endpoint leverages Perplexity's powerful research capabilities to provide
    in-depth information on topics with proper citations and organized findings.
    """
    try:
        # Create research agent
        research_agent_config = {
            "type": "research",
            "model_config": {
                "model": request.model or "sonar-large-online"
            },
            "research_depth": request.depth,
            "require_citations": True
        }
        
        research_agent = ResearchAgent(research_agent_config)
        
        # Process research request
        research_data = {
            "query": request.query,
            "context": request.context.dict() if request.context else {},
            "research_type": request.research_type,
            "depth": request.depth,
            "focus_areas": request.focus_areas or []
        }
        
        # Process asynchronously for better performance
        result = await research_agent.process_async(research_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        await log_activity(
            user_id=str(current_user.id),
            organization_id=current_user.organization_id,
            activity_type="research.query",
            details={"query": request.query, "depth": request.depth, "research_type": request.research_type, "model": result.get("model_used", "")},
            related_resource_type="AGENT",
            agent_name="Research Agent",
        )

        return result
    except Exception as e:
        logger.error("Error processing research query", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research processing failed: {str(e)}"
        )

@router.post("/topic", response_model=ResearchResponse)
async def research_topic(
    request: ResearchRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Conduct topic-based research using Perplexity AI.
    
    This endpoint specializes in providing comprehensive overviews of topics,
    including key concepts, historical context, current developments, and perspectives.
    """
    try:
        # Set research type to topic_research
        request.research_type = "topic_research"
        
        # Create research agent
        research_agent_config = {
            "type": "research",
            "model_config": {
                "model": request.model or "sonar-large-online"
            },
            "research_depth": request.depth,
            "require_citations": True
        }
        
        research_agent = ResearchAgent(research_agent_config)
        
        # Process research request
        research_data = {
            "query": request.query,
            "context": request.context.dict() if request.context else {},
            "research_type": request.research_type,
            "depth": request.depth,
            "focus_areas": request.focus_areas or []
        }
        
        # Process asynchronously for better performance
        result = await research_agent.process_async(research_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        await log_activity(
            user_id=str(current_user.id),
            organization_id=current_user.organization_id,
            activity_type="research.topic",
            details={"query": request.query, "depth": request.depth},
            related_resource_type="AGENT",
            agent_name="Research Agent",
        )

        return result
    except Exception as e:
        logger.error("Error researching topic", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Topic research failed: {str(e)}"
        )

@router.post("/literature-review", response_model=ResearchResponse)
async def conduct_literature_review(
    request: ResearchRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Conduct a literature review on a specific topic using Perplexity AI.
    
    This endpoint specializes in synthesizing scholarly research, identifying key
    publications, researchers, theories, methodologies, trends, and gaps.
    """
    try:
        # Set research type to literature_review
        request.research_type = "literature_review"
        
        # Create research agent
        research_agent_config = {
            "type": "research",
            "model_config": {
                "model": request.model or "sonar-large-online"
            },
            "research_depth": request.depth,
            "require_citations": True
        }
        
        research_agent = ResearchAgent(research_agent_config)
        
        # Process research request
        research_data = {
            "query": request.query,
            "context": request.context.dict() if request.context else {},
            "research_type": request.research_type,
            "depth": request.depth,
            "focus_areas": request.focus_areas or []
        }
        
        # Process asynchronously for better performance
        result = await research_agent.process_async(research_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        await log_activity(
            user_id=str(current_user.id),
            organization_id=current_user.organization_id,
            activity_type="research.literature_review",
            details={"query": request.query, "depth": request.depth},
            related_resource_type="AGENT",
            agent_name="Research Agent",
        )

        return result
    except Exception as e:
        logger.error("Error conducting literature review", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Literature review failed: {str(e)}"
        )

@router.post("/fact-check", response_model=ResearchResponse)
async def fact_check(
    request: ResearchRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Fact-check statements or claims using Perplexity AI.
    
    This endpoint specializes in verifying the accuracy of information,
    providing evidence from reliable sources, and assessing overall veracity.
    """
    try:
        # Set research type to fact_checking
        request.research_type = "fact_checking"
        
        # Create research agent
        research_agent_config = {
            "type": "research",
            "model_config": {
                "model": request.model or "sonar-large-online"
            },
            "research_depth": request.depth,
            "require_citations": True
        }
        
        research_agent = ResearchAgent(research_agent_config)
        
        # Process research request
        research_data = {
            "query": request.query,
            "context": request.context.dict() if request.context else {},
            "research_type": request.research_type,
            "depth": request.depth,
            "focus_areas": request.focus_areas or []
        }
        
        # Process asynchronously for better performance
        result = await research_agent.process_async(research_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        await log_activity(
            user_id=str(current_user.id),
            organization_id=current_user.organization_id,
            activity_type="research.fact_check",
            details={"query": request.query, "depth": request.depth},
            related_resource_type="AGENT",
            agent_name="Research Agent",
        )

        return result
    except Exception as e:
        logger.error("Error fact-checking", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Fact-checking failed: {str(e)}"
        )

@router.post("/compare-sources", response_model=ResearchResponse)
async def compare_sources(
    request: ResearchRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Compare information from multiple sources on a topic using Perplexity AI.
    
    This endpoint specializes in analyzing consensus, conflicts, biases,
    and synthesizing information from diverse sources.
    """
    try:
        # Set research type to source_comparison
        request.research_type = "source_comparison"
        
        # Create research agent
        research_agent_config = {
            "type": "research",
            "model_config": {
                "model": request.model or "sonar-large-online"
            },
            "research_depth": request.depth,
            "require_citations": True
        }
        
        research_agent = ResearchAgent(research_agent_config)
        
        # Process research request
        research_data = {
            "query": request.query,
            "context": request.context.dict() if request.context else {},
            "research_type": request.research_type,
            "depth": request.depth,
            "focus_areas": request.focus_areas or []
        }
        
        # Process asynchronously for better performance
        result = await research_agent.process_async(research_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        await log_activity(
            user_id=str(current_user.id),
            organization_id=current_user.organization_id,
            activity_type="research.compare_sources",
            details={"query": request.query, "depth": request.depth},
            related_resource_type="AGENT",
            agent_name="Research Agent",
        )

        return result
    except Exception as e:
        logger.error("Error comparing sources", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Source comparison failed: {str(e)}"
        )

@router.post("/comprehensive", response_model=ResearchResponse)
async def comprehensive_research(
    request: ResearchRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Conduct deep, comprehensive research on a topic using Perplexity AI.
    
    This endpoint performs multi-stage research that breaks down complex topics
    into sub-questions, researches each thoroughly, and synthesizes findings
    into a comprehensive report with citations.
    """
    try:
        # Set research type and depth for comprehensive analysis
        request.research_type = "comprehensive"
        request.depth = "deep"
        
        # Create research agent
        research_agent_config = {
            "type": "research",
            "model_config": {
                "model": request.model or "sonar-large-online"
            },
            "research_depth": "deep",
            "require_citations": True
        }
        
        research_agent = ResearchAgent(research_agent_config)
        
        # Process research request
        research_data = {
            "query": request.query,
            "context": request.context.dict() if request.context else {},
            "research_type": request.research_type,
            "depth": request.depth,
            "focus_areas": request.focus_areas or []
        }
        
        # Process asynchronously for better performance
        result = await research_agent.process_async(research_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        await log_activity(
            user_id=str(current_user.id),
            organization_id=current_user.organization_id,
            activity_type="research.comprehensive",
            details={"query": request.query, "depth": "deep"},
            related_resource_type="AGENT",
            agent_name="Research Agent",
        )

        return result
    except Exception as e:
        logger.error("Error conducting comprehensive research", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Comprehensive research failed: {str(e)}"
        )
