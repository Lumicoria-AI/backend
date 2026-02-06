from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from datetime import datetime

from backend.api.dependencies import get_current_user, get_agent_service
from backend.agents.agent_service import AgentService
from backend.agents.research_mentor_agent import ResearchMode

router = APIRouter(
    prefix="/research-mentor",
    tags=["research-mentor"],
    dependencies=[Depends(get_current_user)]
)

class ResearchContext(BaseModel):
    """Context for research mentoring requests."""
    research_level: str = Field(
        default="advanced",
        description="Level of research expertise (beginner, intermediate, advanced)"
    )
    field: str = Field(
        default="general",
        description="Field of research or study"
    )
    user_experience: str = Field(
        default="intermediate",
        description="User's experience level (beginner, intermediate, advanced)"
    )

class ProblemAnalysisRequest(BaseModel):
    """Request model for problem analysis."""
    problem: str = Field(..., description="The problem to analyze")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context about the problem"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints or limitations to consider"
    )
    objectives: list[str] = Field(
        default_factory=list,
        description="Specific objectives to achieve"
    )
    research_context: Optional[ResearchContext] = Field(
        default=None,
        description="Context for the research mentoring session"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the analysis"
    )

class ResearchPlanningRequest(BaseModel):
    """Request model for research planning."""
    research_question: str = Field(..., description="The main research question")
    objectives: list[str] = Field(
        default_factory=list,
        description="Specific research objectives"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints or limitations to consider"
    )
    timeline: Dict[str, Any] = Field(
        default_factory=dict,
        description="Timeline requirements or preferences"
    )
    research_context: Optional[ResearchContext] = Field(
        default=None,
        description="Context for the research mentoring session"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the planning"
    )

class LiteratureReviewRequest(BaseModel):
    """Request model for literature review."""
    topic: str = Field(..., description="The topic to review")
    scope: Dict[str, Any] = Field(
        default_factory=dict,
        description="Scope of the literature review"
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Initial sources to consider"
    )
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Specific areas to focus on"
    )
    research_context: Optional[ResearchContext] = Field(
        default=None,
        description="Context for the research mentoring session"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for the review"
    )

class HypothesisDevelopmentRequest(BaseModel):
    """Request model for hypothesis development."""
    research_question: str = Field(..., description="The research question")
    background: Dict[str, Any] = Field(
        default_factory=dict,
        description="Background information"
    )
    variables: list[str] = Field(
        default_factory=list,
        description="Variables to consider"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints or limitations"
    )
    research_context: Optional[ResearchContext] = Field(
        default=None,
        description="Context for the research mentoring session"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for hypothesis development"
    )

class MethodologyGuidanceRequest(BaseModel):
    """Request model for methodology guidance."""
    research_type: str = Field(..., description="Type of research")
    objectives: list[str] = Field(
        default_factory=list,
        description="Research objectives"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints or limitations"
    )
    resources: Dict[str, Any] = Field(
        default_factory=dict,
        description="Available resources"
    )
    research_context: Optional[ResearchContext] = Field(
        default=None,
        description="Context for the research mentoring session"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for methodology guidance"
    )

class CriticalEvaluationRequest(BaseModel):
    """Request model for critical evaluation."""
    research: Dict[str, Any] = Field(..., description="Research to evaluate")
    criteria: list[str] = Field(
        default_factory=list,
        description="Evaluation criteria"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Areas to focus on"
    )
    research_context: Optional[ResearchContext] = Field(
        default=None,
        description="Context for the research mentoring session"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for evaluation"
    )

class SynthesisRequest(BaseModel):
    """Request model for research synthesis."""
    findings: list[Dict[str, Any]] = Field(..., description="Research findings to synthesize")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context"
    )
    objectives: list[str] = Field(
        default_factory=list,
        description="Synthesis objectives"
    )
    constraints: Dict[str, Any] = Field(
        default_factory=dict,
        description="Constraints or limitations"
    )
    research_context: Optional[ResearchContext] = Field(
        default=None,
        description="Context for the research mentoring session"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional parameters for synthesis"
    )

@router.post("/analyze-problem")
async def analyze_problem(
    request: ProblemAnalysisRequest,
    background_tasks: BackgroundTasks,
    agent_service: AgentService = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Analyze and break down complex problems into manageable components."""
    try:
        result = await agent_service.process_research_mentor_request(
            mode=ResearchMode.PROBLEM_ANALYSIS.value,
            data=request.dict(exclude={"research_context", "parameters"}),
            context=request.research_context.dict() if request.research_context else {},
            parameters=request.parameters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/plan-research")
async def plan_research(
    request: ResearchPlanningRequest,
    background_tasks: BackgroundTasks,
    agent_service: AgentService = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Create structured research plans and methodologies."""
    try:
        result = await agent_service.process_research_mentor_request(
            mode=ResearchMode.RESEARCH_PLANNING.value,
            data=request.dict(exclude={"research_context", "parameters"}),
            context=request.research_context.dict() if request.research_context else {},
            parameters=request.parameters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/review-literature")
async def review_literature(
    request: LiteratureReviewRequest,
    background_tasks: BackgroundTasks,
    agent_service: AgentService = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Guide through literature review process with critical analysis."""
    try:
        result = await agent_service.process_research_mentor_request(
            mode=ResearchMode.LITERATURE_REVIEW.value,
            data=request.dict(exclude={"research_context", "parameters"}),
            context=request.research_context.dict() if request.research_context else {},
            parameters=request.parameters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/develop-hypothesis")
async def develop_hypothesis(
    request: HypothesisDevelopmentRequest,
    background_tasks: BackgroundTasks,
    agent_service: AgentService = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Assist in developing and refining research hypotheses."""
    try:
        result = await agent_service.process_research_mentor_request(
            mode=ResearchMode.HYPOTHESIS_DEVELOPMENT.value,
            data=request.dict(exclude={"research_context", "parameters"}),
            context=request.research_context.dict() if request.research_context else {},
            parameters=request.parameters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/guide-methodology")
async def guide_methodology(
    request: MethodologyGuidanceRequest,
    background_tasks: BackgroundTasks,
    agent_service: AgentService = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Provide guidance on research methodology and methods."""
    try:
        result = await agent_service.process_research_mentor_request(
            mode=ResearchMode.METHODOLOGY_GUIDANCE.value,
            data=request.dict(exclude={"research_context", "parameters"}),
            context=request.research_context.dict() if request.research_context else {},
            parameters=request.parameters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate-critically")
async def evaluate_critically(
    request: CriticalEvaluationRequest,
    background_tasks: BackgroundTasks,
    agent_service: AgentService = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Guide critical evaluation of research and evidence."""
    try:
        result = await agent_service.process_research_mentor_request(
            mode=ResearchMode.CRITICAL_EVALUATION.value,
            data=request.dict(exclude={"research_context", "parameters"}),
            context=request.research_context.dict() if request.research_context else {},
            parameters=request.parameters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/synthesize")
async def synthesize(
    request: SynthesisRequest,
    background_tasks: BackgroundTasks,
    agent_service: AgentService = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Assist in synthesizing research findings and insights."""
    try:
        result = await agent_service.process_research_mentor_request(
            mode=ResearchMode.SYNTHESIS.value,
            data=request.dict(exclude={"research_context", "parameters"}),
            context=request.research_context.dict() if request.research_context else {},
            parameters=request.parameters
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 