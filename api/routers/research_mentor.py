from typing import Dict, Any, Optional, List
from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import structlog

from backend.api.deps import get_current_active_user
from backend.core.dependencies import get_agent_service
from backend.agents.agent_service import AgentService
from backend.agents.research_mentor_agent import ResearchMode
from backend.models.user import User
from backend.db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)

router = APIRouter(
    dependencies=[Depends(get_current_active_user)]
)

# ── MongoDB collection ─────────────────────────────────────────────
MENTOR_COLLECTION = "research_mentor_sessions"


async def _save_mentor_session(
    user_id: str,
    mode: str,
    input_data: Dict[str, Any],
    result: Dict[str, Any],
) -> str:
    """Persist research mentor session to MongoDB. Returns the document _id."""
    col = await MongoDB.get_collection(MENTOR_COLLECTION)
    doc_id = str(uuid.uuid4())

    # Extract the raw content from results
    results = result.get("results", {})
    # The content key varies by mode (analysis, plan, review, etc.)
    content_keys = ["analysis", "plan", "review", "hypothesis", "methodology", "evaluation", "synthesis"]
    raw_content = ""
    for key in content_keys:
        val = results.get(key)
        if val:
            raw_content = val.get("content", "") if isinstance(val, dict) else str(val)
            break

    doc = {
        "_id": doc_id,
        "user_id": user_id,
        "mode": mode,
        "input_data": input_data,
        "results": results,
        "raw_content": raw_content,
        "metadata": result.get("metadata", {}),
        "created_at": datetime.utcnow().isoformat(),
    }
    await col.insert_one(doc)
    return doc_id

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
    current_user: User = Depends(get_current_active_user),
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
        doc_id = await _save_mentor_session(
            user_id=str(current_user.id),
            mode=ResearchMode.PROBLEM_ANALYSIS.value,
            input_data=request.dict(exclude={"research_context", "parameters"}),
            result=result,
        )
        result["id"] = doc_id
        return result
    except Exception as e:
        logger.error("analyze_problem_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/plan-research")
async def plan_research(
    request: ResearchPlanningRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
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
        doc_id = await _save_mentor_session(
            user_id=str(current_user.id),
            mode=ResearchMode.RESEARCH_PLANNING.value,
            input_data=request.dict(exclude={"research_context", "parameters"}),
            result=result,
        )
        result["id"] = doc_id
        return result
    except Exception as e:
        logger.error("plan_research_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/review-literature")
async def review_literature(
    request: LiteratureReviewRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
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
        doc_id = await _save_mentor_session(
            user_id=str(current_user.id),
            mode=ResearchMode.LITERATURE_REVIEW.value,
            input_data=request.dict(exclude={"research_context", "parameters"}),
            result=result,
        )
        result["id"] = doc_id
        return result
    except Exception as e:
        logger.error("review_literature_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/develop-hypothesis")
async def develop_hypothesis(
    request: HypothesisDevelopmentRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
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
        doc_id = await _save_mentor_session(
            user_id=str(current_user.id),
            mode=ResearchMode.HYPOTHESIS_DEVELOPMENT.value,
            input_data=request.dict(exclude={"research_context", "parameters"}),
            result=result,
        )
        result["id"] = doc_id
        return result
    except Exception as e:
        logger.error("develop_hypothesis_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/guide-methodology")
async def guide_methodology(
    request: MethodologyGuidanceRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
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
        doc_id = await _save_mentor_session(
            user_id=str(current_user.id),
            mode=ResearchMode.METHODOLOGY_GUIDANCE.value,
            input_data=request.dict(exclude={"research_context", "parameters"}),
            result=result,
        )
        result["id"] = doc_id
        return result
    except Exception as e:
        logger.error("guide_methodology_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/evaluate-critically")
async def evaluate_critically(
    request: CriticalEvaluationRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
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
        doc_id = await _save_mentor_session(
            user_id=str(current_user.id),
            mode=ResearchMode.CRITICAL_EVALUATION.value,
            input_data=request.dict(exclude={"research_context", "parameters"}),
            result=result,
        )
        result["id"] = doc_id
        return result
    except Exception as e:
        logger.error("evaluate_critically_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/synthesize")
async def synthesize(
    request: SynthesisRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user),
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
        doc_id = await _save_mentor_session(
            user_id=str(current_user.id),
            mode=ResearchMode.SYNTHESIS.value,
            input_data=request.dict(exclude={"research_context", "parameters"}),
            result=result,
        )
        result["id"] = doc_id
        return result
    except Exception as e:
        logger.error("synthesize_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ── History / Stats / Delete endpoints ─────────────────────────────

@router.get("/history")
async def get_mentor_history(
    limit: int = Query(default=20, le=50),
    skip: int = Query(default=0, ge=0),
    mode: Optional[str] = Query(default=None),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get the current user's research mentor history from MongoDB."""
    try:
        col = await MongoDB.get_collection(MENTOR_COLLECTION)
        query: Dict[str, Any] = {"user_id": str(current_user.id)}
        if mode:
            query["mode"] = mode

        cursor = col.find(query).sort("created_at", -1).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)

        return [
            {
                "id": doc["_id"],
                "mode": doc.get("mode", ""),
                "input_summary": _extract_input_summary(doc.get("input_data", {}), doc.get("mode", "")),
                "created_at": doc.get("created_at", ""),
            }
            for doc in docs
        ]
    except Exception as e:
        logger.error("mentor_history_fetch_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


@router.get("/history/{session_id}")
async def get_mentor_detail(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get full details of a specific research mentor session."""
    try:
        col = await MongoDB.get_collection(MENTOR_COLLECTION)
        doc = await col.find_one({"_id": session_id, "user_id": str(current_user.id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Session not found")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("mentor_detail_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch session: {str(e)}")


@router.get("/stats")
async def get_mentor_stats(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get aggregated stats for the current user's mentor usage."""
    try:
        col = await MongoDB.get_collection(MENTOR_COLLECTION)
        user_id = str(current_user.id)

        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": None, "total_sessions": {"$sum": 1}}},
        ]
        results = await col.aggregate(pipeline).to_list(length=1)

        type_pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": "$mode", "count": {"$sum": 1}}},
        ]
        type_results = await col.aggregate(type_pipeline).to_list(length=20)
        mode_counts = {r["_id"]: r["count"] for r in type_results if r["_id"]}

        total = results[0].get("total_sessions", 0) if results else 0
        return {
            "total_sessions": total,
            "mode_counts": mode_counts,
        }
    except Exception as e:
        logger.error("mentor_stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


@router.delete("/history/{session_id}")
async def delete_mentor_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Delete a specific research mentor session from history."""
    try:
        col = await MongoDB.get_collection(MENTOR_COLLECTION)
        result = await col.delete_one({"_id": session_id, "user_id": str(current_user.id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_mentor_session_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


def _extract_input_summary(input_data: Dict[str, Any], mode: str) -> str:
    """Extract a short summary from the input data for history display."""
    if mode == "problem_analysis":
        return input_data.get("problem", "")[:150]
    elif mode == "research_planning":
        return input_data.get("research_question", "")[:150]
    elif mode == "literature_review":
        return input_data.get("topic", "")[:150]
    elif mode == "hypothesis_development":
        return input_data.get("research_question", "")[:150]
    elif mode == "methodology_guidance":
        return input_data.get("research_type", "")[:150]
    elif mode == "critical_evaluation":
        research = input_data.get("research", {})
        return str(research)[:150] if research else ""
    elif mode == "synthesis":
        findings = input_data.get("findings", [])
        return str(findings)[:150] if findings else ""
    return ""