from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import structlog

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.agent_universe_repository import agent_universe_repository
from backend.db.mongodb.repositories.permission_repository import permission_repository
from backend.models.user import User
from backend.agents.agent_service import AgentService
from backend.agents.learning_coach_agent import LearningCoachAgent, LearningMode
from backend.services.activity_logger import log_activity
from backend.core.dependencies import get_agent_service as _get_agent_service
from backend.db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)

router = APIRouter()

# Request and Response Models
class LearningRequest(BaseModel):
    """Base model for learning support requests."""
    data: Dict[str, Any] = Field(..., description="Learning data to process")
    mode: str = Field("learning_path", description="Learning support mode")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for learning support")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Mode-specific parameters")
    model: Optional[str] = Field(None, description="AI model to use (defaults to sonar-large-online)")

class LearningResponse(BaseModel):
    """Base model for learning support responses."""
    results: Dict[str, Any]
    metadata: Dict[str, Any]

class LearningPathRequest(LearningRequest):
    """Model for learning path creation requests."""
    goals: List[str] = Field(..., description="Learning goals to achieve")
    current_level: str = Field("beginner", description="Current proficiency level")
    preferences: Optional[Dict[str, Any]] = Field(None, description="Learning preferences")
    constraints: Optional[Dict[str, Any]] = Field(None, description="Time or resource constraints")

class QuizGenerationRequest(LearningRequest):
    """Model for quiz generation requests."""
    topic: str = Field(..., description="Topic to generate quiz for")
    subtopics: Optional[List[str]] = Field(None, description="Specific subtopics to cover")
    previous_performance: Optional[Dict[str, Any]] = Field(None, description="Previous quiz performance")
    learning_style: str = Field("visual", description="Preferred learning style")

class ConceptExplanationRequest(LearningRequest):
    """Model for concept explanation requests."""
    concept: str = Field(..., description="Concept to explain")
    prerequisites: Optional[List[str]] = Field(None, description="Prerequisite concepts")
    learning_style: str = Field("visual", description="Preferred learning style")
    current_understanding: str = Field("basic", description="Current understanding level")

class ProgressTrackingRequest(LearningRequest):
    """Model for progress tracking requests."""
    learning_history: List[Dict[str, Any]] = Field(..., description="Learning activity history")
    assessment_results: Optional[List[Dict[str, Any]]] = Field(None, description="Assessment results")
    goals: List[str] = Field(..., description="Learning goals")
    time_spent: Optional[Dict[str, Any]] = Field(None, description="Time spent on activities")

class ResourceRecommendationRequest(LearningRequest):
    """Model for resource recommendation requests."""
    topics: List[str] = Field(..., description="Topics to find resources for")
    learning_style: str = Field("visual", description="Preferred learning style")
    difficulty_level: str = Field("intermediate", description="Target difficulty level")
    preferred_formats: Optional[List[str]] = Field(None, description="Preferred resource formats")

class AdaptiveLearningRequest(LearningRequest):
    """Model for adaptive learning requests."""
    performance_data: Dict[str, Any] = Field(..., description="User performance data")
    learning_style: str = Field("visual", description="Preferred learning style")
    current_content: Dict[str, Any] = Field(..., description="Current learning content")
    goals: List[str] = Field(..., description="Learning goals")

# ── MongoDB collection ─────────────────────────────────────────────
COACH_COLLECTION = "learning_coach_sessions"


async def _save_learning_session(
    user_id: str,
    mode: str,
    input_data: Dict[str, Any],
    result: Dict[str, Any],
) -> str:
    """Persist learning coach session to MongoDB. Returns the document _id."""
    col = await MongoDB.get_collection(COACH_COLLECTION)
    doc_id = str(uuid.uuid4())

    # Extract raw content from results
    results = result.get("results", {})
    content_keys = [
        "learning_path", "quiz", "explanation", "progress",
        "recommendations", "adaptations", "analysis",
    ]
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


# Helper function to get agent service
def get_agent_service() -> AgentService:
    """Get the global agent service instance (uses pre-initialized agents)."""
    return _get_agent_service()

@router.post("/analyze", response_model=LearningResponse)
async def process_learning_request(
    request: LearningRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Process a learning support request using the Learning Coach Agent.
    
    This endpoint handles various types of learning support tasks including:
    - Learning path creation
    - Quiz generation
    - Concept explanation
    - Progress tracking
    - Resource recommendations
    - Adaptive learning
    """
    try:
        # Check permissions
        org_id = getattr(current_user, "organization_id", None)
        has_permission = await permission_repository.check_permission(
            user_id=current_user.id,
            organization_id=org_id,
            resource_type="AGENT",
            resource_id="learning_coach",
            permission_type="EXECUTE"
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to use the learning coach agent"
            )

        # Process through the global agent service (uses pre-initialized agent)
        result = await agent_service.process_learning_coach_request(
            mode=request.mode,
            data=request.data,
            context=request.context or {},
            parameters=request.parameters or {},
        )

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )

        # Save to MongoDB
        doc_id = await _save_learning_session(
            user_id=str(current_user.id),
            mode=request.mode,
            input_data=request.data,
            result=result,
        )
        result["id"] = doc_id

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="learning.session_started",
            details={"mode": request.mode},
            related_resource_type="AGENT",
            agent_name="Learning Coach Agent",
        )
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error("process_learning_request_failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing learning support request: {str(e)}"
        )

@router.post("/create/learning-path", response_model=LearningResponse)
async def create_learning_path(
    request: LearningPathRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Create a personalized learning path based on user goals and preferences.
    
    This endpoint provides:
    - Structured learning journey
    - Milestones and checkpoints
    - Time estimates
    - Assessment points
    """
    try:
        # Set request type for learning path creation
        request.mode = LearningMode.LEARNING_PATH.value
        
        # Add learning path parameters
        parameters = request.parameters or {}
        parameters.update({
            "goals": request.goals,
            "current_level": request.current_level,
            "preferences": request.preferences,
            "constraints": request.constraints
        })
        
        # Process using the main endpoint
        return await process_learning_request(
            LearningRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating learning path: {str(e)}"
        )

@router.post("/generate/quiz", response_model=LearningResponse)
async def generate_quiz(
    request: QuizGenerationRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Generate quizzes and exercises for knowledge retention.
    
    This endpoint provides:
    - Diverse question types
    - Detailed explanations
    - Difficulty matching
    - Topic coverage
    """
    try:
        # Set request type for quiz generation
        request.mode = LearningMode.QUIZ_GENERATION.value
        
        # Add quiz generation parameters
        parameters = request.parameters or {}
        parameters.update({
            "topic": request.topic,
            "subtopics": request.subtopics,
            "previous_performance": request.previous_performance,
            "learning_style": request.learning_style
        })
        
        # Process using the main endpoint
        return await process_learning_request(
            LearningRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating quiz: {str(e)}"
        )

@router.post("/explain/concept", response_model=LearningResponse)
async def explain_concept(
    request: ConceptExplanationRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Provide explanations of complex concepts.
    
    This endpoint provides:
    - Clear concept breakdowns
    - Practical examples
    - Visual aid recommendations
    - Prerequisite coverage
    """
    try:
        # Set request type for concept explanation
        request.mode = LearningMode.CONCEPT_EXPLANATION.value
        
        # Add concept explanation parameters
        parameters = request.parameters or {}
        parameters.update({
            "concept": request.concept,
            "prerequisites": request.prerequisites,
            "learning_style": request.learning_style,
            "current_understanding": request.current_understanding
        })
        
        # Process using the main endpoint
        return await process_learning_request(
            LearningRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error explaining concept: {str(e)}"
        )

@router.post("/track/progress", response_model=LearningResponse)
async def track_progress(
    request: ProgressTrackingRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Track learning progress and suggest improvements.
    
    This endpoint provides:
    - Progress metrics
    - Achievement tracking
    - Improvement recommendations
    - Goal adjustments
    """
    try:
        # Set request type for progress tracking
        request.mode = LearningMode.PROGRESS_TRACKING.value
        
        # Add progress tracking parameters
        parameters = request.parameters or {}
        parameters.update({
            "learning_history": request.learning_history,
            "assessment_results": request.assessment_results,
            "goals": request.goals,
            "time_spent": request.time_spent
        })
        
        # Process using the main endpoint
        return await process_learning_request(
            LearningRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error tracking progress: {str(e)}"
        )

@router.post("/recommend/resources", response_model=LearningResponse)
async def recommend_resources(
    request: ResourceRecommendationRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Recommend learning resources across different formats.
    
    This endpoint provides:
    - Diverse resource types
    - Difficulty matching
    - Learning style adaptation
    - Resource ratings
    """
    try:
        # Set request type for resource recommendation
        request.mode = LearningMode.RESOURCE_RECOMMENDATION.value
        
        # Add resource recommendation parameters
        parameters = request.parameters or {}
        parameters.update({
            "topics": request.topics,
            "learning_style": request.learning_style,
            "difficulty_level": request.difficulty_level,
            "preferred_formats": request.preferred_formats
        })
        
        # Process using the main endpoint
        return await process_learning_request(
            LearningRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error recommending resources: {str(e)}"
        )

@router.post("/adapt/learning", response_model=LearningResponse)
async def adapt_learning(
    request: AdaptiveLearningRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Adapt learning content based on user performance and preferences.
    
    This endpoint provides:
    - Performance-based adaptations
    - Learning style matching
    - Content difficulty adjustment
    - Personalized feedback
    """
    try:
        # Set request type for adaptive learning
        request.mode = LearningMode.ADAPTIVE_LEARNING.value
        
        # Add adaptive learning parameters
        parameters = request.parameters or {}
        parameters.update({
            "performance_data": request.performance_data,
            "learning_style": request.learning_style,
            "current_content": request.current_content,
            "goals": request.goals
        })
        
        # Process using the main endpoint
        return await process_learning_request(
            LearningRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adapting learning: {str(e)}"
        )

@router.get("/analytics", response_model=Dict[str, Any])
async def get_learning_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get learning analytics and insights.
    
    This endpoint provides analytics about:
    - Learning activity volume
    - Progress metrics
    - Resource usage
    - Performance trends
    """
    try:
        col = await MongoDB.get_collection(COACH_COLLECTION)
        user_id = str(current_user.id)

        # Total sessions
        total_pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": None, "total": {"$sum": 1}}},
        ]
        total_results = await col.aggregate(total_pipeline).to_list(length=1)
        total_sessions = total_results[0]["total"] if total_results else 0

        # Mode breakdown
        mode_pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": "$mode", "count": {"$sum": 1}}},
        ]
        mode_results = await col.aggregate(mode_pipeline).to_list(length=20)
        mode_counts = {r["_id"]: r["count"] for r in mode_results if r["_id"]}

        analytics = {
            "time_range": time_range,
            "total_sessions": total_sessions,
            "mode_counts": mode_counts,
            "learning_activities": {
                "learning_paths": mode_counts.get("learning_path", 0),
                "quizzes": mode_counts.get("quiz_generation", 0),
                "concept_explanations": mode_counts.get("concept_explanation", 0),
                "progress_tracking": mode_counts.get("progress_tracking", 0),
                "resource_recommendations": mode_counts.get("resource_recommendation", 0),
                "adaptive_learning": mode_counts.get("adaptive_learning", 0),
            },
        }
        return analytics

    except Exception as e:
        logger.error("learning_analytics_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch analytics: {str(e)}")


# ── History / Detail / Stats / Delete ──────────────────────────────

@router.get("/history")
async def get_learning_history(
    limit: int = Query(default=20, le=50),
    skip: int = Query(default=0, ge=0),
    mode: Optional[str] = Query(default=None),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get the current user's learning coach history from MongoDB."""
    try:
        col = await MongoDB.get_collection(COACH_COLLECTION)
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
        logger.error("learning_history_fetch_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


@router.get("/history/{session_id}")
async def get_learning_detail(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get full details of a specific learning coach session."""
    try:
        col = await MongoDB.get_collection(COACH_COLLECTION)
        doc = await col.find_one({"_id": session_id, "user_id": str(current_user.id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Session not found")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("learning_detail_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch session: {str(e)}")


@router.get("/stats")
async def get_learning_stats(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get aggregated stats for the current user's learning coach usage."""
    try:
        col = await MongoDB.get_collection(COACH_COLLECTION)
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
        logger.error("learning_stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


@router.delete("/history/{session_id}")
async def delete_learning_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Delete a specific learning coach session from history."""
    try:
        col = await MongoDB.get_collection(COACH_COLLECTION)
        result = await col.delete_one({"_id": session_id, "user_id": str(current_user.id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_learning_session_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")


def _extract_input_summary(input_data: Dict[str, Any], mode: str) -> str:
    """Extract a short summary from the input data for history display."""
    if mode == "learning_path":
        goals = input_data.get("goals", [])
        return ", ".join(goals)[:150] if goals else ""
    elif mode == "quiz_generation":
        return input_data.get("topic", "")[:150]
    elif mode == "concept_explanation":
        return input_data.get("concept", "")[:150]
    elif mode == "progress_tracking":
        goals = input_data.get("goals", [])
        return ", ".join(goals)[:150] if goals else ""
    elif mode == "resource_recommendation":
        topics = input_data.get("topics", [])
        return ", ".join(topics)[:150] if topics else ""
    elif mode == "adaptive_learning":
        goals = input_data.get("goals", [])
        return ", ".join(goals)[:150] if goals else ""
    return str(input_data)[:150]