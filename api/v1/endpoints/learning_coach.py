from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from pydantic import BaseModel, Field
from datetime import datetime

from api.deps import get_current_active_user
from db.mongodb.repositories.agent_universe_repository import agent_universe_repository
from db.mongodb.repositories.permission_repository import permission_repository
from models.user import User
from agents.agent_service import AgentService
from agents.learning_coach_agent import LearningCoachAgent, LearningMode

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

# Helper function to get agent service
def get_agent_service() -> AgentService:
    """Get or create an instance of AgentService."""
    config = {
        "model": "sonar-large-online",
        "model_config": {
            "model": "sonar-large-online",
            "temperature": 0.7,
            "max_tokens": 4096
        }
    }
    return AgentService(config)

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
        has_permission = await permission_repository.check_permission(
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            resource_type="AGENT",
            resource_id="learning_coach",
            permission_type="EXECUTE"
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to use the learning coach agent"
            )

        # Create learning coach agent
        agent_config = {
            "model": request.model or "sonar-large-online",
            "model_config": {
                "model": request.model or "sonar-large-online",
                "temperature": 0.7,
                "max_tokens": 4096
            }
        }
        
        learning_coach_agent = LearningCoachAgent(agent_config)
        
        # Process the request
        result = await learning_coach_agent.process_async({
            "data": request.data,
            "mode": request.mode,
            "context": request.context or {},
            "parameters": request.parameters or {}
        })
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        return result
        
    except Exception as e:
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
    time_range: str = Query("7d", regex="^(1d|7d|30d|90d|1y)$"),
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
    # This would typically fetch from a database
    analytics = {
        "time_range": time_range,
        "total_learning_sessions": 500,
        "learning_activities": {
            "learning_paths": 100,
            "quizzes": 200,
            "concept_explanations": 150,
            "resource_recommendations": 50
        },
        "progress_metrics": {
            "completion_rate": 0.85,
            "average_quiz_score": 0.78,
            "concept_mastery": 0.72,
            "time_spent": 1200  # minutes
        },
        "resource_usage": {
            "videos": 150,
            "articles": 200,
            "exercises": 100,
            "interactive": 50
        },
        "performance_trends": {
            "improvement_rate": 0.15,
            "engagement_score": 0.82,
            "consistency_score": 0.75
        },
        "average_session_duration": 45,  # minutes
        "success_rate": 0.92
    }
    
    return analytics 