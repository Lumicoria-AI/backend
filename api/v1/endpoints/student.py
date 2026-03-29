from typing import Any, List, Optional, Dict, Union
from fastapi import APIRouter, Depends, HTTPException, status, Body
from pydantic import BaseModel, Field
from datetime import datetime
import structlog

from backend.api.deps import get_current_active_user
from backend.agents.student_agent import StudentAgent
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.db.mongodb.repositories.student_repository import student_repository

# Configure logger
logger = structlog.get_logger(__name__)

router = APIRouter()

class StudentContext(BaseModel):
    """Context information about the student."""
    subject: Optional[str] = None
    level: Optional[str] = None
    deadline: Optional[str] = None
    interests: Optional[List[str]] = None
    background: Optional[str] = None
    subjects: Optional[List[str]] = None
    time_available: Optional[str] = None
    learning_style: Optional[str] = None
    focus: Optional[List[str]] = None
    depth: Optional[str] = Field(None, description="Depth of research: 'brief', 'detailed', or 'comprehensive'")

class StudentRequest(BaseModel):
    """Base request model for student agent."""
    content: str = Field(..., description="The content of the student request")
    context: Optional[StudentContext] = Field(None, description="Contextual information about the student")
    request_type: str = Field("general_assistance", 
                             description="Type of student request: assignment_help, study_plan, concept_explanation, research, or general_assistance")
    model: Optional[str] = Field(None, description="AI model to use")
    temperature: Optional[float] = Field(0.7, description="Temperature for model generation")

class StudentFollowUpRequest(BaseModel):
    """Request model for follow-up questions."""
    content: str = Field(..., description="The follow-up question")
    parent_id: str = Field(..., description="ID of the previous interaction")
    model: Optional[str] = Field(None, description="AI model to use")

class StudentResponse(BaseModel):
    """Response model for student agent."""
    response: Dict[str, Any]
    raw_response: Optional[str] = None
    processed_at: str
    model_used: str
    request_type: str
    citations: Optional[List[Dict[str, Any]]] = None

def _make_agent(model: Optional[str] = None) -> StudentAgent:
    """Factory: create a configured StudentAgent.

    When `model` is None the base_agent._resolve_provider() falls through to
    DEFAULT_LLM_PROVIDER (currently Gemini).
    When `model` is e.g. 'sonar-large-online' the provider is auto-inferred
    as Perplexity — no code change required.
    """
    return StudentAgent({
        "type": "student",
        "model_config": {
            # Pass the model name only when explicitly requested.
            # Empty-string or None → system default kicks in.
            "model": model or None,
        },
    })

def _make_student_data(request: StudentRequest) -> Dict[str, Any]:
    """Build the data dict expected by StudentAgent.process_async."""
    return {
        "content": request.content,
        "context": request.context.dict() if request.context else {},
        "request_type": request.request_type,
    }

async def _run_agent(request: StudentRequest, user_id: str, activity_type: str) -> Dict[str, Any]:
    """Shared execution logic for all student endpoints."""
    student_agent = _make_agent(request.model)
    result = await student_agent.process_async(_make_student_data(request))

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"]
        )

    try:
        await log_activity(
            user_id=user_id,
            organization_id=None,  # UserInDB has no organization_id
            activity_type=activity_type,
            details={
                "request_type": request.request_type,
                "content_preview": request.content[:100],
            },
            related_resource_type="AGENT",
            agent_name="Student Agent",
        )
    except Exception as log_err:
        # Don't fail the request if activity logging fails
        logger.warning("Activity logging failed", error=str(log_err), user_id=user_id)

    return result


@router.post("/assignment-help", response_model=StudentResponse)
async def get_assignment_help(
    request: StudentRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get help with academic assignments using Perplexity AI.
    
    This endpoint uses the Perplexity-powered student agent to provide structured
    guidance on approaching assignments, including suggested resources,
    steps, key concepts, and organizational frameworks.
    """
    try:
        # Set request type to assignment_help
        request.request_type = "assignment_help"
        
        # Create student agent
        student_agent_config = {
            "type": "student",
            "model_config": {
                "model": request.model or None  # None → DEFAULT_LLM_PROVIDER (Gemini); 'sonar-large-online' → Perplexity
            },
        }
        
        student_agent = StudentAgent(student_agent_config)
        
        # Process request
        student_data = {
            "content": request.content,
            "context": request.context.dict() if request.context else {},
            "request_type": request.request_type
        }
        
        # Process asynchronously for better performance
        result = await student_agent.process_async(student_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, 'organization_id', None),  # UserInDB has no organization_id
            activity_type="student.assignment_help",
            details={"request_type": "assignment_help", "content_preview": request.content[:100]},
            related_resource_type="AGENT",
            agent_name="Student Agent",
        )

        # Persist interaction
        await student_repository.create_interaction(
            user_id=str(current_user.id),
            request_type=request.request_type,
            content=request.content,
            context=request.context.dict() if request.context else {},
            response=result.get("response", {}),
            raw_response=result.get("raw_response"),
            model_used=result.get("model_used", "unknown"),
            citations=result.get("citations", [])
        )

        return result
    except Exception as e:
        logger.error("Error processing assignment help", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Assignment help processing failed: {str(e)}"
        )

@router.post("/study-plan", response_model=StudentResponse)
async def create_study_plan(
    request: StudentRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Create personalized study plans using Perplexity AI.
    
    This endpoint uses the Perplexity-powered student agent to generate structured
    study plans with schedules, techniques, resources, tracking methods, and
    break strategies tailored to the student's needs.
    """
    try:
        # Set request type to study_plan
        request.request_type = "study_plan"
        
        # Create student agent
        student_agent_config = {
            "type": "student",
            "model_config": {
                "model": request.model or None  # None → DEFAULT_LLM_PROVIDER (Gemini)
            },
        }
        
        student_agent = StudentAgent(student_agent_config)
        
        # Process request
        student_data = {
            "content": request.content,
            "context": request.context.dict() if request.context else {},
            "request_type": request.request_type
        }
        
        # Process asynchronously for better performance
        result = await student_agent.process_async(student_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, 'organization_id', None),  # UserInDB has no organization_id
            activity_type="student.study_plan_created",
            details={"request_type": "study_plan", "content_preview": request.content[:100]},
            related_resource_type="AGENT",
            agent_name="Student Agent",
        )

        # Persist interaction
        await student_repository.create_interaction(
            user_id=str(current_user.id),
            request_type=request.request_type,
            content=request.content,
            context=request.context.dict() if request.context else {},
            response=result.get("response", {}),
            raw_response=result.get("raw_response"),
            model_used=result.get("model_used", "unknown"),
            citations=result.get("citations", [])
        )

        return result
    except Exception as e:
        logger.error("Error creating study plan", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Study plan creation failed: {str(e)}"
        )

@router.post("/explain-concept", response_model=StudentResponse)
async def explain_concept(
    request: StudentRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Explain academic concepts clearly using Perplexity AI.
    
    This endpoint uses the Perplexity-powered student agent to provide
    detailed explanations of concepts with definitions, examples,
    related concepts, misconceptions, and relevant formulas.
    """
    try:
        # Set request type to concept_explanation
        request.request_type = "concept_explanation"
        
        # Create student agent
        student_agent_config = {
            "type": "student",
            "model_config": {
                "model": request.model or None  # None → DEFAULT_LLM_PROVIDER (Gemini)
            },
        }
        
        student_agent = StudentAgent(student_agent_config)
        
        # Process request
        student_data = {
            "content": request.content,
            "context": request.context.dict() if request.context else {},
            "request_type": request.request_type
        }
        
        # Process asynchronously for better performance
        result = await student_agent.process_async(student_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, 'organization_id', None),  # UserInDB has no organization_id
            activity_type="student.concept_explained",
            details={"request_type": "concept_explanation", "content_preview": request.content[:100]},
            related_resource_type="AGENT",
            agent_name="Student Agent",
        )

        # Persist interaction
        await student_repository.create_interaction(
            user_id=str(current_user.id),
            request_type=request.request_type,
            content=request.content,
            context=request.context.dict() if request.context else {},
            response=result.get("response", {}),
            raw_response=result.get("raw_response"),
            model_used=result.get("model_used", "unknown"),
            citations=result.get("citations", [])
        )

        return result
    except Exception as e:
        logger.error("Error explaining concept", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Concept explanation failed: {str(e)}"
        )

@router.post("/research", response_model=StudentResponse)
async def conduct_research(
    request: StudentRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Conduct academic research using Perplexity AI.
    
    This endpoint uses the Perplexity-powered student agent to perform
    comprehensive research with key findings, theories, developments,
    viewpoints, and applications with proper citations.
    """
    try:
        # Set request type to research
        request.request_type = "research"
        
        # Create student agent
        student_agent_config = {
            "type": "student",
            "model_config": {
                "model": request.model or None  # None → DEFAULT_LLM_PROVIDER (Gemini)
            },
        }
        
        student_agent = StudentAgent(student_agent_config)
        
        # Process request
        student_data = {
            "content": request.content,
            "context": request.context.dict() if request.context else {},
            "request_type": request.request_type
        }
        
        # Process asynchronously for better performance
        result = await student_agent.process_async(student_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, 'organization_id', None),  # UserInDB has no organization_id
            activity_type="student.research_conducted",
            details={"request_type": "research", "content_preview": request.content[:100]},
            related_resource_type="AGENT",
            agent_name="Student Agent",
        )

        # Persist interaction
        await student_repository.create_interaction(
            user_id=str(current_user.id),
            request_type=request.request_type,
            content=request.content,
            context=request.context.dict() if request.context else {},
            response=result.get("response", {}),
            raw_response=result.get("raw_response"),
            model_used=result.get("model_used", "unknown"),
            citations=result.get("citations", [])
        )

        return result
    except Exception as e:
        logger.error("Error conducting research", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Research process failed: {str(e)}"
        )

@router.post("/assist", response_model=StudentResponse)
async def provide_general_assistance(
    request: StudentRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Provide general academic assistance using Perplexity AI.
    
    This endpoint uses the Perplexity-powered student agent to deliver
    general academic guidance, recommendations, and resources tailored
    to the student's needs.
    """
    try:
        # Set request type to general_assistance
        request.request_type = "general_assistance"
        
        # Create student agent
        student_agent_config = {
            "type": "student",
            "model_config": {
                "model": request.model or None  # None → DEFAULT_LLM_PROVIDER (Gemini)
            },
        }
        
        student_agent = StudentAgent(student_agent_config)
        
        # Process request
        student_data = {
            "content": request.content,
            "context": request.context.dict() if request.context else {},
            "request_type": request.request_type
        }
        
        # Process asynchronously for better performance
        result = await student_agent.process_async(student_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, 'organization_id', None),  # UserInDB has no organization_id
            activity_type="student.session_started",
            details={"request_type": "general_assistance", "content_preview": request.content[:100]},
            related_resource_type="AGENT",
            agent_name="Student Agent",
        )

        # Persist interaction
        await student_repository.create_interaction(
            user_id=str(current_user.id),
            request_type=request.request_type,
            content=request.content,
            context=request.context.dict() if request.context else {},
            response=result.get("response", {}),
            raw_response=result.get("raw_response"),
            model_used=result.get("model_used", "unknown"),
            citations=result.get("citations", [])
        )

        return result
    except Exception as e:
        logger.error("Error providing general assistance", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"General assistance failed: {str(e)}"
        )

@router.get("/history", response_model=List[Any])
async def get_student_history(
    limit: int = 20,
    skip: int = 0,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get past student agent interactions for the current user."""
    try:
        interactions = await student_repository.get_user_history(
            user_id=str(current_user.id),
            limit=limit,
            skip=skip
        )
        return interactions
    except Exception as e:
        logger.error("Error fetching student history", error=str(e), user_id=str(current_user.id))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch history"
        )

@router.post("/follow-up", response_model=StudentResponse)
async def follow_up_question(
    request: StudentFollowUpRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Ask a follow-up question building on a previous interaction."""
    try:
        # Fetch parent interaction to build history
        parent = await student_repository.get_by_id(request.parent_id)
        if not parent or str(parent.get("user_id")) != str(current_user.id):
            raise HTTPException(status_code=404, detail="Parent interaction not found")

        # Build history from parent
        # For now, we just take the parent's content and response
        # In a more complex system, we could traverse up the tree
        history = [
            {"role": "user", "content": parent.get("content", "")},
            {"role": "assistant", "content": parent.get("raw_response") or str(parent.get("response", ""))}
        ]

        # Create agent
        student_agent = _make_agent(request.model)
        
        # Prepare data
        student_data = {
            "content": request.content,
            "context": parent.get("context", {}),
            "request_type": parent.get("request_type", "general_assistance")
        }

        # Process with history
        result = await student_agent.process_async(student_data, history=history)

        if "error" in result:
            raise HTTPException(status_code=500, detail=result["error"])

        # Persist new interaction
        await student_repository.create_interaction(
            user_id=str(current_user.id),
            request_type=student_data["request_type"],
            content=request.content,
            context=student_data["context"],
            response=result.get("response", {}),
            raw_response=result.get("raw_response"),
            model_used=result.get("model_used", "unknown"),
            citations=result.get("citations", []),
            parent_id=ObjectId(request.parent_id)
        )

        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error in follow-up", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
