from typing import Any, List, Optional, Dict, Union
from fastapi import APIRouter, Depends, HTTPException, status, Body
from pydantic import BaseModel, Field
from datetime import datetime
import structlog

from api.deps import get_current_active_user
from agents.agent_service import AgentService
from agents.student_agent import StudentAgent
from models.user import User

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

class StudentResponse(BaseModel):
    """Response model for student agent."""
    response: Dict[str, Any]
    raw_response: Optional[str] = None
    processed_at: str
    model_used: str
    request_type: str
    citations: Optional[List[Dict[str, Any]]] = None

@router.post("/assignment-help", response_model=StudentResponse)
async def get_assignment_help(
    request: StudentRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
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
                "model": request.model or "sonar-large-online"
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
                "model": request.model or "sonar-large-online"
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
                "model": request.model or "sonar-large-online"
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
                "model": request.model or "sonar-large-online"
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
                "model": request.model or "sonar-large-online"
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
        
        return result
    except Exception as e:
        logger.error("Error providing general assistance", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"General assistance failed: {str(e)}"
        )
