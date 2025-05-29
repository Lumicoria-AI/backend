from typing import Any, List, Optional, Dict, Union
from fastapi import APIRouter, Depends, HTTPException, status, Body, UploadFile, File, Form
from pydantic import BaseModel, Field
from datetime import datetime
import structlog
import json

from backend.api.deps import get_current_active_user
from backend.agents.agent_service import AgentService
from backend.agents.meeting_agent import MeetingAgent
from backend.models.user import User
from backend.services.project_manager import project_manager

# Configure logger
logger = structlog.get_logger(__name__)

router = APIRouter()

class MeetingParticipant(BaseModel):
    """Participant in a meeting."""
    name: str
    role: Optional[str] = None
    email: Optional[str] = None

class MeetingContext(BaseModel):
    """Context information about the meeting."""
    project: Optional[str] = Field(None, description="Project related to the meeting")
    previous_meeting: Optional[str] = Field(None, description="Summary of the previous meeting")
    goals: Optional[List[str]] = Field(None, description="Goals of the meeting")
    team: Optional[str] = Field(None, description="Team or department holding the meeting")
    organization: Optional[str] = Field(None, description="Organization information")

class MeetingMetadata(BaseModel):
    """Metadata about the meeting."""
    id: Optional[str] = None
    title: Optional[str] = None
    date: Optional[str] = None
    duration: Optional[str] = None
    type: str = Field("general", description="Type of meeting: status_update, planning, brainstorming, decision_making, problem_solving, review, team_building, client")
    participants: Optional[List[Union[str, MeetingParticipant]]] = None

class MeetingRequest(BaseModel):
    """Base request model for meeting transcript processing."""
    transcript: str = Field(..., description="The transcript or notes from the meeting")
    metadata: MeetingMetadata = Field(..., description="Metadata about the meeting")
    context: Optional[MeetingContext] = Field(None, description="Additional context for the meeting")
    model: Optional[str] = Field(None, description="AI model to use")

class MeetingActionItem(BaseModel):
    """Action item from a meeting."""
    task: str
    assignee: str
    deadline: str

class MeetingResponse(BaseModel):
    """Response model for meeting processing."""
    meeting_id: str
    summary: str
    action_items: List[MeetingActionItem]
    decisions: List[str]
    key_points: List[str]
    follow_ups: List[str]
    processed_at: str
    model_used: str
    questions: Optional[List[str]] = None
    concerns: Optional[List[str]] = None
    raw_response: Optional[str] = None
    citations: Optional[List[Dict[str, Any]]] = None

@router.post("/process", response_model=MeetingResponse)
async def process_meeting_transcript(
    request: MeetingRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Process a meeting transcript to extract key information using Perplexity AI.
    
    This endpoint analyzes meeting transcripts to extract summaries, action items,
    decisions, and key points to provide structured meeting documentation.
    """
    try:
        # Create meeting agent
        meeting_agent_config = {
            "type": "meeting",
            "model_config": {
                "model": request.model or "sonar-large-online"
            },
            "extraction_targets": [
                "action_items", "decisions", "key_points", "follow_ups", 
                "questions", "concerns", "deadlines"
            ]
        }
        
        meeting_agent = MeetingAgent(meeting_agent_config)
        
        # Format participants if they are complex objects
        participants = []
        if request.metadata.participants:
            for participant in request.metadata.participants:
                if isinstance(participant, str):
                    participants.append(participant)
                else:
                    participants.append(participant.name)
        
        # Process meeting data
        meeting_data = {
            "transcript": request.transcript,
            "metadata": {
                "id": request.metadata.id or str(datetime.utcnow().timestamp()),
                "title": request.metadata.title,
                "date": request.metadata.date,
                "duration": request.metadata.duration,
                "type": request.metadata.type,
                "participants": participants
            },
            "context": request.context.dict() if request.context else {}
        }
        
        # Process asynchronously for better performance
        result = await meeting_agent.process_async(meeting_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return result
    except Exception as e:
        logger.error("Error processing meeting transcript", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Meeting processing failed: {str(e)}"
        )

@router.post("/upload", response_model=MeetingResponse)
async def process_uploaded_transcript(
    file: UploadFile = File(...),
    metadata: str = Form(...),
    context: Optional[str] = Form(None),
    model: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Process a meeting transcript file to extract key information using Perplexity AI.
    
    This endpoint accepts uploaded transcript files (TXT, DOC, DOCX, PDF) and
    extracts structured meeting information including summaries and action items.
    """
    try:
        # Read file content
        file_content = await file.read()
        transcript = file_content.decode("utf-8")
        
        # Parse metadata and context
        try:
            metadata_dict = json.loads(metadata)
            parsed_metadata = MeetingMetadata(**metadata_dict)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid metadata JSON format"
            )
            
        context_dict = {}
        if context:
            try:
                context_dict = json.loads(context)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid context JSON format"
                )
        
        # Create meeting agent
        meeting_agent_config = {
            "type": "meeting",
            "model_config": {
                "model": model or "sonar-large-online"
            }
        }
        
        meeting_agent = MeetingAgent(meeting_agent_config)
        
        # Format participants if they are complex objects
        participants = []
        if parsed_metadata.participants:
            for participant in parsed_metadata.participants:
                if isinstance(participant, str):
                    participants.append(participant)
                else:
                    participants.append(participant.name)
        
        # Process meeting data
        meeting_data = {
            "transcript": transcript,
            "metadata": {
                "id": parsed_metadata.id or str(datetime.utcnow().timestamp()),
                "title": parsed_metadata.title,
                "date": parsed_metadata.date,
                "duration": parsed_metadata.duration,
                "type": parsed_metadata.type,
                "participants": participants
            },
            "context": context_dict
        }
        
        # Process asynchronously for better performance
        result = await meeting_agent.process_async(meeting_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return result
    except Exception as e:
        logger.error("Error processing uploaded transcript", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Meeting processing failed: {str(e)}"
        )

@router.post("/status-update", response_model=MeetingResponse)
async def process_status_meeting(
    request: MeetingRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Process a status update meeting transcript using Perplexity AI.
    
    This specialized endpoint focuses on extracting progress updates, blockers,
    achievements, and next steps from status update meetings.
    """
    try:
        # Set meeting type to status_update
        request.metadata.type = "status_update"
        
        # Create meeting agent
        meeting_agent_config = {
            "type": "meeting",
            "model_config": {
                "model": request.model or "sonar-large-online"
            }
        }
        
        meeting_agent = MeetingAgent(meeting_agent_config)
        
        # Format participants if they are complex objects
        participants = []
        if request.metadata.participants:
            for participant in request.metadata.participants:
                if isinstance(participant, str):
                    participants.append(participant)
                else:
                    participants.append(participant.name)
        
        # Process meeting data
        meeting_data = {
            "transcript": request.transcript,
            "metadata": {
                "id": request.metadata.id or str(datetime.utcnow().timestamp()),
                "title": request.metadata.title,
                "date": request.metadata.date,
                "duration": request.metadata.duration,
                "type": request.metadata.type,
                "participants": participants
            },
            "context": request.context.dict() if request.context else {}
        }
        
        # Process asynchronously for better performance
        result = await meeting_agent.process_async(meeting_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return result
    except Exception as e:
        logger.error("Error processing status update meeting", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Status meeting processing failed: {str(e)}"
        )

@router.post("/planning", response_model=MeetingResponse)
async def process_planning_meeting(
    request: MeetingRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Process a planning meeting transcript using Perplexity AI.
    
    This specialized endpoint focuses on extracting goals, strategies, timelines,
    resource allocations, and risk identification from planning meetings.
    """
    try:
        # Set meeting type to planning
        request.metadata.type = "planning"
        
        # Create meeting agent
        meeting_agent_config = {
            "type": "meeting",
            "model_config": {
                "model": request.model or "sonar-large-online"
            }
        }
        
        meeting_agent = MeetingAgent(meeting_agent_config)
        
        # Format participants if they are complex objects
        participants = []
        if request.metadata.participants:
            for participant in request.metadata.participants:
                if isinstance(participant, str):
                    participants.append(participant)
                else:
                    participants.append(participant.name)
        
        # Process meeting data
        meeting_data = {
            "transcript": request.transcript,
            "metadata": {
                "id": request.metadata.id or str(datetime.utcnow().timestamp()),
                "title": request.metadata.title,
                "date": request.metadata.date,
                "duration": request.metadata.duration,
                "type": request.metadata.type,
                "participants": participants
            },
            "context": request.context.dict() if request.context else {}
        }
        
        # Process asynchronously for better performance
        result = await meeting_agent.process_async(meeting_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return result
    except Exception as e:
        logger.error("Error processing planning meeting", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Planning meeting processing failed: {str(e)}"
        )

@router.post("/decision", response_model=MeetingResponse)
async def process_decision_meeting(
    request: MeetingRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Process a decision-making meeting transcript using Perplexity AI.
    
    This specialized endpoint focuses on extracting options presented, criteria discussed,
    decisions made, and their rationales from decision-making meetings.
    """
    try:
        # Set meeting type to decision_making
        request.metadata.type = "decision_making"
        
        # Create meeting agent
        meeting_agent_config = {
            "type": "meeting",
            "model_config": {
                "model": request.model or "sonar-large-online"
            }
        }
        
        meeting_agent = MeetingAgent(meeting_agent_config)
        
        # Format participants if they are complex objects
        participants = []
        if request.metadata.participants:
            for participant in request.metadata.participants:
                if isinstance(participant, str):
                    participants.append(participant)
                else:
                    participants.append(participant.name)
        
        # Process meeting data
        meeting_data = {
            "transcript": request.transcript,
            "metadata": {
                "id": request.metadata.id or str(datetime.utcnow().timestamp()),
                "title": request.metadata.title,
                "date": request.metadata.date,
                "duration": request.metadata.duration,
                "type": request.metadata.type,
                "participants": participants
            },
            "context": request.context.dict() if request.context else {}
        }
        
        # Process asynchronously for better performance
        result = await meeting_agent.process_async(meeting_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return result
    except Exception as e:
        logger.error("Error processing decision-making meeting", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Decision meeting processing failed: {str(e)}"
        )

@router.post("/client", response_model=MeetingResponse)
async def process_client_meeting(
    request: MeetingRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Process a client meeting transcript using Perplexity AI.
    
    This specialized endpoint focuses on extracting client needs, feedback, concerns,
    agreements, and relationship development steps from client meetings.
    """
    try:
        # Set meeting type to client
        request.metadata.type = "client"
        
        # Create meeting agent
        meeting_agent_config = {
            "type": "meeting",
            "model_config": {
                "model": request.model or "sonar-large-online"
            }
        }
        
        meeting_agent = MeetingAgent(meeting_agent_config)
        
        # Format participants if they are complex objects
        participants = []
        if request.metadata.participants:
            for participant in request.metadata.participants:
                if isinstance(participant, str):
                    participants.append(participant)
                else:
                    participants.append(participant.name)
        
        # Process meeting data
        meeting_data = {
            "transcript": request.transcript,
            "metadata": {
                "id": request.metadata.id or str(datetime.utcnow().timestamp()),
                "title": request.metadata.title,
                "date": request.metadata.date,
                "duration": request.metadata.duration,
                "type": request.metadata.type,
                "participants": participants
            },
            "context": request.context.dict() if request.context else {}
        }
        
        # Process asynchronously for better performance
        result = await meeting_agent.process_async(meeting_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return result
    except Exception as e:
        logger.error("Error processing client meeting", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Client meeting processing failed: {str(e)}"
        )

@router.post("/brainstorming", response_model=MeetingResponse)
async def process_brainstorming_meeting(
    request: MeetingRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Process a brainstorming meeting transcript using Perplexity AI.
    
    This specialized endpoint focuses on extracting ideas generated, concept evaluations,
    and next steps for idea development from brainstorming sessions.
    """
    try:
        # Set meeting type to brainstorming
        request.metadata.type = "brainstorming"
        
        # Create meeting agent
        meeting_agent_config = {
            "type": "meeting",
            "model_config": {
                "model": request.model or "sonar-large-online"
            }
        }
        
        meeting_agent = MeetingAgent(meeting_agent_config)
        
        # Format participants if they are complex objects
        participants = []
        if request.metadata.participants:
            for participant in request.metadata.participants:
                if isinstance(participant, str):
                    participants.append(participant)
                else:
                    participants.append(participant.name)
        
        # Process meeting data
        meeting_data = {
            "transcript": request.transcript,
            "metadata": {
                "id": request.metadata.id or str(datetime.utcnow().timestamp()),
                "title": request.metadata.title,
                "date": request.metadata.date,
                "duration": request.metadata.duration,
                "type": request.metadata.type,
                "participants": participants
            },
            "context": request.context.dict() if request.context else {}
        }
        
        # Process asynchronously for better performance
        result = await meeting_agent.process_async(meeting_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return result
    except Exception as e:
        logger.error("Error processing brainstorming meeting", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Brainstorming meeting processing failed: {str(e)}"
        )

@router.post("/problem-solving", response_model=MeetingResponse)
async def process_problem_solving_meeting(
    request: MeetingRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Process a problem-solving meeting transcript using Perplexity AI.
    
    This specialized endpoint focuses on extracting problems discussed, root causes identified,
    solutions proposed, and implementation plans from problem-solving meetings.
    """
    try:
        # Set meeting type to problem_solving
        request.metadata.type = "problem_solving"
        
        # Create meeting agent
        meeting_agent_config = {
            "type": "meeting",
            "model_config": {
                "model": request.model or "sonar-large-online"
            }
        }
        
        meeting_agent = MeetingAgent(meeting_agent_config)
        
        # Format participants if they are complex objects
        participants = []
        if request.metadata.participants:
            for participant in request.metadata.participants:
                if isinstance(participant, str):
                    participants.append(participant)
                else:
                    participants.append(participant.name)
        
        # Process meeting data
        meeting_data = {
            "transcript": request.transcript,
            "metadata": {
                "id": request.metadata.id or str(datetime.utcnow().timestamp()),
                "title": request.metadata.title,
                "date": request.metadata.date,
                "duration": request.metadata.duration,
                "type": request.metadata.type,
                "participants": participants
            },
            "context": request.context.dict() if request.context else {}
        }
        
        # Process asynchronously for better performance
        result = await meeting_agent.process_async(meeting_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return result
    except Exception as e:
        logger.error("Error processing problem-solving meeting", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Problem-solving meeting processing failed: {str(e)}"
        )

@router.post("/export-to-notion", response_model=MeetingResponse)
async def export_meeting_to_notion(
    request: MeetingRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Process a meeting transcript and export it to Notion.
    
    This endpoint processes a meeting transcript using the MeetingAgent and then
    exports the structured meeting data to Notion for documentation and tracking.
    """
    try:
        # First, process the meeting transcript
        meeting_response = await process_meeting_transcript(request, current_user, agent_service)
        
        # Now, import the project manager to export the meeting to Notion
        from services.project_manager import project_manager
        
        # Export to Notion
        export_result = await project_manager.export_meeting_to_project(
            organization_id=str(current_user.organization_id),
            meeting_data=meeting_response,
            integration_type="notion"
        )
        
        # Add Notion page URL to the meeting response
        if export_result.get("status") == "success" and "page_data" in export_result:
            meeting_response["notion_url"] = export_result["page_data"].get("url", "")
            meeting_response["notion_export_status"] = "success"
        else:
            meeting_response["notion_export_status"] = "failed"
            meeting_response["notion_export_error"] = export_result.get("message", "Unknown error")
        
        return meeting_response
    except Exception as e:
        logger.error("Error processing and exporting meeting", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Meeting processing and export failed: {str(e)}"
        )

@router.post("/export-to-google-workspace", response_model=MeetingResponse)
async def export_meeting_to_google_workspace(
    request: MeetingRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Process a meeting transcript and export it to Google Workspace.
    
    This endpoint processes a meeting transcript using the MeetingAgent and then
    exports the structured meeting data to Google Docs for documentation and tracking.
    """
    try:
        # First, process the meeting transcript
        meeting_response = await process_meeting_transcript(request, current_user, agent_service)
        
        # Now, import the project manager to export the meeting to Google Workspace
        from services.project_manager import project_manager
        
        # Export to Google Workspace
        export_result = await project_manager.export_meeting_to_project(
            organization_id=str(current_user.organization_id),
            meeting_data=meeting_response,
            integration_type="google_workspace"
        )
        
        # Add Google Doc URL to the meeting response
        if export_result.get("status") == "success" and "page_data" in export_result:
            meeting_response["google_doc_url"] = export_result["page_data"].get("url", "")
            meeting_response["google_doc_id"] = export_result["page_data"].get("id", "")
            meeting_response["google_export_status"] = "success"
        else:
            meeting_response["google_export_status"] = "failed"
            meeting_response["google_export_error"] = export_result.get("message", "Unknown error")
        
        return meeting_response
    except Exception as e:
        logger.error("Error processing and exporting meeting to Google Workspace", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Meeting processing and export to Google Workspace failed: {str(e)}"
        )
