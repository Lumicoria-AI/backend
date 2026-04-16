from typing import Any, List, Optional, Dict, Union
from fastapi import APIRouter, Depends, HTTPException, status, Body, UploadFile, File, Form
from pydantic import BaseModel, Field
from datetime import datetime
import structlog
import json
import tempfile
import os

from backend.api.deps import get_current_active_user
from backend.agents.agent_service import AgentService
from backend.agents.meeting_agent import MeetingAgent
from backend.models.user import User
from backend.services.project_manager import project_manager
from backend.services.activity_logger import log_activity
from backend.services.stt_service import stt_service
from backend.core.config import settings
from backend.db.postgres import get_async_sessionmaker
from backend.db.postgres_models import MeetingSQL, MeetingDraftSQL
from sqlalchemy import select, update as sa_update

# Configure logger
logger = structlog.get_logger(__name__)

async def _save_meeting_to_db(
    user_id: str,
    transcript: str,
    result: Dict[str, Any],
    metadata: Dict[str, Any],
    context: Dict[str, Any],
    source: str = "manual",
) -> Optional[str]:
    """Save a processed meeting to Postgres. Returns the meeting row ID or None."""
    try:
        session_factory = get_async_sessionmaker()
        async with session_factory() as session:
            meeting = MeetingSQL(
                user_id=user_id,
                title=metadata.get("title") or (result.get("summary", "")[:80] + "..."),
                meeting_type=metadata.get("type", "general"),
                transcript=transcript,
                summary=result.get("summary", ""),
                raw_response=result.get("raw_response"),
                model_used=result.get("model_used"),
                action_items=result.get("action_items", []),
                decisions=result.get("decisions", []),
                key_points=result.get("key_points", []),
                follow_ups=result.get("follow_ups", []),
                questions=result.get("questions", []),
                concerns=result.get("concerns", []),
                meeting_date=metadata.get("date"),
                duration=metadata.get("duration"),
                participants=metadata.get("participants", []),
                context=context,
                source=source,
                processed_at=datetime.utcnow(),
            )
            session.add(meeting)
            await session.commit()
            await session.refresh(meeting)
            logger.info("meeting_saved_to_db", meeting_id=meeting.id, user_id=user_id)
            return meeting.id
    except Exception as e:
        logger.warning("meeting_save_failed", error=str(e), user_id=user_id)
        return None


def _resolve_model(model: Optional[str] = None) -> str:
    """Return the explicit model or fall back to the default from env."""
    if model:
        return model
    provider = settings.DEFAULT_LLM_PROVIDER
    defaults = {
        "gemini": settings.GEMINI_MODEL,
        "openai": settings.OPENAI_MODEL,
        "anthropic": settings.ANTHROPIC_MODEL,
        "mistral": settings.MISTRAL_MODEL,
        "perplexity": "sonar-large-online",
    }
    return defaults.get(provider, settings.GEMINI_MODEL)

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
            "agent_model_config": {
                "model": _resolve_model(request.model)
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

        # Auto-save to Postgres
        saved_id = await _save_meeting_to_db(
            user_id=str(current_user.id),
            transcript=request.transcript,
            result=result,
            metadata=meeting_data["metadata"],
            context=meeting_data.get("context", {}),
            source="manual",
        )
        if saved_id:
            result["saved_id"] = saved_id

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="meeting.summarized",
            details={"meeting_type": request.metadata.type, "title": request.metadata.title},
            related_resource_type="AGENT",
            agent_name="Meeting Agent",
        )
        return result
    except HTTPException:
        raise
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
    tmp_audio_path = None
    try:
        # Read file content
        file_content = await file.read()
        filename = file.filename or ""

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

        # Determine if file is audio -> transcribe with STT, else decode as text
        if stt_service.is_audio_file(filename):
            # Write to temp file for faster-whisper
            suffix = os.path.splitext(filename)[1] or ".webm"
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                tmp.write(file_content)
                tmp_audio_path = tmp.name

            stt_result = await stt_service.transcribe_file(tmp_audio_path)
            if stt_result.get("error"):
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Audio transcription failed: {stt_result['error']}"
                )
            transcript = stt_result["text"]
            # Set duration from STT if not provided
            if not parsed_metadata.duration and stt_result.get("duration"):
                parsed_metadata.duration = f"{int(stt_result['duration'])}s"
        else:
            # Text-based file (TXT, DOC, etc.)
            transcript = file_content.decode("utf-8")

        # Create meeting agent
        meeting_agent_config = {
            "type": "meeting",
            "agent_model_config": {
                "model": _resolve_model(model)
            }
        }

        meeting_agent = MeetingAgent(meeting_agent_config)

        # Format participants
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

        # Process asynchronously
        result = await meeting_agent.process_async(meeting_data)

        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        # Auto-save to Postgres
        upload_source = "audio_upload" if stt_service.is_audio_file(filename) else "file_upload"
        saved_id = await _save_meeting_to_db(
            user_id=str(current_user.id),
            transcript=transcript,
            result=result,
            metadata=meeting_data["metadata"],
            context=context_dict,
            source=upload_source,
        )
        if saved_id:
            result["saved_id"] = saved_id

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="meeting.summarized",
            details={
                "meeting_type": parsed_metadata.type,
                "title": parsed_metadata.title,
                "source": upload_source,
            },
            related_resource_type="AGENT",
            agent_name="Meeting Agent",
        )
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing uploaded transcript", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Meeting processing failed: {str(e)}"
        )
    finally:
        if tmp_audio_path and os.path.exists(tmp_audio_path):
            try:
                os.unlink(tmp_audio_path)
            except OSError:
                pass

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
            "agent_model_config": {
                "model": _resolve_model(request.model)
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

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="meeting.summarized",
            details={"meeting_type": "status_update", "title": request.metadata.title},
            related_resource_type="AGENT",
            agent_name="Meeting Agent",
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
            "agent_model_config": {
                "model": _resolve_model(request.model)
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

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="meeting.summarized",
            details={"meeting_type": "planning", "title": request.metadata.title},
            related_resource_type="AGENT",
            agent_name="Meeting Agent",
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
            "agent_model_config": {
                "model": _resolve_model(request.model)
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

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="meeting.summarized",
            details={"meeting_type": "decision_making", "title": request.metadata.title},
            related_resource_type="AGENT",
            agent_name="Meeting Agent",
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
            "agent_model_config": {
                "model": _resolve_model(request.model)
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

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="meeting.summarized",
            details={"meeting_type": "client", "title": request.metadata.title},
            related_resource_type="AGENT",
            agent_name="Meeting Agent",
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
            "agent_model_config": {
                "model": _resolve_model(request.model)
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

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="meeting.summarized",
            details={"meeting_type": "brainstorming", "title": request.metadata.title},
            related_resource_type="AGENT",
            agent_name="Meeting Agent",
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
            "agent_model_config": {
                "model": _resolve_model(request.model)
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

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="meeting.summarized",
            details={"meeting_type": "problem_solving", "title": request.metadata.title},
            related_resource_type="AGENT",
            agent_name="Meeting Agent",
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
            organization_id=str(getattr(current_user, "organization_id", "") or ""),
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
        
        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="meeting.exported",
            details={"meeting_type": request.metadata.type, "export_target": "notion"},
            related_resource_type="AGENT",
            agent_name="Meeting Agent",
        )
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
            organization_id=str(getattr(current_user, "organization_id", "") or ""),
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
        
        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None),
            activity_type="meeting.exported",
            details={"meeting_type": request.metadata.type, "export_target": "google_workspace"},
            related_resource_type="AGENT",
            agent_name="Meeting Agent",
        )
        return meeting_response
    except Exception as e:
        logger.error("Error processing and exporting meeting to Google Workspace", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Meeting processing and export to Google Workspace failed: {str(e)}"
        )


# ── Meeting Library (Postgres persistence) ─────────────────────────────

class MeetingLibraryItem(BaseModel):
    """A saved meeting returned from the library."""
    id: str
    title: Optional[str] = None
    meeting_type: str = "general"
    summary: Optional[str] = None
    action_items: List[Dict[str, Any]] = []
    decisions: List[str] = []
    key_points: List[str] = []
    follow_ups: List[str] = []
    model_used: Optional[str] = None
    meeting_date: Optional[str] = None
    duration: Optional[str] = None
    participants: List[str] = []
    source: str = "manual"
    processed_at: Optional[str] = None
    created_at: str


class MeetingLibraryDetail(MeetingLibraryItem):
    """Full meeting detail including transcript and raw response."""
    transcript: str = ""
    raw_response: Optional[str] = None
    questions: List[str] = []
    concerns: List[str] = []
    context: Dict[str, Any] = {}
    tags: List[str] = []


class MeetingLibraryList(BaseModel):
    total: int
    page: int
    limit: int
    meetings: List[MeetingLibraryItem]


@router.get("/library", response_model=MeetingLibraryList)
async def list_meetings(
    page: int = 1,
    limit: int = 20,
    meeting_type: Optional[str] = None,
    search: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """List all saved meetings for the current user, paginated."""
    try:
        session_factory = get_async_sessionmaker()
        async with session_factory() as session:
            # Base query — only non-deleted meetings for this user
            query = select(MeetingSQL).where(
                MeetingSQL.user_id == str(current_user.id),
                MeetingSQL.deleted_at.is_(None),
            )

            if meeting_type:
                query = query.where(MeetingSQL.meeting_type == meeting_type)

            if search:
                pattern = f"%{search}%"
                query = query.where(
                    MeetingSQL.title.ilike(pattern)
                    | MeetingSQL.summary.ilike(pattern)
                )

            # Count
            from sqlalchemy import func
            count_q = select(func.count()).select_from(query.subquery())
            total = (await session.execute(count_q)).scalar() or 0

            # Paginate
            offset = (page - 1) * limit
            query = query.order_by(MeetingSQL.created_at.desc()).offset(offset).limit(limit)
            rows = (await session.execute(query)).scalars().all()

            meetings = [
                MeetingLibraryItem(
                    id=m.id,
                    title=m.title,
                    meeting_type=m.meeting_type,
                    summary=m.summary,
                    action_items=m.action_items or [],
                    decisions=m.decisions or [],
                    key_points=m.key_points or [],
                    follow_ups=m.follow_ups or [],
                    model_used=m.model_used,
                    meeting_date=m.meeting_date,
                    duration=m.duration,
                    participants=m.participants or [],
                    source=m.source,
                    processed_at=m.processed_at.isoformat() if m.processed_at else None,
                    created_at=m.created_at.isoformat(),
                )
                for m in rows
            ]

            return MeetingLibraryList(total=total, page=page, limit=limit, meetings=meetings)

    except Exception as e:
        logger.error("Error listing meetings", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list meetings: {str(e)}")


@router.get("/library/{meeting_id}", response_model=MeetingLibraryDetail)
async def get_meeting(
    meeting_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get a single saved meeting with full details including transcript."""
    try:
        session_factory = get_async_sessionmaker()
        async with session_factory() as session:
            query = select(MeetingSQL).where(
                MeetingSQL.id == meeting_id,
                MeetingSQL.user_id == str(current_user.id),
                MeetingSQL.deleted_at.is_(None),
            )
            m = (await session.execute(query)).scalar_one_or_none()

            if not m:
                raise HTTPException(status_code=404, detail="Meeting not found")

            return MeetingLibraryDetail(
                id=m.id,
                title=m.title,
                meeting_type=m.meeting_type,
                transcript=m.transcript,
                summary=m.summary,
                raw_response=m.raw_response,
                action_items=m.action_items or [],
                decisions=m.decisions or [],
                key_points=m.key_points or [],
                follow_ups=m.follow_ups or [],
                questions=m.questions or [],
                concerns=m.concerns or [],
                model_used=m.model_used,
                meeting_date=m.meeting_date,
                duration=m.duration,
                participants=m.participants or [],
                context=m.context or {},
                tags=m.tags or [],
                source=m.source,
                processed_at=m.processed_at.isoformat() if m.processed_at else None,
                created_at=m.created_at.isoformat(),
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting meeting", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get meeting: {str(e)}")


@router.delete("/library/{meeting_id}")
async def delete_meeting(
    meeting_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Soft-delete a saved meeting."""
    try:
        session_factory = get_async_sessionmaker()
        async with session_factory() as session:
            query = select(MeetingSQL).where(
                MeetingSQL.id == meeting_id,
                MeetingSQL.user_id == str(current_user.id),
                MeetingSQL.deleted_at.is_(None),
            )
            m = (await session.execute(query)).scalar_one_or_none()

            if not m:
                raise HTTPException(status_code=404, detail="Meeting not found")

            m.deleted_at = datetime.utcnow()
            await session.commit()

            return {"status": "deleted", "meeting_id": meeting_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting meeting", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete meeting: {str(e)}")


# ── Meeting Draft (auto-save transcript) ────────────────────────────────

class DraftRequest(BaseModel):
    """Save or update a transcript draft."""
    transcript: str = Field(..., description="The transcript text to save")
    title: Optional[str] = None
    meeting_type: Optional[str] = "general"
    participants: Optional[List[str]] = None
    context: Optional[Dict[str, Any]] = None


class DraftResponse(BaseModel):
    """Saved draft."""
    id: str
    transcript: str
    title: Optional[str] = None
    meeting_type: Optional[str] = None
    participants: List[str] = []
    context: Dict[str, Any] = {}
    updated_at: str


@router.put("/draft", response_model=DraftResponse)
async def save_draft(
    request: DraftRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Save or update the user's meeting transcript draft.

    One draft per user — calling this upserts. The draft persists across
    sessions so even if the user logs out and comes back, the transcript
    is still there.
    """
    try:
        user_id = str(current_user.id)
        session_factory = get_async_sessionmaker()
        async with session_factory() as session:
            # Check if draft exists
            query = select(MeetingDraftSQL).where(MeetingDraftSQL.user_id == user_id)
            draft = (await session.execute(query)).scalar_one_or_none()

            if draft:
                draft.transcript = request.transcript
                draft.title = request.title or draft.title
                draft.meeting_type = request.meeting_type or draft.meeting_type
                if request.participants is not None:
                    draft.participants = request.participants
                if request.context is not None:
                    draft.context = request.context
                draft.updated_at = datetime.utcnow()
            else:
                draft = MeetingDraftSQL(
                    user_id=user_id,
                    transcript=request.transcript,
                    title=request.title,
                    meeting_type=request.meeting_type or "general",
                    participants=request.participants or [],
                    context=request.context or {},
                )
                session.add(draft)

            await session.commit()
            await session.refresh(draft)

            return DraftResponse(
                id=draft.id,
                transcript=draft.transcript,
                title=draft.title,
                meeting_type=draft.meeting_type,
                participants=draft.participants or [],
                context=draft.context or {},
                updated_at=draft.updated_at.isoformat(),
            )

    except Exception as e:
        logger.error("Error saving draft", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to save draft: {str(e)}")


@router.get("/draft", response_model=Optional[DraftResponse])
async def get_draft(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get the current user's saved draft transcript. Returns null if no draft exists."""
    try:
        user_id = str(current_user.id)
        session_factory = get_async_sessionmaker()
        async with session_factory() as session:
            query = select(MeetingDraftSQL).where(MeetingDraftSQL.user_id == user_id)
            draft = (await session.execute(query)).scalar_one_or_none()

            if not draft:
                return None

            return DraftResponse(
                id=draft.id,
                transcript=draft.transcript,
                title=draft.title,
                meeting_type=draft.meeting_type,
                participants=draft.participants or [],
                context=draft.context or {},
                updated_at=draft.updated_at.isoformat(),
            )

    except Exception as e:
        logger.error("Error getting draft", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to get draft: {str(e)}")


@router.delete("/draft")
async def delete_draft(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Delete the current user's draft (e.g. after processing)."""
    try:
        user_id = str(current_user.id)
        session_factory = get_async_sessionmaker()
        async with session_factory() as session:
            query = select(MeetingDraftSQL).where(MeetingDraftSQL.user_id == user_id)
            draft = (await session.execute(query)).scalar_one_or_none()

            if draft:
                await session.delete(draft)
                await session.commit()

            return {"status": "deleted"}

    except Exception as e:
        logger.error("Error deleting draft", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete draft: {str(e)}")
