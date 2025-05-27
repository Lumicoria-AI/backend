from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel
import structlog
from ...services.integration_service import integration_service

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/slack",
    tags=["slack"],
    responses={404: {"description": "Not found"}},
)

# Request/Response Models
class ProjectChannelRequest(BaseModel):
    """Request model for creating a project channel."""
    project_name: str
    description: str
    is_private: bool = False

class ProjectTaskRequest(BaseModel):
    """Request model for adding a project task."""
    channel: str
    task_name: str
    description: str
    assignee: Optional[str] = None
    due_date: Optional[str] = None

class MeetingNotesRequest(BaseModel):
    """Request model for exporting meeting notes."""
    channel: str
    meeting_title: str
    notes: str
    participants: List[str]
    date: str

class ReminderRequest(BaseModel):
    """Request model for creating a reminder."""
    text: str
    time: str
    channel: Optional[str] = None
    user: Optional[str] = None

class SearchRequest(BaseModel):
    """Request model for searching project content."""
    query: str
    channel: Optional[str] = None

class FileUploadRequest(BaseModel):
    """Request model for uploading a file."""
    channel: str
    title: Optional[str] = None
    comment: Optional[str] = None

# Endpoints
@router.post("/project-channel")
async def create_project_channel(request: ProjectChannelRequest) -> Dict[str, Any]:
    """Create a new project channel."""
    try:
        result = await integration_service.execute_integration_action(
            integration_type="slack",
            action="create_project_channel",
            data=request.model_dump()
        )
        return result
    except Exception as e:
        logger.error(f"Error creating project channel: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/project-task")
async def add_project_task(request: ProjectTaskRequest) -> Dict[str, Any]:
    """Add a task to a project channel."""
    try:
        result = await integration_service.execute_integration_action(
            integration_type="slack",
            action="add_project_task",
            data=request.model_dump()
        )
        return result
    except Exception as e:
        logger.error(f"Error adding project task: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/meeting-notes")
async def export_meeting_notes(request: MeetingNotesRequest) -> Dict[str, Any]:
    """Export meeting notes to a channel."""
    try:
        result = await integration_service.execute_integration_action(
            integration_type="slack",
            action="export_meeting_notes",
            data=request.model_dump()
        )
        return result
    except Exception as e:
        logger.error(f"Error exporting meeting notes: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/reminder")
async def create_reminder(request: ReminderRequest) -> Dict[str, Any]:
    """Create a reminder."""
    try:
        result = await integration_service.execute_integration_action(
            integration_type="slack",
            action="create_reminder",
            data=request.model_dump()
        )
        return result
    except Exception as e:
        logger.error(f"Error creating reminder: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_project_content(request: SearchRequest) -> List[Dict[str, Any]]:
    """Search for project content."""
    try:
        result = await integration_service.execute_integration_action(
            integration_type="slack",
            action="search_project_content",
            data=request.model_dump()
        )
        return result
    except Exception as e:
        logger.error(f"Error searching project content: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/upload-file")
async def upload_project_file(
    channel: str,
    file: UploadFile = File(...),
    title: Optional[str] = None,
    comment: Optional[str] = None
) -> Dict[str, Any]:
    """Upload a file to a project channel."""
    try:
        # Save uploaded file temporarily
        file_path = f"temp/{file.filename}"
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
            
        # Upload to Slack
        result = await integration_service.execute_integration_action(
            integration_type="slack",
            action="upload_project_file",
            data={
                "channel": channel,
                "file_path": file_path,
                "title": title,
                "comment": comment
            }
        )
        
        # Clean up temp file
        import os
        os.remove(file_path)
        
        return result
    except Exception as e:
        logger.error(f"Error uploading project file: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/channel-members/{channel}")
async def get_channel_members(channel: str) -> List[Dict[str, Any]]:
    """Get members of a channel."""
    try:
        result = await integration_service.execute_integration_action(
            integration_type="slack",
            action="get_channel_members",
            data={"channel": channel}
        )
        return result
    except Exception as e:
        logger.error(f"Error getting channel members: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/archive-channel/{channel}")
async def archive_project_channel(channel: str) -> Dict[str, Any]:
    """Archive a project channel."""
    try:
        result = await integration_service.execute_integration_action(
            integration_type="slack",
            action="archive_project_channel",
            data={"channel": channel}
        )
        return result
    except Exception as e:
        logger.error(f"Error archiving project channel: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/available-actions")
async def get_available_actions() -> Dict[str, Any]:
    """Get available Slack integration actions."""
    try:
        integrations = integration_service.get_available_integrations()
        return integrations.get("slack", {})
    except Exception as e:
        logger.error(f"Error getting available actions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e)) 