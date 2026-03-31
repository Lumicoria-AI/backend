from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, UploadFile, File
from pydantic import BaseModel
import structlog
import os
import tempfile

from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.services.integration_service import integration_service

logger = structlog.get_logger(__name__)

router = APIRouter(
    prefix="/slack",
    tags=["slack"],
    responses={404: {"description": "Not found"}},
)


# ── Request Models ─────────────────────────────────────────────────────────

class ProjectChannelRequest(BaseModel):
    project_name: str
    description: str
    is_private: bool = False

class ProjectTaskRequest(BaseModel):
    channel: str
    task_name: str
    description: str
    assignee: Optional[str] = None
    due_date: Optional[str] = None

class MeetingNotesRequest(BaseModel):
    channel: str
    meeting_title: str
    notes: str
    participants: List[str]
    date: str

class ReminderRequest(BaseModel):
    text: str
    time: str
    channel: Optional[str] = None
    user: Optional[str] = None

class SearchRequest(BaseModel):
    query: str
    channel: Optional[str] = None


# ── Endpoints (all auth-protected) ─────────────────────────────────────────

@router.post("/project-channel")
async def create_project_channel(
    request: ProjectChannelRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Create a new project channel."""
    try:
        return await integration_service.execute_integration_action(
            integration_type="slack",
            action="create_project_channel",
            data=request.model_dump(),
        )
    except Exception as e:
        logger.error(f"Error creating project channel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/project-task")
async def add_project_task(
    request: ProjectTaskRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Add a task to a project channel."""
    try:
        return await integration_service.execute_integration_action(
            integration_type="slack",
            action="add_project_task",
            data=request.model_dump(),
        )
    except Exception as e:
        logger.error(f"Error adding project task: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/meeting-notes")
async def export_meeting_notes(
    request: MeetingNotesRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Export meeting notes to a channel."""
    try:
        return await integration_service.execute_integration_action(
            integration_type="slack",
            action="export_meeting_notes",
            data=request.model_dump(),
        )
    except Exception as e:
        logger.error(f"Error exporting meeting notes: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/reminder")
async def create_reminder(
    request: ReminderRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Create a reminder."""
    try:
        return await integration_service.execute_integration_action(
            integration_type="slack",
            action="create_reminder",
            data=request.model_dump(),
        )
    except Exception as e:
        logger.error(f"Error creating reminder: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search")
async def search_project_content(
    request: SearchRequest,
    current_user: User = Depends(get_current_active_user),
) -> List[Dict[str, Any]]:
    """Search for project content."""
    try:
        return await integration_service.execute_integration_action(
            integration_type="slack",
            action="search_project_content",
            data=request.model_dump(),
        )
    except Exception as e:
        logger.error(f"Error searching project content: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/upload-file")
async def upload_project_file(
    channel: str,
    current_user: User = Depends(get_current_active_user),
    file: UploadFile = File(...),
    title: Optional[str] = None,
    comment: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload a file to a project channel."""
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=f"_{file.filename}") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        result = await integration_service.execute_integration_action(
            integration_type="slack",
            action="upload_project_file",
            data={
                "channel": channel,
                "file_path": tmp_path,
                "title": title,
                "comment": comment,
            },
        )
        return result
    except Exception as e:
        logger.error(f"Error uploading project file: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


@router.get("/channel-members/{channel}")
async def get_channel_members(
    channel: str,
    current_user: User = Depends(get_current_active_user),
) -> List[Dict[str, Any]]:
    """Get members of a channel."""
    try:
        return await integration_service.execute_integration_action(
            integration_type="slack",
            action="get_channel_members",
            data={"channel": channel},
        )
    except Exception as e:
        logger.error(f"Error getting channel members: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/archive-channel/{channel}")
async def archive_project_channel(
    channel: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Archive a project channel."""
    try:
        return await integration_service.execute_integration_action(
            integration_type="slack",
            action="archive_project_channel",
            data={"channel": channel},
        )
    except Exception as e:
        logger.error(f"Error archiving project channel: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/available-actions")
async def get_available_actions(
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Get available Slack integration actions."""
    integrations = integration_service.get_available_integrations()
    return integrations.get("slack", {})
