from typing import Any, List, Optional, Dict, Union
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from pydantic import BaseModel, Field
from datetime import datetime

from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.services.project_manager import project_manager

router = APIRouter()

class ProjectCreate(BaseModel):
    """Request model for creating a project."""
    title: str = Field(..., description="Project title")
    description: str = Field(..., description="Project description")
    due_date: Optional[Union[datetime, str]] = Field(None, description="Project due date (ISO format)")
    status: Optional[str] = Field("Not Started", description="Project status")
    integration_type: Optional[str] = Field("notion", description="Integration type (notion, etc.)")
    integration_id: Optional[str] = Field(None, description="Specific integration ID to use")

class ProjectDatabaseCreate(BaseModel):
    """Request model for creating a project database."""
    title: str = Field(..., description="Database title")
    integration_type: Optional[str] = Field("notion", description="Integration type (notion, etc.)")
    integration_id: Optional[str] = Field(None, description="Specific integration ID to use")
    parent_id: Optional[str] = Field(None, description="Parent page/folder ID")

class ProjectTaskCreate(BaseModel):
    """Request model for creating a task."""
    database_id: str = Field(..., description="Database ID")
    task_name: str = Field(..., description="Task name")
    description: str = Field(..., description="Task description")
    due_date: Optional[Union[datetime, str]] = Field(None, description="Task due date (ISO format)")
    status: Optional[str] = Field("Not Started", description="Task status")
    priority: Optional[str] = Field("Medium", description="Task priority (Low, Medium, High)")
    assigned_to: Optional[str] = Field(None, description="Person assigned to the task")
    integration_type: Optional[str] = Field("notion", description="Integration type (notion, etc.)")
    integration_id: Optional[str] = Field(None, description="Specific integration ID to use")

class MeetingExport(BaseModel):
    """Request model for exporting a meeting to a project tool."""
    meeting_data: Dict[str, Any] = Field(..., description="Meeting data from MeetingAgent")
    integration_type: Optional[str] = Field("notion", description="Integration type (notion, etc.)")
    integration_id: Optional[str] = Field(None, description="Specific integration ID to use")

class TasksQuery(BaseModel):
    """Request model for querying tasks."""
    database_id: str = Field(..., description="Database ID")
    filter_status: Optional[str] = Field(None, description="Filter by status")
    filter_priority: Optional[str] = Field(None, description="Filter by priority")
    integration_type: Optional[str] = Field("notion", description="Integration type (notion, etc.)")
    integration_id: Optional[str] = Field(None, description="Specific integration ID to use")

class ProjectResponse(BaseModel):
    """Response model for project operations."""
    status: str
    message: str
    project_id: Optional[str] = None
    project_data: Optional[Dict[str, Any]] = None

class DatabaseResponse(BaseModel):
    """Response model for database operations."""
    status: str
    message: str
    database_id: Optional[str] = None
    database_data: Optional[Dict[str, Any]] = None

class TaskResponse(BaseModel):
    """Response model for task operations."""
    status: str
    message: str
    task_id: Optional[str] = None
    task_data: Optional[Dict[str, Any]] = None

class TasksResponse(BaseModel):
    """Response model for task queries."""
    status: str
    tasks: Optional[List[Dict[str, Any]]] = None
    count: Optional[int] = None
    message: Optional[str] = None

class ExportResponse(BaseModel):
    """Response model for meeting export."""
    status: str
    message: str
    page_id: Optional[str] = None
    page_data: Optional[Dict[str, Any]] = None

@router.post("/projects", response_model=ProjectResponse)
async def create_project(
    project_data: ProjectCreate = Body(...),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Create a new project in the specified project management tool.
    
    This endpoint creates a project page or entry in the integrated project management 
    system (e.g., Notion).
    """
    result = await project_manager.create_project(
        organization_id=str(current_user.organization_id),
        title=project_data.title,
        description=project_data.description,
        due_date=project_data.due_date,
        status=project_data.status,
        integration_type=project_data.integration_type,
        integration_id=project_data.integration_id
    )
    
    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("message", "Failed to create project")
        )
        
    return result

@router.post("/databases", response_model=DatabaseResponse)
async def create_project_database(
    database_data: ProjectDatabaseCreate = Body(...),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Create a new project database or board in the specified project management tool.
    
    This endpoint creates a structured database for tracking projects in the integrated
    project management system (e.g., Notion database).
    """
    result = await project_manager.create_project_database(
        organization_id=str(current_user.organization_id),
        title=database_data.title,
        integration_type=database_data.integration_type,
        integration_id=database_data.integration_id,
        parent_id=database_data.parent_id
    )
    
    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("message", "Failed to create project database")
        )
        
    return result

@router.post("/tasks", response_model=TaskResponse)
async def add_task(
    task_data: ProjectTaskCreate = Body(...),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Add a task to a project database or board.
    
    This endpoint adds a task entry to the specified project database in the integrated
    project management system.
    """
    result = await project_manager.add_task(
        organization_id=str(current_user.organization_id),
        database_id=task_data.database_id,
        task_name=task_data.task_name,
        description=task_data.description,
        due_date=task_data.due_date,
        status=task_data.status,
        priority=task_data.priority,
        assigned_to=task_data.assigned_to,
        integration_type=task_data.integration_type,
        integration_id=task_data.integration_id
    )
    
    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("message", "Failed to add task")
        )
        
    return result

@router.post("/export-meeting", response_model=ExportResponse)
async def export_meeting(
    export_data: MeetingExport = Body(...),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Export meeting data to a project management tool.
    
    This endpoint takes meeting data from MeetingAgent and exports it as a structured
    page in the integrated project management system.
    """
    result = await project_manager.export_meeting_to_project(
        organization_id=str(current_user.organization_id),
        meeting_data=export_data.meeting_data,
        integration_type=export_data.integration_type,
        integration_id=export_data.integration_id
    )
    
    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("message", "Failed to export meeting")
        )
        
    return result

@router.post("/query-tasks", response_model=TasksResponse)
async def query_tasks(
    query_data: TasksQuery = Body(...),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Query tasks from a project database or board.
    
    This endpoint retrieves tasks from the specified project database in the integrated
    project management system with optional filtering.
    """
    result = await project_manager.get_tasks(
        organization_id=str(current_user.organization_id),
        database_id=query_data.database_id,
        filter_status=query_data.filter_status,
        filter_priority=query_data.filter_priority,
        integration_type=query_data.integration_type,
        integration_id=query_data.integration_id
    )
    
    if result.get("status") == "error":
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result.get("message", "Failed to query tasks")
        )
        
    return result
