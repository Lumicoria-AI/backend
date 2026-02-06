from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.task_repository import task_repository
from backend.models.user import User
from backend.models.task import Task, TaskCreate, TaskUpdate, TaskStatus, TaskPriority

router = APIRouter()

class TaskResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    status: TaskStatus
    due_date: Optional[datetime]
    priority: Optional[int]
    organization_id: str
    created_by: str
    assigned_to: Optional[str]
    created_at: datetime
    updated_at: Optional[datetime]
    metadata: Optional[Dict[str, Any]]
    document_id: Optional[str] # Link to source document if applicable
    agent_id: Optional[str] # Link to agent that created the task if applicable

class TaskSummaryResponse(BaseModel):
    total_tasks: int
    tasks_by_status: Dict[TaskStatus, int]
    tasks_by_priority: Dict[Optional[int], int]
    overdue_tasks_count: int
    upcoming_tasks_count: int
    # Add other relevant summary fields from TaskRepository.get_task_stats if needed

@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    task_in: TaskCreate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Create a new task.
    """
    try:
        task = await task_repository.create_task(
            task_data=task_in,
            creator_id=current_user.id,
            organization_id=current_user.organization_id
        )
        return task
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{task_id}", response_model=TaskResponse)
async def get_task(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get a task by ID.
    """
    task = await task_repository.get_task_by_id(
        task_id=task_id,
        organization_id=current_user.organization_id
    )
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    if str(task.organization_id) != current_user.organization_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this task")
    return task

@router.put("/{task_id}", response_model=TaskResponse)
async def update_task(
    task_id: str,
    task_in: TaskUpdate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Update a task.
    """
    existing_task = await task_repository.get_task_by_id(task_id, organization_id=current_user.organization_id)
    if not existing_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found or not authorized")

    task = await task_repository.update_task(
        task_id=task_id,
        organization_id=current_user.organization_id,
        update_data=task_in.dict(exclude_unset=True)
    )
    if not task:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update task"
        )
    return task

@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: str,
    current_user: User = Depends(get_current_active_user)
) -> None:
    """
    Delete a task.
    """
    existing_task = await task_repository.get_task_by_id(task_id, organization_id=current_user.organization_id)
    if not existing_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found or not authorized")
    
    deleted = await task_repository.delete_task(
        task_id=task_id,
        organization_id=current_user.organization_id
    )
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete task"
        )
    return None

@router.get("/", response_model=List[TaskResponse])
async def list_tasks(
    status: Optional[TaskStatus] = Query(None),
    assigned_to: Optional[str] = Query(None),
    document_id: Optional[str] = Query(None),
    agent_id: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000), # Increased default limit
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    List tasks for the current user or organization with filters.
    """
    tasks = await task_repository.get_organization_tasks(
        organization_id=current_user.organization_id,
        status=status,
        assigned_to=assigned_to, # Pass filters to repository
        document_id=document_id,
        agent_id=agent_id,
        skip=skip,
        limit=limit
    )
    return tasks

@router.get("/upcoming", response_model=List[TaskResponse])
async def list_upcoming_tasks(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    List upcoming tasks for the current user (due in next 7 days by default).
    """
    tasks = await task_repository.get_upcoming_tasks(
        organization_id=current_user.organization_id,
        user_id=current_user.id
    )
    return tasks

@router.get("/analytics", response_model=Dict[str, Any])
async def get_task_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"), # e.g., "7d", "30d", "1y"
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get task analytics for the current user or organization.
    """
    analytics = await task_repository.get_task_analytics(
        organization_id=current_user.organization_id,
        user_id=current_user.id, # Assuming analytics are often user-specific on a dashboard
        time_range=time_range # Pass time_range as string
    )
    return analytics

@router.get("/summary", response_model=TaskSummaryResponse)
async def get_task_summary(
    current_user: User = Depends(get_current_active_user)
) -> TaskSummaryResponse:
    """
    Get summary statistics for tasks in the organization.
    """
    summary_data = await task_repository.get_task_stats(
        organization_id=current_user.organization_id
        # Pass user_id=current_user.id here if the summary should be personalized for the user
    )
    # Map the dictionary result from the repository to the Pydantic response model
    # This assumes the keys match or need simple mapping
    return TaskSummaryResponse(**summary_data) 