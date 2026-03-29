from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from pydantic import BaseModel
from datetime import datetime
from enum import Enum

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.task_repository import task_repository
from backend.db.postgres import get_optional_async_db
from backend.db.postgres_repositories.task_repository import PostgresTaskRepository
from backend.core.config import settings
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.user import User
from backend.models.task import Task, TaskCreate, TaskUpdate, TaskStatus, TaskPriority
from backend.services.activity_logger import log_activity

router = APIRouter()


def _get_org_id(user: User) -> str:
    """Safely extract organization_id from the user, falling back to user.id."""
    return getattr(user, "organization_id", None) or str(user.id)


class TaskResponse(BaseModel):
    id: str
    title: str
    name: Optional[str] = None  # Alias for title (backwards compat)
    description: Optional[str] = None
    status: TaskStatus
    due_date: Optional[datetime] = None
    priority: Optional[str] = None
    organization_id: Optional[str] = None
    created_by: Optional[str] = None
    assigned_to: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    progress: Optional[int] = None
    document_id: Optional[str] = None  # Link to source document if applicable
    agent_id: Optional[str] = None  # Link to agent that created the task if applicable

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
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    Create a new task.
    """
    try:
        task = await task_repository.create_task(
            task_data=task_in,
            creator_id=current_user.id,
            organization_id=_get_org_id(current_user)
        )

        # Dual-write to Postgres if enabled
        if settings.POSTGRES_ENABLED and settings.POSTGRES_DUAL_WRITE and db is not None:
            try:
                repo = PostgresTaskRepository(db)
                await repo.create_task(
                    task_data=task_in.model_dump() if hasattr(task_in, "model_dump") else task_in.dict(),
                    creator_id=current_user.id,
                    organization_id=_get_org_id(current_user)
                )
            except Exception:
                pass

        await log_activity(
            user_id=str(current_user.id),
            organization_id=_get_org_id(current_user),
            activity_type="task.created",
            details={"title": task_in.title, "status": task_in.status.value if task_in.status else "todo", "priority": str(task_in.priority.value) if task_in.priority else "medium"},
            related_resource_type="TASK",
            related_resource_id=str(task.id) if hasattr(task, "id") else None,
        )

        return task
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{task_id}", response_model=None)
async def get_task(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    Get a task by ID.
    """
    task = await task_repository.get_task_by_id(
        task_id=task_id,
        organization_id=_get_org_id(current_user)
    )
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    if str(task.organization_id) != _get_org_id(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this task")
    return task

@router.put("/{task_id}", response_model=None)
async def update_task(
    task_id: str,
    task_in: TaskUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    Update a task.
    """
    existing_task = await task_repository.get_task_by_id(task_id, organization_id=_get_org_id(current_user))
    if not existing_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found or not authorized")

    task = await task_repository.update_task(
        task_id=task_id,
        organization_id=_get_org_id(current_user),
        update_data=task_in.dict(exclude_unset=True)
    )
    if settings.POSTGRES_ENABLED and settings.POSTGRES_DUAL_WRITE and db is not None:
        try:
            repo = PostgresTaskRepository(db)
            await repo.update_task(
                task_id=task_id,
                organization_id=_get_org_id(current_user),
                update_data=task_in.dict(exclude_unset=True)
            )
        except Exception:
            pass
    if not task:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update task"
        )

    await log_activity(
        user_id=str(current_user.id),
        organization_id=_get_org_id(current_user),
        activity_type="task.updated",
        details={"task_id": task_id, "updated_fields": list(task_in.dict(exclude_unset=True).keys())},
        related_resource_type="TASK",
        related_resource_id=task_id,
    )

    return task

@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> None:
    """
    Delete a task.
    """
    existing_task = await task_repository.get_task_by_id(task_id, organization_id=_get_org_id(current_user))
    if not existing_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found or not authorized")

    deleted = await task_repository.delete_task(
        task_id=task_id,
        organization_id=_get_org_id(current_user)
    )
    if settings.POSTGRES_ENABLED and settings.POSTGRES_DUAL_WRITE and db is not None:
        try:
            repo = PostgresTaskRepository(db)
            await repo.delete_task(
                task_id=task_id,
                organization_id=_get_org_id(current_user)
            )
        except Exception:
            pass
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete task"
        )

    await log_activity(
        user_id=str(current_user.id),
        organization_id=_get_org_id(current_user),
        activity_type="task.deleted",
        details={"task_id": task_id},
        related_resource_type="TASK",
        related_resource_id=task_id,
    )

    return None

@router.get("/", response_model=None)
async def list_tasks(
    status: Optional[TaskStatus] = Query(None),
    assigned_to: Optional[str] = Query(None),
    document_id: Optional[str] = Query(None),
    agent_id: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000), # Increased default limit
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    List tasks for the current user or organization with filters.
    Always reads from MongoDB (source of truth). Postgres is write-only for now.
    """
    tasks = await task_repository.get_organization_tasks(
        organization_id=_get_org_id(current_user),
        status=status,
        assigned_to=assigned_to,
        document_id=document_id,
        agent_id=agent_id,
        skip=skip,
        limit=limit
    )

    # Serialize tasks to dicts for the frontend
    result = []
    for t in tasks:
        t_dict = t.dict() if hasattr(t, "dict") else t
        # Ensure id is a string
        if "_id" in t_dict:
            t_dict["id"] = str(t_dict.pop("_id"))
        elif "id" in t_dict:
            t_dict["id"] = str(t_dict["id"])
        # Stringify ObjectId fields
        for field in ("organization_id", "created_by", "assigned_to", "project_id", "agent_id", "parent_task_id"):
            if t_dict.get(field):
                t_dict[field] = str(t_dict[field])
        # Extract document_id from metadata for convenience
        if not t_dict.get("document_id") and isinstance(t_dict.get("metadata"), dict):
            t_dict["document_id"] = t_dict["metadata"].get("document_id")
        result.append(t_dict)
    return result

@router.get("/upcoming", response_model=List[TaskResponse])
async def list_upcoming_tasks(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    List upcoming tasks for the current user (due in next 7 days by default).
    """
    tasks = await task_repository.get_upcoming_tasks(
        organization_id=_get_org_id(current_user),
        user_id=current_user.id
    )
    return tasks

@router.get("/analytics", response_model=Dict[str, Any])
async def get_task_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"), # e.g., "7d", "30d", "1y"
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    Get task analytics for the current user or organization.
    """
    analytics = await task_repository.get_task_analytics(
        organization_id=_get_org_id(current_user),
        user_id=current_user.id,
        time_range=time_range
    )
    return analytics

@router.get("/summary", response_model=TaskSummaryResponse)
async def get_task_summary(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> TaskSummaryResponse:
    """
    Get summary statistics for tasks in the organization.
    """
    summary_data = await task_repository.get_task_stats(
        organization_id=_get_org_id(current_user)
    )
    return TaskSummaryResponse(**summary_data) 
