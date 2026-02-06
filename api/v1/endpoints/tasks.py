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
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    Create a new task.
    """
    try:
        if settings.POSTGRES_ENABLED and db is not None:
            repo = PostgresTaskRepository(db)
            task = await repo.create_task(
                task_data=task_in.model_dump() if hasattr(task_in, "model_dump") else task_in.dict(),
                creator_id=current_user.id,
                organization_id=current_user.organization_id
            )
            if settings.POSTGRES_DUAL_WRITE:
                try:
                    task_payload = task_in.model_dump() if hasattr(task_in, "model_dump") else task_in.dict()
                    metadata = task_payload.get("metadata") or {}
                    metadata["postgres_id"] = task.id
                    task_payload["metadata"] = metadata
                    await task_repository.create_task_with_postgres_id(
                        task_data=TaskCreate(**task_payload),
                        creator_id=current_user.id,
                        organization_id=current_user.organization_id,
                        postgres_id=task.id
                    )
                except Exception:
                    # Do not fail primary write if secondary fails
                    pass
            return task
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
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    Get a task by ID.
    """
    if settings.POSTGRES_ENABLED and db is not None:
        repo = PostgresTaskRepository(db)
        task = await repo.get_task_by_id(
            task_id=task_id,
            organization_id=current_user.organization_id
        )
    else:
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
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    Update a task.
    """
    if settings.POSTGRES_ENABLED and db is not None:
        repo = PostgresTaskRepository(db)
        existing_task = await repo.get_task_by_id(task_id, organization_id=current_user.organization_id)
    else:
        existing_task = await task_repository.get_task_by_id(task_id, organization_id=current_user.organization_id)
    if not existing_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found or not authorized")

    if settings.POSTGRES_ENABLED and db is not None:
        repo = PostgresTaskRepository(db)
        task = await repo.update_task(
            task_id=task_id,
            organization_id=current_user.organization_id,
            update_data=task_in.dict(exclude_unset=True)
        )
        if settings.POSTGRES_DUAL_WRITE:
            try:
                await task_repository.update_task_by_postgres_id(
                    postgres_id=task_id,
                    update_data=task_in.dict(exclude_unset=True),
                    organization_id=current_user.organization_id
                )
            except Exception:
                pass
    else:
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
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> None:
    """
    Delete a task.
    """
    if settings.POSTGRES_ENABLED and db is not None:
        repo = PostgresTaskRepository(db)
        existing_task = await repo.get_task_by_id(task_id, organization_id=current_user.organization_id)
    else:
        existing_task = await task_repository.get_task_by_id(task_id, organization_id=current_user.organization_id)
    if not existing_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found or not authorized")
    
    if settings.POSTGRES_ENABLED and db is not None:
        repo = PostgresTaskRepository(db)
        deleted = await repo.delete_task(
            task_id=task_id,
            organization_id=current_user.organization_id
        )
        if settings.POSTGRES_DUAL_WRITE:
            try:
                await task_repository.delete_task_by_postgres_id(
                    postgres_id=task_id,
                    organization_id=current_user.organization_id
                )
            except Exception:
                pass
    else:
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
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    List tasks for the current user or organization with filters.
    """
    if settings.POSTGRES_ENABLED and db is not None:
        repo = PostgresTaskRepository(db)
        tasks = await repo.get_organization_tasks(
            organization_id=current_user.organization_id,
            status=status,
            assigned_to=assigned_to,
            document_id=document_id,
            agent_id=agent_id,
            skip=skip,
            limit=limit
        )
    else:
        tasks = await task_repository.get_organization_tasks(
            organization_id=current_user.organization_id,
            status=status,
            assigned_to=assigned_to,
            document_id=document_id,
            agent_id=agent_id,
            skip=skip,
            limit=limit
        )
    return tasks

@router.get("/upcoming", response_model=List[TaskResponse])
async def list_upcoming_tasks(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    List upcoming tasks for the current user (due in next 7 days by default).
    """
    if settings.POSTGRES_ENABLED and db is not None:
        repo = PostgresTaskRepository(db)
        tasks = await repo.get_upcoming_tasks(
            organization_id=current_user.organization_id,
            user_id=current_user.id
        )
    else:
        tasks = await task_repository.get_upcoming_tasks(
            organization_id=current_user.organization_id,
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
    if settings.POSTGRES_ENABLED and db is not None:
        repo = PostgresTaskRepository(db)
        analytics = await repo.get_task_analytics(
            organization_id=current_user.organization_id,
            user_id=current_user.id,
            time_range=time_range
        )
    else:
        analytics = await task_repository.get_task_analytics(
            organization_id=current_user.organization_id,
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
    if settings.POSTGRES_ENABLED and db is not None:
        repo = PostgresTaskRepository(db)
        summary_data = await repo.get_task_stats(
            organization_id=current_user.organization_id
        )
    else:
        summary_data = await task_repository.get_task_stats(
            organization_id=current_user.organization_id
        )
    # Map the dictionary result from the repository to the Pydantic response model
    # This assumes the keys match or need simple mapping
    return TaskSummaryResponse(**summary_data) 
