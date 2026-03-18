from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from pydantic import BaseModel, Field
from datetime import datetime
from bson import ObjectId
import structlog

from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.db.mongodb.mongodb import get_mongodb

router = APIRouter()
logger = structlog.get_logger(__name__)


# ─── Pydantic Models ────────────────────────────────────────────────────────

class ProjectCreate(BaseModel):
    title: str = Field(..., description="Project title")
    description: Optional[str] = Field("", description="Project description")
    due_date: Optional[str] = Field(None, description="Due date (ISO string)")
    status: Optional[str] = Field("Not Started", description="Project status")
    color: Optional[str] = Field("#6366f1", description="Project colour")


class ProjectUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    due_date: Optional[str] = None
    status: Optional[str] = None
    color: Optional[str] = None


class TaskCreate(BaseModel):
    title: str
    description: Optional[str] = ""
    due_date: Optional[str] = None
    status: Optional[str] = "Not Started"
    priority: Optional[str] = "Medium"


class ProjectResponse(BaseModel):
    id: str
    user_id: str
    title: str
    description: Optional[str] = ""
    due_date: Optional[str] = None
    status: str = "Not Started"
    color: str = "#6366f1"
    tasks: List[Dict] = []
    created_at: str
    updated_at: str


# ─── Helpers ─────────────────────────────────────────────────────────────────

async def _get_col():
    db = await get_mongodb()
    return db["lumicoria_projects"]


def _serialize(doc: dict) -> dict:
    doc["id"] = str(doc.pop("_id"))
    return doc


# ─── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/projects", response_model=List[ProjectResponse])
async def list_projects(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """List all projects for the authenticated user."""
    col = await _get_col()
    cursor = col.find(
        {"user_id": str(current_user.id)},
        sort=[("created_at", -1)],
        skip=skip,
        limit=limit,
    )
    docs = await cursor.to_list(length=limit)
    return [_serialize(d) for d in docs]


@router.post("/projects", response_model=ProjectResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    project_data: ProjectCreate = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Create a new project."""
    col = await _get_col()
    now = datetime.utcnow().isoformat()
    doc = {
        "user_id": str(current_user.id),
        "title": project_data.title,
        "description": project_data.description or "",
        "due_date": project_data.due_date,
        "status": project_data.status or "Not Started",
        "color": project_data.color or "#6366f1",
        "tasks": [],
        "created_at": now,
        "updated_at": now,
    }
    result = await col.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _serialize(doc)


@router.get("/projects/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get a project by ID."""
    col = await _get_col()
    try:
        oid = ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID")
    doc = await col.find_one({"_id": oid, "user_id": str(current_user.id)})
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found")
    return _serialize(doc)


@router.put("/projects/{project_id}", response_model=ProjectResponse)
async def update_project(
    project_id: str,
    update: ProjectUpdate = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Update a project."""
    col = await _get_col()
    try:
        oid = ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID")
    update_data = {k: v for k, v in update.model_dump(exclude_unset=True).items() if v is not None}
    update_data["updated_at"] = datetime.utcnow().isoformat()
    doc = await col.find_one_and_update(
        {"_id": oid, "user_id": str(current_user.id)},
        {"$set": update_data},
        return_document=True,
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found")
    return _serialize(doc)


@router.delete("/projects/{project_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_project(
    project_id: str,
    current_user: User = Depends(get_current_active_user),
) -> None:
    """Delete a project."""
    col = await _get_col()
    try:
        oid = ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID")
    result = await col.delete_one({"_id": oid, "user_id": str(current_user.id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Project not found")


@router.post("/projects/{project_id}/tasks", response_model=ProjectResponse)
async def add_task_to_project(
    project_id: str,
    task: TaskCreate = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Add a task to a project."""
    col = await _get_col()
    try:
        oid = ObjectId(project_id)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid project ID")
    task_doc = {
        "id": str(ObjectId()),
        "title": task.title,
        "description": task.description or "",
        "due_date": task.due_date,
        "status": task.status or "Not Started",
        "priority": task.priority or "Medium",
        "created_at": datetime.utcnow().isoformat(),
    }
    doc = await col.find_one_and_update(
        {"_id": oid, "user_id": str(current_user.id)},
        {
            "$push": {"tasks": task_doc},
            "$set": {"updated_at": datetime.utcnow().isoformat()},
        },
        return_document=True,
    )
    if not doc:
        raise HTTPException(status_code=404, detail="Project not found")
    return _serialize(doc)
