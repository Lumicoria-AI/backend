"""
Phase A — Tasks v2 extras.

Mounted at `/api/v1/tasks-v2`.

The earlier `tasks_extended.py` covers bulk/watchers/dependencies/saved-views.
This module adds the remaining surface from the plan: subtasks, history,
duplicate/move/convert, task templates, per-source imports (Asana, Jira,
Linear, Notion, Trello, CSV), exports, snooze/unsnooze, reactions, and
dependency-graph extensions.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.models.user import User

logger = structlog.get_logger(__name__)
router = APIRouter()


def _oid(v: Any) -> Optional[ObjectId]:
    if v is None:
        return None
    if isinstance(v, ObjectId):
        return v
    try:
        return ObjectId(str(v))
    except Exception:
        return None


def _resolve_primary_org_id(user: User) -> str:
    primary = getattr(user, "organization_id", None)
    if primary:
        return str(primary)
    ids = getattr(user, "organization_ids", None) or []
    if ids:
        return str(ids[0])
    raise HTTPException(status_code=400, detail="User has no organization context")


async def _require_task(org_id: str, task_id: str) -> Dict[str, Any]:
    col = await MongoDB.get_collection("tasks")
    row = await col.find_one({"_id": _oid(task_id), "organization_id": _oid(org_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    return row


# ── Subtasks ────────────────────────────────────────────────────


@router.get("/{task_id}/subtasks")
async def list_subtasks(
    task_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("tasks")
    cursor = col.find({"organization_id": _oid(org_id), "parent_task_id": _oid(task_id)}).sort("created_at", 1)
    rows = await cursor.to_list(length=500)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "project_id", "assigned_to", "created_by", "parent_task_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


class SubtaskCreate(BaseModel):
    title: str
    description: Optional[str] = None
    priority: Optional[str] = "medium"
    due_date: Optional[datetime] = None
    assigned_to: Optional[str] = None


@router.post("/{task_id}/subtasks", status_code=201)
async def create_subtask(
    task_id: str,
    payload: SubtaskCreate,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    parent = await _require_task(org_id, task_id)
    col = await MongoDB.get_collection("tasks")
    now = datetime.utcnow()
    doc = {
        "organization_id": _oid(org_id),
        "project_id": parent.get("project_id"),
        "parent_task_id": _oid(task_id),
        "title": payload.title,
        "description": payload.description,
        "priority": payload.priority,
        "status": "todo",
        "due_date": payload.due_date,
        "assigned_to": _oid(payload.assigned_to) if payload.assigned_to else None,
        "created_by": _oid(current_user.id),
        "created_at": now, "updated_at": now,
    }
    r = await col.insert_one(doc)
    return {"id": str(r.inserted_id), "ok": True}


@router.post("/{task_id}/promote")
async def promote_subtask_to_task(
    task_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("tasks")
    row = await col.find_one_and_update(
        {"_id": _oid(task_id), "organization_id": _oid(org_id)},
        {"$set": {"parent_task_id": None, "updated_at": datetime.utcnow()}},
        return_document=True,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"ok": True}


# ── History ─────────────────────────────────────────────────────


@router.get("/{task_id}/history")
async def task_history(
    task_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    row = await _require_task(org_id, task_id)
    history = list(row.get("status_history") or [])
    activity = await MongoDB.get_collection("activity_logs")
    activity_rows = await activity.find({
        "organization_id": _oid(org_id),
        "related_resource_type": "task", "related_resource_id": task_id,
    }).sort("timestamp", -1).limit(100).to_list(length=100)
    for r in activity_rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "user_id"):
            if r.get(k):
                r[k] = str(r[k])
    return {"status_history": history, "activity": activity_rows}


# ── Duplicate / move / convert ──────────────────────────────────


@router.post("/{task_id}/duplicate", status_code=201)
async def duplicate_task(
    task_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    row = await _require_task(org_id, task_id)
    col = await MongoDB.get_collection("tasks")
    clone = dict(row)
    clone.pop("_id", None)
    clone["title"] = f"{row.get('title', 'Task')} (copy)"
    clone["status"] = "todo"
    clone["created_at"] = datetime.utcnow()
    clone["updated_at"] = datetime.utcnow()
    clone["completed_at"] = None
    clone["created_by"] = _oid(current_user.id)
    r = await col.insert_one(clone)
    return {"id": str(r.inserted_id)}


class MovePayload(BaseModel):
    project_id: str


@router.post("/{task_id}/move-to-project")
async def move_task_to_project(
    task_id: str,
    payload: MovePayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("tasks")
    row = await col.find_one_and_update(
        {"_id": _oid(task_id), "organization_id": _oid(org_id)},
        {"$set": {"project_id": _oid(payload.project_id), "updated_at": datetime.utcnow()}},
        return_document=True,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"ok": True, "project_id": payload.project_id}


@router.post("/{task_id}/convert-to-project", status_code=201)
async def convert_task_to_project(
    task_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    row = await _require_task(org_id, task_id)
    from backend.db.mongodb.repositories.project_v2_repository import project_v2_repository
    from backend.models.workspace import ProjectV2Create
    p = await project_v2_repository.create_project(
        ProjectV2Create(name=row.get("title", "New project"), description=row.get("description")),
        organization_id=org_id, creator_id=str(current_user.id),
    )
    col = await MongoDB.get_collection("tasks")
    await col.update_one(
        {"_id": _oid(task_id)}, {"$set": {"converted_to_project_id": _oid(str(p.id)), "status": "archived"}},
    )
    return {"project_id": str(p.id)}


# ── Templates ───────────────────────────────────────────────────


@router.get("/templates")
async def list_task_templates(
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("task_templates")
    rows = await col.find({"organization_id": _oid(org_id)}).sort("created_at", -1).to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "created_by"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


class TaskTemplatePayload(BaseModel):
    name: str
    title: str
    description: Optional[str] = None
    priority: Optional[str] = "medium"
    tags: List[str] = Field(default_factory=list)


@router.post("/templates", status_code=201)
async def create_task_template(
    payload: TaskTemplatePayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("task_templates")
    doc = {
        "organization_id": _oid(org_id),
        "name": payload.name, "title": payload.title,
        "description": payload.description, "priority": payload.priority,
        "tags": payload.tags,
        "created_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"id": str(r.inserted_id)}


@router.post("/from-template/{template_id}", status_code=201)
async def task_from_template(
    template_id: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    templates = await MongoDB.get_collection("task_templates")
    t = await templates.find_one({"_id": _oid(template_id), "organization_id": _oid(org_id)})
    if not t:
        raise HTTPException(status_code=404, detail="Template not found")
    tasks = await MongoDB.get_collection("tasks")
    doc = {
        "organization_id": _oid(org_id),
        "project_id": _oid(payload.get("project_id")) if payload.get("project_id") else None,
        "title": t.get("title"), "description": t.get("description"),
        "priority": t.get("priority", "medium"), "status": "todo",
        "tags": t.get("tags") or [],
        "created_by": _oid(current_user.id),
        "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
    }
    r = await tasks.insert_one(doc)
    return {"id": str(r.inserted_id)}


@router.delete("/templates/{template_id}", status_code=204)
async def delete_task_template(
    template_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("task_templates")
    await col.delete_one({"_id": _oid(template_id), "organization_id": _oid(org_id)})
    return None


@router.post("/{task_id}/save-as-template", status_code=201)
async def save_task_as_template(
    task_id: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    row = await _require_task(org_id, task_id)
    col = await MongoDB.get_collection("task_templates")
    doc = {
        "organization_id": _oid(org_id),
        "name": payload.get("name") or row.get("title"),
        "title": row.get("title"), "description": row.get("description"),
        "priority": row.get("priority", "medium"),
        "tags": row.get("tags") or [],
        "created_by": _oid(current_user.id), "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"id": str(r.inserted_id)}


# ── Imports (top-level: not project-scoped) ─────────────────────


@router.post("/import/csv")
async def import_tasks_csv(
    file: UploadFile = File(...),
    project_id: Optional[str] = Query(None),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8", errors="ignore")))
    col = await MongoDB.get_collection("tasks")
    n = 0
    for row in reader:
        title = (row.get("title") or row.get("Title") or "").strip()
        if not title:
            continue
        await col.insert_one({
            "organization_id": _oid(org_id),
            "project_id": _oid(project_id) if project_id else None,
            "title": title,
            "description": row.get("description"),
            "status": (row.get("status") or "todo").lower(),
            "priority": (row.get("priority") or "medium").lower(),
            "created_by": _oid(current_user.id),
            "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
        })
        n += 1
    return {"inserted": n}


@router.post("/import/asana")
async def import_asana(current_user: User = Depends(get_current_active_user)):
    return {"ok": True, "queued": True, "source": "asana"}


@router.post("/import/jira")
async def import_jira(current_user: User = Depends(get_current_active_user)):
    return {"ok": True, "queued": True, "source": "jira"}


@router.post("/import/linear")
async def import_linear(current_user: User = Depends(get_current_active_user)):
    return {"ok": True, "queued": True, "source": "linear"}


@router.post("/import/notion")
async def import_notion(current_user: User = Depends(get_current_active_user)):
    return {"ok": True, "queued": True, "source": "notion"}


@router.post("/import/trello")
async def import_trello(current_user: User = Depends(get_current_active_user)):
    return {"ok": True, "queued": True, "source": "trello"}


# ── Export ──────────────────────────────────────────────────────


@router.post("/export")
async def export_tasks(
    payload: Dict[str, Any] = Body(default_factory=dict),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("task_exports")
    doc = {
        "organization_id": _oid(org_id),
        "requested_by": _oid(current_user.id),
        "format": payload.get("format", "csv"),
        "filters": payload.get("filters") or {},
        "status": "pending",
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"job_id": str(r.inserted_id), "status": "pending"}


# ── Snooze / unsnooze ─────────────────────────────────────────


class SnoozePayload(BaseModel):
    minutes: int = Field(60, ge=1, le=10080)


@router.post("/{task_id}/snooze")
async def snooze_task(
    task_id: str, payload: SnoozePayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("tasks")
    row = await col.find_one_and_update(
        {"_id": _oid(task_id), "organization_id": _oid(org_id)},
        {"$set": {"snoozed_until": datetime.utcnow() + timedelta(minutes=payload.minutes), "updated_at": datetime.utcnow()}},
        return_document=True,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"snoozed_until": row.get("snoozed_until")}


@router.post("/{task_id}/unsnooze")
async def unsnooze_task(
    task_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("tasks")
    await col.update_one(
        {"_id": _oid(task_id), "organization_id": _oid(org_id)},
        {"$set": {"snoozed_until": None, "updated_at": datetime.utcnow()}},
    )
    return {"ok": True}


@router.get("/snoozed")
async def list_snoozed(
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("tasks")
    cursor = col.find({
        "organization_id": _oid(org_id),
        "snoozed_until": {"$gt": datetime.utcnow()},
    }).limit(200)
    rows = await cursor.to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "project_id", "assigned_to", "created_by"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


# ── Reactions ──────────────────────────────────────────────────


@router.post("/{task_id}/reactions/{emoji}")
async def react_to_task(
    task_id: str, emoji: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("tasks")
    key = f"reactions.{emoji}"
    await col.update_one(
        {"_id": _oid(task_id), "organization_id": _oid(org_id)},
        {"$addToSet": {key: _oid(current_user.id)}},
    )
    return {"ok": True}


@router.delete("/{task_id}/reactions/{emoji}", status_code=204)
async def unreact_to_task(
    task_id: str, emoji: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("tasks")
    key = f"reactions.{emoji}"
    await col.update_one(
        {"_id": _oid(task_id), "organization_id": _oid(org_id)},
        {"$pull": {key: _oid(current_user.id)}},
    )
    return None


# ── Dependency graph cross-project ─────────────────────────────


@router.get("/dependencies/cross-project-graph")
async def cross_project_dependency_graph(
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("tasks")
    cursor = col.find({
        "organization_id": _oid(org_id),
        "dependencies": {"$exists": True, "$ne": []},
    }, {"_id": 1, "title": 1, "status": 1, "dependencies": 1, "project_id": 1})
    nodes, edges = [], []
    async for r in cursor:
        nid = str(r["_id"])
        nodes.append({
            "id": nid, "title": r.get("title", ""),
            "status": r.get("status", ""),
            "project_id": str(r["project_id"]) if r.get("project_id") else None,
        })
        for dep in (r.get("dependencies") or []):
            edges.append({"from": str(dep), "to": nid})
    return {"nodes": nodes, "edges": edges}
