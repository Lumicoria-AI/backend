"""
Phase A4 — Tasks extensions.

Mounted at `/api/v1/tasks-extended`.

Adds bulk operations, watchers, dependencies, and saved-views surface to
the existing tasks router without touching the production-critical
`tasks.py` file.  Each endpoint here is org-scoped and reuses the same
`tasks` collection.

Watchers and dependencies live on the task row itself:
    task.watchers[]       array of user_ids
    task.dependencies[]   array of task_ids that must complete first
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.event_bus import emit

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


def _resolve_org_id(current_user: User) -> str:
    primary = getattr(current_user, "organization_id", None)
    if primary:
        return str(primary)
    ids = getattr(current_user, "organization_ids", None) or []
    if ids:
        return str(ids[0])
    raise HTTPException(status_code=400, detail="User has no organization context")


async def _require_org_member(org_id: str, current_user: User):
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(m) for m in (org.member_ids or [])]:
        raise HTTPException(status_code=403, detail="Not a member of this organization")
    return org


def _serialize_task(row: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    row["id"] = str(row.pop("_id", row.get("id")))
    for k in ("organization_id", "created_by", "assigned_to", "project_id", "parent_task_id",
              "agent_id", "calendar_event_id", "invite_id"):
        if row.get(k) is not None:
            row[k] = str(row[k])
    for k in ("watchers", "dependencies"):
        if isinstance(row.get(k), list):
            row[k] = [str(v) for v in row[k] if v is not None]
    return row


# ── Bulk ─────────────────────────────────────────────────────────────


class BulkUpdatePayload(BaseModel):
    task_ids: List[str] = Field(..., min_length=1, max_length=500)
    patch: Dict[str, Any] = Field(default_factory=dict)


@router.post("/bulk-update")
async def bulk_update_tasks(
    payload: BulkUpdatePayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("tasks")
    oids = [_oid(t) for t in payload.task_ids if _oid(t)]
    if not oids:
        return {"modified": 0}
    patch = dict(payload.patch)
    patch["updated_at"] = datetime.utcnow()
    result = await col.update_many(
        {"_id": {"$in": oids}, "organization_id": _oid(org_id)},
        {"$set": patch},
    )
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="task.bulk_updated",
        details={"count": result.modified_count, "fields": list(payload.patch.keys())},
        related_resource_type="task", related_resource_id="bulk",
    )
    return {"modified": result.modified_count}


class BulkAssignPayload(BaseModel):
    task_ids: List[str] = Field(..., min_length=1, max_length=500)
    assigned_to: Optional[str] = None
    assigned_to_agent: Optional[str] = None


@router.post("/bulk-assign")
async def bulk_assign_tasks(
    payload: BulkAssignPayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("tasks")
    set_doc: Dict[str, Any] = {"updated_at": datetime.utcnow()}
    if payload.assigned_to:
        set_doc["assigned_to"] = _oid(payload.assigned_to)
    if payload.assigned_to_agent:
        set_doc["assigned_to_agent"] = payload.assigned_to_agent
    oids = [_oid(t) for t in payload.task_ids if _oid(t)]
    result = await col.update_many(
        {"_id": {"$in": oids}, "organization_id": _oid(org_id)},
        {"$set": set_doc},
    )
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="task.bulk_assigned",
        details={
            "count": result.modified_count,
            "assigned_to": payload.assigned_to,
            "assigned_to_agent": payload.assigned_to_agent,
        },
        related_resource_type="task", related_resource_id="bulk",
    )
    return {"modified": result.modified_count}


class BulkArchivePayload(BaseModel):
    task_ids: List[str] = Field(..., min_length=1, max_length=500)


@router.post("/bulk-archive")
async def bulk_archive_tasks(
    payload: BulkArchivePayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("tasks")
    oids = [_oid(t) for t in payload.task_ids if _oid(t)]
    result = await col.update_many(
        {"_id": {"$in": oids}, "organization_id": _oid(org_id)},
        {"$set": {"status": "archived", "updated_at": datetime.utcnow()}},
    )
    return {"modified": result.modified_count}


# ── Watchers ─────────────────────────────────────────────────────────


@router.get("/{task_id}/watchers")
async def list_watchers(
    task_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("tasks")
    row = await col.find_one(
        {"_id": _oid(task_id), "organization_id": _oid(org_id)},
        {"watchers": 1},
    )
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"watchers": [str(w) for w in (row.get("watchers") or [])]}


@router.post("/{task_id}/watch")
async def watch_task(
    task_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("tasks")
    result = await col.update_one(
        {"_id": _oid(task_id), "organization_id": _oid(org_id)},
        {"$addToSet": {"watchers": _oid(current_user.id)},
         "$set": {"updated_at": datetime.utcnow()}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"ok": True}


@router.delete("/{task_id}/watch", status_code=204)
async def unwatch_task(
    task_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("tasks")
    await col.update_one(
        {"_id": _oid(task_id), "organization_id": _oid(org_id)},
        {"$pull": {"watchers": _oid(current_user.id)},
         "$set": {"updated_at": datetime.utcnow()}},
    )
    return None


# ── Dependencies ─────────────────────────────────────────────────────


class DependencyPayload(BaseModel):
    depends_on_task_id: str


@router.get("/{task_id}/dependencies")
async def list_dependencies(
    task_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("tasks")
    row = await col.find_one(
        {"_id": _oid(task_id), "organization_id": _oid(org_id)},
        {"dependencies": 1},
    )
    if not row:
        raise HTTPException(status_code=404, detail="Task not found")
    return {"dependencies": [str(d) for d in (row.get("dependencies") or [])]}


@router.post("/{task_id}/dependencies", status_code=201)
async def add_dependency(
    task_id: str,
    payload: DependencyPayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    if task_id == payload.depends_on_task_id:
        raise HTTPException(status_code=400, detail="A task cannot depend on itself")
    col = await MongoDB.get_collection("tasks")
    # Verify both tasks exist in this org.
    found = await col.count_documents({
        "_id": {"$in": [_oid(task_id), _oid(payload.depends_on_task_id)]},
        "organization_id": _oid(org_id),
    })
    if found < 2:
        raise HTTPException(status_code=404, detail="One or both tasks not found")
    await col.update_one(
        {"_id": _oid(task_id), "organization_id": _oid(org_id)},
        {"$addToSet": {"dependencies": _oid(payload.depends_on_task_id)},
         "$set": {"updated_at": datetime.utcnow()}},
    )
    return {"ok": True}


@router.delete("/{task_id}/dependencies/{depends_on_id}", status_code=204)
async def remove_dependency(
    task_id: str,
    depends_on_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("tasks")
    await col.update_one(
        {"_id": _oid(task_id), "organization_id": _oid(org_id)},
        {"$pull": {"dependencies": _oid(depends_on_id)},
         "$set": {"updated_at": datetime.utcnow()}},
    )
    return None


@router.get("/dependencies/graph")
async def dependency_graph(
    project_id: str = Query(..., max_length=64),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("tasks")
    cursor = col.find(
        {"organization_id": _oid(org_id), "project_id": _oid(project_id)},
        {"_id": 1, "title": 1, "status": 1, "dependencies": 1},
    )
    nodes: List[Dict[str, Any]] = []
    edges: List[Dict[str, Any]] = []
    async for row in cursor:
        nid = str(row["_id"])
        nodes.append({"id": nid, "title": row.get("title", ""), "status": row.get("status", "")})
        for dep in (row.get("dependencies") or []):
            edges.append({"from": str(dep), "to": nid})
    return {"nodes": nodes, "edges": edges}


# ── Saved views (lightweight, stored under task_views collection) ────


class SavedViewCreate(BaseModel):
    name: str = Field(..., max_length=120)
    kind: str = Field("list", description="list | board | calendar | gantt | timeline")
    filters: Dict[str, Any] = Field(default_factory=dict)
    sort: Dict[str, Any] = Field(default_factory=dict)


@router.get("/saved-views")
async def list_saved_views(
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("task_saved_views")
    cursor = col.find({"organization_id": _oid(org_id), "user_id": _oid(current_user.id)})
    rows = await cursor.to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "user_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


@router.post("/saved-views", status_code=201)
async def create_saved_view(
    payload: SavedViewCreate,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("task_saved_views")
    doc = {
        "organization_id": _oid(org_id),
        "user_id": _oid(current_user.id),
        "name": payload.name,
        "kind": payload.kind,
        "filters": payload.filters,
        "sort": payload.sort,
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    result = await col.insert_one(doc)
    doc["id"] = str(result.inserted_id)
    for k in ("organization_id", "user_id"):
        if doc.get(k) is not None:
            doc[k] = str(doc[k])
    doc.pop("_id", None)
    return doc


@router.delete("/saved-views/{view_id}", status_code=204)
async def delete_saved_view(
    view_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("task_saved_views")
    await col.delete_one({
        "_id": _oid(view_id),
        "organization_id": _oid(org_id),
        "user_id": _oid(current_user.id),
    })
    return None
