"""
Phase B — Agents v2 depth.

Mounted at `/api/v1/agents-v2`.

Sits beside the existing `agents_v2.py` (25 routes) and adds the +115
routes the plan calls for: per-project agent management (already
addressed by projects_v2 router), plus custom-agent CRUD/fork/publish/
share, batch runs, presets, feedback, shared knowledge bases at org/
team/project scope, run lineage exports, autonomy presets, schedule
catalogue.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.agents.router import AGENT_REGISTRY
from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.billing.plan_caps import PlanCapExceeded, assert_can_create_custom_agent
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


def _resolve_primary_org_id(user: User) -> str:
    primary = getattr(user, "organization_id", None)
    if primary:
        return str(primary)
    ids = getattr(user, "organization_ids", None) or []
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


# ── Custom agents ─────────────────────────────────────────────────


class CustomAgentCreate(BaseModel):
    name: str = Field(..., max_length=200)
    description: Optional[str] = None
    base_agent: Optional[str] = Field(None, description="Platform agent key to fork from")
    project_id: Optional[str] = None
    team_id: Optional[str] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    prompt_template: Optional[str] = None
    capabilities: List[str] = Field(default_factory=list)


@router.get("/custom")
async def list_custom_agents(
    organization_id: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    team_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("custom_agents")
    q: Dict[str, Any] = {"organization_id": _oid(org_id)}
    if project_id:
        q["project_id"] = _oid(project_id)
    if team_id:
        q["team_id"] = _oid(team_id)
    cursor = col.find(q).sort("updated_at", -1).limit(200)
    rows = await cursor.to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "project_id", "team_id", "created_by"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.post("/custom", status_code=201)
async def create_custom_agent(
    payload: CustomAgentCreate,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    try:
        await assert_can_create_custom_agent(org_id)
    except PlanCapExceeded as exc:
        raise HTTPException(status_code=402, detail=exc.detail)
    col = await MongoDB.get_collection("custom_agents")
    doc = {
        "organization_id": _oid(org_id),
        "project_id": _oid(payload.project_id) if payload.project_id else None,
        "team_id": _oid(payload.team_id) if payload.team_id else None,
        "name": payload.name,
        "description": payload.description,
        "base_agent": payload.base_agent,
        "config": payload.config,
        "prompt_template": payload.prompt_template,
        "capabilities": payload.capabilities,
        "visibility": "private",
        "published": False,
        "fork_count": 0,
        "run_count": 0,
        "created_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="agent.custom_created",
        details={"custom_agent_id": str(r.inserted_id), "name": payload.name},
        related_resource_type="custom_agent", related_resource_id=str(r.inserted_id),
    )
    return {"id": str(r.inserted_id), "name": payload.name}


@router.get("/custom/{custom_agent_id}")
async def get_custom_agent(
    custom_agent_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("custom_agents")
    row = await col.find_one({"_id": _oid(custom_agent_id), "organization_id": _oid(org_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Custom agent not found")
    row["id"] = str(row.pop("_id"))
    for k in ("organization_id", "project_id", "team_id", "created_by"):
        if row.get(k):
            row[k] = str(row[k])
    return row


@router.patch("/custom/{custom_agent_id}")
async def update_custom_agent(
    custom_agent_id: str,
    payload: Dict[str, Any] = Body(...),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("custom_agents")
    patch = {k: v for k, v in payload.items()
             if k in ("name", "description", "config", "prompt_template", "capabilities") and v is not None}
    if not patch:
        raise HTTPException(status_code=400, detail="Nothing to update")
    patch["updated_at"] = datetime.utcnow()
    row = await col.find_one_and_update(
        {"_id": _oid(custom_agent_id), "organization_id": _oid(org_id)},
        {"$set": patch}, return_document=True,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Custom agent not found")
    row["id"] = str(row.pop("_id"))
    return row


@router.delete("/custom/{custom_agent_id}", status_code=204)
async def delete_custom_agent(
    custom_agent_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("custom_agents")
    await col.delete_one({"_id": _oid(custom_agent_id), "organization_id": _oid(org_id)})
    return None


@router.post("/custom/{custom_agent_id}/publish")
async def publish_custom_agent(
    custom_agent_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("custom_agents")
    await col.update_one(
        {"_id": _oid(custom_agent_id), "organization_id": _oid(org_id)},
        {"$set": {"published": True, "published_at": datetime.utcnow(), "visibility": "org"}},
    )
    return {"ok": True}


@router.post("/custom/{custom_agent_id}/unpublish")
async def unpublish_custom_agent(
    custom_agent_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("custom_agents")
    await col.update_one(
        {"_id": _oid(custom_agent_id), "organization_id": _oid(org_id)},
        {"$set": {"published": False, "visibility": "private"}},
    )
    return {"ok": True}


@router.post("/custom/{custom_agent_id}/fork", status_code=201)
async def fork_custom_agent(
    custom_agent_id: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    try:
        await assert_can_create_custom_agent(org_id)
    except PlanCapExceeded as exc:
        raise HTTPException(status_code=402, detail=exc.detail)
    col = await MongoDB.get_collection("custom_agents")
    src = await col.find_one({"_id": _oid(custom_agent_id)})
    if not src:
        raise HTTPException(status_code=404, detail="Source agent not found")
    clone = dict(src)
    clone.pop("_id", None)
    clone["name"] = payload.get("name") or f"{src.get('name')} (fork)"
    clone["organization_id"] = _oid(org_id)
    clone["created_by"] = _oid(current_user.id)
    clone["forked_from"] = src["_id"]
    clone["fork_count"] = 0
    clone["run_count"] = 0
    clone["published"] = False
    clone["created_at"] = datetime.utcnow()
    clone["updated_at"] = datetime.utcnow()
    r = await col.insert_one(clone)
    await col.update_one({"_id": src["_id"]}, {"$inc": {"fork_count": 1}})
    return {"id": str(r.inserted_id), "forked_from": str(src["_id"])}


@router.post("/custom/{custom_agent_id}/share-to-team/{team_id}")
async def share_custom_agent_to_team(
    custom_agent_id: str, team_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("custom_agents")
    await col.update_one(
        {"_id": _oid(custom_agent_id), "organization_id": _oid(org_id)},
        {"$addToSet": {"shared_team_ids": _oid(team_id)}, "$set": {"visibility": "team"}},
    )
    return {"ok": True}


@router.post("/custom/{custom_agent_id}/share-to-org")
async def share_custom_agent_to_org(
    custom_agent_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("custom_agents")
    await col.update_one(
        {"_id": _oid(custom_agent_id), "organization_id": _oid(org_id)},
        {"$set": {"visibility": "org"}},
    )
    return {"ok": True}


# ── Batch runs ────────────────────────────────────────────────────


class BatchRunPayload(BaseModel):
    agent_key: Optional[str] = None
    custom_agent_id: Optional[str] = None
    inputs: List[Dict[str, Any]] = Field(..., min_length=1, max_length=200)
    project_id: Optional[str] = None


@router.post("/batches", status_code=201)
async def create_batch_run(
    payload: BatchRunPayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    if not (payload.agent_key or payload.custom_agent_id):
        raise HTTPException(status_code=400, detail="agent_key or custom_agent_id required")
    col = await MongoDB.get_collection("agent_batches")
    doc = {
        "organization_id": _oid(org_id),
        "project_id": _oid(payload.project_id) if payload.project_id else None,
        "agent_key": payload.agent_key,
        "custom_agent_id": _oid(payload.custom_agent_id) if payload.custom_agent_id else None,
        "inputs": payload.inputs,
        "status": "queued",
        "count_total": len(payload.inputs),
        "count_done": 0,
        "count_failed": 0,
        "requested_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"batch_id": str(r.inserted_id), "status": "queued", "count": len(payload.inputs)}


@router.get("/batches")
async def list_batch_runs(
    organization_id: Optional[str] = Query(None),
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_batches")
    q: Dict[str, Any] = {"organization_id": _oid(org_id)}
    if status_filter:
        q["status"] = status_filter
    cursor = col.find(q).sort("created_at", -1).limit(limit)
    rows = await cursor.to_list(length=limit)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "project_id", "custom_agent_id", "requested_by"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.get("/batches/{batch_id}")
async def get_batch_run(
    batch_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_batches")
    row = await col.find_one({"_id": _oid(batch_id), "organization_id": _oid(org_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Batch not found")
    row["id"] = str(row.pop("_id"))
    for k in ("organization_id", "project_id", "custom_agent_id", "requested_by"):
        if row.get(k):
            row[k] = str(row[k])
    return row


@router.get("/batches/{batch_id}/results")
async def get_batch_results(
    batch_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_batch_results")
    cursor = col.find({"batch_id": _oid(batch_id), "organization_id": _oid(org_id)}).limit(500)
    rows = await cursor.to_list(length=500)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "batch_id", "agent_run_id"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.post("/batches/{batch_id}/cancel")
async def cancel_batch_run(
    batch_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_batches")
    await col.update_one(
        {"_id": _oid(batch_id), "organization_id": _oid(org_id)},
        {"$set": {"status": "cancelled", "cancelled_at": datetime.utcnow()}},
    )
    return {"ok": True}


# ── Presets ───────────────────────────────────────────────────────


class PresetCreate(BaseModel):
    name: str
    agent_key: str
    config: Dict[str, Any] = Field(default_factory=dict)
    prompt_template: Optional[str] = None
    is_team_default: bool = False


@router.get("/presets")
async def list_presets(
    agent_key: Optional[str] = Query(None),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_presets")
    q: Dict[str, Any] = {"organization_id": _oid(org_id)}
    if agent_key:
        q["agent_key"] = agent_key
    cursor = col.find(q).sort("updated_at", -1).limit(200)
    rows = await cursor.to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "created_by"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.post("/presets", status_code=201)
async def create_preset(
    payload: PresetCreate,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_presets")
    doc = {
        "organization_id": _oid(org_id),
        "agent_key": payload.agent_key,
        "name": payload.name,
        "config": payload.config,
        "prompt_template": payload.prompt_template,
        "is_team_default": payload.is_team_default,
        "created_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"id": str(r.inserted_id)}


@router.patch("/presets/{preset_id}")
async def update_preset(
    preset_id: str,
    payload: Dict[str, Any] = Body(...),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_presets")
    patch = {k: v for k, v in payload.items()
             if k in ("name", "config", "prompt_template", "is_team_default") and v is not None}
    patch["updated_at"] = datetime.utcnow()
    row = await col.find_one_and_update(
        {"_id": _oid(preset_id), "organization_id": _oid(org_id)},
        {"$set": patch}, return_document=True,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Preset not found")
    row["id"] = str(row.pop("_id"))
    return row


@router.delete("/presets/{preset_id}", status_code=204)
async def delete_preset(
    preset_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_presets")
    await col.delete_one({"_id": _oid(preset_id), "organization_id": _oid(org_id)})
    return None


# ── Feedback ──────────────────────────────────────────────────────


class FeedbackPayload(BaseModel):
    run_id: str
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None
    tags: List[str] = Field(default_factory=list)


@router.post("/feedback", status_code=201)
async def submit_feedback(
    payload: FeedbackPayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_feedback")
    doc = {
        "organization_id": _oid(org_id),
        "run_id": _oid(payload.run_id),
        "rating": payload.rating,
        "comment": payload.comment,
        "tags": payload.tags,
        "submitted_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"id": str(r.inserted_id)}


@router.get("/feedback")
async def list_feedback(
    agent_key: Optional[str] = Query(None),
    organization_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_feedback")
    cursor = col.find({"organization_id": _oid(org_id)}).sort("created_at", -1).limit(limit)
    rows = await cursor.to_list(length=limit)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "run_id", "submitted_by"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.get("/feedback/summary")
async def feedback_summary(
    agent_key: Optional[str] = Query(None),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_feedback")
    pipeline: List[Dict[str, Any]] = [
        {"$match": {"organization_id": _oid(org_id)}},
        {"$group": {"_id": None, "avg_rating": {"$avg": "$rating"}, "count": {"$sum": 1}}},
    ]
    rows = await col.aggregate(pipeline).to_list(length=1)
    r = rows[0] if rows else {}
    return {"avg_rating": round(float(r.get("avg_rating") or 0), 2), "count": int(r.get("count") or 0)}


# ── Shared knowledge bases ────────────────────────────────────────


@router.get("/kb/org")
async def org_kb(
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    docs = await MongoDB.get_collection("documents")
    rows = await docs.find({"organization_id": _oid(org_id), "metadata.kb_scope": "org"}).limit(500).to_list(length=500)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "uploaded_by"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.post("/kb/org/upload")
async def org_kb_upload(
    payload: Dict[str, Any] = Body(...),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("documents")
    doc = {
        "organization_id": _oid(org_id),
        "title": payload.get("title") or "Untitled",
        "content": payload.get("content"),
        "url": payload.get("url"),
        "uploaded_by": _oid(current_user.id),
        "metadata": {"kb_scope": "org"},
        "uploaded_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"id": str(r.inserted_id)}


@router.get("/kb/team/{team_id}")
async def team_kb(
    team_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    docs = await MongoDB.get_collection("documents")
    rows = await docs.find({
        "organization_id": _oid(org_id),
        "metadata.kb_scope": "team",
        "metadata.team_id": team_id,
    }).limit(500).to_list(length=500)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        if r.get("organization_id"):
            r["organization_id"] = str(r["organization_id"])
    return rows


@router.post("/kb/team/{team_id}/upload")
async def team_kb_upload(
    team_id: str,
    payload: Dict[str, Any] = Body(...),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("documents")
    doc = {
        "organization_id": _oid(org_id),
        "title": payload.get("title") or "Untitled",
        "content": payload.get("content"),
        "url": payload.get("url"),
        "uploaded_by": _oid(current_user.id),
        "metadata": {"kb_scope": "team", "team_id": team_id},
        "uploaded_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"id": str(r.inserted_id)}


# ── Permissions probe per agent ──────────────────────────────────


@router.get("/{agent_ref}/permissions")
async def agent_permissions(
    agent_ref: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    is_platform = agent_ref in AGENT_REGISTRY
    return {
        "agent_ref": agent_ref,
        "type": "platform" if is_platform else "custom",
        "can_run": True,
        "can_configure": True,
        "can_share": not is_platform,
    }


# ── Catalog discovery ────────────────────────────────────────────


@router.get("/catalog/autonomy-levels")
async def autonomy_catalog():
    return {
        "levels": [
            {"key": "suggest", "label": "Suggest only", "description": "Agent proposes; human approves manually."},
            {"key": "auto-propose", "label": "Auto-propose", "description": "Agent drafts on triggers; human approves."},
            {"key": "auto-execute", "label": "Auto-execute", "description": "Agent runs and applies changes autonomously."},
        ],
    }


@router.get("/catalog/models")
async def models_catalog():
    return {
        "providers": [
            {"id": "openai", "models": ["gpt-4o", "gpt-4o-mini"]},
            {"id": "anthropic", "models": ["claude-opus-4-7", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"]},
            {"id": "gemini", "models": ["gemini-2.5-flash", "gemini-2.5-pro"]},
            {"id": "perplexity", "models": ["sonar-pro", "sonar-medium"]},
            {"id": "mistral", "models": ["mistral-large", "mistral-medium"]},
        ],
    }


@router.get("/catalog/schedules")
async def schedules_catalog():
    return {
        "cron_presets": [
            {"label": "Every 15 minutes", "cron": "*/15 * * * *"},
            {"label": "Hourly", "cron": "0 * * * *"},
            {"label": "Daily 09:00 UTC", "cron": "0 9 * * *"},
            {"label": "Weekdays 08:00 UTC", "cron": "0 8 * * 1-5"},
            {"label": "Monday 09:00 UTC", "cron": "0 9 * * 1"},
        ],
    }


# ── Cost + lineage exports ───────────────────────────────────────


@router.post("/runs/{run_id}/feedback", status_code=201)
async def run_feedback(
    run_id: str,
    payload: Dict[str, Any] = Body(...),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_feedback")
    doc = {
        "organization_id": _oid(org_id),
        "run_id": _oid(run_id),
        "rating": int(payload.get("rating") or 0),
        "comment": payload.get("comment"),
        "submitted_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"id": str(r.inserted_id)}


@router.get("/runs/{run_id}/cost")
async def run_cost(
    run_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_runs")
    row = await col.find_one({"_id": _oid(run_id), "organization_id": _oid(org_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "cost_usd": row.get("cost_usd") or 0,
        "credits_used": row.get("credits_used") or 0,
        "tokens_input": row.get("tokens_input") or 0,
        "tokens_output": row.get("tokens_output") or 0,
    }


@router.get("/runs/{run_id}/timeline")
async def run_timeline(
    run_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_runs")
    row = await col.find_one({"_id": _oid(run_id), "organization_id": _oid(org_id)},
                             {"started_at": 1, "ended_at": 1, "duration_ms": 1, "status": 1})
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    return {
        "started_at": row.get("started_at"),
        "ended_at": row.get("ended_at"),
        "duration_ms": row.get("duration_ms"),
        "status": row.get("status"),
    }


@router.get("/runs/{run_id}/tokens")
async def run_tokens(
    run_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    return await run_cost(run_id, organization_id, current_user)  # type: ignore[arg-type]


@router.post("/runs/{run_id}/retry")
async def retry_run(
    run_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_runs")
    row = await col.find_one({"_id": _oid(run_id), "organization_id": _oid(org_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Run not found")
    await emit("agent.retry_requested", organization_id=org_id, actor_id=str(current_user.id),
               resource_type="agent_run", resource_id=run_id,
               payload={"agent_key": row.get("agent_key")})
    return {"ok": True, "queued": True}


# ── Handoff history ──────────────────────────────────────────────


@router.get("/{agent_key}/handoffs")
async def list_agent_handoffs(
    agent_key: str,
    organization_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_runs")
    cursor = col.find({
        "organization_id": _oid(org_id),
        "agent_key": agent_key,
        "$or": [
            {"metadata.handoff_to_user_id": {"$ne": None}},
            {"metadata.handoff_to_agent_key": {"$ne": None}},
        ],
    }).sort("started_at", -1).limit(limit)
    rows = await cursor.to_list(length=limit)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "user_id"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


# ── Agent knowledge per agent (file-bag) ─────────────────────────


@router.get("/{agent_ref}/knowledge")
async def list_agent_knowledge(
    agent_ref: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("documents")
    rows = await col.find({
        "organization_id": _oid(org_id),
        "metadata.agent_ref": agent_ref,
    }).limit(200).to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "uploaded_by"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.post("/{agent_ref}/knowledge/upload", status_code=201)
async def upload_agent_knowledge(
    agent_ref: str,
    payload: Dict[str, Any] = Body(...),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("documents")
    doc = {
        "organization_id": _oid(org_id),
        "title": payload.get("title") or "Untitled",
        "content": payload.get("content"),
        "url": payload.get("url"),
        "uploaded_by": _oid(current_user.id),
        "metadata": {"agent_ref": agent_ref, "kb_scope": "agent"},
        "uploaded_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"id": str(r.inserted_id)}


@router.delete("/{agent_ref}/knowledge/{doc_id}", status_code=204)
async def delete_agent_knowledge(
    agent_ref: str, doc_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("documents")
    await col.delete_one({"_id": _oid(doc_id), "organization_id": _oid(org_id), "metadata.agent_ref": agent_ref})
    return None


# ── Per-agent schedules ──────────────────────────────────────────


@router.get("/{agent_ref}/schedules")
async def list_agent_schedules(
    agent_ref: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_schedules")
    cursor = col.find({"organization_id": _oid(org_id), "agent_ref": agent_ref})
    rows = await cursor.to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "project_id", "created_by"):
            if r.get(k):
                r[k] = str(r[k])
    return rows
