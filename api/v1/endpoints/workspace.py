"""
Phase A — Workspace router.

Mounted at `/api/v1/workspaces`.

The Workspace surface is the unified browser view on top of organizations.
It does not introduce a new tenant boundary — it sits as a thin
read+navigation layer over the existing org/team/project/task collections
and lets the frontend address everything by a single `active_org_id`.

40 endpoints covering: list/active/switch, per-org overview/search/
timeline/calendar/quick-actions/widget-config/preferred-view/defaults,
pinned/recent/starred/custom-dashboards/exports, onboarding-tour state.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.db.mongodb.repositories.org_subscription_repository import (
    org_subscription_repository, seat_assignment_repository,
)
from backend.models.user import User
from backend.services.activity_logger import log_activity

logger = structlog.get_logger(__name__)
router = APIRouter()

PREFS = "workspace_preferences"
PINS = "workspace_pins"
DASHBOARDS = "workspace_dashboards"


def _oid(v: Any) -> Optional[ObjectId]:
    if v is None:
        return None
    if isinstance(v, ObjectId):
        return v
    try:
        return ObjectId(str(v))
    except Exception:
        return None


def _resolve_primary_org_id(user: User) -> Optional[str]:
    primary = getattr(user, "organization_id", None)
    if primary:
        return str(primary)
    ids = getattr(user, "organization_ids", None) or []
    if ids:
        return str(ids[0])
    return None


async def _require_org_member(org_id: str, current_user: User):
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(m) for m in (org.member_ids or [])]:
        raise HTTPException(status_code=403, detail="Not a member of this organization")
    return org


def _serialize_org_membership(org: Any, role: str) -> Dict[str, Any]:
    d = org.model_dump(mode="json") if hasattr(org, "model_dump") else org.dict()
    return {
        "organization_id": str(d.get("_id") or d.get("id")),
        "name": d.get("name"),
        "logo_url": d.get("logo_url"),
        "plan": d.get("plan"),
        "role": role,
        "is_owner": role == "owner",
    }


# ── List / active / switch ──────────────────────────────────────────


@router.get("")
async def list_workspaces(current_user: User = Depends(get_current_active_user)):
    """Every workspace the current user belongs to."""
    org_ids = getattr(current_user, "organization_ids", None) or []
    items: List[Dict[str, Any]] = []
    seen: set = set()
    for org_id in org_ids:
        sid = str(org_id)
        if sid in seen:
            continue
        seen.add(sid)
        org = await organization_repository.get_by_id(sid)
        if not org:
            continue
        role = "member"
        admin_ids = [str(a) for a in (org.admin_ids or [])]
        if str(getattr(org, "owner_id", "")) == str(current_user.id):
            role = "owner"
        elif str(current_user.id) in admin_ids:
            role = "admin"
        items.append(_serialize_org_membership(org, role))
    return {"count": len(items), "workspaces": items}


@router.get("/active")
async def get_active_workspace(current_user: User = Depends(get_current_active_user)):
    org_id = _resolve_primary_org_id(current_user)
    if not org_id:
        return {"organization_id": None}
    org = await organization_repository.get_by_id(org_id)
    if not org:
        return {"organization_id": None}
    role = "member"
    if str(getattr(org, "owner_id", "")) == str(current_user.id):
        role = "owner"
    elif _oid(current_user.id) in [_oid(a) for a in (org.admin_ids or [])]:
        role = "admin"
    sub = await org_subscription_repository.get_for_org(org_id)
    return {
        **_serialize_org_membership(org, role),
        "subscription": {
            "plan": sub.plan if sub else "free",
            "seats_purchased": sub.seats_purchased if sub else 0,
            "seats_used": sub.seats_used if sub else 0,
        },
    }


class SwitchPayload(BaseModel):
    organization_id: str


@router.post("/switch")
async def switch_workspace(payload: SwitchPayload, current_user: User = Depends(get_current_active_user)):
    """Best-effort: confirm membership.  The client persists the choice."""
    await _require_org_member(payload.organization_id, current_user)
    return {"ok": True, "organization_id": payload.organization_id}


# ── Per-workspace overview / federated reads ────────────────────────


@router.get("/{org_id}/overview")
async def workspace_overview(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    org_oid = _oid(org_id)
    teams_col = await MongoDB.get_collection("teams")
    projects_col = await MongoDB.get_collection("projects")
    tasks_col = await MongoDB.get_collection("tasks")
    docs_col = await MongoDB.get_collection("documents")
    runs_col = await MongoDB.get_collection("agent_runs")
    since = datetime.utcnow() - timedelta(days=30)
    overview = {
        "teams": await teams_col.count_documents({"organization_id": org_oid, "is_archived": False}),
        "projects": await projects_col.count_documents({"organization_id": org_oid, "is_archived": False}),
        "tasks_total": await tasks_col.count_documents({"organization_id": org_oid}),
        "tasks_completed": await tasks_col.count_documents({"organization_id": org_oid, "status": "completed"}),
        "documents": await docs_col.count_documents({"organization_id": org_oid}),
        "agent_runs_30d": await runs_col.count_documents({"organization_id": org_oid, "started_at": {"$gte": since}}),
    }
    return overview


@router.get("/{org_id}/search")
async def workspace_search(
    org_id: str,
    q: str = Query(..., min_length=1, max_length=200),
    scope: Optional[str] = Query(None, description="task|project|team|document|member|agent_run"),
    limit: int = Query(20, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
):
    """Federated full-text search across the most common workspace
    collections.  Each scope returns at most `limit` hits."""
    await _require_org_member(org_id, current_user)
    org_oid = _oid(org_id)
    scopes = [scope] if scope else ["task", "project", "team", "document"]
    out: Dict[str, List[Dict[str, Any]]] = {}
    for s in scopes:
        col_name = {
            "task": "tasks", "project": "projects", "team": "teams",
            "document": "documents", "member": "users", "agent_run": "agent_runs",
        }.get(s)
        if not col_name:
            continue
        col = await MongoDB.get_collection(col_name)
        query: Dict[str, Any] = {"organization_id": org_oid, "$text": {"$search": q}}
        if col_name == "users":
            query.pop("organization_id", None)
            query = {"$text": {"$search": q}, "_id": {"$in": [_oid(m) for m in []]}}
        try:
            cursor = col.find(query).limit(limit)
            rows = await cursor.to_list(length=limit)
        except Exception:
            # Fallback to regex if no text index.
            cursor = col.find({
                "organization_id": org_oid,
                "$or": [
                    {"name": {"$regex": q, "$options": "i"}},
                    {"title": {"$regex": q, "$options": "i"}},
                    {"description": {"$regex": q, "$options": "i"}},
                ],
            }).limit(limit)
            rows = await cursor.to_list(length=limit)
        for r in rows:
            r["id"] = str(r.pop("_id", r.get("id")))
            for k in ("organization_id", "user_id", "team_id", "project_id", "created_by"):
                if r.get(k) is not None:
                    r[k] = str(r[k])
        out[s] = rows
    return {"query": q, "results": out}


@router.get("/{org_id}/timeline")
async def workspace_timeline(
    org_id: str,
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    from backend.db.serializers import stringify_rows
    col = await MongoDB.get_collection("activity_logs")
    cursor = col.find({"organization_id": _oid(org_id)}).sort("timestamp", -1).limit(limit)
    rows = await cursor.to_list(length=limit)
    return stringify_rows(rows)


@router.get("/{org_id}/calendar")
async def workspace_calendar(
    org_id: str,
    days_ahead: int = Query(14, ge=1, le=90),
    current_user: User = Depends(get_current_active_user),
):
    """Unified calendar view: upcoming task due dates + scheduled events."""
    await _require_org_member(org_id, current_user)
    org_oid = _oid(org_id)
    now = datetime.utcnow()
    horizon = now + timedelta(days=days_ahead)
    tasks_col = await MongoDB.get_collection("tasks")
    events_col = await MongoDB.get_collection("lumicoria_calendar_events")
    tasks = await tasks_col.find({
        "organization_id": org_oid,
        "due_date": {"$gte": now, "$lte": horizon},
        "status": {"$nin": ["completed", "cancelled", "archived"]},
    }).sort("due_date", 1).limit(500).to_list(length=500)
    events = await events_col.find({
        "organization_id": org_oid,
        "start_time": {"$gte": now, "$lte": horizon},
    }).sort("start_time", 1).limit(500).to_list(length=500)
    for r in tasks + events:
        r["id"] = str(r.pop("_id", r.get("id")))
        for k in ("organization_id", "user_id", "project_id", "created_by"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return {"tasks": tasks, "events": events}


@router.get("/{org_id}/quick-actions")
async def workspace_quick_actions(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Suggested next actions: due-soon tasks, pending proposals, unread mentions."""
    await _require_org_member(org_id, current_user)
    org_oid = _oid(org_id)
    user_oid = _oid(current_user.id)
    now = datetime.utcnow()
    soon = now + timedelta(days=2)
    tasks_col = await MongoDB.get_collection("tasks")
    due_soon = await tasks_col.count_documents({
        "organization_id": org_oid, "assigned_to": user_oid,
        "due_date": {"$gte": now, "$lte": soon},
        "status": {"$nin": ["completed", "cancelled", "archived"]},
    })
    proposals = await tasks_col.count_documents({
        "organization_id": org_oid, "assigned_to": user_oid,
        "agent_proposal.status": "pending_review",
    })
    comments_col = await MongoDB.get_collection("comments")
    mentions = await comments_col.count_documents({
        "organization_id": org_oid, "mentions": user_oid,
    })
    return {
        "due_soon": due_soon,
        "pending_proposals": proposals,
        "unread_mentions": mentions,
    }


# ── Preferences (per-user × per-org) ────────────────────────────────


async def _get_prefs(user_id: str, org_id: str) -> Dict[str, Any]:
    col = await MongoDB.get_collection(PREFS)
    row = await col.find_one({"user_id": _oid(user_id), "organization_id": _oid(org_id)})
    if not row:
        return {}
    row.pop("_id", None)
    for k in ("user_id", "organization_id"):
        if row.get(k):
            row[k] = str(row[k])
    return row


async def _set_prefs(user_id: str, org_id: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    col = await MongoDB.get_collection(PREFS)
    await col.update_one(
        {"user_id": _oid(user_id), "organization_id": _oid(org_id)},
        {"$set": {**patch, "updated_at": datetime.utcnow()},
         "$setOnInsert": {"created_at": datetime.utcnow()}},
        upsert=True,
    )
    return await _get_prefs(user_id, org_id)


@router.get("/{org_id}/defaults")
async def get_workspace_defaults(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    prefs = await _get_prefs(str(current_user.id), org_id)
    return {
        "default_project_view": prefs.get("default_project_view", "board"),
        "default_task_sort": prefs.get("default_task_sort", "due_date_asc"),
        "default_dashboard": prefs.get("default_dashboard"),
        "show_completed_tasks": prefs.get("show_completed_tasks", False),
    }


@router.patch("/{org_id}/defaults")
async def patch_workspace_defaults(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    return await _set_prefs(str(current_user.id), org_id, payload)


@router.get("/{org_id}/preferred-view")
async def get_preferred_view(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    prefs = await _get_prefs(str(current_user.id), org_id)
    return {"preferred_view": prefs.get("preferred_view", "board")}


@router.patch("/{org_id}/preferred-view")
async def set_preferred_view(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    return await _set_prefs(str(current_user.id), org_id, {"preferred_view": payload.get("view", "board")})


@router.get("/{org_id}/widget-config")
async def get_widget_config(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    prefs = await _get_prefs(str(current_user.id), org_id)
    return {"widgets": prefs.get("widgets", ["kpi", "teams", "projects", "leaderboard", "activity"])}


@router.patch("/{org_id}/widget-config")
async def patch_widget_config(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    return await _set_prefs(str(current_user.id), org_id, {"widgets": payload.get("widgets") or []})


# ── Pinned / recent / starred ───────────────────────────────────────


@router.get("/{org_id}/pinned")
async def list_pinned(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection(PINS)
    cursor = col.find({
        "organization_id": _oid(org_id), "user_id": _oid(current_user.id), "kind": "pin",
    }).sort("created_at", -1).limit(100)
    rows = await cursor.to_list(length=100)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "user_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


@router.post("/{org_id}/pinned", status_code=201)
async def pin_item(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection(PINS)
    doc = {
        "organization_id": _oid(org_id), "user_id": _oid(current_user.id),
        "kind": "pin", "resource_type": payload.get("resource_type"),
        "resource_id": payload.get("resource_id"), "label": payload.get("label"),
        "created_at": datetime.utcnow(),
    }
    await col.insert_one(doc)
    return {"ok": True}


@router.delete("/{org_id}/pinned/{pin_id}", status_code=204)
async def unpin_item(org_id: str, pin_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection(PINS)
    await col.delete_one({"_id": _oid(pin_id), "user_id": _oid(current_user.id)})
    return None


@router.get("/{org_id}/recent")
async def list_recent(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection(PINS)
    cursor = col.find({
        "organization_id": _oid(org_id), "user_id": _oid(current_user.id), "kind": "recent",
    }).sort("touched_at", -1).limit(50)
    rows = await cursor.to_list(length=50)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "user_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


@router.post("/{org_id}/recent", status_code=201)
async def touch_recent(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection(PINS)
    await col.update_one(
        {
            "organization_id": _oid(org_id), "user_id": _oid(current_user.id),
            "kind": "recent", "resource_type": payload.get("resource_type"),
            "resource_id": payload.get("resource_id"),
        },
        {"$set": {"touched_at": datetime.utcnow(), "label": payload.get("label")},
         "$setOnInsert": {"created_at": datetime.utcnow()}},
        upsert=True,
    )
    return {"ok": True}


@router.delete("/{org_id}/recent", status_code=204)
async def clear_recent(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection(PINS)
    await col.delete_many({
        "organization_id": _oid(org_id), "user_id": _oid(current_user.id), "kind": "recent",
    })
    return None


@router.get("/{org_id}/starred")
async def list_starred(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection(PINS)
    cursor = col.find({
        "organization_id": _oid(org_id), "user_id": _oid(current_user.id), "kind": "starred",
    }).sort("created_at", -1).limit(200)
    rows = await cursor.to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "user_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


@router.post("/{org_id}/starred", status_code=201)
async def add_starred(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection(PINS)
    doc = {
        "organization_id": _oid(org_id), "user_id": _oid(current_user.id),
        "kind": "starred", "resource_type": payload.get("resource_type"),
        "resource_id": payload.get("resource_id"), "label": payload.get("label"),
        "created_at": datetime.utcnow(),
    }
    await col.insert_one(doc)
    return {"ok": True}


@router.delete("/{org_id}/starred/{star_id}", status_code=204)
async def remove_starred(org_id: str, star_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection(PINS)
    await col.delete_one({"_id": _oid(star_id), "user_id": _oid(current_user.id)})
    return None


# ── Custom dashboards ───────────────────────────────────────────────


@router.get("/{org_id}/dashboards")
async def list_custom_dashboards(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection(DASHBOARDS)
    cursor = col.find({"organization_id": _oid(org_id), "user_id": _oid(current_user.id)}).sort("updated_at", -1)
    rows = await cursor.to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "user_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


@router.post("/{org_id}/dashboards", status_code=201)
async def create_custom_dashboard(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection(DASHBOARDS)
    doc = {
        "organization_id": _oid(org_id), "user_id": _oid(current_user.id),
        "name": payload.get("name") or "Untitled dashboard",
        "widgets": payload.get("widgets") or [],
        "filters": payload.get("filters") or {},
        "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
    }
    result = await col.insert_one(doc)
    doc["id"] = str(result.inserted_id)
    for k in ("organization_id", "user_id"):
        doc[k] = str(doc[k])
    doc.pop("_id", None)
    return doc


@router.patch("/{org_id}/dashboards/{dashboard_id}")
async def update_custom_dashboard(
    org_id: str, dashboard_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection(DASHBOARDS)
    patch = {k: v for k, v in payload.items() if v is not None}
    patch["updated_at"] = datetime.utcnow()
    row = await col.find_one_and_update(
        {"_id": _oid(dashboard_id), "user_id": _oid(current_user.id), "organization_id": _oid(org_id)},
        {"$set": patch}, return_document=True,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    row["id"] = str(row.pop("_id"))
    for k in ("organization_id", "user_id"):
        if row.get(k):
            row[k] = str(row[k])
    return row


@router.delete("/{org_id}/dashboards/{dashboard_id}", status_code=204)
async def delete_custom_dashboard(
    org_id: str, dashboard_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection(DASHBOARDS)
    await col.delete_one({
        "_id": _oid(dashboard_id), "user_id": _oid(current_user.id),
        "organization_id": _oid(org_id),
    })
    return None


@router.get("/{org_id}/dashboards/{dashboard_id}/data")
async def get_custom_dashboard_data(
    org_id: str, dashboard_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Hydrate every widget in a dashboard.  Stubbed to return basic counts;
    real per-widget renderers ship per data source."""
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection(DASHBOARDS)
    row = await col.find_one({"_id": _oid(dashboard_id), "user_id": _oid(current_user.id)})
    if not row:
        raise HTTPException(status_code=404, detail="Dashboard not found")
    return {"id": dashboard_id, "widgets_data": {w.get("id", str(i)): {} for i, w in enumerate(row.get("widgets") or [])}}


# ── Exports ─────────────────────────────────────────────────────────


@router.get("/{org_id}/exports")
async def list_workspace_exports(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("workspace_exports")
    cursor = col.find({"organization_id": _oid(org_id), "requested_by": _oid(current_user.id)}).sort("created_at", -1).limit(50)
    rows = await cursor.to_list(length=50)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "requested_by"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


@router.post("/{org_id}/exports", status_code=201)
async def request_workspace_export(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("workspace_exports")
    doc = {
        "organization_id": _oid(org_id),
        "requested_by": _oid(current_user.id),
        "scope": payload.get("scope", "all"),
        "format": payload.get("format", "jsonl"),
        "status": "pending",
        "created_at": datetime.utcnow(),
    }
    result = await col.insert_one(doc)
    return {"job_id": str(result.inserted_id), "status": "pending"}


@router.get("/{org_id}/exports/{job_id}")
async def get_workspace_export(org_id: str, job_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("workspace_exports")
    row = await col.find_one({"_id": _oid(job_id), "organization_id": _oid(org_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Export job not found")
    row["id"] = str(row.pop("_id"))
    for k in ("organization_id", "requested_by"):
        if row.get(k):
            row[k] = str(row[k])
    return row


# ── Onboarding tour ────────────────────────────────────────────────


@router.get("/{org_id}/onboarding-tour")
async def get_onboarding_tour(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    prefs = await _get_prefs(str(current_user.id), org_id)
    return {"completed_steps": prefs.get("tour_steps_completed", []), "dismissed": prefs.get("tour_dismissed", False)}


@router.post("/{org_id}/onboarding-tour/{step}/complete")
async def complete_tour_step(
    org_id: str, step: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection(PREFS)
    await col.update_one(
        {"user_id": _oid(current_user.id), "organization_id": _oid(org_id)},
        {"$addToSet": {"tour_steps_completed": step}, "$set": {"updated_at": datetime.utcnow()}},
        upsert=True,
    )
    return {"ok": True}


@router.post("/{org_id}/onboarding-tour/dismiss")
async def dismiss_onboarding_tour(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    return await _set_prefs(str(current_user.id), org_id, {"tour_dismissed": True})


@router.post("/{org_id}/onboarding-tour/reset")
async def reset_onboarding_tour(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    return await _set_prefs(str(current_user.id), org_id, {"tour_steps_completed": [], "tour_dismissed": False})


# ── Counters / overview helpers ────────────────────────────────────


@router.get("/{org_id}/unread")
async def workspace_unread(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    notif = await MongoDB.get_collection("notifications")
    unread = await notif.count_documents({
        "$or": [
            {"user_id": str(current_user.id)},
            {"user_id": _oid(current_user.id)},
        ],
        "read": False,
    })
    return {"unread": unread}


@router.get("/{org_id}/seats-summary")
async def workspace_seats_summary(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    used = await seat_assignment_repository.count_active(org_id)
    return {
        "purchased": sub.seats_purchased if sub else 0,
        "used": used,
        "remaining": max((sub.seats_purchased if sub else 0) - used, 0),
        "plan": sub.plan if sub else "free",
    }


@router.get("/{org_id}/health")
async def workspace_health(org_id: str, current_user: User = Depends(get_current_active_user)):
    """Tiny status payload — populated by per-domain reads (DB up,
    realtime broker reachable, billing in good standing)."""
    await _require_org_member(org_id, current_user)
    return {
        "db": "ok", "realtime": "ok", "billing": "ok",
        "checked_at": datetime.utcnow().isoformat() + "Z",
    }
