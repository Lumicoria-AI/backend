"""
Phase B — Agents v2 collaboration surface.

Mounted at `/api/v1/agents-v2`.

Layers on top of the existing `agents.router` (the 21 platform agents
catalogue) without replacing it.  This module focuses on the *collaboration*
surface: per-org / per-team / per-project metrics, leaderboards, run history
search, schedules, handoffs, autonomy controls, batches, and platform-agent
introspection.

Per-project activation lives on `projects_v2.router` (see Phase A2).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.agents.router import AGENT_REGISTRY
from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.agent_metrics_repository import agent_metrics_repository
from backend.db.mongodb.repositories.agent_run_repository import agent_run_repository
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.db.mongodb.repositories.project_v2_repository import project_v2_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.billing.plan_caps import PlanCapExceeded, assert_can_run_agent
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


# ── Platform agent catalogue ─────────────────────────────────────────


@router.get("/platform")
async def list_platform_agents():
    """Public catalogue of the 21 platform agents.  Open to any caller —
    every plan can run these subject to monthly run caps."""
    out: List[Dict[str, Any]] = []
    for key, meta in AGENT_REGISTRY.items():
        if isinstance(meta, dict):
            out.append({
                "key": key,
                "name": meta.get("name") or key.replace("_", " ").title(),
                "description": meta.get("description"),
                "capabilities": meta.get("capabilities") or [],
                "default_model": meta.get("default_model"),
            })
        else:
            out.append({
                "key": key,
                "name": key.replace("_", " ").title(),
                "description": None,
                "capabilities": [],
            })
    return {"count": len(out), "agents": out}


@router.get("/platform/{agent_key}")
async def get_platform_agent(agent_key: str):
    meta = AGENT_REGISTRY.get(agent_key)
    if meta is None:
        raise HTTPException(status_code=404, detail="Unknown agent key")
    base = {"key": agent_key}
    if isinstance(meta, dict):
        base.update(meta)
    return base


# ── Metrics + leaderboards ───────────────────────────────────────────


@router.get("/metrics")
async def get_agent_metrics(
    organization_id: Optional[str] = Query(None),
    agent_key: Optional[str] = Query(None),
    custom_agent_id: Optional[str] = Query(None),
    team_id: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    window: str = Query("month", description="day|week|month|all"),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    return await agent_metrics_repository.get(
        organization_id=org_id,
        agent_key=agent_key,
        custom_agent_id=custom_agent_id,
        team_id=team_id,
        project_id=project_id,
        window=window,
    )


@router.get("/leaderboard")
async def agent_leaderboard(
    organization_id: Optional[str] = Query(None),
    window: str = Query("month"),
    limit: int = Query(25, ge=1, le=100),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    return await agent_metrics_repository.leaderboard(
        organization_id=org_id, window=window, limit=limit,
    )


@router.post("/metrics/rebuild")
async def rebuild_agent_metrics(
    organization_id: Optional[str] = Query(None),
    window: str = Query("month"),
    current_user: User = Depends(get_current_active_user),
):
    """On-demand metrics rebuild.  Normally driven by the Celery beat
    `materialise-agent-metrics` job; this endpoint lets admins force it."""
    org_id = organization_id or _resolve_org_id(current_user)
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(a) for a in (org.admin_ids or [])]:
        raise HTTPException(status_code=403, detail="Org admin permission required")
    n = await agent_metrics_repository.rebuild(organization_id=org_id, window=window)
    return {"window": window, "rows_written": n}


@router.get("/cost-breakdown")
async def agent_cost_breakdown(
    organization_id: Optional[str] = Query(None),
    window: str = Query("month"),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    rows = await agent_metrics_repository.get(organization_id=org_id, window=window)
    by_agent: Dict[str, Dict[str, Any]] = {}
    total_credits = 0
    for r in rows:
        key = r.get("agent_key") or r.get("custom_agent_id") or "unknown"
        slot = by_agent.setdefault(key, {"agent_key": r.get("agent_key"),
                                         "custom_agent_id": r.get("custom_agent_id"),
                                         "cost_usd": 0.0, "credits_used": 0, "runs": 0})
        slot["cost_usd"] += float(r.get("cost_usd") or 0)
        slot["credits_used"] += int(r.get("credits_used") or 0)
        slot["runs"] += int(r.get("runs") or 0)
        total_credits += int(r.get("credits_used") or 0)
    items = sorted(by_agent.values(), key=lambda x: x["credits_used"], reverse=True)
    for it in items:
        it["cost_usd"] = round(it["cost_usd"], 4)
        it["share_pct"] = round((it["credits_used"] / total_credits * 100) if total_credits else 0.0, 2)
    return {"window": window, "total_credits": total_credits, "by_agent": items}


@router.get("/token-burn")
async def agent_token_burn(
    organization_id: Optional[str] = Query(None),
    window: str = Query("month"),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    rows = await agent_metrics_repository.get(organization_id=org_id, window=window)
    tokens_in = sum(int(r.get("tokens_in") or 0) for r in rows)
    tokens_out = sum(int(r.get("tokens_out") or 0) for r in rows)
    return {"window": window, "tokens_in": tokens_in, "tokens_out": tokens_out,
            "total": tokens_in + tokens_out}


@router.get("/error-rate")
async def agent_error_rate(
    organization_id: Optional[str] = Query(None),
    window: str = Query("month"),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    rows = await agent_metrics_repository.get(organization_id=org_id, window=window)
    by_agent: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        key = r.get("agent_key") or r.get("custom_agent_id") or "unknown"
        slot = by_agent.setdefault(key, {"runs": 0, "errors": 0, "agent_key": r.get("agent_key"),
                                         "custom_agent_id": r.get("custom_agent_id")})
        slot["runs"] += int(r.get("runs") or 0)
        slot["errors"] += int(r.get("errors") or 0)
    items = list(by_agent.values())
    for it in items:
        it["error_rate"] = round((it["errors"] / it["runs"]) if it["runs"] else 0.0, 4)
    return sorted(items, key=lambda x: x["error_rate"], reverse=True)


# ── Run history + lineage ────────────────────────────────────────────


@router.get("/runs")
async def list_agent_runs(
    organization_id: Optional[str] = Query(None),
    agent_key: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_runs")
    query: Dict[str, Any] = {"organization_id": _oid(org_id)}
    if agent_key:
        query["agent_key"] = agent_key
    if status:
        query["status"] = status
    if project_id:
        query["project_id"] = _oid(project_id)
    if user_id:
        query["user_id"] = _oid(user_id)
    cursor = col.find(query).sort("started_at", -1).skip(skip).limit(limit)
    rows = await cursor.to_list(length=limit)
    for r in rows:
        r["id"] = str(r.pop("_id", r.get("id")))
        for k in ("organization_id", "user_id", "task_id", "project_id", "parent_run_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


@router.get("/runs/{run_id}")
async def get_run(
    run_id: str,
    current_user: User = Depends(get_current_active_user),
):
    run = await agent_run_repository.get_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    org_id = str(run.organization_id) if run.organization_id else None
    if not org_id:
        raise HTTPException(status_code=404, detail="Run is not org-scoped")
    await _require_org_member(org_id, current_user)
    return run.model_dump(mode="json") if hasattr(run, "model_dump") else run.dict()


@router.get("/runs/{run_id}/lineage")
async def get_run_lineage(
    run_id: str,
    current_user: User = Depends(get_current_active_user),
):
    parent = await agent_run_repository.get_by_id(run_id)
    if not parent:
        raise HTTPException(status_code=404, detail="Run not found")
    org_id = str(parent.organization_id) if parent.organization_id else None
    if org_id:
        await _require_org_member(org_id, current_user)
    children = await agent_run_repository.list_children(run_id)
    return {
        "parent": parent.model_dump(mode="json") if hasattr(parent, "model_dump") else parent.dict(),
        "children": [c.model_dump(mode="json") if hasattr(c, "model_dump") else c.dict() for c in children],
    }


@router.post("/runs/{run_id}/cancel")
async def cancel_run(
    run_id: str,
    current_user: User = Depends(get_current_active_user),
):
    run = await agent_run_repository.get_by_id(run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Run not found")
    if run.organization_id:
        await _require_org_member(str(run.organization_id), current_user)
    ok = await agent_run_repository.cancel_run(run_id)
    return {"ok": bool(ok)}


# ── Per-project agent surface (shortcuts to projects_v2 surface) ────


@router.get("/projects/{project_id}/agents")
async def list_project_agents_shortcut(
    project_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("project_agents")
    cursor = col.find({"organization_id": _oid(org_id), "project_id": _oid(project_id)})
    rows = await cursor.to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "project_id", "attached_by", "custom_agent_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


@router.get("/projects/{project_id}/runs")
async def list_project_runs(
    project_id: str,
    organization_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_runs")
    cursor = col.find({
        "organization_id": _oid(org_id),
        "project_id": _oid(project_id),
    }).sort("started_at", -1).skip(skip).limit(limit)
    rows = await cursor.to_list(length=limit)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "user_id", "task_id", "project_id", "parent_run_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


# ── Autonomy + model selection ──────────────────────────────────────


class AutonomyPatch(BaseModel):
    level: str = Field(..., description="suggest | auto-propose | auto-execute")


@router.patch("/projects/{project_id}/agents/{agent_ref}/autonomy")
async def update_agent_autonomy(
    project_id: str,
    agent_ref: str,
    payload: AutonomyPatch,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    if payload.level not in ("suggest", "auto-propose", "auto-execute"):
        raise HTTPException(status_code=400, detail="Invalid autonomy level")
    from backend.db.mongodb.repositories.project_agent_repository import project_agent_repository
    updated = await project_agent_repository.update(
        project_id=project_id, organization_id=org_id, agent_ref=agent_ref,
        patch={"autonomy_level": payload.level},
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Agent not attached to project")
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="agent.autonomy_changed",
        details={"project_id": project_id, "agent_ref": agent_ref, "level": payload.level},
        related_resource_type="project", related_resource_id=project_id,
    )
    return {"ok": True, "level": payload.level}


class ModelPatch(BaseModel):
    model: str = Field(..., max_length=120)


@router.patch("/projects/{project_id}/agents/{agent_ref}/model")
async def update_agent_model(
    project_id: str,
    agent_ref: str,
    payload: ModelPatch,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    from backend.db.mongodb.repositories.project_agent_repository import project_agent_repository
    updated = await project_agent_repository.update(
        project_id=project_id, organization_id=org_id, agent_ref=agent_ref,
        patch={"config_overrides": {"model": payload.model}},
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Agent not attached to project")
    return {"ok": True, "model": payload.model}


class FallbackChainPatch(BaseModel):
    chain: List[str] = Field(..., max_length=10)


@router.patch("/projects/{project_id}/agents/{agent_ref}/fallback-chain")
async def update_agent_fallback_chain(
    project_id: str,
    agent_ref: str,
    payload: FallbackChainPatch,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    from backend.db.mongodb.repositories.project_agent_repository import project_agent_repository
    updated = await project_agent_repository.update(
        project_id=project_id, organization_id=org_id, agent_ref=agent_ref,
        patch={"fallback_chain": payload.chain},
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Agent not attached to project")
    return {"ok": True, "fallback_chain": payload.chain}


# ── Schedules (cron-driven runs) ─────────────────────────────────────


class SchedulePayload(BaseModel):
    project_id: str
    agent_ref: str
    cron: str = Field(..., max_length=128, description="Cron expression in UTC")
    input: Dict[str, Any] = Field(default_factory=dict)
    enabled: bool = True
    name: Optional[str] = None


@router.get("/schedules")
async def list_schedules(
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_schedules")
    cursor = col.find({"organization_id": _oid(org_id)}).sort("created_at", -1)
    rows = await cursor.to_list(length=500)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "project_id", "created_by"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


@router.post("/schedules", status_code=201)
async def create_schedule(
    payload: SchedulePayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_schedules")
    doc: Dict[str, Any] = {
        "organization_id": _oid(org_id),
        "project_id": _oid(payload.project_id),
        "agent_ref": payload.agent_ref,
        "cron": payload.cron,
        "input": payload.input or {},
        "enabled": bool(payload.enabled),
        "name": payload.name,
        "created_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
        "next_run_at": None,
        "last_run_at": None,
    }
    result = await col.insert_one(doc)
    doc["id"] = str(result.inserted_id)
    doc.pop("_id", None)
    for k in ("organization_id", "project_id", "created_by"):
        if doc.get(k) is not None:
            doc[k] = str(doc[k])
    await emit("agent.schedule_created", organization_id=org_id, actor_id=str(current_user.id),
               project_id=payload.project_id, payload={"agent_ref": payload.agent_ref})
    return doc


@router.post("/schedules/{schedule_id}/pause")
async def pause_schedule(
    schedule_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_schedules")
    await col.update_one(
        {"_id": _oid(schedule_id), "organization_id": _oid(org_id)},
        {"$set": {"enabled": False, "updated_at": datetime.utcnow()}},
    )
    return {"ok": True}


@router.post("/schedules/{schedule_id}/resume")
async def resume_schedule(
    schedule_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_schedules")
    await col.update_one(
        {"_id": _oid(schedule_id), "organization_id": _oid(org_id)},
        {"$set": {"enabled": True, "updated_at": datetime.utcnow()}},
    )
    return {"ok": True}


@router.delete("/schedules/{schedule_id}", status_code=204)
async def delete_schedule(
    schedule_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_schedules")
    await col.delete_one({"_id": _oid(schedule_id), "organization_id": _oid(org_id)})
    return None


# ── Handoff ──────────────────────────────────────────────────────────


class HandoffToUserPayload(BaseModel):
    run_id: str
    user_id: str
    note: Optional[str] = None


@router.post("/handoffs/to-user")
async def handoff_run_to_user(
    payload: HandoffToUserPayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("agent_runs")
    result = await col.update_one(
        {"_id": _oid(payload.run_id), "organization_id": _oid(org_id)},
        {"$set": {
            "metadata.handoff_to_user_id": payload.user_id,
            "metadata.handoff_note": payload.note,
            "metadata.handoff_at": datetime.utcnow(),
        }},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Run not found")
    await emit("agent.handoff_to_user", organization_id=org_id, actor_id=str(current_user.id),
               resource_type="agent_run", resource_id=payload.run_id,
               payload={"to_user_id": payload.user_id})
    return {"ok": True}


class HandoffToAgentPayload(BaseModel):
    run_id: str
    target_agent_key: str
    input: Dict[str, Any] = Field(default_factory=dict)


@router.post("/handoffs/to-agent")
async def handoff_run_to_agent(
    payload: HandoffToAgentPayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    if payload.target_agent_key not in AGENT_REGISTRY:
        raise HTTPException(status_code=400, detail="Unknown target agent")
    col = await MongoDB.get_collection("agent_runs")
    result = await col.update_one(
        {"_id": _oid(payload.run_id), "organization_id": _oid(org_id)},
        {"$set": {
            "metadata.handoff_to_agent_key": payload.target_agent_key,
            "metadata.handoff_input": payload.input,
            "metadata.handoff_at": datetime.utcnow(),
        }},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Run not found")
    await emit("agent.handoff_to_agent", organization_id=org_id, actor_id=str(current_user.id),
               resource_type="agent_run", resource_id=payload.run_id,
               payload={"to_agent_key": payload.target_agent_key})
    return {"ok": True}


# ── Run quota probe ──────────────────────────────────────────────────


@router.get("/quota")
async def get_run_quota(
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    try:
        await assert_can_run_agent(org_id)
        return {"ok": True, "blocked": False}
    except PlanCapExceeded as exc:
        return {"ok": True, "blocked": True, "detail": exc.detail}
