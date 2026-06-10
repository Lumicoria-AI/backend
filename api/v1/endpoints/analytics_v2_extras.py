"""
Phase D — Analytics v2 depth.

Adds retention, funnel, cohorts, and per-scope exports for the metric
families the plan calls out.  Each endpoint reuses the helpers in
`backend/services/analytics_v2.py` and the materialised `agent_metrics`
collection so reads stay cheap.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.models.user import User
from backend.services import analytics_v2 as svc

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


def _window(time_range: str) -> datetime:
    days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}.get(time_range, 30)
    return datetime.utcnow() - timedelta(days=days)


# ── Org-level deep dives ──────────────────────────────────────────


@router.get("/org/{org_id}/tasks")
async def org_tasks(org_id: str, time_range: str = Query("30d"),
                    current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    tasks = await MongoDB.get_collection("tasks")
    since = _window(time_range)
    match = {"organization_id": _oid(org_id), "created_at": {"$gte": since}}
    by_status = {r["_id"] or "unknown": r["count"]
                 async for r in tasks.aggregate([
                     {"$match": match},
                     {"$group": {"_id": "$status", "count": {"$sum": 1}}},
                 ])}
    by_priority = {r["_id"] or "unknown": r["count"]
                   async for r in tasks.aggregate([
                       {"$match": match},
                       {"$group": {"_id": "$priority", "count": {"$sum": 1}}},
                   ])}
    return {"time_range": time_range, "by_status": by_status, "by_priority": by_priority}


@router.get("/org/{org_id}/agents")
async def org_agents(org_id: str, window: str = Query("month"),
                     current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    from backend.db.mongodb.repositories.agent_metrics_repository import agent_metrics_repository
    return await agent_metrics_repository.leaderboard(organization_id=org_id, window=window, limit=50)


@router.get("/org/{org_id}/members")
async def org_members(org_id: str, time_range: str = Query("30d"),
                      current_user: User = Depends(get_current_active_user)):
    org = await _require_org_member(org_id, current_user)
    since = _window(time_range)
    activity = await MongoDB.get_collection("activity_logs")
    pipeline = [
        {"$match": {"organization_id": _oid(org_id), "timestamp": {"$gte": since}}},
        {"$group": {"_id": "$user_id", "actions": {"$sum": 1}}},
        {"$sort": {"actions": -1}}, {"$limit": 50},
    ]
    rows = await activity.aggregate(pipeline).to_list(length=50)
    return {
        "total_members": len(org.member_ids or []),
        "active_in_window": len(rows),
        "top_actors": [{"user_id": str(r["_id"]) if r["_id"] else None, "actions": r["actions"]} for r in rows],
        "time_range": time_range,
    }


@router.get("/org/{org_id}/documents")
async def org_documents(org_id: str, time_range: str = Query("30d"),
                        current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    since = _window(time_range)
    docs = await MongoDB.get_collection("documents")
    return {
        "total": await docs.count_documents({"organization_id": _oid(org_id)}),
        "uploaded_in_window": await docs.count_documents({
            "organization_id": _oid(org_id), "uploaded_at": {"$gte": since},
        }),
        "time_range": time_range,
    }


@router.get("/org/{org_id}/billing")
async def org_billing_analytics(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    from backend.db.mongodb.repositories.org_subscription_repository import (
        org_subscription_repository, seat_assignment_repository,
    )
    sub = await org_subscription_repository.get_for_org(org_id)
    used = await seat_assignment_repository.count_active(org_id)
    return {
        "plan": sub.plan if sub else "free",
        "seats_purchased": sub.seats_purchased if sub else 0,
        "seats_used": used,
        "billing_status": sub.status if sub else "free",
    }


@router.get("/org/{org_id}/usage")
async def org_usage(org_id: str, time_range: str = Query("30d"),
                    current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    since = _window(time_range)
    runs = await MongoDB.get_collection("agent_runs")
    pipeline = [
        {"$match": {"organization_id": _oid(org_id), "started_at": {"$gte": since}}},
        {"$group": {"_id": None,
                    "runs": {"$sum": 1},
                    "tokens_in": {"$sum": {"$ifNull": ["$tokens_input", 0]}},
                    "tokens_out": {"$sum": {"$ifNull": ["$tokens_output", 0]}},
                    "cost": {"$sum": {"$ifNull": ["$cost_usd", 0]}}}},
    ]
    rows = await runs.aggregate(pipeline).to_list(length=1)
    r = rows[0] if rows else {}
    return {
        "time_range": time_range,
        "runs": int(r.get("runs") or 0),
        "tokens_in": int(r.get("tokens_in") or 0),
        "tokens_out": int(r.get("tokens_out") or 0),
        "cost_usd": round(float(r.get("cost") or 0), 4),
    }


@router.get("/org/{org_id}/forecast")
async def org_forecast(org_id: str, horizon_days: int = Query(90, ge=7, le=365),
                       current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    return await svc.org_seat_forecast(org_id, horizon_days=horizon_days)


@router.get("/org/{org_id}/retention")
async def org_retention(org_id: str, time_range: str = Query("90d"),
                        current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    since = _window(time_range)
    activity = await MongoDB.get_collection("activity_logs")
    pipeline = [
        {"$match": {"organization_id": _oid(org_id), "timestamp": {"$gte": since},
                    "activity_type": "user.login"}},
        {"$group": {"_id": {"$dateToString": {"format": "%Y-W%U", "date": "$timestamp"}},
                    "distinct_users": {"$addToSet": "$user_id"}}},
        {"$project": {"week": "$_id", "active_users": {"$size": "$distinct_users"}}},
        {"$sort": {"week": 1}},
    ]
    return {"time_range": time_range, "weekly_active": await activity.aggregate(pipeline).to_list(length=100)}


@router.get("/org/{org_id}/funnel")
async def org_funnel(org_id: str, time_range: str = Query("30d"),
                     current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    activity = await MongoDB.get_collection("activity_logs")
    since = _window(time_range)
    steps = ["user.login", "project.created", "task.created", "agent.executed", "task.completed"]
    counts = []
    for step in steps:
        n = await activity.count_documents({
            "organization_id": _oid(org_id), "activity_type": step,
            "timestamp": {"$gte": since},
        })
        counts.append({"step": step, "count": n})
    return {"time_range": time_range, "funnel": counts}


@router.get("/org/{org_id}/cohorts")
async def org_cohorts(org_id: str, time_range: str = Query("90d"),
                      current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    since = _window(time_range)
    users = await MongoDB.get_collection("users")
    pipeline = [
        {"$match": {"created_at": {"$gte": since}}},
        {"$group": {"_id": {"$dateToString": {"format": "%Y-%m", "date": "$created_at"}},
                    "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}},
    ]
    rows = await users.aggregate(pipeline).to_list(length=24)
    return {"time_range": time_range, "cohorts_by_month": rows}


@router.get("/org/{org_id}/timeline")
async def org_timeline(org_id: str, time_range: str = Query("30d"),
                       current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    return await svc.org_task_throughput(org_id, time_range=time_range)


@router.post("/org/{org_id}/export")
async def org_export(org_id: str, format: str = Query("csv"),
                     current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("analytics_exports")
    doc = {
        "organization_id": _oid(org_id), "scope": "org",
        "requested_by": _oid(current_user.id),
        "format": format, "status": "pending",
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"job_id": str(r.inserted_id), "status": "pending"}


# ── Team-level dives ──────────────────────────────────────────────


@router.get("/team/{team_id}/tasks")
async def team_tasks(team_id: str, organization_id: Optional[str] = Query(None),
                     time_range: str = Query("30d"),
                     current_user: User = Depends(get_current_active_user)):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    return await svc.team_overview(org_id, team_id, time_range=time_range)


@router.get("/team/{team_id}/agents")
async def team_agents(team_id: str, organization_id: Optional[str] = Query(None),
                      window: str = Query("month"),
                      current_user: User = Depends(get_current_active_user)):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    from backend.db.mongodb.repositories.agent_metrics_repository import agent_metrics_repository
    return await agent_metrics_repository.get(organization_id=org_id, team_id=team_id, window=window)


@router.get("/team/{team_id}/throughput")
async def team_throughput(team_id: str, organization_id: Optional[str] = Query(None),
                          time_range: str = Query("30d"),
                          current_user: User = Depends(get_current_active_user)):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    overview = await svc.team_overview(org_id, team_id, time_range=time_range)
    return {"time_range": time_range, "tasks_completed": overview.get("tasks", {}).get("completed", 0)}


@router.post("/team/{team_id}/export")
async def team_export(team_id: str, organization_id: Optional[str] = Query(None),
                      format: str = Query("csv"),
                      current_user: User = Depends(get_current_active_user)):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("analytics_exports")
    doc = {
        "organization_id": _oid(org_id), "scope": "team", "scope_id": team_id,
        "requested_by": _oid(current_user.id),
        "format": format, "status": "pending",
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"job_id": str(r.inserted_id), "status": "pending"}


# ── Project-level dives ───────────────────────────────────────────


@router.get("/project/{project_id}/cost")
async def project_cost(project_id: str, organization_id: Optional[str] = Query(None),
                       time_range: str = Query("30d"),
                       current_user: User = Depends(get_current_active_user)):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    runs = await MongoDB.get_collection("agent_runs")
    since = _window(time_range)
    pipeline = [
        {"$match": {"organization_id": _oid(org_id), "project_id": _oid(project_id),
                    "started_at": {"$gte": since}}},
        {"$group": {"_id": None, "cost": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
                    "credits": {"$sum": {"$ifNull": ["$credits_used", 0]}}}},
    ]
    rows = await runs.aggregate(pipeline).to_list(length=1)
    r = rows[0] if rows else {}
    return {"cost_usd": round(float(r.get("cost") or 0), 4), "credits_used": int(r.get("credits") or 0)}


@router.get("/project/{project_id}/agents")
async def project_agents_analytics(project_id: str, organization_id: Optional[str] = Query(None),
                                   window: str = Query("month"),
                                   current_user: User = Depends(get_current_active_user)):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    from backend.db.mongodb.repositories.agent_metrics_repository import agent_metrics_repository
    return await agent_metrics_repository.get(organization_id=org_id, project_id=project_id, window=window)


@router.get("/project/{project_id}/cycle-time")
async def project_cycle(project_id: str, organization_id: Optional[str] = Query(None),
                        time_range: str = Query("30d"),
                        current_user: User = Depends(get_current_active_user)):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    tasks = await MongoDB.get_collection("tasks")
    since = _window(time_range)
    pipeline = [
        {"$match": {"organization_id": _oid(org_id), "project_id": _oid(project_id),
                    "status": "completed", "completed_at": {"$gte": since}}},
        {"$project": {"d": {"$divide": [{"$subtract": ["$completed_at", "$created_at"]}, 1000]}}},
        {"$group": {"_id": None, "avg": {"$avg": "$d"}, "count": {"$sum": 1}}},
    ]
    rows = await tasks.aggregate(pipeline).to_list(length=1)
    r = rows[0] if rows else {}
    return {"avg_hours": round(float(r.get("avg") or 0) / 3600, 2), "count": int(r.get("count") or 0)}


@router.post("/project/{project_id}/export")
async def project_export(project_id: str, organization_id: Optional[str] = Query(None),
                         format: str = Query("csv"),
                         current_user: User = Depends(get_current_active_user)):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("analytics_exports")
    doc = {
        "organization_id": _oid(org_id), "scope": "project", "scope_id": project_id,
        "requested_by": _oid(current_user.id),
        "format": format, "status": "pending",
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"job_id": str(r.inserted_id), "status": "pending"}


# ── Cost roll-ups ────────────────────────────────────────────────


@router.get("/cost/agents")
async def cost_by_agent(organization_id: Optional[str] = Query(None),
                        window: str = Query("month"),
                        current_user: User = Depends(get_current_active_user)):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    from backend.db.mongodb.repositories.agent_metrics_repository import agent_metrics_repository
    return await agent_metrics_repository.leaderboard(organization_id=org_id, window=window, limit=50)


@router.get("/cost/models")
async def cost_by_model(organization_id: Optional[str] = Query(None),
                        time_range: str = Query("30d"),
                        current_user: User = Depends(get_current_active_user)):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    runs = await MongoDB.get_collection("agent_runs")
    since = _window(time_range)
    pipeline = [
        {"$match": {"organization_id": _oid(org_id), "started_at": {"$gte": since}}},
        {"$group": {"_id": "$metadata.model", "runs": {"$sum": 1},
                    "cost": {"$sum": {"$ifNull": ["$cost_usd", 0]}}}},
        {"$sort": {"cost": -1}}, {"$limit": 20},
    ]
    rows = await runs.aggregate(pipeline).to_list(length=20)
    return [{"model": r["_id"] or "default", "runs": r["runs"], "cost_usd": round(float(r["cost"] or 0), 4)} for r in rows]


@router.get("/cost/per-seat")
async def cost_per_seat(organization_id: Optional[str] = Query(None),
                        time_range: str = Query("30d"),
                        current_user: User = Depends(get_current_active_user)):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    from backend.db.mongodb.repositories.org_subscription_repository import seat_assignment_repository
    cost = await svc.org_cost(org_id, time_range=time_range)
    used = await seat_assignment_repository.count_active(org_id)
    return {
        "time_range": time_range, "cost_usd": cost.get("cost_usd", 0),
        "seats_used": used,
        "cost_per_seat_usd": round((cost.get("cost_usd", 0) / used) if used else 0, 4),
    }


@router.get("/cost/forecast")
async def cost_forecast(organization_id: Optional[str] = Query(None),
                        horizon_days: int = Query(90, ge=7, le=365),
                        current_user: User = Depends(get_current_active_user)):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    cost = await svc.org_cost(org_id, time_range="30d")
    monthly = float(cost.get("cost_usd") or 0)
    return {"monthly_baseline_usd": monthly, "horizon_days": horizon_days,
            "projected_usd": round(monthly * (horizon_days / 30), 4)}


# ── User-level summary ───────────────────────────────────────────


@router.get("/user/{user_id}/summary")
async def user_summary(user_id: str, organization_id: Optional[str] = Query(None),
                       time_range: str = Query("30d"),
                       current_user: User = Depends(get_current_active_user)):
    if organization_id:
        await _require_org_member(organization_id, current_user)
    return await svc.user_summary(user_id, organization_id, time_range=time_range)


@router.get("/user/{user_id}/agents")
async def user_agents(user_id: str, organization_id: Optional[str] = Query(None),
                      time_range: str = Query("30d"),
                      current_user: User = Depends(get_current_active_user)):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    runs = await MongoDB.get_collection("agent_runs")
    since = _window(time_range)
    pipeline = [
        {"$match": {"organization_id": _oid(org_id), "user_id": _oid(user_id),
                    "started_at": {"$gte": since}}},
        {"$group": {"_id": "$agent_key", "runs": {"$sum": 1}}},
        {"$sort": {"runs": -1}}, {"$limit": 20},
    ]
    return [{"agent_key": r["_id"], "runs": r["runs"]}
            for r in await runs.aggregate(pipeline).to_list(length=20)]


@router.get("/user/{user_id}/tasks")
async def user_tasks(user_id: str, organization_id: Optional[str] = Query(None),
                     time_range: str = Query("30d"),
                     current_user: User = Depends(get_current_active_user)):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    tasks = await MongoDB.get_collection("tasks")
    since = _window(time_range)
    match = {"organization_id": _oid(org_id), "assigned_to": _oid(user_id),
             "created_at": {"$gte": since}}
    return {
        "total": await tasks.count_documents(match),
        "completed": await tasks.count_documents({**match, "status": "completed"}),
    }


# ── Agent-level dives ────────────────────────────────────────────


@router.get("/agent/{agent_key}/global")
async def agent_global(agent_key: str, window: str = Query("month"),
                       current_user: User = Depends(get_current_active_user)):
    runs = await MongoDB.get_collection("agent_runs")
    since = _window({"day": "1d", "week": "7d", "month": "30d", "all": "1y"}.get(window, "30d"))
    pipeline = [
        {"$match": {"agent_key": agent_key, "started_at": {"$gte": since}}},
        {"$group": {"_id": None, "runs": {"$sum": 1},
                    "errors": {"$sum": {"$cond": [{"$eq": ["$status", "error"]}, 1, 0]}}}},
    ]
    rows = await runs.aggregate(pipeline).to_list(length=1)
    r = rows[0] if rows else {}
    return {"agent_key": agent_key, "window": window,
            "runs": int(r.get("runs") or 0), "errors": int(r.get("errors") or 0)}


@router.get("/agent/{agent_key}/by-org")
async def agent_by_org(agent_key: str, current_user: User = Depends(get_current_active_user)):
    org_id = _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    from backend.db.mongodb.repositories.agent_metrics_repository import agent_metrics_repository
    return await agent_metrics_repository.get(organization_id=org_id, agent_key=agent_key, window="month")
