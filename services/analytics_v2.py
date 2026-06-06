"""
Lumicoria AI — Analytics v2 service.

Computes per-level (org / team / project / user / agent / cost) aggregations
across the existing collections (`tasks`, `agent_runs`, `documents`,
`activity_logs`, `agent_metrics`, `org_subscriptions`, `seat_assignments`).

Heavy lifting that's expensive enough to cache lives in this module so the
router stays thin.  Cache TTL is 5 minutes by default — long enough to soak
up burst traffic, short enough that admins see fresh numbers.
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId

from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.agent_metrics_repository import agent_metrics_repository
from backend.db.mongodb.repositories.agent_run_repository import agent_run_repository
from backend.db.redis.redis import RedisClient

logger = structlog.get_logger(__name__)


CACHE_PREFIX = "analytics_v2:"
CACHE_TTL_SECONDS = 300  # 5 min


def _oid(v: Any) -> Optional[ObjectId]:
    if v is None:
        return None
    if isinstance(v, ObjectId):
        return v
    try:
        return ObjectId(str(v))
    except Exception:
        return None


def _window_since(time_range: str) -> datetime:
    days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}.get(time_range, 30)
    return datetime.utcnow() - timedelta(days=days)


async def _cache_get(key: str) -> Optional[Dict[str, Any]]:
    try:
        client = await RedisClient.get_client()
        raw = await client.get(CACHE_PREFIX + key)
        return json.loads(raw) if raw else None
    except Exception:  # noqa: BLE001
        return None


async def _cache_set(key: str, payload: Dict[str, Any]) -> None:
    try:
        client = await RedisClient.get_client()
        await client.set(CACHE_PREFIX + key, json.dumps(payload, default=str), ex=CACHE_TTL_SECONDS)
    except Exception:  # noqa: BLE001
        pass


# ── Org level ────────────────────────────────────────────────────────


async def org_overview(organization_id: str, *, time_range: str = "30d") -> Dict[str, Any]:
    cache_key = f"org:{organization_id}:overview:{time_range}"
    cached = await _cache_get(cache_key)
    if cached:
        return cached

    since = _window_since(time_range)
    org_oid = _oid(organization_id)

    tasks_col = await MongoDB.get_collection("tasks")
    runs_col = await MongoDB.get_collection("agent_runs")
    docs_col = await MongoDB.get_collection("documents")
    teams_col = await MongoDB.get_collection("teams")
    projects_col = await MongoDB.get_collection("projects")

    tasks_total, tasks_completed, tasks_overdue, runs_total, docs_total, teams_count, projects_count = await asyncio.gather(
        tasks_col.count_documents({"organization_id": org_oid}),
        tasks_col.count_documents({"organization_id": org_oid, "status": "completed"}),
        tasks_col.count_documents({
            "organization_id": org_oid,
            "status": {"$nin": ["completed", "cancelled", "archived"]},
            "due_date": {"$lt": datetime.utcnow()},
        }),
        runs_col.count_documents({"organization_id": org_oid, "started_at": {"$gte": since}}),
        docs_col.count_documents({"organization_id": org_oid}),
        teams_col.count_documents({"organization_id": org_oid, "is_archived": False}),
        projects_col.count_documents({"organization_id": org_oid, "is_archived": False}),
    )

    payload: Dict[str, Any] = {
        "time_range": time_range,
        "since": since.isoformat() + "Z",
        "tasks": {
            "total": tasks_total,
            "completed": tasks_completed,
            "overdue": tasks_overdue,
            "completion_rate": (tasks_completed / tasks_total) if tasks_total else 0.0,
        },
        "agent_runs": {"total": runs_total},
        "documents": {"total": docs_total},
        "teams": {"total": teams_count},
        "projects": {"total": projects_count},
    }
    await _cache_set(cache_key, payload)
    return payload


async def org_task_throughput(organization_id: str, *, time_range: str = "30d") -> Dict[str, Any]:
    cache_key = f"org:{organization_id}:throughput:{time_range}"
    cached = await _cache_get(cache_key)
    if cached:
        return cached
    since = _window_since(time_range)
    org_oid = _oid(organization_id)
    tasks_col = await MongoDB.get_collection("tasks")
    pipeline = [
        {"$match": {
            "organization_id": org_oid,
            "completed_at": {"$gte": since},
            "status": "completed",
        }},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$completed_at"}},
            "completed": {"$sum": 1},
        }},
        {"$sort": {"_id": 1}},
    ]
    series = await tasks_col.aggregate(pipeline).to_list(length=400)
    payload = {
        "time_range": time_range,
        "since": since.isoformat() + "Z",
        "series": [{"day": s["_id"], "completed": s["completed"]} for s in series if s["_id"]],
    }
    await _cache_set(cache_key, payload)
    return payload


async def org_cycle_time(organization_id: str, *, time_range: str = "30d") -> Dict[str, Any]:
    """Average time from task.created → task.completed."""
    since = _window_since(time_range)
    org_oid = _oid(organization_id)
    tasks_col = await MongoDB.get_collection("tasks")
    pipeline = [
        {"$match": {
            "organization_id": org_oid,
            "completed_at": {"$gte": since},
            "status": "completed",
        }},
        {"$project": {
            "duration_seconds": {
                "$divide": [{"$subtract": ["$completed_at", "$created_at"]}, 1000],
            },
        }},
        {"$group": {
            "_id": None,
            "avg_seconds": {"$avg": "$duration_seconds"},
            "max_seconds": {"$max": "$duration_seconds"},
            "min_seconds": {"$min": "$duration_seconds"},
            "count": {"$sum": 1},
        }},
    ]
    rows = await tasks_col.aggregate(pipeline).to_list(length=1)
    if not rows:
        return {"avg_hours": 0, "count": 0}
    r = rows[0]
    return {
        "time_range": time_range,
        "count": int(r.get("count") or 0),
        "avg_hours": round(float(r.get("avg_seconds") or 0) / 3600.0, 2),
        "max_hours": round(float(r.get("max_seconds") or 0) / 3600.0, 2),
        "min_hours": round(float(r.get("min_seconds") or 0) / 3600.0, 2),
    }


async def org_cost(organization_id: str, *, time_range: str = "30d") -> Dict[str, Any]:
    cache_key = f"org:{organization_id}:cost:{time_range}"
    cached = await _cache_get(cache_key)
    if cached:
        return cached
    since = _window_since(time_range)
    org_oid = _oid(organization_id)
    runs_col = await MongoDB.get_collection("agent_runs")
    pipeline = [
        {"$match": {"organization_id": org_oid, "started_at": {"$gte": since}}},
        {"$group": {
            "_id": None,
            "cost_usd": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
            "credits_used": {"$sum": {"$ifNull": ["$credits_used", 0]}},
            "tokens_in": {"$sum": {"$ifNull": ["$tokens_input", 0]}},
            "tokens_out": {"$sum": {"$ifNull": ["$tokens_output", 0]}},
            "runs": {"$sum": 1},
        }},
    ]
    rows = await runs_col.aggregate(pipeline).to_list(length=1)
    r = rows[0] if rows else {}
    payload = {
        "time_range": time_range,
        "cost_usd": round(float(r.get("cost_usd") or 0), 4),
        "credits_used": int(r.get("credits_used") or 0),
        "tokens_in": int(r.get("tokens_in") or 0),
        "tokens_out": int(r.get("tokens_out") or 0),
        "runs": int(r.get("runs") or 0),
    }
    await _cache_set(cache_key, payload)
    return payload


# ── Team level ───────────────────────────────────────────────────────


async def team_overview(organization_id: str, team_id: str, *, time_range: str = "30d") -> Dict[str, Any]:
    cache_key = f"team:{team_id}:overview:{time_range}"
    cached = await _cache_get(cache_key)
    if cached:
        return cached
    since = _window_since(time_range)
    org_oid = _oid(organization_id)
    team_oid = _oid(team_id)

    projects_col = await MongoDB.get_collection("projects")
    tasks_col = await MongoDB.get_collection("tasks")
    runs_col = await MongoDB.get_collection("agent_runs")

    project_ids: List[ObjectId] = []
    async for p in projects_col.find({"organization_id": org_oid, "team_id": team_oid}, {"_id": 1}):
        project_ids.append(p["_id"])

    task_match: Dict[str, Any] = {"organization_id": org_oid}
    runs_match: Dict[str, Any] = {"organization_id": org_oid, "started_at": {"$gte": since}}
    if project_ids:
        task_match["project_id"] = {"$in": project_ids}
        runs_match["project_id"] = {"$in": project_ids}
    else:
        # Fallback: use metadata.team_id when a project is not linked
        task_match["metadata.team_id"] = str(team_id)
        runs_match["metadata.team_id"] = str(team_id)

    tasks_total, tasks_completed, runs_total = await asyncio.gather(
        tasks_col.count_documents(task_match),
        tasks_col.count_documents({**task_match, "status": "completed"}),
        runs_col.count_documents(runs_match),
    )
    payload = {
        "time_range": time_range,
        "projects": {"total": len(project_ids)},
        "tasks": {
            "total": tasks_total,
            "completed": tasks_completed,
            "completion_rate": (tasks_completed / tasks_total) if tasks_total else 0.0,
        },
        "agent_runs": {"total": runs_total},
    }
    await _cache_set(cache_key, payload)
    return payload


# ── Project level ────────────────────────────────────────────────────


async def project_burnup(organization_id: str, project_id: str, *, time_range: str = "30d") -> Dict[str, Any]:
    since = _window_since(time_range)
    org_oid = _oid(organization_id)
    proj_oid = _oid(project_id)
    tasks_col = await MongoDB.get_collection("tasks")
    pipeline = [
        {"$match": {"organization_id": org_oid, "project_id": proj_oid, "created_at": {"$gte": since}}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$created_at"}},
            "created": {"$sum": 1},
            "completed": {"$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}},
        }},
        {"$sort": {"_id": 1}},
    ]
    rows = await tasks_col.aggregate(pipeline).to_list(length=400)
    cumulative_created = 0
    cumulative_completed = 0
    series: List[Dict[str, Any]] = []
    for r in rows:
        cumulative_created += int(r.get("created") or 0)
        cumulative_completed += int(r.get("completed") or 0)
        series.append({
            "day": r["_id"],
            "created": cumulative_created,
            "completed": cumulative_completed,
        })
    return {"time_range": time_range, "since": since.isoformat() + "Z", "series": series}


async def project_throughput(organization_id: str, project_id: str, *, time_range: str = "30d") -> Dict[str, Any]:
    since = _window_since(time_range)
    org_oid = _oid(organization_id)
    proj_oid = _oid(project_id)
    tasks_col = await MongoDB.get_collection("tasks")
    pipeline = [
        {"$match": {
            "organization_id": org_oid,
            "project_id": proj_oid,
            "completed_at": {"$gte": since},
            "status": "completed",
        }},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$completed_at"}},
            "completed": {"$sum": 1},
        }},
        {"$sort": {"_id": 1}},
    ]
    rows = await tasks_col.aggregate(pipeline).to_list(length=400)
    return {
        "time_range": time_range,
        "series": [{"day": r["_id"], "completed": r["completed"]} for r in rows if r["_id"]],
    }


# ── User level ───────────────────────────────────────────────────────


async def user_summary(user_id: str, organization_id: Optional[str] = None, *, time_range: str = "30d") -> Dict[str, Any]:
    since = _window_since(time_range)
    user_oid = _oid(user_id)
    tasks_col = await MongoDB.get_collection("tasks")
    runs_col = await MongoDB.get_collection("agent_runs")

    task_match: Dict[str, Any] = {
        "$or": [
            {"assigned_to": user_oid},
            {"created_by": user_oid},
        ],
    }
    runs_match: Dict[str, Any] = {"user_id": user_oid, "started_at": {"$gte": since}}
    if organization_id:
        task_match["organization_id"] = _oid(organization_id)
        runs_match["organization_id"] = _oid(organization_id)

    tasks_assigned, tasks_completed, runs_total = await asyncio.gather(
        tasks_col.count_documents(task_match),
        tasks_col.count_documents({**task_match, "status": "completed"}),
        runs_col.count_documents(runs_match),
    )
    return {
        "time_range": time_range,
        "tasks": {
            "assigned_or_created": tasks_assigned,
            "completed": tasks_completed,
            "completion_rate": (tasks_completed / tasks_assigned) if tasks_assigned else 0.0,
        },
        "agent_runs": {"total": runs_total},
    }


# ── Forecast (very simple linear) ────────────────────────────────────


async def org_seat_forecast(organization_id: str, *, horizon_days: int = 90) -> Dict[str, Any]:
    from backend.db.mongodb.repositories.org_subscription_repository import (
        org_subscription_repository, seat_assignment_repository,
    )
    sub = await org_subscription_repository.get_for_org(organization_id)
    used = await seat_assignment_repository.count_active(organization_id)
    purchased = sub.seats_purchased if sub else 0
    utilisation = (used / purchased) if purchased else 0.0
    return {
        "purchased": purchased,
        "used_today": used,
        "projected_used": int(round(used * (1 + (utilisation * horizon_days / 30) * 0.05))),
        "horizon_days": horizon_days,
        "utilisation_pct": round(utilisation * 100, 2),
    }
