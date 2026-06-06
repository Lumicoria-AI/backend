"""
AgentMetricsRepository — materialised per-agent metrics.

Rebuilt by the `materialise-agent-metrics` Celery beat task every 10 minutes
from raw `agent_runs` rows.  Queries against this collection are O(1) per
(scope, agent, window) instead of the O(N) aggregation the live `agent_run`
analytics endpoint runs.

A row uniquely identifies one slice:
    (agent_key | custom_agent_id, organization_id, team_id?, project_id?, user_id?, window)

`window` is one of: `day` (last 24h), `week` (last 7d), `month` (last 30d),
`all` (lifetime).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from backend.db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)

COLLECTION = "agent_metrics"
WINDOWS = {
    "day": timedelta(days=1),
    "week": timedelta(days=7),
    "month": timedelta(days=30),
    "all": None,
}


class AgentMetricsRepository:
    def __init__(self) -> None:
        self._initialised = False

    async def _ensure_indexes(self) -> None:
        if self._initialised:
            return
        col = await MongoDB.get_collection(COLLECTION)
        await col.create_index([
            ("agent_key", ASCENDING),
            ("organization_id", ASCENDING),
            ("window", ASCENDING),
        ])
        await col.create_index([
            ("custom_agent_id", ASCENDING),
            ("organization_id", ASCENDING),
            ("window", ASCENDING),
        ])
        await col.create_index([("project_id", ASCENDING), ("agent_key", ASCENDING)])
        await col.create_index([("team_id", ASCENDING), ("agent_key", ASCENDING)])
        await col.create_index("updated_at")
        self._initialised = True

    async def rebuild(
        self,
        *,
        organization_id: Optional[str] = None,
        window: str = "month",
    ) -> int:
        """Rebuild metric rows for a window.  Returns rows written."""
        await self._ensure_indexes()
        if window not in WINDOWS:
            raise ValueError(f"Unknown window {window}")
        delta = WINDOWS[window]
        since = (datetime.utcnow() - delta) if delta else datetime(1970, 1, 1)

        runs_col = await MongoDB.get_collection("agent_runs")
        metrics_col = await MongoDB.get_collection(COLLECTION)

        match: Dict[str, Any] = {"started_at": {"$gte": since}}
        if organization_id:
            match["organization_id"] = ObjectId(organization_id)

        pipeline: List[Dict[str, Any]] = [
            {"$match": match},
            {"$group": {
                "_id": {
                    "agent_key": "$agent_key",
                    "custom_agent_id": "$custom_agent_id",
                    "organization_id": "$organization_id",
                    "team_id": "$metadata.team_id",
                    "project_id": "$project_id",
                },
                "runs": {"$sum": 1},
                "completed": {"$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}},
                "errors": {"$sum": {"$cond": [{"$eq": ["$status", "error"]}, 1, 0]}},
                "avg_ms": {"$avg": "$duration_ms"},
                "max_ms": {"$max": "$duration_ms"},
                "tokens_in": {"$sum": {"$ifNull": ["$tokens_input", 0]}},
                "tokens_out": {"$sum": {"$ifNull": ["$tokens_output", 0]}},
                "cost": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
                "credits": {"$sum": {"$ifNull": ["$credits_used", 0]}},
                "last_run_at": {"$max": "$started_at"},
            }},
        ]
        written = 0
        now = datetime.utcnow()
        async for grp in runs_col.aggregate(pipeline):
            k = grp["_id"]
            doc: Dict[str, Any] = {
                "agent_key": k.get("agent_key"),
                "custom_agent_id": k.get("custom_agent_id"),
                "organization_id": k.get("organization_id"),
                "team_id": k.get("team_id"),
                "project_id": k.get("project_id"),
                "window": window,
                "runs": int(grp.get("runs") or 0),
                "completed": int(grp.get("completed") or 0),
                "errors": int(grp.get("errors") or 0),
                "avg_duration_ms": int(grp.get("avg_ms") or 0) if grp.get("avg_ms") else None,
                "max_duration_ms": int(grp.get("max_ms") or 0) if grp.get("max_ms") else None,
                "tokens_in": int(grp.get("tokens_in") or 0),
                "tokens_out": int(grp.get("tokens_out") or 0),
                "cost_usd": float(grp.get("cost") or 0),
                "credits_used": int(grp.get("credits") or 0),
                "last_run_at": grp.get("last_run_at"),
                "updated_at": now,
            }
            filt: Dict[str, Any] = {
                "agent_key": doc["agent_key"],
                "custom_agent_id": doc["custom_agent_id"],
                "organization_id": doc["organization_id"],
                "team_id": doc["team_id"],
                "project_id": doc["project_id"],
                "window": window,
            }
            await metrics_col.update_one(filt, {"$set": doc}, upsert=True)
            written += 1
        return written

    async def get(
        self,
        *,
        organization_id: str,
        agent_key: Optional[str] = None,
        custom_agent_id: Optional[str] = None,
        team_id: Optional[str] = None,
        project_id: Optional[str] = None,
        window: str = "month",
    ) -> List[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        query: Dict[str, Any] = {
            "organization_id": ObjectId(organization_id),
            "window": window,
        }
        if agent_key:
            query["agent_key"] = agent_key
        if custom_agent_id:
            query["custom_agent_id"] = ObjectId(custom_agent_id)
        if team_id:
            query["team_id"] = str(team_id)
        if project_id:
            query["project_id"] = ObjectId(project_id)
        rows = await col.find(query).sort("runs", DESCENDING).to_list(length=200)
        out: List[Dict[str, Any]] = []
        for r in rows:
            r = dict(r)
            r["id"] = str(r.pop("_id"))
            if r.get("organization_id") is not None:
                r["organization_id"] = str(r["organization_id"])
            if r.get("project_id") is not None:
                r["project_id"] = str(r["project_id"])
            if r.get("custom_agent_id") is not None:
                r["custom_agent_id"] = str(r["custom_agent_id"])
            out.append(r)
        return out

    async def leaderboard(
        self,
        *,
        organization_id: str,
        window: str = "month",
        limit: int = 25,
    ) -> List[Dict[str, Any]]:
        rows = await self.get(organization_id=organization_id, window=window)
        # Aggregate across team/project rows for the same agent_key.
        bucket: Dict[str, Dict[str, Any]] = {}
        for r in rows:
            key = r.get("agent_key") or r.get("custom_agent_id") or "unknown"
            slot = bucket.setdefault(key, {
                "agent_key": r.get("agent_key"),
                "custom_agent_id": r.get("custom_agent_id"),
                "runs": 0, "completed": 0, "errors": 0,
                "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0, "credits_used": 0,
                "last_run_at": r.get("last_run_at"),
            })
            for n in ("runs", "completed", "errors", "tokens_in", "tokens_out", "credits_used"):
                slot[n] += int(r.get(n) or 0)
            slot["cost_usd"] += float(r.get("cost_usd") or 0)
            if r.get("last_run_at") and (not slot["last_run_at"] or r["last_run_at"] > slot["last_run_at"]):
                slot["last_run_at"] = r["last_run_at"]
        ordered = sorted(bucket.values(), key=lambda x: x["runs"], reverse=True)[:limit]
        for r in ordered:
            r["success_rate"] = (r["completed"] / r["runs"]) if r["runs"] else 0.0
            r["cost_usd"] = round(r["cost_usd"], 4)
        return ordered


agent_metrics_repository = AgentMetricsRepository()
