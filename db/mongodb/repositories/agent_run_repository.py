"""
AgentRun repository — production-grade CRUD + analytics over `agent_runs`.

Used by:
  • Phase 6 task executor — opens a run, executes the agent, closes the run.
  • Phase 7 step graph — opens a parent run for the orchestrator, then a
    child run per sub-agent it invokes.
  • Phase 9 dashboard analytics — aggregations for the runs/perf charts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from backend.db.mongodb.models.agent_run import (
    AgentRun,
    AgentRunCreate,
    AgentRunStatus,
    AgentRunTrigger,
)
from backend.db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)

COLLECTION_NAME = "agent_runs"


def _coerce_oid(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value))
    except Exception:
        return str(value)


class AgentRunRepository:
    def __init__(self) -> None:
        self._collection = None

    async def _get_collection(self):
        if self._collection is None:
            self._collection = await MongoDB.get_collection(COLLECTION_NAME)
            await self._create_indexes()
        return self._collection

    async def _create_indexes(self) -> None:
        col = self._collection
        await col.create_index("user_id")
        await col.create_index("organization_id")
        await col.create_index("agent_key")
        await col.create_index("status")
        await col.create_index("trigger")
        await col.create_index("task_id")
        await col.create_index("conversation_id")
        await col.create_index("parent_run_id")
        await col.create_index([("started_at", DESCENDING)])
        # Compound — most common analytics filter.
        await col.create_index([
            ("organization_id", ASCENDING),
            ("agent_key", ASCENDING),
            ("started_at", DESCENDING),
        ])
        await col.create_index([
            ("user_id", ASCENDING),
            ("started_at", DESCENDING),
        ])

    # ── Lifecycle ──────────────────────────────────────────────────────

    async def start_run(self, payload: AgentRunCreate) -> AgentRun:
        """Open a run in RUNNING state.  Returns the persisted row."""
        now = datetime.utcnow()
        doc: Dict[str, Any] = payload.model_dump()
        for key in ("user_id", "organization_id", "task_id", "parent_run_id"):
            if doc.get(key):
                doc[key] = _coerce_oid(doc[key])
        # Coerce trigger / status to string enum values
        if isinstance(doc.get("trigger"), AgentRunTrigger):
            doc["trigger"] = doc["trigger"].value
        doc["status"] = AgentRunStatus.RUNNING.value
        doc["started_at"] = now
        doc["created_at"] = now

        col = await self._get_collection()
        result = await col.insert_one(doc)
        created = await col.find_one({"_id": result.inserted_id})
        if not created:
            raise RuntimeError("AgentRun insert succeeded but find_one returned None")
        return AgentRun(**created)

    async def complete_run(
        self,
        run_id: str,
        *,
        output: Optional[Dict[str, Any]] = None,
        tokens_input: Optional[int] = None,
        tokens_output: Optional[int] = None,
        cost_usd: Optional[float] = None,
        metadata_patch: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentRun]:
        """Close a run as COMPLETED.  Auto-computes duration_ms."""
        oid = _coerce_oid(run_id)
        if not isinstance(oid, ObjectId):
            return None

        col = await self._get_collection()
        existing = await col.find_one({"_id": oid})
        if not existing:
            return None

        started = existing.get("started_at") or datetime.utcnow()
        now = datetime.utcnow()
        duration_ms = int((now - started).total_seconds() * 1000)

        patch: Dict[str, Any] = {
            "status": AgentRunStatus.COMPLETED.value,
            "ended_at": now,
            "duration_ms": duration_ms,
        }
        if output is not None:
            patch["output"] = output
        if tokens_input is not None:
            patch["tokens_input"] = tokens_input
        if tokens_output is not None:
            patch["tokens_output"] = tokens_output
        if cost_usd is not None:
            patch["cost_usd"] = cost_usd

        update: Dict[str, Any] = {"$set": patch}
        if metadata_patch:
            for k, v in metadata_patch.items():
                update.setdefault("$set", {})[f"metadata.{k}"] = v

        result = await col.find_one_and_update(
            {"_id": oid}, update, return_document=True
        )
        return AgentRun(**result) if result else None

    async def fail_run(
        self,
        run_id: str,
        error: str,
        *,
        metadata_patch: Optional[Dict[str, Any]] = None,
    ) -> Optional[AgentRun]:
        """Close a run as ERROR."""
        oid = _coerce_oid(run_id)
        if not isinstance(oid, ObjectId):
            return None

        col = await self._get_collection()
        existing = await col.find_one({"_id": oid})
        if not existing:
            return None

        started = existing.get("started_at") or datetime.utcnow()
        now = datetime.utcnow()
        duration_ms = int((now - started).total_seconds() * 1000)

        patch: Dict[str, Any] = {
            "status": AgentRunStatus.ERROR.value,
            "error": error,
            "ended_at": now,
            "duration_ms": duration_ms,
        }
        update: Dict[str, Any] = {"$set": patch}
        if metadata_patch:
            for k, v in metadata_patch.items():
                update.setdefault("$set", {})[f"metadata.{k}"] = v

        result = await col.find_one_and_update(
            {"_id": oid}, update, return_document=True
        )
        return AgentRun(**result) if result else None

    async def cancel_run(self, run_id: str) -> bool:
        oid = _coerce_oid(run_id)
        if not isinstance(oid, ObjectId):
            return False
        col = await self._get_collection()
        result = await col.update_one(
            {"_id": oid, "status": AgentRunStatus.RUNNING.value},
            {"$set": {"status": AgentRunStatus.CANCELLED.value, "ended_at": datetime.utcnow()}},
        )
        return result.modified_count > 0

    # ── Discovery ─────────────────────────────────────────────────────

    async def get_by_id(self, run_id: str) -> Optional[AgentRun]:
        oid = _coerce_oid(run_id)
        if not isinstance(oid, ObjectId):
            return None
        col = await self._get_collection()
        doc = await col.find_one({"_id": oid})
        return AgentRun(**doc) if doc else None

    async def list_for_task(self, task_id: str, limit: int = 50) -> List[AgentRun]:
        col = await self._get_collection()
        cursor = col.find({"task_id": _coerce_oid(task_id)}).sort("started_at", DESCENDING).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [AgentRun(**d) for d in docs]

    async def list_for_conversation(
        self,
        conversation_id: str,
        limit: int = 200,
    ) -> List[AgentRun]:
        col = await self._get_collection()
        cursor = col.find({"conversation_id": conversation_id}).sort("started_at", ASCENDING).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [AgentRun(**d) for d in docs]

    async def list_children(self, parent_run_id: str) -> List[AgentRun]:
        col = await self._get_collection()
        cursor = col.find({"parent_run_id": _coerce_oid(parent_run_id)}).sort("step_index", ASCENDING)
        docs = await cursor.to_list(length=200)
        return [AgentRun(**d) for d in docs]

    async def list_recent(
        self,
        *,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        agent_key: Optional[str] = None,
        status: Optional[AgentRunStatus] = None,
        limit: int = 50,
        skip: int = 0,
    ) -> List[AgentRun]:
        query: Dict[str, Any] = {}
        if user_id:
            query["user_id"] = _coerce_oid(user_id)
        if organization_id:
            query["organization_id"] = _coerce_oid(organization_id)
        if agent_key:
            query["agent_key"] = agent_key
        if status:
            query["status"] = status.value if isinstance(status, AgentRunStatus) else status
        col = await self._get_collection()
        cursor = col.find(query).sort("started_at", DESCENDING).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [AgentRun(**d) for d in docs]

    # ── Analytics (powers the dashboard in Phase 9) ───────────────────

    async def analytics(
        self,
        *,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        time_range: str = "7d",
    ) -> Dict[str, Any]:
        """Run-volume + success rate + per-agent breakdown for a window."""
        window_days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}.get(time_range, 7)
        since = datetime.utcnow() - timedelta(days=window_days)

        match: Dict[str, Any] = {"started_at": {"$gte": since}}
        if organization_id:
            match["organization_id"] = _coerce_oid(organization_id)
        if user_id:
            match["user_id"] = _coerce_oid(user_id)

        col = await self._get_collection()

        # Totals by status
        status_rows = await col.aggregate([
            {"$match": match},
            {"$group": {"_id": "$status", "count": {"$sum": 1}}},
        ]).to_list(length=None)
        by_status = {r["_id"] or "unknown": r["count"] for r in status_rows}
        total = sum(by_status.values())
        completed = by_status.get("completed", 0)
        errors = by_status.get("error", 0)

        # Per-agent breakdown
        agent_rows = await col.aggregate([
            {"$match": match},
            {
                "$group": {
                    "_id": "$agent_key",
                    "runs": {"$sum": 1},
                    "completed": {"$sum": {"$cond": [{"$eq": ["$status", "completed"]}, 1, 0]}},
                    "errors": {"$sum": {"$cond": [{"$eq": ["$status", "error"]}, 1, 0]}},
                    "avg_ms": {"$avg": "$duration_ms"},
                    "p95_ms": {"$max": "$duration_ms"},   # rough; Mongo lacks native p95
                    "tokens_in": {"$sum": {"$ifNull": ["$tokens_input", 0]}},
                    "tokens_out": {"$sum": {"$ifNull": ["$tokens_output", 0]}},
                    "cost": {"$sum": {"$ifNull": ["$cost_usd", 0]}},
                }
            },
            {"$sort": {"runs": -1}},
        ]).to_list(length=None)

        # Per-day series
        series_rows = await col.aggregate([
            {"$match": match},
            {
                "$group": {
                    "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$started_at"}},
                    "runs": {"$sum": 1},
                    "errors": {"$sum": {"$cond": [{"$eq": ["$status", "error"]}, 1, 0]}},
                }
            },
            {"$sort": {"_id": 1}},
        ]).to_list(length=None)

        return {
            "time_range": time_range,
            "since": since.isoformat() + "Z",
            "total_runs": total,
            "completed": completed,
            "errors": errors,
            "success_rate": (completed / total) if total else 0.0,
            "by_status": by_status,
            "by_agent": [
                {
                    "agent_key": r["_id"] or "unknown",
                    "runs": r["runs"],
                    "completed": r["completed"],
                    "errors": r["errors"],
                    "avg_duration_ms": int(r["avg_ms"]) if r.get("avg_ms") else None,
                    "max_duration_ms": int(r["p95_ms"]) if r.get("p95_ms") else None,
                    "success_rate": (r["completed"] / r["runs"]) if r["runs"] else 0.0,
                    "tokens_input": r["tokens_in"],
                    "tokens_output": r["tokens_out"],
                    "cost_usd": round(float(r["cost"] or 0), 4),
                }
                for r in agent_rows
            ],
            "series_by_day": [{"day": r["_id"], "runs": r["runs"], "errors": r["errors"]} for r in series_rows if r["_id"]],
        }


agent_run_repository = AgentRunRepository()
