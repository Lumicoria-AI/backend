"""
AutomationsRepository — rules engine storage.

Each automation row:
    organization_id  ObjectId
    team_id?         ObjectId
    project_id?      ObjectId
    name             str
    description      str
    trigger          { type: "event"|"schedule"|"manual", config: dict }
    conditions       list[ { field, op, value } ]
    actions          list[ { type, config } ]
    enabled          bool
    last_run_at      datetime
    run_count        int
    error_count      int
    created_by       ObjectId
    created_at, updated_at

Each automation_runs row:
    automation_id    ObjectId
    organization_id  ObjectId
    status           pending | running | completed | error
    trigger_payload  dict
    actions_executed list[ { type, result, error? } ]
    error?           str
    started_at, ended_at
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from backend.db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)

AUTOMATIONS = "automations"
RUNS = "automation_runs"


class AutomationsRepository:
    def __init__(self) -> None:
        self._initialised = False

    async def _ensure_indexes(self) -> None:
        if self._initialised:
            return
        a = await MongoDB.get_collection(AUTOMATIONS)
        r = await MongoDB.get_collection(RUNS)
        await a.create_index([("organization_id", ASCENDING), ("enabled", ASCENDING), ("trigger.type", ASCENDING)])
        await a.create_index([("organization_id", ASCENDING), ("project_id", ASCENDING)])
        await a.create_index([("organization_id", ASCENDING), ("team_id", ASCENDING)])
        await r.create_index([("automation_id", ASCENDING), ("started_at", DESCENDING)])
        await r.create_index([("organization_id", ASCENDING), ("started_at", DESCENDING)])
        self._initialised = True

    # ---- automations ----

    async def create(
        self,
        *,
        organization_id: str,
        name: str,
        trigger: Dict[str, Any],
        actions: List[Dict[str, Any]],
        conditions: Optional[List[Dict[str, Any]]] = None,
        team_id: Optional[str] = None,
        project_id: Optional[str] = None,
        description: Optional[str] = None,
        enabled: bool = True,
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(AUTOMATIONS)
        now = datetime.utcnow()
        doc = {
            "organization_id": ObjectId(organization_id),
            "team_id": ObjectId(team_id) if team_id else None,
            "project_id": ObjectId(project_id) if project_id else None,
            "name": name,
            "description": description,
            "trigger": trigger,
            "conditions": list(conditions or []),
            "actions": list(actions or []),
            "enabled": bool(enabled),
            "last_run_at": None,
            "run_count": 0,
            "error_count": 0,
            "created_by": ObjectId(created_by) if created_by else None,
            "metadata": metadata or {},
            "created_at": now,
            "updated_at": now,
        }
        result = await col.insert_one(doc)
        doc["_id"] = result.inserted_id
        return self._serialize(doc)

    async def get(self, automation_id: str, organization_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
        col = await MongoDB.get_collection(AUTOMATIONS)
        q: Dict[str, Any] = {"_id": ObjectId(automation_id)}
        if organization_id:
            q["organization_id"] = ObjectId(organization_id)
        row = await col.find_one(q)
        return self._serialize(row) if row else None

    async def list(
        self,
        *,
        organization_id: str,
        enabled: Optional[bool] = None,
        project_id: Optional[str] = None,
        team_id: Optional[str] = None,
        trigger_type: Optional[str] = None,
        limit: int = 200,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(AUTOMATIONS)
        q: Dict[str, Any] = {"organization_id": ObjectId(organization_id)}
        if enabled is not None:
            q["enabled"] = bool(enabled)
        if project_id:
            q["project_id"] = ObjectId(project_id)
        if team_id:
            q["team_id"] = ObjectId(team_id)
        if trigger_type:
            q["trigger.type"] = trigger_type
        cursor = col.find(q).sort("created_at", DESCENDING).skip(skip).limit(limit)
        return [self._serialize(r) for r in await cursor.to_list(length=limit)]

    async def update(
        self,
        automation_id: str,
        organization_id: str,
        *,
        patch: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        col = await MongoDB.get_collection(AUTOMATIONS)
        update = {k: v for k, v in patch.items() if v is not None}
        if not update:
            return await self.get(automation_id, organization_id)
        update["updated_at"] = datetime.utcnow()
        row = await col.find_one_and_update(
            {"_id": ObjectId(automation_id), "organization_id": ObjectId(organization_id)},
            {"$set": update},
            return_document=True,
        )
        return self._serialize(row) if row else None

    async def delete(self, automation_id: str, organization_id: str) -> bool:
        col = await MongoDB.get_collection(AUTOMATIONS)
        result = await col.delete_one({
            "_id": ObjectId(automation_id),
            "organization_id": ObjectId(organization_id),
        })
        return result.deleted_count > 0

    async def list_for_event(
        self,
        *,
        organization_id: str,
        event_type: str,
    ) -> List[Dict[str, Any]]:
        """List enabled event-triggered automations matching an event type."""
        await self._ensure_indexes()
        col = await MongoDB.get_collection(AUTOMATIONS)
        cursor = col.find({
            "organization_id": ObjectId(organization_id),
            "enabled": True,
            "trigger.type": "event",
            "trigger.config.event_type": event_type,
        })
        return [self._serialize(r) for r in await cursor.to_list(length=200)]

    # ---- runs ----

    async def record_run(
        self,
        *,
        automation_id: str,
        organization_id: str,
        status: str,
        trigger_payload: Optional[Dict[str, Any]] = None,
        actions_executed: Optional[List[Dict[str, Any]]] = None,
        error: Optional[str] = None,
        started_at: Optional[datetime] = None,
        ended_at: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        await self._ensure_indexes()
        runs = await MongoDB.get_collection(RUNS)
        automations = await MongoDB.get_collection(AUTOMATIONS)
        now = datetime.utcnow()
        doc = {
            "automation_id": ObjectId(automation_id),
            "organization_id": ObjectId(organization_id),
            "status": status,
            "trigger_payload": trigger_payload or {},
            "actions_executed": actions_executed or [],
            "error": error,
            "started_at": started_at or now,
            "ended_at": ended_at,
            "created_at": now,
        }
        result = await runs.insert_one(doc)
        doc["_id"] = result.inserted_id
        # Mirror counters on the parent.
        inc: Dict[str, int] = {"run_count": 1}
        if status == "error":
            inc["error_count"] = 1
        await automations.update_one(
            {"_id": ObjectId(automation_id)},
            {"$inc": inc, "$set": {"last_run_at": doc["started_at"], "updated_at": now}},
        )
        return self._serialize(doc)

    async def list_runs(
        self,
        *,
        organization_id: str,
        automation_id: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(RUNS)
        q: Dict[str, Any] = {"organization_id": ObjectId(organization_id)}
        if automation_id:
            q["automation_id"] = ObjectId(automation_id)
        if status:
            q["status"] = status
        cursor = col.find(q).sort("started_at", DESCENDING).skip(skip).limit(limit)
        return [self._serialize(r) for r in await cursor.to_list(length=limit)]

    @staticmethod
    def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc:
            return doc
        d = dict(doc)
        d["id"] = str(d.pop("_id"))
        for k in ("organization_id", "team_id", "project_id", "automation_id", "created_by"):
            if d.get(k) is not None:
                d[k] = str(d[k])
        return d


automations_repository = AutomationsRepository()
