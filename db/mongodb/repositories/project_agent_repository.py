"""
ProjectAgentRepository — per-project agent activations.

A `project_agents` row binds a platform agent (one of the 21 keys) OR a
custom agent (Agent Studio) to a project.  Carries per-project overrides:
model selection, autonomy level, fallback chain, and config overrides.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING

from backend.db.mongodb.base_repository import BaseRepository
from backend.db.scoping import require_org, scoped_filter, to_oid
from backend.models.workspace import ProjectAgent, ProjectAgentAdd

logger = structlog.get_logger(__name__)


class ProjectAgentRepository(BaseRepository[ProjectAgent]):
    def __init__(self) -> None:
        super().__init__("project_agents", ProjectAgent)

    async def _create_indexes(self) -> None:
        col = await self.collection
        # Sparse uniqueness: a project can only have one row per (agent_key)
        # or one per (custom_agent_id).
        await col.create_index(
            [("project_id", ASCENDING), ("agent_key", ASCENDING)],
            unique=True, sparse=True,
        )
        await col.create_index(
            [("project_id", ASCENDING), ("custom_agent_id", ASCENDING)],
            unique=True, sparse=True,
        )
        await col.create_index("organization_id")
        await col.create_index("project_id")
        await col.create_index("enabled")

    @require_org
    async def attach(
        self,
        *,
        project_id: str,
        organization_id: str,
        payload: ProjectAgentAdd,
        actor_id: str,
    ) -> ProjectAgent:
        if not payload.agent_key and not payload.custom_agent_id:
            raise ValueError("agent_key or custom_agent_id is required")
        if payload.agent_key and payload.custom_agent_id:
            raise ValueError("Only one of agent_key or custom_agent_id can be set")

        col = await self.collection
        now = datetime.utcnow()
        doc: Dict[str, Any] = {
            "project_id": ObjectId(project_id),
            "organization_id": ObjectId(organization_id),
            "agent_key": payload.agent_key,
            "custom_agent_id": ObjectId(payload.custom_agent_id) if payload.custom_agent_id else None,
            "enabled": bool(payload.enabled),
            "autonomy_level": payload.autonomy_level or "suggest",
            "config_overrides": payload.config_overrides or {},
            "fallback_chain": list(payload.fallback_chain or []),
            "attached_by": ObjectId(actor_id),
            "updated_at": now,
        }
        # Upsert by (project, agent_key) or (project, custom_agent_id)
        if payload.agent_key:
            filt = {"project_id": doc["project_id"], "agent_key": payload.agent_key}
        else:
            filt = {"project_id": doc["project_id"], "custom_agent_id": doc["custom_agent_id"]}
        await col.update_one(
            filt,
            {"$set": doc, "$setOnInsert": {"created_at": now}},
            upsert=True,
        )
        row = await col.find_one(filt)
        return ProjectAgent(**row)

    @require_org
    async def detach(
        self,
        *,
        project_id: str,
        organization_id: str,
        agent_key: Optional[str] = None,
        custom_agent_id: Optional[str] = None,
    ) -> bool:
        if not agent_key and not custom_agent_id:
            return False
        col = await self.collection
        filt: Dict[str, Any] = {"project_id": ObjectId(project_id)}
        if agent_key:
            filt["agent_key"] = agent_key
        else:
            filt["custom_agent_id"] = ObjectId(custom_agent_id)
        result = await col.delete_one(filt)
        return result.deleted_count > 0

    @require_org
    async def update(
        self,
        *,
        project_id: str,
        organization_id: str,
        agent_ref: str,
        patch: Dict[str, Any],
    ) -> Optional[ProjectAgent]:
        """`agent_ref` is either a platform key or an ObjectId string for a
        custom agent.  We try both lookup shapes."""
        col = await self.collection
        update = {k: v for k, v in patch.items() if v is not None}
        if not update:
            return None
        update["updated_at"] = datetime.utcnow()
        # Try platform first.
        result = await col.find_one_and_update(
            {"project_id": ObjectId(project_id), "agent_key": agent_ref},
            {"$set": update},
            return_document=True,
        )
        if result:
            return ProjectAgent(**result)
        # Then custom.
        oid = to_oid(agent_ref)
        if oid is None:
            return None
        result = await col.find_one_and_update(
            {"project_id": ObjectId(project_id), "custom_agent_id": oid},
            {"$set": update},
            return_document=True,
        )
        return ProjectAgent(**result) if result else None

    @require_org
    async def list_for_project(
        self,
        *,
        project_id: str,
        organization_id: str,
        enabled_only: bool = False,
    ) -> List[ProjectAgent]:
        query: Dict[str, Any] = scoped_filter(
            {"project_id": ObjectId(project_id)}, organization_id,
        )
        if enabled_only:
            query["enabled"] = True
        col = await self.collection
        cursor = col.find(query).sort("created_at", ASCENDING)
        return [ProjectAgent(**r) for r in await cursor.to_list(length=200)]

    @require_org
    async def get_by_ref(
        self,
        *,
        project_id: str,
        organization_id: str,
        agent_ref: str,
    ) -> Optional[ProjectAgent]:
        col = await self.collection
        row = await col.find_one({
            "project_id": ObjectId(project_id),
            "agent_key": agent_ref,
        })
        if row:
            return ProjectAgent(**row)
        oid = to_oid(agent_ref)
        if oid is None:
            return None
        row = await col.find_one({
            "project_id": ObjectId(project_id),
            "custom_agent_id": oid,
        })
        return ProjectAgent(**row) if row else None


project_agent_repository = ProjectAgentRepository()
