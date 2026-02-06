from __future__ import annotations

from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId
import structlog

from ..base_repository import BaseRepository
from backend.models.mongodb_models import AgentWorkflow, AgentStatus

logger = structlog.get_logger()


class WorkflowRepository(BaseRepository[AgentWorkflow]):
    def __init__(self):
        super().__init__("agent_workflows", AgentWorkflow)

    async def _create_indexes(self):
        collection = await self.collection
        await collection.create_index("organization_id")
        await collection.create_index("created_by")
        await collection.create_index("is_public")
        await collection.create_index("status")
        await collection.create_index("created_at")
        await collection.create_index("tags")
        await collection.create_index("postgres_id")
        await collection.create_index([
            ("organization_id", 1),
            ("created_at", -1)
        ])
        await collection.create_index([
            ("created_by", 1),
            ("created_at", -1)
        ])

    async def create_workflow(
        self,
        data: Dict[str, Any],
        organization_id: str,
        created_by: str,
        postgres_id: Optional[str] = None
    ) -> Dict[str, Any]:
        entry = {
            "name": data.get("name"),
            "description": data.get("description"),
            "components": data.get("components", []),
            "nodes": data.get("nodes", []),
            "connections": data.get("connections", []),
            "organization_id": ObjectId(organization_id) if organization_id else None,
            "created_by": ObjectId(created_by) if created_by else None,
            "version": data.get("version", "1.0.0"),
            "is_public": data.get("is_public", False),
            "tags": data.get("tags", []),
            "status": data.get("status", AgentStatus.DRAFT),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        if postgres_id:
            entry["postgres_id"] = postgres_id

        created = await self.create(entry)
        return self._serialize(created)

    async def get_workflow_by_id(
        self,
        workflow_id: str,
        organization_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        collection = await self.collection
        query: Dict[str, Any] = {"_id": ObjectId(workflow_id)}
        if organization_id:
            query["organization_id"] = ObjectId(organization_id)
        doc = await collection.find_one(query)
        return self._serialize(doc) if doc else None

    async def get_workflow_by_postgres_id(
        self,
        postgres_id: str,
        organization_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        collection = await self.collection
        query: Dict[str, Any] = {"postgres_id": postgres_id}
        if organization_id:
            query["organization_id"] = ObjectId(organization_id)
        doc = await collection.find_one(query)
        return self._serialize(doc) if doc else None

    async def list_workflows(
        self,
        organization_id: Optional[str] = None,
        created_by: Optional[str] = None,
        include_public: bool = True,
        skip: int = 0,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        collection = await self.collection
        query: Dict[str, Any] = {}
        if organization_id:
            query["organization_id"] = ObjectId(organization_id)
        if created_by:
            query["created_by"] = ObjectId(created_by)
        if not include_public:
            query["is_public"] = False

        cursor = collection.find(query).sort("created_at", -1).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [self._serialize(d) for d in docs]

    async def update_workflow(
        self,
        workflow_id: str,
        update_data: Dict[str, Any],
        organization_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        collection = await self.collection
        query: Dict[str, Any] = {"_id": ObjectId(workflow_id)}
        if organization_id:
            query["organization_id"] = ObjectId(organization_id)
        update_data["updated_at"] = datetime.utcnow()
        await collection.update_one(query, {"$set": update_data})
        doc = await collection.find_one(query)
        return self._serialize(doc) if doc else None

    async def update_workflow_by_postgres_id(
        self,
        postgres_id: str,
        update_data: Dict[str, Any],
        organization_id: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        collection = await self.collection
        query: Dict[str, Any] = {"postgres_id": postgres_id}
        if organization_id:
            query["organization_id"] = ObjectId(organization_id)
        update_data["updated_at"] = datetime.utcnow()
        await collection.update_one(query, {"$set": update_data})
        doc = await collection.find_one(query)
        return self._serialize(doc) if doc else None

    async def delete_workflow(
        self,
        workflow_id: str,
        organization_id: Optional[str] = None
    ) -> bool:
        collection = await self.collection
        query: Dict[str, Any] = {"_id": ObjectId(workflow_id)}
        if organization_id:
            query["organization_id"] = ObjectId(organization_id)
        result = await collection.delete_one(query)
        return result.deleted_count > 0

    async def delete_workflow_by_postgres_id(
        self,
        postgres_id: str,
        organization_id: Optional[str] = None
    ) -> bool:
        collection = await self.collection
        query: Dict[str, Any] = {"postgres_id": postgres_id}
        if organization_id:
            query["organization_id"] = ObjectId(organization_id)
        result = await collection.delete_one(query)
        return result.deleted_count > 0

    def _serialize(self, doc: Any) -> Dict[str, Any]:
        if not doc:
            return {}
        if isinstance(doc, AgentWorkflow):
            doc = doc.model_dump(by_alias=True)
        return {
            "id": str(doc.get("_id")) if doc.get("_id") else None,
            "name": doc.get("name"),
            "description": doc.get("description"),
            "components": doc.get("components", []),
            "nodes": doc.get("nodes", []),
            "connections": doc.get("connections", []),
            "organization_id": str(doc.get("organization_id")) if doc.get("organization_id") else None,
            "created_by": str(doc.get("created_by")) if doc.get("created_by") else None,
            "version": doc.get("version", "1.0.0"),
            "is_public": doc.get("is_public", False),
            "tags": doc.get("tags", []),
            "status": doc.get("status"),
            "created_at": doc.get("created_at"),
            "updated_at": doc.get("updated_at"),
            "postgres_id": doc.get("postgres_id"),
        }


workflow_repository = WorkflowRepository()
