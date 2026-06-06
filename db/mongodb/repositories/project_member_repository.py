"""
ProjectMemberRepository — normalised project membership rows.

Holds the per-project role (lead / editor / reviewer / viewer).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING

from backend.db.mongodb.base_repository import BaseRepository
from backend.db.scoping import require_org, scoped_filter
from backend.models.workspace import ProjectMember, ProjectRoleEnum

logger = structlog.get_logger(__name__)


class ProjectMemberRepository(BaseRepository[ProjectMember]):
    def __init__(self) -> None:
        super().__init__("project_members", ProjectMember)

    async def _create_indexes(self) -> None:
        col = await self.collection
        await col.create_index(
            [("project_id", ASCENDING), ("user_id", ASCENDING)], unique=True,
        )
        await col.create_index("user_id")
        await col.create_index([("project_id", ASCENDING), ("role", ASCENDING)])
        await col.create_index("organization_id")

    @require_org
    async def add_or_update(
        self,
        *,
        project_id: str,
        user_id: str,
        organization_id: str,
        role: ProjectRoleEnum = ProjectRoleEnum.EDITOR,
        invited_by: Optional[str] = None,
    ) -> ProjectMember:
        col = await self.collection
        now = datetime.utcnow()
        doc = {
            "project_id": ObjectId(project_id),
            "user_id": ObjectId(user_id),
            "organization_id": ObjectId(organization_id),
            "role": role.value if isinstance(role, ProjectRoleEnum) else str(role),
            "invited_by": ObjectId(invited_by) if invited_by else None,
        }
        await col.update_one(
            {"project_id": doc["project_id"], "user_id": doc["user_id"]},
            {"$set": {k: v for k, v in doc.items()},
             "$setOnInsert": {"joined_at": now}},
            upsert=True,
        )
        row = await col.find_one(
            {"project_id": doc["project_id"], "user_id": doc["user_id"]}
        )
        return ProjectMember(**row)

    @require_org
    async def remove(
        self,
        *,
        project_id: str,
        user_id: str,
        organization_id: str,  # noqa: ARG002
    ) -> bool:
        col = await self.collection
        result = await col.delete_one({
            "project_id": ObjectId(project_id),
            "user_id": ObjectId(user_id),
        })
        return result.deleted_count > 0

    @require_org
    async def list_for_project(
        self,
        *,
        project_id: str,
        organization_id: str,
        role: Optional[ProjectRoleEnum] = None,
        skip: int = 0,
        limit: int = 200,
    ) -> List[ProjectMember]:
        query: Dict[str, Any] = scoped_filter(
            {"project_id": ObjectId(project_id)}, organization_id,
        )
        if role:
            query["role"] = role.value if isinstance(role, ProjectRoleEnum) else str(role)
        col = await self.collection
        cursor = col.find(query).sort("joined_at", ASCENDING).skip(skip).limit(limit)
        return [ProjectMember(**r) for r in await cursor.to_list(length=limit)]

    @require_org
    async def get_role(
        self,
        *,
        project_id: str,
        user_id: str,
        organization_id: str,  # noqa: ARG002
    ) -> Optional[str]:
        col = await self.collection
        row = await col.find_one({
            "project_id": ObjectId(project_id),
            "user_id": ObjectId(user_id),
        }, {"role": 1})
        return row["role"] if row else None

    @require_org
    async def update_role(
        self,
        *,
        project_id: str,
        user_id: str,
        organization_id: str,  # noqa: ARG002
        role: ProjectRoleEnum,
    ) -> Optional[ProjectMember]:
        col = await self.collection
        result = await col.find_one_and_update(
            {"project_id": ObjectId(project_id), "user_id": ObjectId(user_id)},
            {"$set": {"role": role.value if isinstance(role, ProjectRoleEnum) else str(role)}},
            return_document=True,
        )
        return ProjectMember(**result) if result else None

    async def list_for_user(
        self,
        *,
        user_id: str,
        organization_id: Optional[str] = None,
        limit: int = 500,
    ) -> List[ProjectMember]:
        query: Dict[str, Any] = {"user_id": ObjectId(user_id)}
        if organization_id:
            query["organization_id"] = ObjectId(organization_id)
        col = await self.collection
        cursor = col.find(query).limit(limit)
        return [ProjectMember(**r) for r in await cursor.to_list(length=limit)]


project_member_repository = ProjectMemberRepository()
