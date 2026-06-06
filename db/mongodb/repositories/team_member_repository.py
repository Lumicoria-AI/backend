"""
TeamMemberRepository — normalised membership rows for `team_members`.

The Team document keeps `member_ids[]` + `admin_ids[]` for cheap "is X a
member?" checks.  This collection adds a per-membership row that carries
the team-scoped role and join metadata.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING

from backend.db.mongodb.base_repository import BaseRepository
from backend.db.scoping import require_org, scoped_filter, to_oid
from backend.models.workspace import TeamMember, TeamRoleEnum

logger = structlog.get_logger(__name__)


class TeamMemberRepository(BaseRepository[TeamMember]):
    def __init__(self) -> None:
        super().__init__("team_members", TeamMember)

    async def _create_indexes(self) -> None:
        col = await self.collection
        await col.create_index(
            [("team_id", ASCENDING), ("user_id", ASCENDING)],
            unique=True,
        )
        await col.create_index("user_id")
        await col.create_index([("team_id", ASCENDING), ("role", ASCENDING)])
        await col.create_index("organization_id")

    @require_org
    async def add_or_update(
        self,
        *,
        team_id: str,
        user_id: str,
        organization_id: str,
        role: TeamRoleEnum = TeamRoleEnum.EDITOR,
        invited_by: Optional[str] = None,
    ) -> TeamMember:
        col = await self.collection
        now = datetime.utcnow()
        doc = {
            "team_id": ObjectId(team_id),
            "user_id": ObjectId(user_id),
            "organization_id": ObjectId(organization_id),
            "role": role.value if isinstance(role, TeamRoleEnum) else str(role),
            "joined_at": now,
            "invited_by": ObjectId(invited_by) if invited_by else None,
        }
        await col.update_one(
            {"team_id": doc["team_id"], "user_id": doc["user_id"]},
            {"$set": {k: v for k, v in doc.items() if k not in ("joined_at",)},
             "$setOnInsert": {"joined_at": now}},
            upsert=True,
        )
        row = await col.find_one(
            {"team_id": doc["team_id"], "user_id": doc["user_id"]}
        )
        return TeamMember(**row)

    @require_org
    async def remove(
        self,
        *,
        team_id: str,
        user_id: str,
        organization_id: str,  # noqa: ARG002 — for require_org
    ) -> bool:
        col = await self.collection
        result = await col.delete_one({
            "team_id": ObjectId(team_id),
            "user_id": ObjectId(user_id),
        })
        return result.deleted_count > 0

    @require_org
    async def list_for_team(
        self,
        *,
        team_id: str,
        organization_id: str,
        role: Optional[TeamRoleEnum] = None,
        skip: int = 0,
        limit: int = 200,
    ) -> List[TeamMember]:
        query: Dict[str, Any] = scoped_filter({"team_id": ObjectId(team_id)}, organization_id)
        if role:
            query["role"] = role.value if isinstance(role, TeamRoleEnum) else str(role)
        col = await self.collection
        cursor = col.find(query).sort("joined_at", ASCENDING).skip(skip).limit(limit)
        return [TeamMember(**r) for r in await cursor.to_list(length=limit)]

    @require_org
    async def get_role(
        self,
        *,
        team_id: str,
        user_id: str,
        organization_id: str,  # noqa: ARG002
    ) -> Optional[str]:
        col = await self.collection
        row = await col.find_one({
            "team_id": ObjectId(team_id),
            "user_id": ObjectId(user_id),
        }, {"role": 1})
        return row["role"] if row else None

    @require_org
    async def update_role(
        self,
        *,
        team_id: str,
        user_id: str,
        organization_id: str,  # noqa: ARG002
        role: TeamRoleEnum,
    ) -> Optional[TeamMember]:
        col = await self.collection
        result = await col.find_one_and_update(
            {"team_id": ObjectId(team_id), "user_id": ObjectId(user_id)},
            {"$set": {"role": role.value if isinstance(role, TeamRoleEnum) else str(role)}},
            return_document=True,
        )
        return TeamMember(**result) if result else None

    @require_org
    async def count_for_team(
        self,
        *,
        team_id: str,
        organization_id: str,
    ) -> int:
        col = await self.collection
        return await col.count_documents(scoped_filter({"team_id": ObjectId(team_id)}, organization_id))

    async def list_for_user(
        self,
        *,
        user_id: str,
        organization_id: Optional[str] = None,
        limit: int = 200,
    ) -> List[TeamMember]:
        query: Dict[str, Any] = {"user_id": ObjectId(user_id)}
        if organization_id:
            query["organization_id"] = ObjectId(organization_id)
        col = await self.collection
        cursor = col.find(query).limit(limit)
        return [TeamMember(**r) for r in await cursor.to_list(length=limit)]


team_member_repository = TeamMemberRepository()
