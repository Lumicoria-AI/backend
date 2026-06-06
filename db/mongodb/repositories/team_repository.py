"""
TeamRepository — production-grade CRUD for the `teams` collection.

Org-scoped, slug-unique-per-org, with sparse index on member_ids for fast
membership lookup.  Pairs with `team_member_repository` for the normalised
membership rows that carry team-specific roles.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from backend.db.mongodb.base_repository import BaseRepository
from backend.db.scoping import require_org, scoped_filter, to_oid
from backend.models.workspace import Team, TeamCreate, slugify

logger = structlog.get_logger(__name__)


class TeamRepository(BaseRepository[Team]):
    def __init__(self) -> None:
        super().__init__("teams", Team)

    async def _create_indexes(self) -> None:
        col = await self.collection
        await col.create_index([("organization_id", ASCENDING), ("slug", ASCENDING)], unique=True)
        await col.create_index([("organization_id", ASCENDING), ("is_archived", ASCENDING)])
        await col.create_index([("organization_id", ASCENDING), ("department_tag", ASCENDING)])
        await col.create_index("member_ids")
        await col.create_index("admin_ids")
        await col.create_index("owner_id")
        await col.create_index([("created_at", DESCENDING)])
        await col.create_index([("name", "text"), ("description", "text")])

    # ───────────────────────────────────────── lifecycle

    @require_org
    async def create_team(
        self,
        payload: TeamCreate,
        *,
        organization_id: str,
        creator_id: str,
    ) -> Team:
        slug = payload.slug or slugify(payload.name)
        # Ensure uniqueness within org by appending a numeric suffix if needed
        slug = await self._unique_slug(organization_id, slug)

        member_ids = list({creator_id, *payload.member_ids})
        admin_ids = list({creator_id, *payload.admin_ids})

        doc: Dict[str, Any] = {
            "organization_id": ObjectId(organization_id),
            "name": payload.name,
            "slug": slug,
            "description": payload.description,
            "department_tag": payload.department_tag,
            "color": payload.color or "#6C4AB0",
            "logo_url": payload.logo_url,
            "cover_url": payload.cover_url,
            "owner_id": ObjectId(creator_id),
            "admin_ids": [ObjectId(uid) for uid in admin_ids],
            "member_ids": [ObjectId(uid) for uid in member_ids],
            "settings": payload.settings or {},
            "branding": {},
            "is_archived": False,
            "archived_at": None,
            "metadata": payload.metadata or {},
            "created_by": ObjectId(creator_id),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        col = await self.collection
        result = await col.insert_one(doc)
        created = await col.find_one({"_id": result.inserted_id})
        return Team(**created)

    @require_org
    async def get_team(
        self,
        team_id: str,
        *,
        organization_id: str,
    ) -> Optional[Team]:
        oid = to_oid(team_id)
        if oid is None:
            return None
        col = await self.collection
        doc = await col.find_one(scoped_filter({"_id": oid}, organization_id))
        return Team(**doc) if doc else None

    @require_org
    async def get_team_by_slug(
        self,
        slug: str,
        *,
        organization_id: str,
    ) -> Optional[Team]:
        col = await self.collection
        doc = await col.find_one(scoped_filter({"slug": slug}, organization_id))
        return Team(**doc) if doc else None

    @require_org
    async def list_teams(
        self,
        *,
        organization_id: str,
        include_archived: bool = False,
        search: Optional[str] = None,
        department_tag: Optional[str] = None,
        member_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
        sort_field: str = "updated_at",
        sort_dir: int = -1,
    ) -> List[Team]:
        query: Dict[str, Any] = scoped_filter(None, organization_id)
        if not include_archived:
            query["is_archived"] = False
        if department_tag:
            query["department_tag"] = department_tag
        if member_id:
            mid = to_oid(member_id)
            if mid is not None:
                query["member_ids"] = mid
        if search:
            query["$text"] = {"$search": search}

        col = await self.collection
        cursor = col.find(query).sort(sort_field, sort_dir).skip(skip).limit(limit)
        rows = await cursor.to_list(length=limit)
        return [Team(**r) for r in rows]

    @require_org
    async def count_teams(
        self,
        *,
        organization_id: str,
        include_archived: bool = False,
    ) -> int:
        query: Dict[str, Any] = scoped_filter(None, organization_id)
        if not include_archived:
            query["is_archived"] = False
        col = await self.collection
        return await col.count_documents(query)

    @require_org
    async def update_team(
        self,
        team_id: str,
        *,
        organization_id: str,
        patch: Dict[str, Any],
    ) -> Optional[Team]:
        oid = to_oid(team_id)
        if oid is None:
            return None
        update = {k: v for k, v in patch.items() if v is not None}
        if not update:
            return await self.get_team(team_id, organization_id=organization_id)
        update["updated_at"] = datetime.utcnow()
        col = await self.collection
        result = await col.find_one_and_update(
            scoped_filter({"_id": oid}, organization_id),
            {"$set": update},
            return_document=True,
        )
        return Team(**result) if result else None

    @require_org
    async def archive_team(
        self,
        team_id: str,
        *,
        organization_id: str,
        archived: bool = True,
    ) -> Optional[Team]:
        return await self.update_team(
            team_id,
            organization_id=organization_id,
            patch={
                "is_archived": archived,
                "archived_at": datetime.utcnow() if archived else None,
            },
        )

    @require_org
    async def delete_team(
        self,
        team_id: str,
        *,
        organization_id: str,
    ) -> bool:
        oid = to_oid(team_id)
        if oid is None:
            return False
        col = await self.collection
        result = await col.delete_one(scoped_filter({"_id": oid}, organization_id))
        return result.deleted_count > 0

    # ───────────────────────────────────────── membership helpers

    @require_org
    async def add_member(
        self,
        team_id: str,
        *,
        organization_id: str,
        user_id: str,
        promote_to_admin: bool = False,
    ) -> Optional[Team]:
        oid = to_oid(team_id)
        if oid is None:
            return None
        uid = ObjectId(user_id)
        update: Dict[str, Any] = {"$addToSet": {"member_ids": uid}}
        if promote_to_admin:
            update["$addToSet"]["admin_ids"] = uid
        update["$set"] = {"updated_at": datetime.utcnow()}
        col = await self.collection
        result = await col.find_one_and_update(
            scoped_filter({"_id": oid}, organization_id),
            update,
            return_document=True,
        )
        return Team(**result) if result else None

    @require_org
    async def remove_member(
        self,
        team_id: str,
        *,
        organization_id: str,
        user_id: str,
    ) -> Optional[Team]:
        oid = to_oid(team_id)
        if oid is None:
            return None
        uid = ObjectId(user_id)
        col = await self.collection
        result = await col.find_one_and_update(
            scoped_filter({"_id": oid}, organization_id),
            {
                "$pull": {"member_ids": uid, "admin_ids": uid},
                "$set": {"updated_at": datetime.utcnow()},
            },
            return_document=True,
        )
        return Team(**result) if result else None

    @require_org
    async def promote(
        self,
        team_id: str,
        *,
        organization_id: str,
        user_id: str,
    ) -> Optional[Team]:
        oid = to_oid(team_id)
        if oid is None:
            return None
        uid = ObjectId(user_id)
        col = await self.collection
        result = await col.find_one_and_update(
            scoped_filter({"_id": oid}, organization_id),
            {
                "$addToSet": {"admin_ids": uid, "member_ids": uid},
                "$set": {"updated_at": datetime.utcnow()},
            },
            return_document=True,
        )
        return Team(**result) if result else None

    @require_org
    async def demote(
        self,
        team_id: str,
        *,
        organization_id: str,
        user_id: str,
    ) -> Optional[Team]:
        oid = to_oid(team_id)
        if oid is None:
            return None
        uid = ObjectId(user_id)
        col = await self.collection
        result = await col.find_one_and_update(
            scoped_filter({"_id": oid}, organization_id),
            {"$pull": {"admin_ids": uid}, "$set": {"updated_at": datetime.utcnow()}},
            return_document=True,
        )
        return Team(**result) if result else None

    @require_org
    async def transfer_ownership(
        self,
        team_id: str,
        *,
        organization_id: str,
        new_owner_id: str,
    ) -> Optional[Team]:
        return await self.update_team(
            team_id,
            organization_id=organization_id,
            patch={"owner_id": ObjectId(new_owner_id)},
        )

    @require_org
    async def get_teams_for_user(
        self,
        *,
        organization_id: str,
        user_id: str,
        include_archived: bool = False,
        limit: int = 200,
    ) -> List[Team]:
        query: Dict[str, Any] = scoped_filter({"member_ids": ObjectId(user_id)}, organization_id)
        if not include_archived:
            query["is_archived"] = False
        col = await self.collection
        cursor = col.find(query).sort("updated_at", DESCENDING).limit(limit)
        return [Team(**r) for r in await cursor.to_list(length=limit)]

    # ───────────────────────────────────────── internals

    async def _unique_slug(self, organization_id: str, base: str) -> str:
        col = await self.collection
        candidate = base
        n = 1
        while await col.count_documents(
            scoped_filter({"slug": candidate}, organization_id), limit=1
        ):
            n += 1
            candidate = f"{base}-{n}"
        return candidate


team_repository = TeamRepository()
