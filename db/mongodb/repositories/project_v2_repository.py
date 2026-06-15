"""
ProjectV2Repository — org-scoped projects collection (`projects`).

Replaces the legacy `lumicoria_projects` collection (user-scoped, embedded
tasks).  Projects are now first-class org resources that can optionally
belong to a team; tasks live in their own collection and link via
`project_id` (already supported in the Task model).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from backend.db.mongodb.base_repository import BaseRepository
from backend.db.scoping import require_org, scoped_filter, to_oid
from backend.models.workspace import (
    ProjectStatus,
    ProjectV2,
    ProjectV2Create,
    ProjectVisibility,
    slugify,
)

logger = structlog.get_logger(__name__)


class ProjectV2Repository(BaseRepository[ProjectV2]):
    def __init__(self) -> None:
        super().__init__("projects", ProjectV2)

    async def _create_indexes(self) -> None:
        col = await self.collection
        await col.create_index([("organization_id", ASCENDING), ("slug", ASCENDING)], unique=True)
        await col.create_index([("organization_id", ASCENDING), ("team_id", ASCENDING)])
        await col.create_index([("organization_id", ASCENDING), ("status", ASCENDING), ("updated_at", DESCENDING)])
        await col.create_index([("organization_id", ASCENDING), ("is_archived", ASCENDING)])
        await col.create_index("member_ids")
        await col.create_index("lead_id")
        await col.create_index("agent_keys")
        await col.create_index("custom_agent_ids")
        await col.create_index("tag_ids")
        await col.create_index([("created_at", DESCENDING)])
        await col.create_index([("name", "text"), ("description", "text")])

    # ───────────────────────────────────────── lifecycle

    @require_org
    async def create_project(
        self,
        payload: ProjectV2Create,
        *,
        organization_id: str,
        creator_id: str,
    ) -> ProjectV2:
        slug = payload.slug or slugify(payload.name)
        slug = await self._unique_slug(organization_id, slug)

        member_ids = list({creator_id, *payload.member_ids})
        if payload.lead_id:
            member_ids = list({*member_ids, payload.lead_id})

        doc: Dict[str, Any] = {
            "organization_id": ObjectId(organization_id),
            "team_id": ObjectId(payload.team_id) if payload.team_id else None,
            "name": payload.name,
            "slug": slug,
            "description": payload.description,
            "status": payload.status.value if isinstance(payload.status, ProjectStatus) else str(payload.status),
            "priority": payload.priority or "medium",
            "color": payload.color or "#6C4AB0",
            "cover_image_url": payload.cover_image_url,
            "due_date": payload.due_date,
            "lead_id": ObjectId(payload.lead_id) if payload.lead_id else ObjectId(creator_id),
            "member_ids": [ObjectId(uid) for uid in member_ids],
            "agent_keys": list(payload.agent_keys or []),
            "custom_agent_ids": [],
            "tag_ids": [],
            "strict_mode": bool(payload.strict_mode),
            "visibility": payload.visibility.value if isinstance(payload.visibility, ProjectVisibility) else str(payload.visibility),
            "settings": payload.settings or {},
            "branding": {},
            "metadata": payload.metadata or {},
            "is_archived": False,
            "archived_at": None,
            "created_by": ObjectId(creator_id),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        col = await self.collection
        result = await col.insert_one(doc)
        created = await col.find_one({"_id": result.inserted_id})
        return ProjectV2(**created)

    @require_org
    async def get_project(
        self,
        project_id: str,
        *,
        organization_id: str,
    ) -> Optional[ProjectV2]:
        oid = to_oid(project_id)
        if oid is None:
            return None
        col = await self.collection
        doc = await col.find_one(scoped_filter({"_id": oid}, organization_id))
        return ProjectV2(**doc) if doc else None

    @require_org
    async def get_project_by_slug(
        self,
        slug: str,
        *,
        organization_id: str,
    ) -> Optional[ProjectV2]:
        col = await self.collection
        doc = await col.find_one(scoped_filter({"slug": slug}, organization_id))
        return ProjectV2(**doc) if doc else None

    @require_org
    async def list_projects(
        self,
        *,
        organization_id: str,
        team_id: Optional[str] = None,
        member_id: Optional[str] = None,
        status: Optional[str] = None,
        include_archived: bool = False,
        search: Optional[str] = None,
        skip: int = 0,
        limit: int = 100,
    ) -> List[ProjectV2]:
        query: Dict[str, Any] = scoped_filter(None, organization_id)
        if not include_archived:
            query["is_archived"] = False
        if team_id is not None:
            tid = to_oid(team_id)
            query["team_id"] = tid
        if member_id is not None:
            mid = to_oid(member_id)
            if mid is not None:
                query["member_ids"] = mid
        if status:
            query["status"] = status
        if search:
            query["$text"] = {"$search": search}
        col = await self.collection
        cursor = col.find(query).sort("updated_at", DESCENDING).skip(skip).limit(limit)
        return [ProjectV2(**r) for r in await cursor.to_list(length=limit)]

    @require_org
    async def count_projects(
        self,
        *,
        organization_id: str,
        team_id: Optional[str] = None,
        include_archived: bool = False,
    ) -> int:
        query: Dict[str, Any] = scoped_filter(None, organization_id)
        if not include_archived:
            query["is_archived"] = False
        if team_id is not None:
            query["team_id"] = to_oid(team_id)
        col = await self.collection
        return await col.count_documents(query)

    # Fields that callers may legitimately want to unset via PATCH.
    # For these, an explicit None in the patch becomes a Mongo $unset
    # rather than being silently dropped (the default for every other
    # field, where None means "no change").
    _UNSETTABLE_FIELDS = frozenset(("team_id", "lead_id", "due_date", "description", "logo_url", "cover_image_url"))

    @require_org
    async def update_project(
        self,
        project_id: str,
        *,
        organization_id: str,
        patch: Dict[str, Any],
    ) -> Optional[ProjectV2]:
        oid = to_oid(project_id)
        if oid is None:
            return None

        # Split the incoming patch into $set + $unset based on which
        # field is being touched and whether the value is None.
        set_update: Dict[str, Any] = {}
        unset_update: Dict[str, str] = {}
        for k, v in patch.items():
            if v is None:
                if k in self._UNSETTABLE_FIELDS:
                    unset_update[k] = ""
                # else: silently ignore Nones for fields we don't support unsetting
                continue
            set_update[k] = v

        # Coerce ObjectId-typed sets
        for key in ("team_id", "lead_id"):
            if key in set_update and set_update[key] is not None:
                set_update[key] = ObjectId(str(set_update[key]))

        if not set_update and not unset_update:
            return await self.get_project(project_id, organization_id=organization_id)

        set_update["updated_at"] = datetime.utcnow()
        update_doc: Dict[str, Any] = {"$set": set_update}
        if unset_update:
            update_doc["$unset"] = unset_update

        col = await self.collection
        result = await col.find_one_and_update(
            scoped_filter({"_id": oid}, organization_id),
            update_doc,
            return_document=True,
        )
        return ProjectV2(**result) if result else None

    @require_org
    async def archive_project(
        self,
        project_id: str,
        *,
        organization_id: str,
        archived: bool = True,
    ) -> Optional[ProjectV2]:
        return await self.update_project(
            project_id,
            organization_id=organization_id,
            patch={
                "is_archived": archived,
                "archived_at": datetime.utcnow() if archived else None,
                "status": ProjectStatus.ARCHIVED.value if archived else ProjectStatus.ACTIVE.value,
            },
        )

    @require_org
    async def delete_project(
        self,
        project_id: str,
        *,
        organization_id: str,
    ) -> bool:
        oid = to_oid(project_id)
        if oid is None:
            return False
        col = await self.collection
        result = await col.delete_one(scoped_filter({"_id": oid}, organization_id))
        return result.deleted_count > 0

    # ───────────────────────────────────────── membership

    @require_org
    async def add_member(
        self,
        project_id: str,
        *,
        organization_id: str,
        user_id: str,
    ) -> Optional[ProjectV2]:
        oid = to_oid(project_id)
        if oid is None:
            return None
        col = await self.collection
        result = await col.find_one_and_update(
            scoped_filter({"_id": oid}, organization_id),
            {"$addToSet": {"member_ids": ObjectId(user_id)}, "$set": {"updated_at": datetime.utcnow()}},
            return_document=True,
        )
        return ProjectV2(**result) if result else None

    @require_org
    async def remove_member(
        self,
        project_id: str,
        *,
        organization_id: str,
        user_id: str,
    ) -> Optional[ProjectV2]:
        oid = to_oid(project_id)
        if oid is None:
            return None
        col = await self.collection
        result = await col.find_one_and_update(
            scoped_filter({"_id": oid}, organization_id),
            {"$pull": {"member_ids": ObjectId(user_id)}, "$set": {"updated_at": datetime.utcnow()}},
            return_document=True,
        )
        return ProjectV2(**result) if result else None

    # ───────────────────────────────────────── agents

    @require_org
    async def attach_agent_key(
        self,
        project_id: str,
        *,
        organization_id: str,
        agent_key: str,
    ) -> Optional[ProjectV2]:
        oid = to_oid(project_id)
        if oid is None:
            return None
        col = await self.collection
        result = await col.find_one_and_update(
            scoped_filter({"_id": oid}, organization_id),
            {"$addToSet": {"agent_keys": agent_key}, "$set": {"updated_at": datetime.utcnow()}},
            return_document=True,
        )
        return ProjectV2(**result) if result else None

    @require_org
    async def detach_agent_key(
        self,
        project_id: str,
        *,
        organization_id: str,
        agent_key: str,
    ) -> Optional[ProjectV2]:
        oid = to_oid(project_id)
        if oid is None:
            return None
        col = await self.collection
        result = await col.find_one_and_update(
            scoped_filter({"_id": oid}, organization_id),
            {"$pull": {"agent_keys": agent_key}, "$set": {"updated_at": datetime.utcnow()}},
            return_document=True,
        )
        return ProjectV2(**result) if result else None

    @require_org
    async def attach_custom_agent(
        self,
        project_id: str,
        *,
        organization_id: str,
        custom_agent_id: str,
    ) -> Optional[ProjectV2]:
        oid = to_oid(project_id)
        if oid is None:
            return None
        col = await self.collection
        result = await col.find_one_and_update(
            scoped_filter({"_id": oid}, organization_id),
            {"$addToSet": {"custom_agent_ids": ObjectId(custom_agent_id)},
             "$set": {"updated_at": datetime.utcnow()}},
            return_document=True,
        )
        return ProjectV2(**result) if result else None

    @require_org
    async def detach_custom_agent(
        self,
        project_id: str,
        *,
        organization_id: str,
        custom_agent_id: str,
    ) -> Optional[ProjectV2]:
        oid = to_oid(project_id)
        if oid is None:
            return None
        col = await self.collection
        result = await col.find_one_and_update(
            scoped_filter({"_id": oid}, organization_id),
            {"$pull": {"custom_agent_ids": ObjectId(custom_agent_id)},
             "$set": {"updated_at": datetime.utcnow()}},
            return_document=True,
        )
        return ProjectV2(**result) if result else None

    # ───────────────────────────────────────── helpers

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


project_v2_repository = ProjectV2Repository()
