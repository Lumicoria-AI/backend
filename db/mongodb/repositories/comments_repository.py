"""
CommentsRepository — cross-resource comments.

A single `comments` collection backs comments on tasks, projects, documents,
agent runs, and any future resource type that wants a threaded comment surface.

Schema:
    organization_id   ObjectId (required, scoping)
    resource_type     str      ("task" | "project" | "document" | "agent_run")
    resource_id       str      (the resource's _id as a string)
    user_id           ObjectId (commenter — None for agents)
    agent_key         str      (set when the commenter is a platform agent)
    body              str
    mentions          list[ObjectId]  user_ids @-mentioned
    reactions         dict     { ":fire:": [user_id, ...], ... }
    parent_id         ObjectId (None for top-level; set for thread replies)
    resolved          bool
    edited_at         datetime
    created_at, updated_at
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from backend.db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)

COLLECTION = "comments"


class CommentsRepository:
    def __init__(self) -> None:
        self._initialised = False

    async def _ensure_indexes(self) -> None:
        if self._initialised:
            return
        col = await MongoDB.get_collection(COLLECTION)
        await col.create_index([
            ("resource_type", ASCENDING),
            ("resource_id", ASCENDING),
            ("created_at", DESCENDING),
        ])
        await col.create_index("organization_id")
        await col.create_index("parent_id")
        await col.create_index("mentions")
        await col.create_index("user_id")
        self._initialised = True

    async def create(
        self,
        *,
        organization_id: str,
        resource_type: str,
        resource_id: str,
        body: str,
        user_id: Optional[str] = None,
        agent_key: Optional[str] = None,
        mentions: Optional[List[str]] = None,
        parent_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        doc: Dict[str, Any] = {
            "organization_id": ObjectId(organization_id),
            "resource_type": resource_type,
            "resource_id": str(resource_id),
            "user_id": ObjectId(user_id) if user_id else None,
            "agent_key": agent_key,
            "body": body,
            "mentions": [ObjectId(m) for m in (mentions or []) if ObjectId.is_valid(m)],
            "reactions": {},
            "parent_id": ObjectId(parent_id) if parent_id and ObjectId.is_valid(parent_id) else None,
            "resolved": False,
            "edited_at": None,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        result = await col.insert_one(doc)
        doc["_id"] = result.inserted_id
        return self._serialize(doc)

    async def list(
        self,
        *,
        organization_id: str,
        resource_type: str,
        resource_id: str,
        parent_id: Optional[str] = None,
        limit: int = 200,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        query: Dict[str, Any] = {
            "organization_id": ObjectId(organization_id),
            "resource_type": resource_type,
            "resource_id": str(resource_id),
        }
        if parent_id is None:
            query["parent_id"] = None
        else:
            query["parent_id"] = ObjectId(parent_id)
        cursor = col.find(query).sort("created_at", ASCENDING).skip(skip).limit(limit)
        rows = await cursor.to_list(length=limit)
        return [self._serialize(r) for r in rows]

    async def get(self, comment_id: str) -> Optional[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        row = await col.find_one({"_id": ObjectId(comment_id)})
        return self._serialize(row) if row else None

    async def update(
        self,
        comment_id: str,
        *,
        body: Optional[str] = None,
        mentions: Optional[List[str]] = None,
        resolved: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        patch: Dict[str, Any] = {"updated_at": datetime.utcnow()}
        if body is not None:
            patch["body"] = body
            patch["edited_at"] = datetime.utcnow()
        if mentions is not None:
            patch["mentions"] = [ObjectId(m) for m in mentions if ObjectId.is_valid(m)]
        if resolved is not None:
            patch["resolved"] = bool(resolved)
        if len(patch) == 1:
            return await self.get(comment_id)
        row = await col.find_one_and_update(
            {"_id": ObjectId(comment_id)},
            {"$set": patch},
            return_document=True,
        )
        return self._serialize(row) if row else None

    async def delete(self, comment_id: str) -> bool:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        result = await col.delete_one({"_id": ObjectId(comment_id)})
        return result.deleted_count > 0

    async def react(
        self,
        comment_id: str,
        *,
        emoji: str,
        user_id: str,
        add: bool = True,
    ) -> Optional[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        key = f"reactions.{emoji}"
        if add:
            await col.update_one(
                {"_id": ObjectId(comment_id)},
                {"$addToSet": {key: ObjectId(user_id)}},
            )
        else:
            await col.update_one(
                {"_id": ObjectId(comment_id)},
                {"$pull": {key: ObjectId(user_id)}},
            )
        return await self.get(comment_id)

    async def list_mentions_for_user(
        self,
        *,
        user_id: str,
        organization_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(COLLECTION)
        query: Dict[str, Any] = {"mentions": ObjectId(user_id)}
        if organization_id:
            query["organization_id"] = ObjectId(organization_id)
        cursor = col.find(query).sort("created_at", DESCENDING).limit(limit)
        return [self._serialize(r) for r in await cursor.to_list(length=limit)]

    @staticmethod
    def _serialize(doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc:
            return doc
        d = dict(doc)
        d["id"] = str(d.pop("_id"))
        d["organization_id"] = str(d["organization_id"]) if d.get("organization_id") else None
        d["user_id"] = str(d["user_id"]) if d.get("user_id") else None
        d["parent_id"] = str(d["parent_id"]) if d.get("parent_id") else None
        d["mentions"] = [str(m) for m in (d.get("mentions") or [])]
        rx = d.get("reactions") or {}
        d["reactions"] = {k: [str(u) for u in v] for k, v in rx.items()}
        return d


comments_repository = CommentsRepository()
