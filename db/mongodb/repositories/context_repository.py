from typing import Optional, List, Dict, Any, Union
from motor.motor_asyncio import ASCENDING, DESCENDING
from bson import ObjectId
from datetime import datetime, timedelta
from ..base_repository import BaseRepository
from ...models.mongodb_models import (
    Context,
    ContextStrategy,
    ContextStrategyType,
    Conversation,
    Message
)
import structlog
import json

logger = structlog.get_logger()

class ContextRepository(BaseRepository[Context]):
    def __init__(self):
        super().__init__("contexts", Context)

    async def _create_indexes(self):
        collection = await self.collection
        # Create indexes for common queries
        await collection.create_index("organization_id")
        await collection.create_index("created_by")
        await collection.create_index("is_global")
        await collection.create_index("tags")
        await collection.create_index("last_used")
        # Compound indexes for common queries
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("is_global", ASCENDING),
            ("last_used", DESCENDING)
        ])
        await collection.create_index([
            ("created_by", ASCENDING),
            ("tags", ASCENDING),
            ("version", DESCENDING)
        ])
        # Text search index for name and description
        await collection.create_index([
            ("name", "text"),
            ("description", "text")
        ])

    async def create_context(
        self,
        name: str,
        content: Dict[str, Any],
        organization_id: str,
        created_by: str,
        description: Optional[str] = None,
        is_global: bool = False,
        tags: List[str] = None
    ) -> Context:
        """Create a new context entry."""
        entry_dict = {
            "name": name,
            "description": description,
            "content": content,
            "organization_id": ObjectId(organization_id),
            "created_by": ObjectId(created_by),
            "is_global": is_global,
            "tags": tags or [],
            "version": 1,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "last_used": None
        }

        try:
            return await self.create(entry_dict)
        except Exception as e:
            logger.error(
                "Failed to create context",
                error=str(e),
                organization_id=organization_id,
                name=name
            )
            raise

    async def update_context(
        self,
        context_id: str,
        update_data: Dict[str, Any],
        increment_version: bool = True
    ) -> Optional[Context]:
        """Update context with optional version increment."""
        if increment_version:
            update_data["$inc"] = {"version": 1}
        update_data["updated_at"] = datetime.utcnow()
        return await self.update(context_id, update_data)

    async def get_context_by_id(
        self,
        context_id: str,
        update_last_used: bool = True
    ) -> Optional[Context]:
        """Get context by ID with optional last_used update."""
        context = await self.find_one({"_id": ObjectId(context_id)})
        if context and update_last_used:
            await self.update(context_id, {
                "last_used": datetime.utcnow()
            })
        return context

    async def get_organization_contexts(
        self,
        organization_id: str,
        include_global: bool = True,
        tags: Optional[List[str]] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Context]:
        """Get all contexts for an organization with filtering."""
        filters = {
            "$or": [
                {"organization_id": ObjectId(organization_id)},
                {"is_global": True} if include_global else {}
            ]
        }
        if tags:
            filters["tags"] = {"$all": tags}

        return await self.find_many(
            filters,
            skip=skip,
            limit=limit,
            sort=[("last_used", DESCENDING)]
        )

    async def merge_contexts(
        self,
        context_ids: List[str],
        merge_strategy: str = "deep_merge"
    ) -> Optional[Context]:
        """Merge multiple contexts using specified strategy."""
        contexts = []
        for context_id in context_ids:
            context = await self.get_context_by_id(context_id, update_last_used=False)
            if context:
                contexts.append(context)

        if not contexts:
            return None

        # Get the first context as base
        base_context = contexts[0]
        merged_content = base_context["content"]

        # Merge remaining contexts
        for context in contexts[1:]:
            if merge_strategy == "deep_merge":
                merged_content = self._deep_merge(merged_content, context["content"])
            elif merge_strategy == "shallow_merge":
                merged_content.update(context["content"])
            elif merge_strategy == "append":
                for key, value in context["content"].items():
                    if key in merged_content:
                        if isinstance(merged_content[key], list):
                            merged_content[key].extend(value)
                        else:
                            merged_content[key] = [merged_content[key], value]
                    else:
                        merged_content[key] = value

        # Create new merged context
        return await self.create_context(
            name=f"Merged Context {datetime.utcnow().isoformat()}",
            content=merged_content,
            organization_id=str(base_context["organization_id"]),
            created_by=str(base_context["created_by"]),
            description=f"Merged from contexts: {', '.join(str(c['_id']) for c in contexts)}",
            tags=list(set(tag for c in contexts for tag in c.get("tags", [])))
        )

    def _deep_merge(self, dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
        """Recursively merge two dictionaries."""
        result = dict1.copy()
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            elif key in result and isinstance(result[key], list) and isinstance(value, list):
                result[key] = result[key] + value
            else:
                result[key] = value
        return result

    async def get_conversation_context(
        self,
        conversation_id: str,
        max_messages: int = 10,
        include_metadata: bool = True
    ) -> Dict[str, Any]:
        """Get context from a conversation including recent messages."""
        conversation = await self.database["conversations"].find_one(
            {"_id": ObjectId(conversation_id)}
        )
        if not conversation:
            return {}

        # Get recent messages
        messages = conversation.get("messages", [])[-max_messages:]
        
        # Build context
        context = {
            "conversation_id": str(conversation["_id"]),
            "title": conversation.get("title"),
            "participants": [str(p) for p in conversation.get("participants", [])],
            "agent_id": str(conversation["agent_id"]) if conversation.get("agent_id") else None,
            "messages": [
                {
                    "role": msg.get("role"),
                    "content": msg.get("content"),
                    "timestamp": msg.get("timestamp")
                }
                for msg in messages
            ]
        }

        if include_metadata:
            context.update({
                "organization_id": str(conversation["organization_id"]),
                "tags": conversation.get("tags", []),
                "created_at": conversation.get("created_at"),
                "is_archived": conversation.get("is_archived", False)
            })

        return context

    async def get_agent_context(
        self,
        agent_id: str,
        include_config: bool = True,
        include_workflow: bool = True
    ) -> Dict[str, Any]:
        """Get context for an agent including configuration and workflow."""
        agent = await self.database["agents"].find_one(
            {"_id": ObjectId(agent_id)}
        )
        if not agent:
            return {}

        context = {
            "agent_id": str(agent["_id"]),
            "name": agent.get("name"),
            "agent_type": agent.get("agent_type"),
            "status": agent.get("status"),
            "capabilities": agent.get("capabilities", []),
            "version": agent.get("version")
        }

        if include_config and "configuration" in agent:
            context["configuration"] = agent["configuration"]

        if include_workflow and "workflow" in agent:
            workflow = agent["workflow"]
            context["workflow"] = {
                "nodes": workflow.get("nodes", []),
                "connections": workflow.get("connections", []),
                "context_strategy": workflow.get("context_strategy")
            }

        return context

    async def get_user_context(
        self,
        user_id: str,
        include_profile: bool = True,
        include_settings: bool = True
    ) -> Dict[str, Any]:
        """Get context for a user including profile and settings."""
        user = await self.database["users"].find_one(
            {"_id": ObjectId(user_id)}
        )
        if not user:
            return {}

        context = {
            "user_id": str(user["_id"]),
            "email": user.get("email"),
            "role": user.get("role")
        }

        if include_profile and "profile" in user:
            context["profile"] = user["profile"]

        if include_settings and "settings" in user:
            context["settings"] = user["settings"]

        return context

    async def search_contexts(
        self,
        query: str,
        organization_id: str,
        include_global: bool = True,
        tags: Optional[List[str]] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Context]:
        """Search contexts using text search and filters."""
        filters = {
            "$text": {"$search": query},
            "$or": [
                {"organization_id": ObjectId(organization_id)},
                {"is_global": True} if include_global else {}
            ]
        }
        if tags:
            filters["tags"] = {"$all": tags}

        return await self.find_many(
            filters,
            skip=skip,
            limit=limit,
            sort=[("score", {"$meta": "textScore"})]
        )

    async def get_context_usage_stats(
        self,
        organization_id: str,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get statistics about context usage in an organization."""
        match = {"organization_id": ObjectId(organization_id)}
        if time_range:
            match["last_used"] = {
                "$gte": datetime.utcnow() - time_range
            }

        pipeline = [
            {"$match": match},
            {"$group": {
                "_id": {
                    "is_global": "$is_global",
                    "tags": "$tags"
                },
                "count": {"$sum": 1},
                "avg_version": {"$avg": "$version"},
                "last_used_count": {
                    "$sum": {
                        "$cond": [
                            {"$ne": ["$last_used", None]},
                            1,
                            0
                        ]
                    }
                }
            }},
            {"$group": {
                "_id": "$_id.is_global",
                "tag_stats": {
                    "$push": {
                        "tags": "$_id.tags",
                        "count": "$count",
                        "avg_version": "$avg_version",
                        "last_used_count": "$last_used_count"
                    }
                }
            }}
        ]

        results = await self.aggregate(pipeline)
        return {
            "global": next((r["tag_stats"] for r in results if r["_id"]), []),
            "organization": next((r["tag_stats"] for r in results if not r["_id"]), [])
        }

# Create a singleton instance
context_repository = ContextRepository() 