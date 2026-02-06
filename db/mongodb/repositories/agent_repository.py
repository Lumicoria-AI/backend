from typing import Optional, List, Dict, Any, Union
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId
from datetime import datetime, timedelta
from ..base_repository import BaseRepository
from backend.models.mongodb_models import (
    Agent,
    AgentCreate,
    AgentStatus,
    AgentType,
    AgentConfig,
    AgentState,
    AgentCapability,
    AgentModel
)
import structlog

logger = structlog.get_logger()

class AgentRepository(BaseRepository[Agent]):
    def __init__(self):
        super().__init__("agents", Agent)

    async def _create_indexes(self):
        collection = await self.collection
        # Create indexes for common queries
        await collection.create_index("created_by")
        await collection.create_index("organization_id")
        await collection.create_index("status")
        await collection.create_index("type")
        await collection.create_index("created_at")
        await collection.create_index("last_active")
        await collection.create_index("model_id")
        # Compound indexes for common queries
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("status", ASCENDING),
            ("type", ASCENDING)
        ])
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("capabilities", ASCENDING)
        ])
        # Text search index
        await collection.create_index([
            ("name", "text"),
            ("description", "text"),
            ("tags", "text")
        ])

    async def create_agent(
        self,
        agent_data: AgentCreate,
        creator_id: str,
        organization_id: str
    ) -> Agent:
        """Create a new agent with initial configuration."""
        agent_dict = agent_data.dict()
        initial_state = AgentState(
            status=AgentStatus.INACTIVE,
            last_active=datetime.utcnow(),
            current_task=None,
            memory_usage=0,
            error_count=0
        )

        agent_dict.update({
            "created_by": ObjectId(creator_id),
            "organization_id": ObjectId(organization_id),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "state": initial_state.dict(),
            "status": AgentStatus.INACTIVE,
            "version": "1.0.0"
        })

        try:
            return await self.create(agent_dict)
        except Exception as e:
            logger.error(
                "Failed to create agent",
                error=str(e),
                creator_id=creator_id,
                organization_id=organization_id
            )
            raise

    async def get_organization_agents(
        self,
        organization_id: str,
        status: Optional[AgentStatus] = None,
        agent_type: Optional[AgentType] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Agent]:
        """Get agents in an organization."""
        filters = {"organization_id": ObjectId(organization_id)}
        if status:
            filters["status"] = status
        if agent_type:
            filters["type"] = agent_type

        return await self.find_many(
            filters,
            skip=skip,
            limit=limit,
            sort=[("created_at", DESCENDING)]
        )

    async def search_agents(
        self,
        query: str,
        organization_id: Optional[str] = None,
        status: Optional[AgentStatus] = None,
        agent_type: Optional[AgentType] = None,
        capabilities: Optional[List[AgentCapability]] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Agent]:
        """Search agents by name, description, or capabilities."""
        search_filter = {
            "$text": {"$search": query}
        }

        if organization_id:
            search_filter["organization_id"] = ObjectId(organization_id)
        if status:
            search_filter["status"] = status
        if agent_type:
            search_filter["type"] = agent_type
        if capabilities:
            search_filter["capabilities"] = {"$all": capabilities}

        return await self.find_many(
            search_filter,
            skip=skip,
            limit=limit,
            sort=[("score", {"$meta": "textScore"})]
        )

    async def update_agent_state(
        self,
        agent_id: str,
        state_update: Dict[str, Any]
    ) -> Optional[Agent]:
        """Update agent state and status."""
        update_data = {
            "state": state_update,
            "last_active": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }
        
        if "status" in state_update:
            update_data["status"] = state_update["status"]
            update_data["$push"] = {
                "status_history": {
                    "status": state_update["status"],
                    "changed_at": datetime.utcnow()
                }
            }

        return await self.update(agent_id, update_data)

    async def update_agent_config(
        self,
        agent_id: str,
        config_update: AgentConfig
    ) -> Optional[Agent]:
        """Update agent configuration."""
        update_data = {
            "config": config_update.dict(),
            "updated_at": datetime.utcnow(),
            "$push": {
                "config_history": {
                    "config": config_update.dict(),
                    "changed_at": datetime.utcnow()
                }
            }
        }
        return await self.update(agent_id, update_data)

    async def add_capability(
        self,
        agent_id: str,
        capability: AgentCapability,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Agent]:
        """Add a new capability to an agent."""
        capability_data = {
            "name": capability,
            "added_at": datetime.utcnow(),
            "config": config or {}
        }
        
        return await self.update(
            agent_id,
            {
                "$push": {"capabilities": capability_data},
                "updated_at": datetime.utcnow()
            }
        )

    async def remove_capability(
        self,
        agent_id: str,
        capability: AgentCapability
    ) -> Optional[Agent]:
        """Remove a capability from an agent."""
        return await self.update(
            agent_id,
            {
                "$pull": {"capabilities": {"name": capability}},
                "updated_at": datetime.utcnow()
            }
        )

    async def get_agent_stats(
        self,
        organization_id: str,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get agent statistics and performance metrics."""
        match = {"organization_id": ObjectId(organization_id)}
        if time_range:
            match["last_active"] = {
                "$gte": datetime.utcnow() - time_range
            }

        pipeline = [
            {"$match": match},
            {"$group": {
                "_id": {
                    "status": "$status",
                    "type": "$type"
                },
                "count": {"$sum": 1},
                "active_agents": {
                    "$sum": {
                        "$cond": [
                            {"$eq": ["$status", AgentStatus.ACTIVE]},
                            1,
                            0
                        ]
                    }
                },
                "total_tasks": {"$sum": "$state.task_count"},
                "error_count": {"$sum": "$state.error_count"},
                "agents": {"$push": {
                    "id": "$_id",
                    "name": "$name",
                    "status": "$status",
                    "last_active": "$last_active"
                }}
            }},
            {"$group": {
                "_id": None,
                "total": {"$sum": "$count"},
                "active_agents": {"$sum": "$active_agents"},
                "total_tasks": {"$sum": "$total_tasks"},
                "total_errors": {"$sum": "$error_count"},
                "categories": {
                    "$push": {
                        "status": "$_id.status",
                        "type": "$_id.type",
                        "count": "$count",
                        "agents": "$agents"
                    }
                }
            }}
        ]

        results = await self.aggregate(pipeline)
        return results[0] if results else {
            "total": 0,
            "active_agents": 0,
            "total_tasks": 0,
            "total_errors": 0,
            "categories": []
        }

    async def get_active_agents(
        self,
        organization_id: str,
        min_activity: Optional[timedelta] = None
    ) -> List[Agent]:
        """Get currently active agents."""
        filters = {
            "organization_id": ObjectId(organization_id),
            "status": AgentStatus.ACTIVE
        }
        
        if min_activity:
            filters["last_active"] = {
                "$gte": datetime.utcnow() - min_activity
            }

        return await self.find_many(
            filters,
            sort=[("last_active", DESCENDING)]
        )

    async def get_agent_by_model(
        self,
        model_id: str,
        organization_id: Optional[str] = None
    ) -> List[Agent]:
        """Get agents using a specific AI model."""
        filters = {"model_id": ObjectId(model_id)}
        if organization_id:
            filters["organization_id"] = ObjectId(organization_id)

        return await self.find_many(
            filters,
            sort=[("created_at", DESCENDING)]
        )

    async def get_agent_capabilities(
        self,
        agent_id: str
    ) -> List[Dict[str, Any]]:
        """Get all capabilities of an agent with their configurations."""
        agent = await self.get_by_id(agent_id)
        if not agent:
            return []
        return agent.capabilities

    async def update_agent_model(
        self,
        agent_id: str,
        model_id: str,
        config: Optional[Dict[str, Any]] = None
    ) -> Optional[Agent]:
        """Update the AI model used by an agent."""
        update_data = {
            "model_id": ObjectId(model_id),
            "updated_at": datetime.utcnow()
        }
        
        if config:
            update_data["model_config"] = config

        update_data["$push"] = {
            "model_history": {
                "model_id": ObjectId(model_id),
                "config": config or {},
                "changed_at": datetime.utcnow()
            }
        }

        return await self.update(agent_id, update_data)

# Create a singleton instance
agent_repository = AgentRepository() 