from typing import Optional, List, Dict, Any, Union
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId
from datetime import datetime, timedelta
from ..base_repository import BaseRepository
from backend.models.mongodb_models import (
    AgentComponent,
    AgentComponentType,
    AgentWorkflow,
    AgentWorkflowNode,
    AgentWorkflowConnection
)
import structlog
import json

logger = structlog.get_logger()

class ComponentRepository(BaseRepository[AgentComponent]):
    def __init__(self):
        super().__init__("agent_components", AgentComponent)

    async def _create_indexes(self):
        collection = await self.collection
        # Create indexes for common queries
        await collection.create_index("organization_id")
        await collection.create_index("created_by")
        await collection.create_index("component_type")
        await collection.create_index("is_public")
        await collection.create_index("tags")
        await collection.create_index("version")
        # Compound indexes for common queries
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("component_type", ASCENDING),
            ("is_public", ASCENDING)
        ])
        await collection.create_index([
            ("created_by", ASCENDING),
            ("tags", ASCENDING),
            ("version", DESCENDING)
        ])
        # Text search index for name and description
        await collection.create_index([
            ("name", "text"),
            ("description", "text"),
            ("category", "text")
        ])

    async def create_component(
        self,
        name: str,
        component_type: AgentComponentType,
        configuration: Dict[str, Any],
        organization_id: str,
        created_by: str,
        description: Optional[str] = None,
        category: Optional[str] = None,
        is_public: bool = False,
        tags: List[str] = None,
        dependencies: List[Dict[str, Any]] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None
    ) -> AgentComponent:
        """Create a new agent component."""
        entry_dict = {
            "name": name,
            "description": description,
            "component_type": component_type,
            "configuration": configuration,
            "organization_id": ObjectId(organization_id),
            "created_by": ObjectId(created_by),
            "category": category,
            "is_public": is_public,
            "tags": tags or [],
            "dependencies": dependencies or [],
            "input_schema": input_schema or {},
            "output_schema": output_schema or {},
            "version": "1.0.0",
            "usage_count": 0,
            "rating": 0.0,
            "rating_count": 0,
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "last_used": None
        }

        try:
            return await self.create(entry_dict)
        except Exception as e:
            logger.error(
                "Failed to create component",
                error=str(e),
                organization_id=organization_id,
                name=name,
                component_type=component_type
            )
            raise

    async def update_component(
        self,
        component_id: str,
        update_data: Dict[str, Any],
        increment_version: bool = True
    ) -> Optional[AgentComponent]:
        """Update component with optional version increment."""
        if increment_version:
            # Parse current version and increment minor version
            component = await self.get_component_by_id(component_id)
            if component:
                version_parts = component["version"].split(".")
                if len(version_parts) == 3:
                    version_parts[1] = str(int(version_parts[1]) + 1)
                    update_data["version"] = ".".join(version_parts)

        update_data["updated_at"] = datetime.utcnow()
        return await self.update(component_id, update_data)

    async def get_component_by_id(
        self,
        component_id: str,
        update_last_used: bool = True
    ) -> Optional[AgentComponent]:
        """Get component by ID with optional last_used update."""
        component = await self.find_one({"_id": ObjectId(component_id)})
        if component and update_last_used:
            await self.update(component_id, {
                "last_used": datetime.utcnow(),
                "$inc": {"usage_count": 1}
            })
        return component

    async def get_organization_components(
        self,
        organization_id: str,
        component_type: Optional[AgentComponentType] = None,
        include_public: bool = True,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[AgentComponent]:
        """Get all components for an organization with filtering."""
        filters = {
            "$or": [
                {"organization_id": ObjectId(organization_id)},
                {"is_public": True} if include_public else {}
            ]
        }
        if component_type:
            filters["component_type"] = component_type
        if category:
            filters["category"] = category
        if tags:
            filters["tags"] = {"$all": tags}

        return await self.find_many(
            filters,
            skip=skip,
            limit=limit,
            sort=[("usage_count", DESCENDING), ("rating", DESCENDING)]
        )

    async def search_components(
        self,
        query: str,
        organization_id: str,
        include_public: bool = True,
        component_type: Optional[AgentComponentType] = None,
        category: Optional[str] = None,
        tags: Optional[List[str]] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[AgentComponent]:
        """Search components using text search and filters."""
        filters = {
            "$text": {"$search": query},
            "$or": [
                {"organization_id": ObjectId(organization_id)},
                {"is_public": True} if include_public else {}
            ]
        }
        if component_type:
            filters["component_type"] = component_type
        if category:
            filters["category"] = category
        if tags:
            filters["tags"] = {"$all": tags}

        return await self.find_many(
            filters,
            skip=skip,
            limit=limit,
            sort=[("score", {"$meta": "textScore"})]
        )

    async def update_component_rating(
        self,
        component_id: str,
        rating: float,
        user_id: str
    ) -> Optional[AgentComponent]:
        """Update component rating."""
        component = await self.get_component_by_id(component_id, update_last_used=False)
        if not component:
            return None

        # Calculate new average rating
        current_rating = component.get("rating", 0.0)
        current_count = component.get("rating_count", 0)
        new_count = current_count + 1
        new_rating = ((current_rating * current_count) + rating) / new_count

        update_data = {
            "rating": new_rating,
            "rating_count": new_count,
            "$push": {
                "ratings": {
                    "user_id": ObjectId(user_id),
                    "rating": rating,
                    "timestamp": datetime.utcnow()
                }
            }
        }
        return await self.update(component_id, update_data)

    async def get_component_dependencies(
        self,
        component_id: str,
        recursive: bool = True
    ) -> List[Dict[str, Any]]:
        """Get component dependencies, optionally recursively."""
        component = await self.get_component_by_id(component_id, update_last_used=False)
        if not component:
            return []

        dependencies = component.get("dependencies", [])
        if not recursive:
            return dependencies

        # Recursively get dependencies of dependencies
        all_dependencies = []
        for dep in dependencies:
            dep_component = await self.get_component_by_id(dep["component_id"], update_last_used=False)
            if dep_component:
                dep_info = {
                    "component_id": str(dep_component["_id"]),
                    "name": dep_component["name"],
                    "version": dep_component["version"],
                    "component_type": dep_component["component_type"],
                    "dependencies": await self.get_component_dependencies(
                        str(dep_component["_id"]),
                        recursive=True
                    )
                }
                all_dependencies.append(dep_info)

        return all_dependencies

    async def validate_component_configuration(
        self,
        component_id: str,
        configuration: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Validate component configuration against its schema."""
        component = await self.get_component_by_id(component_id, update_last_used=False)
        if not component:
            return {"valid": False, "error": "Component not found"}

        input_schema = component.get("input_schema", {})
        output_schema = component.get("output_schema", {})
        errors = []

        # Validate required fields
        for field, schema in input_schema.items():
            if schema.get("required", False) and field not in configuration:
                errors.append(f"Missing required field: {field}")
            elif field in configuration:
                # Validate field type
                expected_type = schema.get("type")
                if expected_type and not isinstance(configuration[field], eval(expected_type)):
                    errors.append(f"Invalid type for field {field}: expected {expected_type}")

        # Validate field constraints
        for field, value in configuration.items():
            if field in input_schema:
                schema = input_schema[field]
                if "min" in schema and value < schema["min"]:
                    errors.append(f"Value for {field} below minimum: {schema['min']}")
                if "max" in schema and value > schema["max"]:
                    errors.append(f"Value for {field} above maximum: {schema['max']}")
                if "enum" in schema and value not in schema["enum"]:
                    errors.append(f"Value for {field} not in allowed values: {schema['enum']}")

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "output_schema": output_schema
        }

    async def get_component_usage_stats(
        self,
        organization_id: str,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get statistics about component usage in an organization."""
        match = {"organization_id": ObjectId(organization_id)}
        if time_range:
            match["last_used"] = {
                "$gte": datetime.utcnow() - time_range
            }

        pipeline = [
            {"$match": match},
            {"$group": {
                "_id": {
                    "component_type": "$component_type",
                    "category": "$category"
                },
                "count": {"$sum": 1},
                "total_usage": {"$sum": "$usage_count"},
                "avg_rating": {"$avg": "$rating"},
                "total_ratings": {"$sum": "$rating_count"}
            }},
            {"$group": {
                "_id": "$_id.component_type",
                "categories": {
                    "$push": {
                        "category": "$_id.category",
                        "count": "$count",
                        "total_usage": "$total_usage",
                        "avg_rating": "$avg_rating",
                        "total_ratings": "$total_ratings"
                    }
                }
            }}
        ]

        results = await self.aggregate(pipeline)
        return {
            result["_id"]: result["categories"]
            for result in results
        }

    async def get_component_categories(
        self,
        organization_id: str,
        include_public: bool = True
    ) -> List[Dict[str, Any]]:
        """Get all component categories with usage statistics."""
        filters = {
            "$or": [
                {"organization_id": ObjectId(organization_id)},
                {"is_public": True} if include_public else {}
            ]
        }

        pipeline = [
            {"$match": filters},
            {"$group": {
                "_id": "$category",
                "count": {"$sum": 1},
                "components": {
                    "$push": {
                        "id": "$_id",
                        "name": "$name",
                        "component_type": "$component_type",
                        "usage_count": "$usage_count",
                        "rating": "$rating"
                    }
                }
            }},
            {"$sort": {"count": -1}}
        ]

        return await self.aggregate(pipeline)

# Create a singleton instance
component_repository = ComponentRepository() 