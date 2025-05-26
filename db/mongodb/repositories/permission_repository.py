from typing import Optional, List, Dict, Any, Union
from motor.motor_asyncio import ASCENDING, DESCENDING
from bson import ObjectId
from datetime import datetime
from ..base_repository import BaseRepository
from ...models.mongodb_models import (
    Permission,
    PermissionType,
    ResourceType,
    RolePermission
)
import structlog
import json

logger = structlog.get_logger()

class PermissionRepository(BaseRepository[Permission]):
    def __init__(self):
        super().__init__("permissions", Permission)

    async def _create_indexes(self):
        collection = await self.collection
        # Create indexes for common queries
        await collection.create_index("organization_id")
        await collection.create_index("user_id")
        await collection.create_index("resource_type")
        await collection.create_index("resource_id")
        await collection.create_index("permission_type")
        # Compound indexes for common queries
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("user_id", ASCENDING),
            ("resource_type", ASCENDING)
        ])
        await collection.create_index([
            ("resource_type", ASCENDING),
            ("resource_id", ASCENDING),
            ("permission_type", ASCENDING)
        ])
        # Role-based permissions index
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("role_id", ASCENDING),
            ("resource_type", ASCENDING)
        ])

    async def create_permission(
        self,
        organization_id: str,
        user_id: str,
        resource_type: ResourceType,
        resource_id: str,
        permission_type: PermissionType,
        granted_by: str,
        role_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Permission:
        """Create a new permission entry."""
        entry_dict = {
            "organization_id": ObjectId(organization_id),
            "user_id": ObjectId(user_id),
            "resource_type": resource_type,
            "resource_id": ObjectId(resource_id),
            "permission_type": permission_type,
            "granted_by": ObjectId(granted_by),
            "role_id": ObjectId(role_id) if role_id else None,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow()
        }

        try:
            return await self.create(entry_dict)
        except Exception as e:
            logger.error(
                "Failed to create permission",
                error=str(e),
                organization_id=organization_id,
                user_id=user_id,
                resource_type=resource_type
            )
            raise

    async def check_permission(
        self,
        user_id: str,
        organization_id: str,
        resource_type: ResourceType,
        resource_id: str,
        permission_type: PermissionType
    ) -> bool:
        """Check if a user has a specific permission."""
        # Check direct permissions
        direct_permission = await self.find_one({
            "user_id": ObjectId(user_id),
            "organization_id": ObjectId(organization_id),
            "resource_type": resource_type,
            "resource_id": ObjectId(resource_id),
            "permission_type": permission_type
        })

        if direct_permission:
            return True

        # Check role-based permissions
        role_permission = await self.find_one({
            "organization_id": ObjectId(organization_id),
            "resource_type": resource_type,
            "resource_id": ObjectId(resource_id),
            "permission_type": permission_type,
            "role_id": {"$exists": True},
            "user_id": ObjectId(user_id)
        })

        return bool(role_permission)

    async def get_user_permissions(
        self,
        user_id: str,
        organization_id: str,
        resource_type: Optional[ResourceType] = None,
        include_role_permissions: bool = True
    ) -> List[Permission]:
        """Get all permissions for a user in an organization."""
        filters = {
            "user_id": ObjectId(user_id),
            "organization_id": ObjectId(organization_id)
        }
        
        if resource_type:
            filters["resource_type"] = resource_type

        if not include_role_permissions:
            filters["role_id"] = None

        return await self.find_many(filters)

    async def get_resource_permissions(
        self,
        organization_id: str,
        resource_type: ResourceType,
        resource_id: str
    ) -> List[Dict[str, Any]]:
        """Get all permissions for a specific resource."""
        pipeline = [
            {"$match": {
                "organization_id": ObjectId(organization_id),
                "resource_type": resource_type,
                "resource_id": ObjectId(resource_id)
            }},
            {"$lookup": {
                "from": "users",
                "localField": "user_id",
                "foreignField": "_id",
                "as": "user"
            }},
            {"$lookup": {
                "from": "roles",
                "localField": "role_id",
                "foreignField": "_id",
                "as": "role"
            }},
            {"$unwind": {
                "path": "$user",
                "preserveNullAndEmptyArrays": True
            }},
            {"$unwind": {
                "path": "$role",
                "preserveNullAndEmptyArrays": True
            }},
            {"$project": {
                "permission_type": 1,
                "granted_by": 1,
                "created_at": 1,
                "user": {
                    "id": "$user._id",
                    "name": "$user.name",
                    "email": "$user.email"
                },
                "role": {
                    "id": "$role._id",
                    "name": "$role.name"
                }
            }}
        ]

        return await self.aggregate(pipeline)

    async def grant_role_permissions(
        self,
        organization_id: str,
        role_id: str,
        permissions: List[Dict[str, Any]],
        granted_by: str
    ) -> List[Permission]:
        """Grant multiple permissions to a role."""
        permission_entries = []
        for perm in permissions:
            entry = {
                "organization_id": ObjectId(organization_id),
                "role_id": ObjectId(role_id),
                "resource_type": perm["resource_type"],
                "resource_id": ObjectId(perm["resource_id"]),
                "permission_type": perm["permission_type"],
                "granted_by": ObjectId(granted_by),
                "metadata": perm.get("metadata", {}),
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            permission_entries.append(entry)

        try:
            return await self.create_many(permission_entries)
        except Exception as e:
            logger.error(
                "Failed to grant role permissions",
                error=str(e),
                organization_id=organization_id,
                role_id=role_id
            )
            raise

    async def revoke_permissions(
        self,
        organization_id: str,
        user_id: Optional[str] = None,
        role_id: Optional[str] = None,
        resource_type: Optional[ResourceType] = None,
        resource_id: Optional[str] = None,
        permission_type: Optional[PermissionType] = None
    ) -> int:
        """Revoke permissions based on filters."""
        filters = {"organization_id": ObjectId(organization_id)}
        
        if user_id:
            filters["user_id"] = ObjectId(user_id)
        if role_id:
            filters["role_id"] = ObjectId(role_id)
        if resource_type:
            filters["resource_type"] = resource_type
        if resource_id:
            filters["resource_id"] = ObjectId(resource_id)
        if permission_type:
            filters["permission_type"] = permission_type

        result = await self.delete_many(filters)
        return result.deleted_count

    async def get_permission_analytics(
        self,
        organization_id: str
    ) -> Dict[str, Any]:
        """Get analytics about permissions in an organization."""
        pipeline = [
            {"$match": {"organization_id": ObjectId(organization_id)}},
            {"$facet": {
                "permissions_by_type": [
                    {"$group": {
                        "_id": "$permission_type",
                        "count": {"$sum": 1}
                    }},
                    {"$sort": {"count": -1}}
                ],
                "permissions_by_resource": [
                    {"$group": {
                        "_id": {
                            "resource_type": "$resource_type",
                            "permission_type": "$permission_type"
                        },
                        "count": {"$sum": 1}
                    }},
                    {"$group": {
                        "_id": "$_id.resource_type",
                        "permissions": {
                            "$push": {
                                "type": "$_id.permission_type",
                                "count": "$count"
                            }
                        }
                    }},
                    {"$sort": {"_id": 1}}
                ],
                "role_permissions": [
                    {"$match": {"role_id": {"$exists": True}}},
                    {"$group": {
                        "_id": "$role_id",
                        "permission_count": {"$sum": 1},
                        "resource_types": {"$addToSet": "$resource_type"}
                    }},
                    {"$lookup": {
                        "from": "roles",
                        "localField": "_id",
                        "foreignField": "_id",
                        "as": "role"
                    }},
                    {"$unwind": "$role"},
                    {"$project": {
                        "role_name": "$role.name",
                        "permission_count": 1,
                        "resource_types": 1
                    }},
                    {"$sort": {"permission_count": -1}}
                ]
            }}
        ]

        results = await self.aggregate(pipeline)
        return results[0] if results else {}

    async def transfer_permissions(
        self,
        organization_id: str,
        from_user_id: str,
        to_user_id: str,
        resource_type: Optional[ResourceType] = None,
        resource_id: Optional[str] = None
    ) -> int:
        """Transfer permissions from one user to another."""
        filters = {
            "organization_id": ObjectId(organization_id),
            "user_id": ObjectId(from_user_id)
        }
        
        if resource_type:
            filters["resource_type"] = resource_type
        if resource_id:
            filters["resource_id"] = ObjectId(resource_id)

        update_data = {
            "user_id": ObjectId(to_user_id),
            "updated_at": datetime.utcnow()
        }

        result = await self.update_many(filters, update_data)
        return result.modified_count

    async def get_inherited_permissions(
        self,
        user_id: str,
        organization_id: str
    ) -> List[Dict[str, Any]]:
        """Get all permissions for a user, including those inherited from roles."""
        pipeline = [
            {"$match": {
                "organization_id": ObjectId(organization_id),
                "$or": [
                    {"user_id": ObjectId(user_id)},
                    {"role_id": {"$exists": True}}
                ]
            }},
            {"$lookup": {
                "from": "user_roles",
                "let": {"role_id": "$role_id"},
                "pipeline": [
                    {"$match": {
                        "$expr": {
                            "$and": [
                                {"$eq": ["$user_id", ObjectId(user_id)]},
                                {"$eq": ["$role_id", "$$role_id"]}
                            ]
                        }
                    }}
                ],
                "as": "user_role"
            }},
            {"$match": {
                "$or": [
                    {"user_id": ObjectId(user_id)},
                    {"user_role": {"$ne": []}}
                ]
            }},
            {"$project": {
                "resource_type": 1,
                "resource_id": 1,
                "permission_type": 1,
                "role_id": 1,
                "metadata": 1,
                "inherited_from": {
                    "$cond": {
                        "if": {"$eq": ["$user_id", ObjectId(user_id)]},
                        "then": "direct",
                        "else": "role"
                    }
                }
            }}
        ]

        return await self.aggregate(pipeline)

# Create a singleton instance
permission_repository = PermissionRepository() 