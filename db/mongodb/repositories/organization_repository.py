from typing import Optional, List, Dict, Any
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId
from ..base_repository import BaseRepository
from backend.models.mongodb_models import (
    Organization,
    OrganizationCreate,
    User
)
from .user_repository import user_repository
import structlog

logger = structlog.get_logger()

class OrganizationRepository(BaseRepository[Organization]):
    def __init__(self):
        super().__init__("organizations", Organization)

    async def _create_indexes(self):
        collection = await self.collection
        # Create indexes for common queries
        await collection.create_index("name", unique=True)
        await collection.create_index("created_at")
        await collection.create_index("member_ids")
        await collection.create_index("admin_ids")
        # Compound index for search
        await collection.create_index([
            ("name", "text"),
            ("description", "text"),
            ("industry", "text")
        ])

    async def create_organization(
        self,
        org_data: OrganizationCreate,
        creator_id: str
    ) -> Organization:
        """Create a new organization and add the creator as admin."""
        org_dict = org_data.dict()
        org_dict["admin_ids"] = [ObjectId(creator_id)]
        org_dict["member_ids"] = [ObjectId(creator_id)]
        
        try:
            organization = await self.create(org_dict)
            # Add organization to creator's organizations
            await user_repository.add_to_organization(creator_id, str(organization.id))
            return organization
        except Exception as e:
            logger.error("Failed to create organization", error=str(e), creator_id=creator_id)
            raise

    async def get_organization_by_name(self, name: str) -> Optional[Organization]:
        """Get organization by exact name match."""
        return await self.find_one({"name": name})

    async def search_organizations(
        self,
        query: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Organization]:
        """Search organizations by name, description, or industry."""
        search_filter = {
            "$text": {"$search": query}
        }
        return await self.find_many(
            search_filter,
            skip=skip,
            limit=limit,
            sort=[("score", {"$meta": "textScore"})]
        )

    async def get_user_organizations(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Organization]:
        """Get all organizations a user is a member of."""
        return await self.find_many(
            {"member_ids": ObjectId(user_id)},
            skip=skip,
            limit=limit,
            sort=[("created_at", DESCENDING)]
        )

    async def get_user_admin_organizations(
        self,
        user_id: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[Organization]:
        """Get all organizations a user is an admin of."""
        return await self.find_many(
            {"admin_ids": ObjectId(user_id)},
            skip=skip,
            limit=limit,
            sort=[("created_at", DESCENDING)]
        )

    async def add_member(
        self,
        organization_id: str,
        user_id: str,
        is_admin: bool = False
    ) -> Optional[Organization]:
        """Add a user to an organization, optionally as an admin."""
        update_data = {"$addToSet": {"member_ids": ObjectId(user_id)}}
        if is_admin:
            update_data["$addToSet"]["admin_ids"] = ObjectId(user_id)
        
        try:
            org = await self.update(organization_id, update_data)
            if org:
                await user_repository.add_to_organization(user_id, organization_id)
            return org
        except Exception as e:
            logger.error(
                "Failed to add member to organization",
                error=str(e),
                organization_id=organization_id,
                user_id=user_id
            )
            raise

    async def remove_member(
        self,
        organization_id: str,
        user_id: str
    ) -> Optional[Organization]:
        """Remove a user from an organization."""
        update_data = {
            "$pull": {
                "member_ids": ObjectId(user_id),
                "admin_ids": ObjectId(user_id)
            }
        }
        
        try:
            org = await self.update(organization_id, update_data)
            if org:
                await user_repository.remove_from_organization(user_id, organization_id)
            return org
        except Exception as e:
            logger.error(
                "Failed to remove member from organization",
                error=str(e),
                organization_id=organization_id,
                user_id=user_id
            )
            raise

    async def promote_to_admin(
        self,
        organization_id: str,
        user_id: str
    ) -> Optional[Organization]:
        """Promote a member to admin."""
        return await self.update(
            organization_id,
            {"$addToSet": {"admin_ids": ObjectId(user_id)}}
        )

    async def demote_from_admin(
        self,
        organization_id: str,
        user_id: str
    ) -> Optional[Organization]:
        """Demote an admin to regular member."""
        return await self.update(
            organization_id,
            {"$pull": {"admin_ids": ObjectId(user_id)}}
        )

    async def get_organization_members(
        self,
        organization_id: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """Get all members of an organization."""
        org = await self.get_by_id(organization_id)
        if not org:
            return []
        
        return await user_repository.find_many(
            {"_id": {"$in": org.member_ids}},
            skip=skip,
            limit=limit,
            sort=[("created_at", ASCENDING)]
        )

    async def get_organization_admins(
        self,
        organization_id: str,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        """Get all admins of an organization."""
        org = await self.get_by_id(organization_id)
        if not org:
            return []
        
        return await user_repository.find_many(
            {"_id": {"$in": org.admin_ids}},
            skip=skip,
            limit=limit,
            sort=[("created_at", ASCENDING)]
        )

    async def update_organization_settings(
        self,
        organization_id: str,
        settings: Dict[str, Any]
    ) -> Optional[Organization]:
        """Update organization settings."""
        return await self.update(
            organization_id,
            {"settings": settings}
        )

    async def get_organization_stats(self, organization_id: str) -> Dict[str, Any]:
        """Get organization statistics."""
        pipeline = [
            {"$match": {"_id": ObjectId(organization_id)}},
            {"$lookup": {
                "from": "users",
                "localField": "member_ids",
                "foreignField": "_id",
                "as": "members"
            }},
            {"$project": {
                "total_members": {"$size": "$members"},
                "total_admins": {"$size": "$admin_ids"},
                "created_at": 1,
                "name": 1
            }}
        ]
        
        results = await self.aggregate(pipeline)
        return results[0] if results else {}

# Create a singleton instance
organization_repository = OrganizationRepository() 