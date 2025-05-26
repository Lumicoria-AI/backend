from typing import Optional, List, Dict, Any
from motor.motor_asyncio import ASCENDING
from ..base_repository import BaseRepository
from ...models.mongodb_models import User, UserCreate, UserUpdate
import structlog

logger = structlog.get_logger()

class UserRepository(BaseRepository[User]):
    def __init__(self):
        super().__init__("users", User)

    async def _create_indexes(self):
        collection = await self.collection
        await collection.create_index("email", unique=True)
        await collection.create_index("created_at")
        await collection.create_index("organization_ids")
        await collection.create_index("role_ids")
        await collection.create_index([("full_name", ASCENDING), ("email", ASCENDING)])

    async def create_user(self, user_data: UserCreate, hashed_password: str) -> User:
        user_dict = user_data.dict(exclude={"password", "organization_name"})
        user_dict["hashed_password"] = hashed_password
        return await self.create(user_dict)

    async def get_by_email(self, email: str) -> Optional[User]:
        return await self.find_one({"email": email})

    async def update_user(self, user_id: str, update_data: UserUpdate) -> Optional[User]:
        update_dict = update_data.dict(exclude_unset=True)
        return await self.update(user_id, update_dict)

    async def get_users_by_organization(self, organization_id: str, skip: int = 0, limit: int = 100) -> List[User]:
        return await self.find_many(
            {"organization_ids": organization_id},
            skip=skip,
            limit=limit,
            sort=[("created_at", ASCENDING)]
        )

    async def get_users_by_role(self, role_id: str, skip: int = 0, limit: int = 100) -> List[User]:
        return await self.find_many(
            {"role_ids": role_id},
            skip=skip,
            limit=limit,
            sort=[("created_at", ASCENDING)]
        )

    async def add_to_organization(self, user_id: str, organization_id: str) -> Optional[User]:
        return await self.update(
            user_id,
            {"$addToSet": {"organization_ids": organization_id}}
        )

    async def remove_from_organization(self, user_id: str, organization_id: str) -> Optional[User]:
        return await self.update(
            user_id,
            {"$pull": {"organization_ids": organization_id}}
        )

    async def add_role(self, user_id: str, role_id: str) -> Optional[User]:
        return await self.update(
            user_id,
            {"$addToSet": {"role_ids": role_id}}
        )

    async def remove_role(self, user_id: str, role_id: str) -> Optional[User]:
        return await self.update(
            user_id,
            {"$pull": {"role_ids": role_id}}
        )

    async def search_users(
        self,
        query: str,
        organization_id: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[User]:
        search_filter = {
            "$or": [
                {"full_name": {"$regex": query, "$options": "i"}},
                {"email": {"$regex": query, "$options": "i"}},
                {"profile.job_title": {"$regex": query, "$options": "i"}},
                {"profile.company": {"$regex": query, "$options": "i"}}
            ]
        }
        
        if organization_id:
            search_filter["organization_ids"] = organization_id

        return await self.find_many(
            search_filter,
            skip=skip,
            limit=limit,
            sort=[("created_at", ASCENDING)]
        )

    async def get_active_users(self, skip: int = 0, limit: int = 100) -> List[User]:
        return await self.find_many(
            {"is_active": True},
            skip=skip,
            limit=limit,
            sort=[("created_at", ASCENDING)]
        )

    async def get_superusers(self, skip: int = 0, limit: int = 100) -> List[User]:
        return await self.find_many(
            {"is_superuser": True},
            skip=skip,
            limit=limit,
            sort=[("created_at", ASCENDING)]
        )

# Create a singleton instance
user_repository = UserRepository() 