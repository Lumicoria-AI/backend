from typing import Optional, List, Dict, Any
from pymongo import ASCENDING
from ..base_repository import BaseRepository
from ...models.mongodb_models import User, UserCreate, UserUpdate
import structlog
from datetime import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from db.mongodb.mongodb import get_mongodb
from db.mongodb.models.user import UserInDB, UserProfile, UserSettings

logger = structlog.get_logger()

class UserRepository(BaseRepository[User]):
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__("users", User)
        self.db = db
        self.profile_collection = db.user_profiles
        self.settings_collection = db.user_settings

    async def _create_indexes(self):
        collection = await self.collection
        await collection.create_index("email", unique=True)
        await collection.create_index("created_at")
        await collection.create_index("organization_ids")
        await collection.create_index("role_ids")
        await collection.create_index([("full_name", ASCENDING), ("email", ASCENDING)])

    async def create_user(self, user: UserCreate) -> UserInDB:
        user_dict = user.model_dump(exclude={"password"})
        user_dict["created_at"] = datetime.utcnow()
        result = await self.collection.insert_one(user_dict)
        user_dict["_id"] = result.inserted_id
        return UserInDB(**user_dict)

    async def get_user_by_email(self, email: str) -> Optional[UserInDB]:
        user_dict = await self.collection.find_one({"email": email})
        if user_dict:
            return UserInDB(**user_dict)
        return None

    async def get_user_by_id(self, user_id: str) -> Optional[UserInDB]:
        try:
            user_dict = await self.collection.find_one({"_id": ObjectId(user_id)})
            if user_dict:
                return UserInDB(**user_dict)
        except:
            pass
        return None

    async def get_user_by_firebase_uid(self, firebase_uid: str) -> Optional[UserInDB]:
        user_dict = await self.collection.find_one({"firebase_uid": firebase_uid})
        if user_dict:
            return UserInDB(**user_dict)
        return None

    async def update_user(self, user_id: str, user_update: UserUpdate) -> Optional[UserInDB]:
        update_data = user_update.model_dump(exclude_unset=True)
        if update_data:
            update_data["updated_at"] = datetime.utcnow()
            result = await self.collection.find_one_and_update(
                {"_id": ObjectId(user_id)},
                {"$set": update_data},
                return_document=True
            )
            if result:
                return UserInDB(**result)
        return None

    async def create_user_profile(self, user_id: str, profile: UserProfile) -> UserProfile:
        profile_dict = profile.model_dump()
        profile_dict["user_id"] = ObjectId(user_id)
        profile_dict["created_at"] = datetime.utcnow()
        result = await self.profile_collection.insert_one(profile_dict)
        profile_dict["_id"] = result.inserted_id
        return UserProfile(**profile_dict)

    async def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        profile_dict = await self.profile_collection.find_one({"user_id": ObjectId(user_id)})
        if profile_dict:
            return UserProfile(**profile_dict)
        return None

    async def create_user_settings(self, user_id: str, settings: UserSettings) -> UserSettings:
        settings_dict = settings.model_dump()
        settings_dict["user_id"] = ObjectId(user_id)
        settings_dict["created_at"] = datetime.utcnow()
        result = await self.settings_collection.insert_one(settings_dict)
        settings_dict["_id"] = result.inserted_id
        return UserSettings(**settings_dict)

    async def get_user_settings(self, user_id: str) -> Optional[UserSettings]:
        settings_dict = await self.settings_collection.find_one({"user_id": ObjectId(user_id)})
        if settings_dict:
            return UserSettings(**settings_dict)
        return None

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
user_repository = UserRepository(get_mongodb()) 