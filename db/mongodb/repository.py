from typing import Optional, List, Dict, Any
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from datetime import datetime
from ...models.mongodb_models import User, UserCreate, UserUpdate
from .mongodb import MongoDB
import structlog

logger = structlog.get_logger()

class UserRepository:
    def __init__(self):
        self.collection_name = "users"
        self._collection: Optional[AsyncIOMotorCollection] = None

    @property
    async def collection(self) -> AsyncIOMotorCollection:
        if not self._collection:
            self._collection = await MongoDB.get_collection(self.collection_name)
            # Create indexes
            await self._collection.create_index("email", unique=True)
            await self._collection.create_index("created_at")
        return self._collection

    async def create_user(self, user_data: UserCreate, hashed_password: str) -> User:
        collection = await self.collection
        user_dict = user_data.dict()
        user_dict["hashed_password"] = hashed_password
        user_dict["created_at"] = datetime.utcnow()
        
        result = await collection.insert_one(user_dict)
        created_user = await collection.find_one({"_id": result.inserted_id})
        return User(**created_user)

    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        collection = await self.collection
        user = await collection.find_one({"_id": ObjectId(user_id)})
        return User(**user) if user else None

    async def get_user_by_email(self, email: str) -> Optional[User]:
        collection = await self.collection
        user = await collection.find_one({"email": email})
        return User(**user) if user else None

    async def update_user(self, user_id: str, update_data: UserUpdate) -> Optional[User]:
        collection = await self.collection
        update_dict = {k: v for k, v in update_data.dict(exclude_unset=True).items()}
        if update_dict:
            update_dict["updated_at"] = datetime.utcnow()
            await collection.update_one(
                {"_id": ObjectId(user_id)},
                {"$set": update_dict}
            )
        return await self.get_user_by_id(user_id)

    async def delete_user(self, user_id: str) -> bool:
        collection = await self.collection
        result = await collection.delete_one({"_id": ObjectId(user_id)})
        return result.deleted_count > 0

    async def list_users(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[User]:
        collection = await self.collection
        query = filters or {}
        cursor = collection.find(query).skip(skip).limit(limit)
        users = await cursor.to_list(length=limit)
        return [User(**user) for user in users]

    async def watch_users(self):
        """Watch for changes in the users collection (real-time updates)"""
        collection = await self.collection
        async with collection.watch() as stream:
            async for change in stream:
                yield change

    async def count_users(self, filters: Optional[Dict[str, Any]] = None) -> int:
        collection = await self.collection
        query = filters or {}
        return await collection.count_documents(query)

# Create a singleton instance
user_repository = UserRepository() 