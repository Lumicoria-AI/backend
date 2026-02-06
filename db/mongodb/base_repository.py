from typing import TypeVar, Generic, Optional, List, Dict, Any, Type
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from datetime import datetime
from backend.models.mongodb_models import MongoBaseModel
# Remove MongoDB import as it will be handled in the dependency
# from .mongodb import MongoDB
import structlog
# Import UpdateOne for bulk_update
from pymongo import UpdateOne

logger = structlog.get_logger()

T = TypeVar('T', bound=MongoBaseModel)

class BaseRepository(Generic[T]):
    # Accept collection instance in constructor
    def __init__(self, collection: AsyncIOMotorCollection, model_class: Type[T]):
        self.collection = collection
        self.model_class = model_class
        # Remove _collection and collection_name
        # self.collection_name = collection_name
        # self._collection: Optional[AsyncIOMotorCollection] = None

    # Remove the async collection property
    # @property
    # async def collection(self) -> AsyncIOMotorCollection:
    #     if not self._collection:
    #         self._collection = await MongoDB.get_collection(self.collection_name)
    #         # Create indexes
    #         await self._create_indexes()
    #     return self._collection

    async def _create_indexes(self):
        """Override this method in child classes to create specific indexes"""
        pass

    async def create(self, data: Dict[str, Any]) -> T:
        # Access collection directly
        data["created_at"] = datetime.utcnow()
        result = await self.collection.insert_one(data)
        # Use self.collection explicitly for find_one
        created_doc = await self.collection.find_one({"_id": result.inserted_id})
        if created_doc:
            return self.model_class(**created_doc)
        # Handle case where insert_one succeeds but find_one fails immediately after
        raise Exception("Failed to retrieve document after creation")

    async def get_by_id(self, id: str) -> Optional[T]:
        # Access collection directly
        doc = await self.collection.find_one({"_id": ObjectId(id)})
        return self.model_class(**doc) if doc else None

    async def update(self, id: str, data: Dict[str, Any]) -> Optional[T]:
        # Access collection directly
        update_data = {k: v for k, v in data.items() if v is not None}
        if update_data:
            update_data["updated_at"] = datetime.utcnow()
            await self.collection.update_one(
                {"_id": ObjectId(id)},
                {"$set": update_data}
            )
        # Fetch the updated document
        return await self.get_by_id(id)

    async def delete(self, id: str) -> bool:
        # Access collection directly
        result = await self.collection.delete_one({"_id": ObjectId(id)})
        return result.deleted_count > 0

    async def list(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort: Optional[List[tuple]] = None
    ) -> List[T]:
        # Access collection directly
        collection = self.collection
        query = filters or {}
        cursor = collection.find(query)
        
        if sort:
            cursor = cursor.sort(sort)
        
        cursor = cursor.skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [self.model_class(**doc) for doc in docs]

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        # Access collection directly
        collection = self.collection
        query = filters or {}
        return await collection.count_documents(query)

    async def watch(self):
        """Watch for changes in the collection (real-time updates)"""
        # Access collection directly
        collection = self.collection
        # Ensure change streams are supported and properly configured
        async with collection.watch() as stream:
            async for change in stream:
                yield change

    async def bulk_create(self, items: List[Dict[str, Any]]) -> List[T]:
        # Access collection directly
        collection = self.collection
        now = datetime.utcnow()
        for item in items:
            item["created_at"] = now
        result = await collection.insert_many(items)
        created_ids = result.inserted_ids
        # Fetch created documents
        docs = await collection.find({"_id": {"$in": created_ids}}).to_list(length=len(created_ids))
        return [self.model_class(**doc) for doc in docs]

    async def bulk_update(self, updates: List[Dict[str, Any]]) -> bool:
        # Access collection directly
        collection = self.collection
        now = datetime.utcnow()
        operations = []
        # Import UpdateOne here if not globally imported in this file
        from pymongo import UpdateOne
        for update in updates:
            id = update.pop("_id", None)
            if id and update:
                # Use ObjectId for _id in the update filter
                operations.append(
                    UpdateOne(
                        {"_id": ObjectId(id)},
                        {"$set": {**update, "updated_at": now}}
                    )
                )
        if operations:
            result = await collection.bulk_write(operations)
            return result.modified_count > 0
        return False

    async def bulk_delete(self, ids: List[str]) -> bool:
        # Access collection directly
        collection = self.collection
        # Use ObjectId for _id in the delete filter
        result = await collection.delete_many({"_id": {"$in": [ObjectId(id) for id in ids]}})
        return result.deleted_count > 0

    async def find_one(self, filters: Dict[str, Any]) -> Optional[T]:
        # Access collection directly
        collection = self.collection
        doc = await collection.find_one(filters)
        return self.model_class(**doc) if doc else None

    async def find_many(
        self,
        filters: Dict[str, Any],
        skip: int = 0,
        limit: int = 100,
        sort: Optional[List[tuple]] = None
    ) -> List[T]:
        # Access collection directly
        collection = self.collection
        cursor = collection.find(filters)
        
        if sort:
            cursor = cursor.sort(sort)
        
        cursor = cursor.skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [self.model_class(**doc) for doc in docs]

    async def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        # Access collection directly
        collection = self.collection
        cursor = collection.aggregate(pipeline)
        return await cursor.to_list(length=None) 