from typing import TypeVar, Generic, Optional, List, Dict, Any, Type, Union
from motor.motor_asyncio import AsyncIOMotorCollection
from bson import ObjectId
from datetime import datetime
from backend.models.mongodb_models import MongoBaseModel
import structlog
from pymongo import UpdateOne

logger = structlog.get_logger()

T = TypeVar('T', bound=MongoBaseModel)

class _CollectionAccessor:
    def __init__(self, repo: "BaseRepository"):
        self._repo = repo

    def __await__(self):
        return self._repo._get_collection().__await__()

    def __getattr__(self, item: str):
        if self._repo._collection is None:
            raise RuntimeError("MongoDB collection not initialized; use `await self.collection` first")
        return getattr(self._repo._collection, item)


class BaseRepository(Generic[T]):
    """
    Base repository supporting both legacy (collection name) and modern
    (collection instance) initialization patterns.
    """
    def __init__(self, collection: Union[str, AsyncIOMotorCollection], model_class: Type[T]):
        self._collection_name: Optional[str] = None
        self._collection: Optional[AsyncIOMotorCollection] = None
        if isinstance(collection, AsyncIOMotorCollection):
            self._collection = collection
        else:
            self._collection_name = str(collection)
        self.model_class = model_class
        # Expose a dual-mode accessor so both `await self.collection` and
        # `self.collection.insert_one(...)` can work depending on initialization.
        self.collection = _CollectionAccessor(self)

    async def _get_collection(self) -> AsyncIOMotorCollection:
        if self._collection is None:
            if not self._collection_name:
                raise RuntimeError("MongoDB collection is not configured")
            from .mongodb import MongoDB
            self._collection = await MongoDB.get_collection(self._collection_name)
            await self._create_indexes()
        return self._collection

    async def _create_indexes(self):
        """Override this method in child classes to create specific indexes"""
        pass

    async def create(self, data: Dict[str, Any]) -> T:
        collection = await self._get_collection()
        data["created_at"] = datetime.utcnow()
        result = await collection.insert_one(data)
        created_doc = await collection.find_one({"_id": result.inserted_id})
        if created_doc:
            return self.model_class(**created_doc)
        # Handle case where insert_one succeeds but find_one fails immediately after
        raise Exception("Failed to retrieve document after creation")

    async def get_by_id(self, id: str) -> Optional[T]:
        collection = await self._get_collection()
        doc = await collection.find_one({"_id": ObjectId(id)})
        return self.model_class(**doc) if doc else None

    async def update(self, id: str, data: Dict[str, Any]) -> Optional[T]:
        """Update a document by id.

        `data` may be:
          - A plain dict of fields to set (e.g. {"name": "x"}) — auto-wrapped
            in $set plus an updated_at timestamp.
          - A MongoDB update specification (any key starting with $, e.g.
            {"$addToSet": {"member_ids": uid}}). Passed through verbatim;
            we just merge an $set.updated_at into it.

        Without this branch, callers like add_to_organization that pass
        operator-style payloads hit pymongo's "$addToSet in $addToSet not
        allowed in replacement document" WriteError.
        """
        collection = await self._get_collection()

        if not data:
            return await self.get_by_id(id)

        is_operator_update = any(isinstance(k, str) and k.startswith("$") for k in data.keys())

        if is_operator_update:
            update_spec: Dict[str, Any] = {k: v for k, v in data.items() if v is not None}
            existing_set = update_spec.get("$set", {})
            if not isinstance(existing_set, dict):
                existing_set = {}
            existing_set["updated_at"] = datetime.utcnow()
            update_spec["$set"] = existing_set
            await collection.update_one({"_id": ObjectId(id)}, update_spec)
        else:
            update_data = {k: v for k, v in data.items() if v is not None}
            update_data["updated_at"] = datetime.utcnow()
            await collection.update_one(
                {"_id": ObjectId(id)},
                {"$set": update_data},
            )

        return await self.get_by_id(id)

    async def delete(self, id: str) -> bool:
        collection = await self._get_collection()
        result = await collection.delete_one({"_id": ObjectId(id)})
        return result.deleted_count > 0

    async def list(
        self,
        skip: int = 0,
        limit: int = 100,
        filters: Optional[Dict[str, Any]] = None,
        sort: Optional[List[tuple]] = None
    ) -> List[T]:
        collection = await self._get_collection()
        query = filters or {}
        cursor = collection.find(query)
        
        if sort:
            cursor = cursor.sort(sort)
        
        cursor = cursor.skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [self.model_class(**doc) for doc in docs]

    async def count(self, filters: Optional[Dict[str, Any]] = None) -> int:
        collection = await self._get_collection()
        query = filters or {}
        return await collection.count_documents(query)

    async def watch(self):
        """Watch for changes in the collection (real-time updates)"""
        collection = await self._get_collection()
        # Ensure change streams are supported and properly configured
        async with collection.watch() as stream:
            async for change in stream:
                yield change

    async def bulk_create(self, items: List[Dict[str, Any]]) -> List[T]:
        collection = await self._get_collection()
        now = datetime.utcnow()
        for item in items:
            item["created_at"] = now
        result = await collection.insert_many(items)
        created_ids = result.inserted_ids
        # Fetch created documents
        docs = await collection.find({"_id": {"$in": created_ids}}).to_list(length=len(created_ids))
        return [self.model_class(**doc) for doc in docs]

    async def bulk_update(self, updates: List[Dict[str, Any]]) -> bool:
        collection = await self._get_collection()
        now = datetime.utcnow()
        operations = []
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
        collection = await self._get_collection()
        # Use ObjectId for _id in the delete filter
        result = await collection.delete_many({"_id": {"$in": [ObjectId(id) for id in ids]}})
        return result.deleted_count > 0

    async def find_one(self, filters: Dict[str, Any]) -> Optional[T]:
        collection = await self._get_collection()
        doc = await collection.find_one(filters)
        return self.model_class(**doc) if doc else None

    async def find_many(
        self,
        filters: Dict[str, Any],
        skip: int = 0,
        limit: int = 100,
        sort: Optional[List[tuple]] = None
    ) -> List[T]:
        collection = await self.collection
        cursor = collection.find(filters)
        
        if sort:
            cursor = cursor.sort(sort)
        
        cursor = cursor.skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        return [self.model_class(**doc) for doc in docs]

    async def aggregate(self, pipeline: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        collection = await self.collection
        cursor = collection.aggregate(pipeline)
        return await cursor.to_list(length=None)
