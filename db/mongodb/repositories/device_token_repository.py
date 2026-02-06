"""
Repository for device token CRUD operations.
"""

from typing import List, Optional
from datetime import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
import structlog

from backend.db.mongodb.mongodb import get_mongodb
from backend.db.mongodb.models.device_token import (
    DeviceToken,
    DeviceTokenCreate,
    DeviceTokenUpdate,
    DevicePlatform
)

logger = structlog.get_logger()


class DeviceTokenRepository:
    """Repository for managing device push notification tokens."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        self.db = db
        self._collection = db["device_tokens"]
    
    @property
    def collection(self):
        return self._collection
    
    async def register_token(
        self,
        user_id: str,
        token: str,
        platform: DevicePlatform = DevicePlatform.UNKNOWN,
        device_name: Optional[str] = None,
        app_version: Optional[str] = None
    ) -> DeviceToken:
        """
        Register a new device token or update existing one.
        
        If the token already exists for this user, update it.
        If the token exists for a different user, reassign it.
        """
        now = datetime.utcnow()
        
        # Check if token already exists
        existing = await self.collection.find_one({"token": token})
        
        if existing:
            # Token exists - update it
            if existing["user_id"] != user_id:
                logger.info(
                    "device_token_reassigned",
                    old_user_id=existing["user_id"],
                    new_user_id=user_id
                )
            
            await self.collection.update_one(
                {"token": token},
                {
                    "$set": {
                        "user_id": user_id,
                        "platform": platform.value,
                        "device_name": device_name,
                        "app_version": app_version,
                        "updated_at": now,
                        "is_active": True
                    }
                }
            )
            
            updated = await self.collection.find_one({"token": token})
            return DeviceToken(**updated)
        
        # New token - create it
        device_token = DeviceToken(
            user_id=user_id,
            token=token,
            platform=platform,
            device_name=device_name,
            app_version=app_version,
            created_at=now,
            updated_at=now,
            is_active=True
        )
        
        result = await self.collection.insert_one(
            device_token.model_dump(by_alias=True)
        )
        device_token.id = result.inserted_id
        
        logger.info("device_token_registered", user_id=user_id, platform=platform.value)
        return device_token
    
    async def get_user_tokens(
        self,
        user_id: str,
        active_only: bool = True
    ) -> List[DeviceToken]:
        """Get all device tokens for a user."""
        query = {"user_id": user_id}
        if active_only:
            query["is_active"] = True
        
        cursor = self.collection.find(query)
        tokens = []
        async for doc in cursor:
            tokens.append(DeviceToken(**doc))
        
        return tokens
    
    async def get_token_by_id(self, token_id: str) -> Optional[DeviceToken]:
        """Get a device token by its ID."""
        doc = await self.collection.find_one({"_id": ObjectId(token_id)})
        if doc:
            return DeviceToken(**doc)
        return None
    
    async def update_last_used(self, token: str) -> bool:
        """Update the last_used timestamp for a token."""
        result = await self.collection.update_one(
            {"token": token},
            {"$set": {"last_used": datetime.utcnow()}}
        )
        return result.modified_count > 0
    
    async def deactivate_token(self, user_id: str, token: str) -> bool:
        """Deactivate a device token (soft delete)."""
        result = await self.collection.update_one(
            {"user_id": user_id, "token": token},
            {"$set": {"is_active": False, "updated_at": datetime.utcnow()}}
        )
        return result.modified_count > 0
    
    async def delete_token(self, user_id: str, token: str) -> bool:
        """Delete a device token (hard delete)."""
        result = await self.collection.delete_one({
            "user_id": user_id,
            "token": token
        })
        if result.deleted_count > 0:
            logger.info("device_token_deleted", user_id=user_id)
            return True
        return False
    
    async def delete_all_user_tokens(self, user_id: str) -> int:
        """Delete all device tokens for a user."""
        result = await self.collection.delete_many({"user_id": user_id})
        return result.deleted_count
    
    async def cleanup_inactive_tokens(self, days_inactive: int = 90) -> int:
        """Delete tokens that haven't been used in the specified days."""
        from datetime import timedelta
        cutoff = datetime.utcnow() - timedelta(days=days_inactive)
        
        result = await self.collection.delete_many({
            "$or": [
                {"last_used": {"$lt": cutoff}},
                {"last_used": None, "created_at": {"$lt": cutoff}}
            ]
        })
        
        if result.deleted_count > 0:
            logger.info(
                "inactive_tokens_cleaned",
                deleted_count=result.deleted_count,
                days_inactive=days_inactive
            )
        
        return result.deleted_count
    
    async def get_platform_stats(self, user_id: Optional[str] = None) -> dict:
        """Get statistics about device tokens by platform."""
        match_stage = {"is_active": True}
        if user_id:
            match_stage["user_id"] = user_id
        
        pipeline = [
            {"$match": match_stage},
            {"$group": {
                "_id": "$platform",
                "count": {"$sum": 1}
            }}
        ]
        
        stats = {"ios": 0, "android": 0, "web": 0, "unknown": 0, "total": 0}
        async for doc in self.collection.aggregate(pipeline):
            platform = doc["_id"]
            count = doc["count"]
            stats[platform] = count
            stats["total"] += count
        
        return stats


# Singleton instance - lazily initialized
_device_token_repository: Optional[DeviceTokenRepository] = None


async def get_device_token_repository() -> DeviceTokenRepository:
    """Get or create the device token repository singleton."""
    global _device_token_repository
    if _device_token_repository is None:
        db = await get_mongodb()
        _device_token_repository = DeviceTokenRepository(db)
    return _device_token_repository
