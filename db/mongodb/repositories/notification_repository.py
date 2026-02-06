from typing import List, Optional, Dict, Any
from pymongo import ASCENDING
from backend.db.mongodb.base_repository import BaseRepository
from backend.db.mongodb.models.notification import Notification, NotificationCreate, NotificationUpdate
import structlog
from datetime import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase
from backend.db.mongodb.mongodb import get_mongodb

logger = structlog.get_logger()

class NotificationRepository(BaseRepository[Notification]):
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__("notifications", Notification)
        self.db = db

    @classmethod
    async def create(cls) -> 'NotificationRepository':
        db = await get_mongodb()
        return cls(db)

# Create a singleton instance
notification_repository: Optional[NotificationRepository] = None

async def get_notification_repository() -> NotificationRepository:
    global notification_repository
    if notification_repository is None:
        notification_repository = await NotificationRepository.create()
    return notification_repository

    async def create_notification(self, notification: Notification) -> Notification:
        """Create a new notification."""
        notification_dict = notification.dict(by_alias=True)
        result = await self.collection.insert_one(notification_dict)
        notification.id = result.inserted_id
        return notification

    async def get_user_notifications(
        self,
        user_id: str,
        unread_only: bool = False,
        limit: int = 50,
        skip: int = 0
    ) -> List[Notification]:
        """Get notifications for a user."""
        query = {"user_id": user_id}
        if unread_only:
            query["read"] = False

        cursor = self.collection.find(query) \
            .sort("created_at", -1) \
            .skip(skip) \
            .limit(limit)

        notifications = []
        async for doc in cursor:
            notifications.append(Notification(**doc))
        return notifications

    async def mark_as_read(
        self,
        notification_id: str,
        user_id: str
    ) -> bool:
        """Mark a notification as read."""
        result = await self.collection.update_one(
            {
                "_id": ObjectId(notification_id),
                "user_id": user_id
            },
            {
                "$set": {
                    "read": True,
                    "read_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0

    async def mark_all_as_read(self, user_id: str) -> bool:
        """Mark all notifications as read for a user."""
        result = await self.collection.update_many(
            {
                "user_id": user_id,
                "read": False
            },
            {
                "$set": {
                    "read": True,
                    "read_at": datetime.utcnow()
                }
            }
        )
        return result.modified_count > 0

    async def delete_notification(
        self,
        notification_id: str,
        user_id: str
    ) -> bool:
        """Delete a notification."""
        result = await self.collection.delete_one({
            "_id": ObjectId(notification_id),
            "user_id": user_id
        })
        return result.deleted_count > 0

    async def get_unread_count(self, user_id: str) -> int:
        """Get count of unread notifications for a user."""
        return await self.collection.count_documents({
            "user_id": user_id,
            "read": False
        })

    async def get_notifications_by_type(
        self,
        user_id: str,
        notification_type: NotificationType,
        limit: int = 50,
        skip: int = 0
    ) -> List[Notification]:
        """Get notifications of a specific type for a user."""
        cursor = self.collection.find({
            "user_id": user_id,
            "notification_type": notification_type
        }) \
            .sort("created_at", -1) \
            .skip(skip) \
            .limit(limit)

        notifications = []
        async for doc in cursor:
            notifications.append(Notification(**doc))
        return notifications 