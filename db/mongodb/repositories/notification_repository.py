from typing import List, Optional
from datetime import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorDatabase

from backend.db.mongodb.base_repository import BaseRepository
from backend.db.mongodb.models.notification import (
    Notification, 
    NotificationCreate, 
    NotificationUpdate,
    NotificationType
)
from backend.db.mongodb.mongodb import get_mongodb
import structlog

logger = structlog.get_logger()


class NotificationRepository(BaseRepository[Notification]):
    """Repository for notification CRUD operations."""
    
    def __init__(self, db: AsyncIOMotorDatabase):
        super().__init__("notifications", Notification)
        self.db = db
        self._collection = db["notifications"]

    @property
    def collection(self):
        return self._collection

    async def create_notification(self, notification: Notification) -> Notification:
        """Create a new notification."""
        notification_dict = notification.model_dump(by_alias=True)
        result = await self.collection.insert_one(notification_dict)
        notification.id = result.inserted_id
        logger.info("notification_created", notification_id=str(result.inserted_id))
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
            "notification_type": notification_type.value
        }) \
            .sort("created_at", -1) \
            .skip(skip) \
            .limit(limit)

        notifications = []
        async for doc in cursor:
            notifications.append(Notification(**doc))
        return notifications

    async def get_notification_by_id(
        self,
        notification_id: str,
        user_id: str
    ) -> Optional[Notification]:
        """Get a specific notification by ID."""
        doc = await self.collection.find_one({
            "_id": ObjectId(notification_id),
            "user_id": user_id
        })
        if doc:
            return Notification(**doc)
        return None

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

    async def mark_all_as_read(self, user_id: str) -> int:
        """Mark all notifications as read for a user. Returns count of updated notifications."""
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
        return result.modified_count

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

    async def delete_all_notifications(self, user_id: str) -> int:
        """Delete all notifications for a user. Returns count of deleted notifications."""
        result = await self.collection.delete_many({"user_id": user_id})
        return result.deleted_count

    async def get_unread_count(self, user_id: str) -> int:
        """Get count of unread notifications for a user."""
        return await self.collection.count_documents({
            "user_id": user_id,
            "read": False
        })

    async def cleanup_old_notifications(
        self,
        older_than_days: int = 30,
        read_only: bool = True
    ) -> int:
        """Delete old notifications. Returns count of deleted notifications."""
        from datetime import timedelta
        cutoff_date = datetime.utcnow() - timedelta(days=older_than_days)
        
        query = {"created_at": {"$lt": cutoff_date}}
        if read_only:
            query["read"] = True
            
        result = await self.collection.delete_many(query)
        logger.info(
            "old_notifications_cleaned", 
            deleted_count=result.deleted_count,
            older_than_days=older_than_days
        )
        return result.deleted_count


# Singleton instance - lazily initialized
_notification_repository: Optional[NotificationRepository] = None


async def get_notification_repository() -> NotificationRepository:
    """Get or create the notification repository singleton."""
    global _notification_repository
    if _notification_repository is None:
        db = await get_mongodb()
        _notification_repository = NotificationRepository(db)
    return _notification_repository


# For backward compatibility - will be lazily initialized on first use
notification_repository = None


async def init_notification_repository() -> NotificationRepository:
    """Initialize the notification repository. Call this at app startup."""
    global notification_repository
    notification_repository = await get_notification_repository()
    return notification_repository