from typing import Optional, List, Dict, Any, Union
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId
from datetime import datetime
from ..base_repository import BaseRepository
from backend.models.mongodb_models import (
    ActivityLogEntry
)
import structlog

logger = structlog.get_logger()


_OBJECT_ID_HEX = frozenset("0123456789abcdefABCDEF")


def _maybe_object_id(value: Optional[Any]) -> Optional[Any]:
    """Coerce to ObjectId only when the value is a real 24-char hex
    string.  Otherwise return the raw value so non-ObjectId resource
    ids (e.g. 'TK-abc12345', UUIDs, slugs) survive without raising.

    MongoDB accepts mixed types in a single field, so storing some
    related_resource_id values as ObjectId and others as plain strings
    is supported and queries on the field still work.
    """
    if value is None or value == "":
        return None
    if isinstance(value, ObjectId):
        return value
    s = str(value)
    if len(s) == 24 and all(c in _OBJECT_ID_HEX for c in s):
        try:
            return ObjectId(s)
        except Exception:
            return s
    return s

class ActivityRepository(BaseRepository[ActivityLogEntry]):
    def __init__(self):
        super().__init__("activity_logs", ActivityLogEntry)

    async def _create_indexes(self):
        collection = await self.collection
        # Create indexes for common queries
        await collection.create_index("organization_id")
        await collection.create_index("user_id")
        await collection.create_index("activity_type")
        await collection.create_index([("timestamp", DESCENDING)])
        # Compound indexes for filtering and sorting
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("timestamp", DESCENDING)
        ])
        await collection.create_index([
            ("user_id", ASCENDING),
            ("timestamp", DESCENDING)
        ])
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("activity_type", ASCENDING),
            ("timestamp", DESCENDING)
        ])
        # Index for related resources (if frequently queried this way)
        await collection.create_index([
            ("related_resource_type", ASCENDING),
            ("related_resource_id", ASCENDING),
            ("timestamp", DESCENDING)
        ])

    async def create_log_entry(
        self,
        organization_id: str,
        user_id: str,
        activity_type: str,
        details: Dict[str, Any],
        related_resource_type: Optional[str] = None,
        related_resource_id: Optional[str] = None
    ) -> ActivityLogEntry:
        """
        Create a new activity log entry.
        """
        entry_data = {
            "organization_id": _maybe_object_id(organization_id),
            "user_id": _maybe_object_id(user_id),
            "activity_type": activity_type,
            "details": details,
            "timestamp": datetime.utcnow(),
            "related_resource_type": related_resource_type,
            "related_resource_id": _maybe_object_id(related_resource_id),
        }

        try:
            return await self.create(entry_data)
        except Exception as e:
            logger.error(
                "Failed to create activity log entry",
                error=str(e),
                organization_id=organization_id,
                user_id=user_id,
                activity_type=activity_type
            )
            raise

    async def get_recent_activity(
        self,
        organization_id: str,
        user_id: Optional[str] = None,
        activity_type: Optional[str] = None,
        limit: int = 10,
        skip: int = 0
    ) -> List[ActivityLogEntry]:
        """
        Get a list of recent activity log entries for an organization or user.
        """
        filters = {"organization_id": _maybe_object_id(organization_id)}
        if user_id:
            filters["user_id"] = _maybe_object_id(user_id)
        if activity_type:
            filters["activity_type"] = activity_type

        return await self.find_many(
            filters,
            limit=limit,
            skip=skip,
            sort=[("timestamp", DESCENDING)]
        )

    async def get_activity_by_resource(
        self,
        organization_id: str,
        related_resource_type: str,
        related_resource_id: str,
        limit: int = 10,
        skip: int = 0
    ) -> List[ActivityLogEntry]:
        """
        Get activity log entries related to a specific resource.
        """
        filters = {
            "organization_id": _maybe_object_id(organization_id),
            "related_resource_type": related_resource_type,
            "related_resource_id": _maybe_object_id(related_resource_id),
        }

        return await self.find_many(
            filters,
            limit=limit,
            skip=skip,
            sort=[("timestamp", DESCENDING)]
        )

# Create a singleton instance
activity_repository = ActivityRepository() 