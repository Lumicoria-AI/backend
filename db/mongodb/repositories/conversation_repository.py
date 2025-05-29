from typing import Optional, List, Dict, Any, Union
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId
from datetime import datetime, timedelta
from ..base_repository import BaseRepository
from backend.models.mongodb_models import (
    Conversation,
    ConversationCreate,
    Message,
    MessageCreate,
    ConversationStatus,
    MessageType,
    MessageStatus,
    ContextData
)
import structlog

logger = structlog.get_logger()

class ConversationRepository(BaseRepository[Conversation]):
    def __init__(self):
        super().__init__("conversations", Conversation)

    async def _create_indexes(self):
        collection = await self.collection
        # Create indexes for common queries
        await collection.create_index("created_by")
        await collection.create_index("organization_id")
        await collection.create_index("status")
        await collection.create_index("created_at")
        await collection.create_index("updated_at")
        await collection.create_index("participants")
        await collection.create_index("agent_id")
        # Compound indexes for common queries
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("status", ASCENDING),
            ("created_at", DESCENDING)
        ])
        await collection.create_index([
            ("participants", ASCENDING),
            ("status", ASCENDING),
            ("updated_at", DESCENDING)
        ])
        # Text search index for messages
        await collection.create_index([
            ("messages.content", "text"),
            ("messages.metadata", "text")
        ])

    async def create_conversation(
        self,
        conversation_data: ConversationCreate,
        creator_id: str,
        organization_id: str,
        agent_id: Optional[str] = None
    ) -> Conversation:
        """Create a new conversation with initial message."""
        conversation_dict = conversation_data.dict()
        initial_message = Message(
            content=conversation_dict.get("initial_message", ""),
            type=MessageType.TEXT,
            status=MessageStatus.SENT,
            sender_id=ObjectId(creator_id),
            created_at=datetime.utcnow()
        )

        conversation_dict.update({
            "created_by": ObjectId(creator_id),
            "organization_id": ObjectId(organization_id),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "status": ConversationStatus.ACTIVE,
            "participants": [ObjectId(creator_id)],
            "messages": [initial_message.dict()],
            "context": {},
            "metadata": {}
        })

        if agent_id:
            conversation_dict["agent_id"] = ObjectId(agent_id)
            conversation_dict["participants"].append(ObjectId(agent_id))

        try:
            return await self.create(conversation_dict)
        except Exception as e:
            logger.error(
                "Failed to create conversation",
                error=str(e),
                creator_id=creator_id,
                organization_id=organization_id
            )
            raise

    async def add_message(
        self,
        conversation_id: str,
        message_data: MessageCreate,
        sender_id: str
    ) -> Optional[Conversation]:
        """Add a new message to a conversation."""
        message = Message(
            content=message_data.content,
            type=message_data.type,
            status=MessageStatus.SENT,
            sender_id=ObjectId(sender_id),
            created_at=datetime.utcnow(),
            metadata=message_data.metadata or {}
        )

        update_data = {
            "$push": {"messages": message.dict()},
            "updated_at": datetime.utcnow()
        }

        return await self.update(conversation_id, update_data)

    async def get_conversation_messages(
        self,
        conversation_id: str,
        skip: int = 0,
        limit: int = 100,
        before_date: Optional[datetime] = None
    ) -> List[Message]:
        """Get messages from a conversation with pagination."""
        conversation = await self.get_by_id(conversation_id)
        if not conversation:
            return []

        messages = conversation.messages
        if before_date:
            messages = [m for m in messages if m["created_at"] < before_date]

        messages.sort(key=lambda x: x["created_at"], reverse=True)
        return [Message(**m) for m in messages[skip:skip + limit]]

    async def get_user_conversations(
        self,
        user_id: str,
        status: Optional[ConversationStatus] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Conversation]:
        """Get conversations for a user."""
        filters = {"participants": ObjectId(user_id)}
        if status:
            filters["status"] = status

        return await self.find_many(
            filters,
            skip=skip,
            limit=limit,
            sort=[("updated_at", DESCENDING)]
        )

    async def get_agent_conversations(
        self,
        agent_id: str,
        status: Optional[ConversationStatus] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Conversation]:
        """Get conversations for an agent."""
        filters = {"agent_id": ObjectId(agent_id)}
        if status:
            filters["status"] = status

        return await self.find_many(
            filters,
            skip=skip,
            limit=limit,
            sort=[("updated_at", DESCENDING)]
        )

    async def search_conversations(
        self,
        query: str,
        organization_id: Optional[str] = None,
        user_id: Optional[str] = None,
        status: Optional[ConversationStatus] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Conversation]:
        """Search conversations by message content."""
        search_filter = {
            "messages": {
                "$elemMatch": {
                    "$text": {"$search": query}
                }
            }
        }

        if organization_id:
            search_filter["organization_id"] = ObjectId(organization_id)
        if user_id:
            search_filter["participants"] = ObjectId(user_id)
        if status:
            search_filter["status"] = status

        return await self.find_many(
            search_filter,
            skip=skip,
            limit=limit,
            sort=[("updated_at", DESCENDING)]
        )

    async def update_conversation_status(
        self,
        conversation_id: str,
        status: ConversationStatus
    ) -> Optional[Conversation]:
        """Update conversation status."""
        update_data = {
            "status": status,
            "updated_at": datetime.utcnow(),
            "$push": {
                "status_history": {
                    "status": status,
                    "changed_at": datetime.utcnow()
                }
            }
        }
        return await self.update(conversation_id, update_data)

    async def add_participant(
        self,
        conversation_id: str,
        user_id: str,
        added_by: str
    ) -> Optional[Conversation]:
        """Add a participant to a conversation."""
        participant_data = {
            "user_id": ObjectId(user_id),
            "added_by": ObjectId(added_by),
            "added_at": datetime.utcnow()
        }

        update_data = {
            "$addToSet": {"participants": ObjectId(user_id)},
            "$push": {"participant_history": participant_data},
            "updated_at": datetime.utcnow()
        }

        return await self.update(conversation_id, update_data)

    async def remove_participant(
        self,
        conversation_id: str,
        user_id: str,
        removed_by: str
    ) -> Optional[Conversation]:
        """Remove a participant from a conversation."""
        removal_data = {
            "user_id": ObjectId(user_id),
            "removed_by": ObjectId(removed_by),
            "removed_at": datetime.utcnow()
        }

        update_data = {
            "$pull": {"participants": ObjectId(user_id)},
            "$push": {"participant_history": removal_data},
            "updated_at": datetime.utcnow()
        }

        return await self.update(conversation_id, update_data)

    async def update_context(
        self,
        conversation_id: str,
        context_data: ContextData
    ) -> Optional[Conversation]:
        """Update conversation context."""
        update_data = {
            "context": context_data.dict(),
            "updated_at": datetime.utcnow(),
            "$push": {
                "context_history": {
                    "context": context_data.dict(),
                    "updated_at": datetime.utcnow()
                }
            }
        }
        return await self.update(conversation_id, update_data)

    async def get_conversation_stats(
        self,
        organization_id: str,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get conversation statistics and metrics."""
        match = {"organization_id": ObjectId(organization_id)}
        if time_range:
            match["created_at"] = {
                "$gte": datetime.utcnow() - time_range
            }

        pipeline = [
            {"$match": match},
            {"$group": {
                "_id": {
                    "status": "$status",
                    "has_agent": {"$cond": [{"$ifNull": ["$agent_id", false]}, true, false]}
                },
                "count": {"$sum": 1},
                "total_messages": {"$sum": {"$size": "$messages"}},
                "avg_messages": {"$avg": {"$size": "$messages"}},
                "conversations": {"$push": {
                    "id": "$_id",
                    "created_at": "$created_at",
                    "updated_at": "$updated_at",
                    "message_count": {"$size": "$messages"}
                }}
            }},
            {"$group": {
                "_id": None,
                "total": {"$sum": "$count"},
                "total_messages": {"$sum": "$total_messages"},
                "avg_messages_per_conversation": {"$avg": "$avg_messages"},
                "categories": {
                    "$push": {
                        "status": "$_id.status",
                        "has_agent": "$_id.has_agent",
                        "count": "$count",
                        "conversations": "$conversations"
                    }
                }
            }}
        ]

        results = await self.aggregate(pipeline)
        return results[0] if results else {
            "total": 0,
            "total_messages": 0,
            "avg_messages_per_conversation": 0,
            "categories": []
        }

    async def get_active_conversations(
        self,
        organization_id: str,
        min_activity: Optional[timedelta] = None
    ) -> List[Conversation]:
        """Get currently active conversations."""
        filters = {
            "organization_id": ObjectId(organization_id),
            "status": ConversationStatus.ACTIVE
        }
        
        if min_activity:
            filters["updated_at"] = {
                "$gte": datetime.utcnow() - min_activity
            }

        return await self.find_many(
            filters,
            sort=[("updated_at", DESCENDING)]
        )

    async def get_conversation_thread(
        self,
        conversation_id: str,
        message_id: str
    ) -> List[Message]:
        """Get a thread of messages starting from a specific message."""
        conversation = await self.get_by_id(conversation_id)
        if not conversation:
            return []

        messages = conversation.messages
        thread_messages = []
        current_message = next(
            (m for m in messages if str(m["_id"]) == message_id),
            None
        )

        if current_message:
            thread_messages.append(Message(**current_message))
            # Get messages that reference this message
            thread_messages.extend([
                Message(**m) for m in messages
                if m.get("metadata", {}).get("reply_to") == message_id
            ])

        return sorted(thread_messages, key=lambda x: x.created_at)

# Create a singleton instance
conversation_repository = ConversationRepository() 