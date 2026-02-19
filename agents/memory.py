"""
Conversation Memory — Stores and retrieves conversation history from MongoDB.

Schema per conversation document:
{
    "_id": ObjectId,
    "conversation_id": "uuid-string",
    "user_id": "user-uuid",
    "title": "Auto-generated title",
    "messages": [
        {"role": "user", "content": "...", "timestamp": "ISO8601"},
        {"role": "assistant", "content": "...", "timestamp": "ISO8601", "agent": "research"},
        ...
    ],
    "agent_history": ["general", "research", "research"],
    "created_at": "ISO8601",
    "updated_at": "ISO8601",
    "metadata": {}
}
"""

import structlog
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

logger = structlog.get_logger(__name__)

COLLECTION_NAME = "conversations"
MAX_MESSAGES_PER_CONVERSATION = 200  # Safety limit


async def _get_collection():
    """Get the MongoDB conversations collection."""
    from backend.db.mongodb import get_mongodb
    db = await get_mongodb()
    return db.db[COLLECTION_NAME]


async def save_message(
    conversation_id: str,
    user_id: str,
    role: str,       # "user" or "assistant"
    content: str,
    agent: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Append a message to a conversation. Creates the conversation document if it doesn't exist.
    """
    try:
        collection = await _get_collection()
    except Exception as e:
        logger.warning("memory_db_unavailable", error=str(e))
        return
    
    message = {
        "role": role,
        "content": content,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    if agent:
        message["agent"] = agent
    if metadata:
        message["metadata"] = metadata
    
    now = datetime.now(timezone.utc).isoformat()
    
    update_ops: Dict[str, Any] = {
        "$push": {
            "messages": {
                "$each": [message],
                "$slice": -MAX_MESSAGES_PER_CONVERSATION,
            },
        },
        "$set": {"updated_at": now, "user_id": user_id},
        "$setOnInsert": {"created_at": now, "metadata": {}},
    }
    
    # $addToSet and $setOnInsert cannot both touch 'agent_history' in MongoDB.
    # Instead, use $push with $addToSet semantics via a separate pipeline, or
    # simply use $addToSet when agent is present (no $setOnInsert on agent_history).
    # The field is created as [] on insert via the first $addToSet call itself.
    if agent:
        update_ops["$addToSet"] = {"agent_history": agent}
    
    await collection.update_one(
        {"conversation_id": conversation_id},
        update_ops,
        upsert=True,
    )
    
    logger.debug(
        "message_saved",
        conversation_id=conversation_id,
        role=role,
        agent=agent,
        content_length=len(content),
    )


async def get_conversation_history(
    conversation_id: str,
    limit: int = 10,
) -> List[Dict[str, str]]:
    """
    Retrieve recent messages from a conversation.
    Returns empty list if conversation not found or DB unavailable.
    """
    try:
        collection = await _get_collection()
    except Exception:
        return []
    
    doc = await collection.find_one(
        {"conversation_id": conversation_id},
        {"messages": {"$slice": -limit}},
    )
    
    if not doc or "messages" not in doc:
        return []
    
    return [
        {"role": msg["role"], "content": msg["content"]}
        for msg in doc["messages"]
    ]


async def get_full_conversation(conversation_id: str) -> Optional[Dict[str, Any]]:
    """Get the full conversation document including metadata."""
    try:
        collection = await _get_collection()
    except Exception:
        return None
    return await collection.find_one({"conversation_id": conversation_id})


async def list_user_conversations(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
) -> List[Dict[str, Any]]:
    """List all conversations for a user, sorted by most recent."""
    try:
        collection = await _get_collection()
    except Exception:
        return []
    
    cursor = collection.find(
        {"user_id": user_id},
        {
            "conversation_id": 1,
            "title": 1,
            "created_at": 1,
            "updated_at": 1,
            "agent_history": 1,
            "messages": {"$slice": 1},
        },
    ).sort("updated_at", -1).skip(offset).limit(limit)
    
    conversations = []
    async for doc in cursor:
        first_msg = doc.get("messages", [{}])[0] if doc.get("messages") else {}
        conversations.append({
            "conversation_id": doc["conversation_id"],
            "title": doc.get("title", first_msg.get("content", "")[:50]),
            "preview": first_msg.get("content", "")[:100],
            "created_at": doc.get("created_at"),
            "updated_at": doc.get("updated_at"),
            "agents_used": doc.get("agent_history", []),
        })
    
    return conversations


async def delete_conversation(conversation_id: str, user_id: str) -> bool:
    """Delete a conversation. Returns True if deleted, False if not found."""
    try:
        collection = await _get_collection()
    except Exception:
        return False
    result = await collection.delete_one({"conversation_id": conversation_id, "user_id": user_id})
    return result.deleted_count > 0


async def generate_conversation_title(
    conversation_id: str,
    first_message: str,
    llm_client=None,
) -> str:
    """Generate a short, descriptive title from the first user message."""
    if llm_client:
        try:
            from backend.ai_models import LLMConfig
            config = LLMConfig(temperature=0.3, max_tokens=20)
            response = await llm_client.generate(
                messages=[{"role": "user", "content": f'Generate a very short title (3-6 words, no quotes) for a conversation that starts with: "{first_message[:200]}"'}],
                config=config,
            )
            title = response.content.strip().strip('"\'')
        except Exception:
            title = first_message[:50] + ("..." if len(first_message) > 50 else "")
    else:
        title = first_message[:50] + ("..." if len(first_message) > 50 else "")
    
    try:
        collection = await _get_collection()
        await collection.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"title": title}},
        )
    except Exception:
        pass
    return title


async def ensure_indexes():
    """Create MongoDB indexes for conversations collection."""
    try:
        collection = await _get_collection()
        await collection.create_index("conversation_id", unique=True)
        await collection.create_index("user_id")
        await collection.create_index("updated_at")
        logger.info("conversation_indexes_created")
    except Exception as e:
        logger.warning("conversation_indexes_failed", error=str(e))
