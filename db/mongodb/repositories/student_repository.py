from typing import List, Optional, Dict, Any
from bson import ObjectId
from datetime import datetime
from backend.db.mongodb.base_repository import BaseRepository
from backend.models.mongodb_models import StudentInteraction
import structlog

logger = structlog.get_logger()

class StudentRepository(BaseRepository[StudentInteraction]):
    def __init__(self):
        super().__init__("student_interactions", StudentInteraction)

    async def _create_indexes(self):
        collection = await self.collection
        await collection.create_index("user_id")
        await collection.create_index("request_type")
        await collection.create_index("created_at")

    async def create_interaction(
        self,
        user_id: str,
        request_type: str,
        content: str,
        context: Dict[str, Any],
        response: Dict[str, Any],
        model_used: str,
        raw_response: Optional[str] = None,
        citations: Optional[List[Dict[str, Any]]] = None
    ) -> StudentInteraction:
        """Create and save a new student interaction."""
        interaction_data = {
            "user_id": ObjectId(user_id),
            "request_type": request_type,
            "content": content,
            "context": context,
            "response": response,
            "raw_response": raw_response,
            "model_used": model_used,
            "citations": citations or [],
            "created_at": datetime.utcnow()
        }
        
        try:
            return await self.create(interaction_data)
        except Exception as e:
            logger.error("Failed to create student interaction", error=str(e), user_id=user_id)
            raise

    async def get_user_history(
        self,
        user_id: str,
        limit: int = 20,
        skip: int = 0
    ) -> List[StudentInteraction]:
        """Get past interactions for a specific user."""
        filters = {"user_id": ObjectId(user_id)}
        return await self.find_many(
            filters,
            limit=limit,
            skip=skip,
            sort=[("created_at", -1)]
        )

student_repository = StudentRepository()
