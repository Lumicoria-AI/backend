from typing import Optional, List, Dict, Any
from datetime import datetime
from bson import ObjectId
import structlog

from backend.db.mongodb.mongodb import MongoDB

logger = structlog.get_logger()


class WellbeingGoalRepository:
    def __init__(self):
        self.collection = None

    async def _get_collection(self):
        if self.collection is None:
            self.collection = await MongoDB.get_collection("wellbeing_goals")
        return self.collection

    async def create_goal(
        self,
        user_id: str,
        organization_id: str,
        goal_type: str,
        target_value: float,
        start_date: datetime,
        end_date: datetime,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        collection = await self._get_collection()
        goal = {
            "user_id": ObjectId(user_id),
            "organization_id": ObjectId(organization_id),
            "goal_type": goal_type,
            "target_value": target_value,
            "current_value": 0.0,
            "status": "active",
            "progress": 0.0,
            "start_date": start_date,
            "end_date": end_date,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        result = await collection.insert_one(goal)
        goal["_id"] = result.inserted_id
        return self._serialize(goal)

    async def get_user_goals(
        self,
        user_id: str,
        organization_id: str,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        collection = await self._get_collection()
        query = {
            "user_id": ObjectId(user_id),
            "organization_id": ObjectId(organization_id),
        }
        if status:
            query["status"] = status
        docs = await collection.find(query).to_list(length=200)
        return [self._serialize(d) for d in docs]

    async def get_goal_by_id(self, goal_id: str, organization_id: str) -> Optional[Dict[str, Any]]:
        collection = await self._get_collection()
        doc = await collection.find_one({
            "_id": ObjectId(goal_id),
            "organization_id": ObjectId(organization_id),
        })
        return self._serialize(doc) if doc else None

    async def update_goal(
        self,
        goal_id: str,
        organization_id: str,
        update_data: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        collection = await self._get_collection()
        update_data["updated_at"] = datetime.utcnow()
        await collection.update_one(
            {"_id": ObjectId(goal_id), "organization_id": ObjectId(organization_id)},
            {"$set": update_data}
        )
        doc = await collection.find_one({"_id": ObjectId(goal_id)})
        return self._serialize(doc) if doc else None

    def _serialize(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        if not doc:
            return {}
        return {
            "id": str(doc.get("_id")),
            "user_id": str(doc.get("user_id")) if doc.get("user_id") else None,
            "organization_id": str(doc.get("organization_id")) if doc.get("organization_id") else None,
            "goal_type": doc.get("goal_type"),
            "target_value": doc.get("target_value"),
            "current_value": doc.get("current_value"),
            "status": doc.get("status"),
            "progress": doc.get("progress"),
            "start_date": doc.get("start_date"),
            "end_date": doc.get("end_date"),
            "metadata": doc.get("metadata", {}),
            "created_at": doc.get("created_at"),
            "updated_at": doc.get("updated_at"),
        }


wellbeing_goal_repository = WellbeingGoalRepository()
