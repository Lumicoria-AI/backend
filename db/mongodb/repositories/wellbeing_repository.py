from typing import Optional, List, Dict, Any, Union
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId
from datetime import datetime, timedelta
from ..base_repository import BaseRepository
from backend.models.mongodb_models import (
    WellbeingData,
    WellbeingCreate,
    WellbeingMetric,
    WellbeingStatus,
    WellbeingCategory,
    WellbeingRecommendation,
    WellbeingGoal
)
import structlog

logger = structlog.get_logger()

class WellbeingRepository(BaseRepository[WellbeingData]):
    def __init__(self):
        super().__init__("wellbeing", WellbeingData)

    async def _create_indexes(self):
        collection = await self.collection
        # Create indexes for common queries
        await collection.create_index("user_id")
        await collection.create_index("organization_id")
        await collection.create_index("category")
        await collection.create_index("created_at")
        await collection.create_index("status")
        # Compound indexes for common queries
        await collection.create_index([
            ("user_id", ASCENDING),
            ("category", ASCENDING),
            ("created_at", DESCENDING)
        ])
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("category", ASCENDING),
            ("status", ASCENDING)
        ])
        # Text search index for notes and recommendations
        await collection.create_index([
            ("notes", "text"),
            ("recommendations.title", "text"),
            ("goals.description", "text")
        ])

    async def create_wellbeing_entry(
        self,
        wellbeing_data: WellbeingCreate,
        user_id: str,
        organization_id: str
    ) -> WellbeingData:
        """Create a new wellbeing entry with metrics and status."""
        entry_dict = wellbeing_data.dict()
        entry_dict.update({
            "user_id": ObjectId(user_id),
            "organization_id": ObjectId(organization_id),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "status": WellbeingStatus.ACTIVE,
            "metrics_history": [{
                "metrics": entry_dict.get("metrics", {}),
                "recorded_at": datetime.utcnow()
            }]
        })

        try:
            return await self.create(entry_dict)
        except Exception as e:
            logger.error(
                "Failed to create wellbeing entry",
                error=str(e),
                user_id=user_id,
                organization_id=organization_id
            )
            raise

    async def update_wellbeing_metrics(
        self,
        entry_id: str,
        metrics: Dict[str, float],
        notes: Optional[str] = None
    ) -> Optional[WellbeingData]:
        """Update wellbeing metrics with history tracking."""
        update_data = {
            "metrics": metrics,
            "updated_at": datetime.utcnow(),
            "$push": {
                "metrics_history": {
                    "metrics": metrics,
                    "recorded_at": datetime.utcnow(),
                    "notes": notes
                }
            }
        }
        return await self.update(entry_id, update_data)

    async def get_user_wellbeing_history(
        self,
        user_id: str,
        category: Optional[WellbeingCategory] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[WellbeingData]:
        """Get wellbeing history for a user with filtering."""
        filters = {"user_id": ObjectId(user_id)}
        if category:
            filters["category"] = category
        if start_date or end_date:
            filters["created_at"] = {}
            if start_date:
                filters["created_at"]["$gte"] = start_date
            if end_date:
                filters["created_at"]["$lte"] = end_date

        return await self.find_many(
            filters,
            skip=skip,
            limit=limit,
            sort=[("created_at", DESCENDING)]
        )

    async def add_wellbeing_goal(
        self,
        entry_id: str,
        goal: WellbeingGoal
    ) -> Optional[WellbeingData]:
        """Add a wellbeing goal to an entry."""
        goal_data = goal.dict()
        goal_data.update({
            "created_at": datetime.utcnow(),
            "status": "active"
        })

        update_data = {
            "$push": {"goals": goal_data},
            "updated_at": datetime.utcnow()
        }
        return await self.update(entry_id, update_data)

    async def update_goal_status(
        self,
        entry_id: str,
        goal_id: str,
        status: str,
        completion_notes: Optional[str] = None
    ) -> Optional[WellbeingData]:
        """Update the status of a wellbeing goal."""
        update_data = {
            "$set": {
                f"goals.$[goal].status": status,
                f"goals.$[goal].updated_at": datetime.utcnow()
            },
            "updated_at": datetime.utcnow()
        }

        if completion_notes:
            update_data["$set"][f"goals.$[goal].completion_notes"] = completion_notes

        return await self.update(
            entry_id,
            update_data,
            array_filters=[{"goal._id": ObjectId(goal_id)}]
        )

    async def add_recommendation(
        self,
        entry_id: str,
        recommendation: WellbeingRecommendation
    ) -> Optional[WellbeingData]:
        """Add a wellbeing recommendation."""
        recommendation_data = recommendation.dict()
        recommendation_data.update({
            "created_at": datetime.utcnow(),
            "status": "active"
        })

        update_data = {
            "$push": {"recommendations": recommendation_data},
            "updated_at": datetime.utcnow()
        }
        return await self.update(entry_id, update_data)

    async def get_wellbeing_stats(
        self,
        organization_id: str,
        time_range: Optional[timedelta] = None,
        category: Optional[WellbeingCategory] = None
    ) -> Dict[str, Any]:
        """Get wellbeing statistics and trends."""
        match = {"organization_id": ObjectId(organization_id)}
        if time_range:
            match["created_at"] = {
                "$gte": datetime.utcnow() - time_range
            }
        if category:
            match["category"] = category

        pipeline = [
            {"$match": match},
            {"$group": {
                "_id": {
                    "category": "$category",
                    "status": "$status"
                },
                "count": {"$sum": 1},
                "avg_metrics": {
                    "$avg": {
                        "$map": {
                            "input": "$metrics_history",
                            "as": "metric",
                            "in": "$$metric.metrics"
                        }
                    }
                },
                "active_goals": {
                    "$sum": {
                        "$size": {
                            "$filter": {
                                "input": "$goals",
                                "as": "goal",
                                "cond": {"$eq": ["$$goal.status", "active"]}
                            }
                        }
                    }
                },
                "completed_goals": {
                    "$sum": {
                        "$size": {
                            "$filter": {
                                "input": "$goals",
                                "as": "goal",
                                "cond": {"$eq": ["$$goal.status", "completed"]}
                            }
                        }
                    }
                }
            }},
            {"$group": {
                "_id": None,
                "total_entries": {"$sum": "$count"},
                "categories": {
                    "$push": {
                        "category": "$_id.category",
                        "status": "$_id.status",
                        "count": "$count",
                        "avg_metrics": "$avg_metrics",
                        "active_goals": "$active_goals",
                        "completed_goals": "$completed_goals"
                    }
                }
            }}
        ]

        results = await self.aggregate(pipeline)
        return results[0] if results else {
            "total_entries": 0,
            "categories": []
        }

    async def get_user_wellbeing_summary(
        self,
        user_id: str,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get a summary of user's wellbeing data."""
        match = {"user_id": ObjectId(user_id)}
        if time_range:
            match["created_at"] = {
                "$gte": datetime.utcnow() - time_range
            }

        pipeline = [
            {"$match": match},
            {"$sort": {"created_at": -1}},
            {"$group": {
                "_id": None,
                "latest_metrics": {"$first": "$metrics"},
                "metric_history": {"$push": "$metrics_history"},
                "active_goals": {
                    "$push": {
                        "$filter": {
                            "input": "$goals",
                            "as": "goal",
                            "cond": {"$eq": ["$$goal.status", "active"]}
                        }
                    }
                },
                "completed_goals": {
                    "$push": {
                        "$filter": {
                            "input": "$goals",
                            "as": "goal",
                            "cond": {"$eq": ["$$goal.status", "completed"]}
                        }
                    }
                },
                "recommendations": {"$push": "$recommendations"},
                "categories": {"$addToSet": "$category"}
            }},
            {"$project": {
                "latest_metrics": 1,
                "metric_trends": {
                    "$map": {
                        "input": "$metric_history",
                        "as": "history",
                        "in": {
                            "date": "$$history.recorded_at",
                            "metrics": "$$history.metrics"
                        }
                    }
                },
                "active_goals": {"$setUnion": ["$active_goals"]},
                "completed_goals": {"$setUnion": ["$completed_goals"]},
                "recommendations": {"$setUnion": ["$recommendations"]},
                "categories": 1
            }}
        ]

        results = await self.aggregate(pipeline)
        return results[0] if results else {
            "latest_metrics": {},
            "metric_trends": [],
            "active_goals": [],
            "completed_goals": [],
            "recommendations": [],
            "categories": []
        }

    async def get_organization_wellbeing_trends(
        self,
        organization_id: str,
        time_range: timedelta,
        category: Optional[WellbeingCategory] = None
    ) -> Dict[str, Any]:
        """Get wellbeing trends across an organization."""
        match = {
            "organization_id": ObjectId(organization_id),
            "created_at": {
                "$gte": datetime.utcnow() - time_range
            }
        }
        if category:
            match["category"] = category

        pipeline = [
            {"$match": match},
            {"$unwind": "$metrics_history"},
            {"$group": {
                "_id": {
                    "date": {
                        "$dateToString": {
                            "format": "%Y-%m-%d",
                            "date": "$metrics_history.recorded_at"
                        }
                    },
                    "category": "$category"
                },
                "avg_metrics": {"$avg": "$metrics_history.metrics"},
                "count": {"$sum": 1}
            }},
            {"$group": {
                "_id": "$_id.category",
                "trends": {
                    "$push": {
                        "date": "$_id.date",
                        "avg_metrics": "$avg_metrics",
                        "count": "$count"
                    }
                }
            }},
            {"$sort": {"trends.date": 1}}
        ]

        results = await self.aggregate(pipeline)
        return {
            "categories": {
                result["_id"]: result["trends"]
                for result in results
            }
        }

    async def get_latest_user_metrics(
        self,
        user_id: str,
        metrics: List[str]
    ) -> Dict[str, float]:
        """Get the most recent value for specified metrics for a user."""
        match = {"user_id": ObjectId(user_id)}
        projection = {
            "metrics_history": {
                "$elemMatch": {
                    "metrics": {
                        "$elemMatch": {
                            "$in": metrics
                        }
                    }
                }
            }
        }

        results = await self.find_one(
            match,
            projection=projection
        )

        if results:
            metrics_history = results["metrics_history"]
            latest_metrics = {}
            for history in metrics_history:
                for metric, value in history["metrics"].items():
                    if metric in metrics:
                        latest_metrics[metric] = value
            return latest_metrics
        else:
            return {}

# Create a singleton instance
wellbeing_repository = WellbeingRepository() 