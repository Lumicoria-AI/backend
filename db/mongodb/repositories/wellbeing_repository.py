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
        self._metrics_collection = None

    async def _get_metrics_collection(self):
        if self._metrics_collection is None:
            from backend.db.mongodb.mongodb import MongoDB
            self._metrics_collection = await MongoDB.get_collection("wellbeing_metrics")
        return self._metrics_collection

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

    async def create_metric(
        self,
        user_id: str,
        organization_id: str,
        metric_type: str,
        value: float,
        metadata: Optional[Dict[str, Any]],
        source: str,
        timestamp: Optional[datetime] = None
    ) -> Dict[str, Any]:
        collection = await self._get_metrics_collection()

        def _to_object_id(value: Any) -> Any:
            try:
                return ObjectId(str(value))
            except Exception:
                return str(value)

        user_oid = _to_object_id(user_id)
        org_oid = _to_object_id(organization_id)

        doc = {
            "user_id": user_oid,
            "organization_id": org_oid,
            "metric_type": metric_type,
            "value": value,
            "metadata": metadata or {},
            "source": source,
            "timestamp": timestamp or datetime.utcnow(),
            "created_at": datetime.utcnow()
        }
        result = await collection.insert_one(doc)
        doc["_id"] = result.inserted_id
        return {
            "id": str(doc["_id"]),
            "user_id": str(user_id),
            "organization_id": str(organization_id),
            "metric_type": metric_type,
            "value": value,
            "metadata": metadata or {},
            "source": source,
            "timestamp": doc["timestamp"],
        }

    async def get_user_metrics(
        self,
        user_id: str,
        organization_id: str,
        metric_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        collection = await self._get_metrics_collection()
        query = {
            "user_id": ObjectId(user_id),
            "organization_id": ObjectId(organization_id)
        }
        if metric_type:
            query["metric_type"] = metric_type
        if start_date or end_date:
            query["timestamp"] = {}
            if start_date:
                query["timestamp"]["$gte"] = start_date
            if end_date:
                query["timestamp"]["$lte"] = end_date
        docs = await collection.find(query).sort("timestamp", -1).to_list(length=limit)
        results = []
        for doc in docs:
            results.append({
                "id": str(doc.get("_id")),
                "user_id": str(doc.get("user_id")),
                "organization_id": str(doc.get("organization_id")),
                "metric_type": doc.get("metric_type"),
                "value": doc.get("value"),
                "metadata": doc.get("metadata", {}),
                "source": doc.get("source"),
                "timestamp": doc.get("timestamp"),
            })
        return results

    async def get_user_wellbeing_data(
        self,
        user_id: str,
        organization_id: str
    ) -> Dict[str, Any]:
        # Provide a minimal aggregate for agent context
        latest_metrics = {}
        metrics = await self.get_user_metrics(
            user_id=user_id,
            organization_id=organization_id,
            limit=50
        )
        for metric in metrics:
            mtype = metric.get("metric_type")
            if mtype and mtype not in latest_metrics:
                latest_metrics[mtype] = metric.get("value")

        return {
            "latest_metrics": latest_metrics,
            "activity_log": [],
            "screen_time": 0,
            "breaks_taken": 0,
            "focus_sessions": 0
        }

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
        """Get the most recent value for specified metrics for a user.

        Driven directly off the raw metrics collection so we can pass
        a projection — the base repository's ``find_one`` does not
        accept one.
        """
        try:
            object_user_id: Any = ObjectId(user_id)
        except Exception:
            object_user_id = user_id

        collection = await self._get_metrics_collection()
        projection = {
            "metric_type": 1,
            "value": 1,
            "timestamp": 1,
            "_id": 0,
        }

        latest_metrics: Dict[str, float] = {}
        for metric_type in metrics:
            doc = await collection.find_one(
                {
                    "$and": [
                        {"user_id": {"$in": [object_user_id, user_id]}},
                        {"metric_type": metric_type},
                    ]
                },
                projection=projection,
                sort=[("timestamp", -1)],
            )
            if doc and "value" in doc:
                try:
                    latest_metrics[metric_type] = float(doc["value"])
                except (TypeError, ValueError):
                    continue
        return latest_metrics

    # ── Methods required by the wellbeing API endpoints ─────────────────

    async def get_recommendations(
        self,
        user_id: str,
        organization_id: str
    ) -> List[Dict[str, Any]]:
        """Get stored wellbeing recommendations for a user."""
        collection = await self._get_metrics_collection()
        cursor = collection.find(
            {"user_id": str(user_id), "record_type": "recommendation"},
            sort=[("created_at", DESCENDING)],
            limit=20,
        )
        return await cursor.to_list(length=20)

    async def save_recommendation(self, recommendation) -> None:
        """Persist a wellbeing recommendation to the database."""
        collection = await self._get_metrics_collection()
        doc = recommendation.dict() if hasattr(recommendation, "dict") else dict(recommendation)
        doc["record_type"] = "recommendation"
        doc["created_at"] = datetime.utcnow()
        await collection.insert_one(doc)

    async def log_break_recommendation(
        self,
        user_id: str,
        organization_id: str,
        recommendation: Dict[str, Any],
    ) -> None:
        """Log a break recommendation for auditing / analytics."""
        collection = await self._get_metrics_collection()
        await collection.insert_one({
            "user_id": str(user_id),
            "organization_id": str(organization_id),
            "record_type": "break_recommendation",
            "recommendation": recommendation,
            "created_at": datetime.utcnow(),
        })

    async def get_break_recommendation(
        self,
        user_id: str,
        organization_id: str,
    ) -> Dict[str, Any]:
        """Return a sensible default break recommendation when the AI agent is unavailable."""
        return {
            "break_type": "micro_break",
            "duration_minutes": 5,
            "reason": "You've been working for a while. A short break will help you recharge.",
            "suggested_activities": [
                "Take a short walk",
                "Do some stretching",
                "Drink a glass of water",
            ],
            "metadata": {"source": "default"},
        }

    async def record_activity(
        self,
        user_id: str,
        organization_id: str,
        activity_type: str,
        duration_minutes: int,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record a wellbeing activity (break, exercise, etc.)."""
        collection = await self._get_metrics_collection()
        doc = {
            "user_id": str(user_id),
            "organization_id": str(organization_id),
            "record_type": "activity",
            "activity_type": str(activity_type),
            "duration_minutes": duration_minutes,
            "metadata": metadata or {},
            "created_at": datetime.utcnow(),
        }
        result = await collection.insert_one(doc)
        doc["id"] = str(result.inserted_id)
        # ``insert_one`` mutates the doc and adds a Mongo ObjectId in
        # ``_id`` — strip it so the dict is JSON-safe for FastAPI.
        doc.pop("_id", None)
        if isinstance(doc.get("created_at"), datetime):
            doc["created_at"] = doc["created_at"].isoformat()
        return doc

    async def get_recent_activities(
        self,
        user_id: str,
        organization_id: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """Get recent wellbeing activities for a user."""
        collection = await self._get_metrics_collection()
        cursor = collection.find(
            {
                "user_id": str(user_id),
                "record_type": "activity",
            },
            sort=[("created_at", DESCENDING)],
            limit=limit,
        )
        docs = await cursor.to_list(length=limit)
        cleaned: List[Dict[str, Any]] = []
        for doc in docs:
            cleaned.append({
                "id": str(doc.get("_id")) if doc.get("_id") is not None else None,
                "user_id": str(doc.get("user_id")) if doc.get("user_id") is not None else None,
                "organization_id": str(doc.get("organization_id"))
                if doc.get("organization_id") is not None
                else None,
                "record_type": doc.get("record_type"),
                "activity_type": doc.get("activity_type"),
                "duration_minutes": doc.get("duration_minutes"),
                "metadata": doc.get("metadata") or {},
                "created_at": doc.get("created_at").isoformat()
                if isinstance(doc.get("created_at"), datetime)
                else doc.get("created_at"),
                "timestamp": doc.get("timestamp").isoformat()
                if isinstance(doc.get("timestamp"), datetime)
                else doc.get("timestamp"),
            })
        return cleaned

    async def get_last_break_time(
        self,
        user_id: str,
        organization_id: str,
    ) -> Optional[datetime]:
        """Get the timestamp of the user's last recorded break."""
        collection = await self._get_metrics_collection()
        doc = await collection.find_one(
            {
                "user_id": str(user_id),
                "record_type": "activity",
                "activity_type": {"$in": ["physical", "relaxation", "mindfulness"]},
            },
            sort=[("created_at", DESCENDING)],
        )
        return doc["created_at"] if doc else None

    async def get_wellbeing_analytics(
        self,
        user_id: str,
        organization_id: str,
        time_range: str = "7d",
    ) -> Dict[str, Any]:
        """Get wellbeing analytics for a user over a time range."""
        days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}.get(time_range, 7)
        since = datetime.utcnow() - timedelta(days=days)

        metrics = await self.get_user_metrics(
            user_id=user_id,
            organization_id=organization_id,
            start_date=since,
            limit=1000,
        )

        # Aggregate by metric_type
        by_type: Dict[str, list] = {}
        for m in metrics:
            mt = m.get("metric_type", "unknown")
            by_type.setdefault(mt, []).append(float(m.get("value", 0)))

        summary = {}
        for mt, values in by_type.items():
            summary[mt] = {
                "count": len(values),
                "avg": round(sum(values) / len(values), 2) if values else 0,
                "min": round(min(values), 2) if values else 0,
                "max": round(max(values), 2) if values else 0,
                "trend": values[-7:] if len(values) >= 7 else values,
            }

        # Get activities in range
        collection = await self._get_metrics_collection()
        activity_cursor = collection.find(
            {
                "user_id": str(user_id),
                "record_type": "activity",
                "created_at": {"$gte": since},
            },
            sort=[("created_at", DESCENDING)],
        )
        activities = await activity_cursor.to_list(length=500)

        return {
            "time_range": time_range,
            "total_metrics": len(metrics),
            "total_activities": len(activities),
            "metrics_summary": summary,
            "recent_activities": [
                {
                    "type": a.get("activity_type"),
                    "duration": a.get("duration_minutes"),
                    "timestamp": a.get("created_at", "").isoformat() if hasattr(a.get("created_at", ""), "isoformat") else str(a.get("created_at", "")),
                }
                for a in activities[:20]
            ],
        }

    async def get_organization_analytics(
        self,
        organization_id: str,
        time_range: str = "7d",
    ) -> Dict[str, Any]:
        """Get organization-wide wellbeing analytics."""
        days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}.get(time_range, 7)
        since = datetime.utcnow() - timedelta(days=days)

        collection = await self._get_metrics_collection()
        pipeline = [
            {"$match": {
                "organization_id": str(organization_id),
                "record_type": {"$in": ["metric", None]},
                "timestamp": {"$gte": since},
            }},
            {"$group": {
                "_id": "$metric_type",
                "count": {"$sum": 1},
                "avg_value": {"$avg": "$value"},
                "unique_users": {"$addToSet": "$user_id"},
            }},
        ]
        results = await collection.aggregate(pipeline).to_list(length=100)

        return {
            "time_range": time_range,
            "metrics": {
                r["_id"]: {
                    "count": r["count"],
                    "avg_value": round(r["avg_value"], 2) if r["avg_value"] else 0,
                    "unique_users": len(r["unique_users"]),
                }
                for r in results if r["_id"]
            },
        }

    async def get_user_goals(
        self,
        user_id: str,
        organization_id: str,
        status: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get wellbeing goals from the metrics collection (fallback for recommendations endpoint)."""
        collection = await self._get_metrics_collection()
        query: Dict[str, Any] = {
            "user_id": str(user_id),
            "record_type": "goal",
        }
        if status:
            query["status"] = status
        cursor = collection.find(query, sort=[("created_at", DESCENDING)])
        return await cursor.to_list(length=50)


# Create a singleton instance
wellbeing_repository = WellbeingRepository()
