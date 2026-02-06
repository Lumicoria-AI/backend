from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from .cassandra import CassandraClient
from ...core.config import settings
from bson import ObjectId


class CassandraActivityRepository:
    async def create_log_entry(
        self,
        organization_id: str,
        user_id: str,
        activity_type: str,
        details: Dict[str, Any],
        related_resource_type: Optional[str] = None,
        related_resource_id: Optional[str] = None,
        severity: str = "info",
        agent_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        ts = timestamp or datetime.utcnow()
        query = (
            "INSERT INTO activity_logs (organization_id, user_id, timestamp, activity_type, details, "
            "related_resource_type, related_resource_id, severity, agent_id, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
        )
        await CassandraClient.execute(
            query,
            {
                "organization_id": organization_id,
                "user_id": user_id,
                "timestamp": ts,
                "activity_type": activity_type,
                "details": json.dumps(details),
                "related_resource_type": related_resource_type,
                "related_resource_id": related_resource_id,
                "severity": severity,
                "agent_id": agent_id,
                "metadata": json.dumps(metadata or {}),
            },
        )
        if settings.db.CASSANDRA_DUAL_WRITE:
            try:
                from backend.db.mongodb.mongodb import MongoDB
                collection = await MongoDB.get_collection("activity_logs")
                await collection.insert_one({
                    "organization_id": ObjectId(organization_id) if ObjectId.is_valid(organization_id) else organization_id,
                    "user_id": ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id,
                    "activity_type": activity_type,
                    "details": details,
                    "related_resource_type": related_resource_type,
                    "related_resource_id": ObjectId(related_resource_id) if related_resource_id and ObjectId.is_valid(related_resource_id) else related_resource_id,
                    "severity": severity,
                    "agent_id": ObjectId(agent_id) if agent_id and ObjectId.is_valid(agent_id) else agent_id,
                    "metadata": metadata or {},
                    "timestamp": ts,
                    "created_at": datetime.utcnow()
                })
            except Exception:
                pass
        return {
            "organization_id": organization_id,
            "user_id": user_id,
            "timestamp": ts,
            "activity_type": activity_type,
            "details": details,
            "related_resource_type": related_resource_type,
            "related_resource_id": related_resource_id,
            "severity": severity,
            "agent_id": agent_id,
            "metadata": metadata or {},
        }

    async def get_recent_activity(
        self,
        organization_id: str,
        user_id: Optional[str] = None,
        activity_type: Optional[str] = None,
        limit: int = 10,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        if user_id:
            query = (
                "SELECT * FROM activity_logs WHERE organization_id=%s AND user_id=%s"
            )
            params = {"organization_id": organization_id, "user_id": user_id}
        else:
            query = (
                "SELECT * FROM activity_logs WHERE organization_id=%s ALLOW FILTERING"
            )
            params = {"organization_id": organization_id}

        rows = await CassandraClient.execute(query, params)
        filtered = []
        for row in rows:
            if activity_type and row.get("activity_type") != activity_type:
                continue
            details = {}
            if row.get("details"):
                try:
                    details = json.loads(row["details"])
                except Exception:
                    details = {}
            metadata = {}
            if row.get("metadata"):
                try:
                    metadata = json.loads(row["metadata"])
                except Exception:
                    metadata = {}
            filtered.append({
                "organization_id": row.get("organization_id"),
                "user_id": row.get("user_id"),
                "timestamp": row.get("timestamp"),
                "activity_type": row.get("activity_type"),
                "details": details,
                "related_resource_type": row.get("related_resource_type"),
                "related_resource_id": row.get("related_resource_id"),
                "severity": row.get("severity"),
                "agent_id": row.get("agent_id"),
                "metadata": metadata,
            })
        # Cassandra already returns clustered DESC; apply skip/limit
        return filtered[skip:skip + limit]

    async def get_activity_summary(
        self,
        organization_id: str,
        user_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        activities = await self.get_recent_activity(
            organization_id=organization_id,
            user_id=user_id,
            limit=limit,
            skip=0,
        )
        by_type: Dict[str, int] = {}
        by_severity: Dict[str, int] = {}
        for entry in activities:
            ts = entry.get("timestamp")
            if start_date and ts and ts < start_date:
                continue
            if end_date and ts and ts > end_date:
                continue
            atype = entry.get("activity_type") or "unknown"
            by_type[atype] = by_type.get(atype, 0) + 1
            severity = entry.get("severity") or "info"
            by_severity[severity] = by_severity.get(severity, 0) + 1

        return {
            "total_events": sum(by_type.values()),
            "by_type": by_type,
            "by_severity": by_severity,
            "time_range": {"start": start_date, "end": end_date},
        }


cassandra_activity_repository = CassandraActivityRepository()
