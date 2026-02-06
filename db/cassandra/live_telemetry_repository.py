from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import json

from .cassandra import CassandraClient
from ...core.config import settings
from bson import ObjectId


class CassandraLiveTelemetryRepository:
    async def log_event(
        self,
        organization_id: str,
        session_id: str,
        user_id: str,
        data_type: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        ts = timestamp or datetime.utcnow()
        query = (
            "INSERT INTO live_telemetry (organization_id, session_id, timestamp, user_id, data_type, payload, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)"
        )
        await CassandraClient.execute(
            query,
            {
                "organization_id": organization_id,
                "session_id": session_id,
                "timestamp": ts,
                "user_id": user_id,
                "data_type": data_type,
                "payload": json.dumps(payload),
                "metadata": json.dumps(metadata or {}),
            },
        )
        if settings.db.CASSANDRA_DUAL_WRITE:
            try:
                from backend.db.mongodb.mongodb import MongoDB
                collection = await MongoDB.get_collection("live_telemetry")
                await collection.insert_one({
                    "organization_id": ObjectId(organization_id) if ObjectId.is_valid(organization_id) else organization_id,
                    "session_id": session_id,
                    "user_id": ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id,
                    "data_type": data_type,
                    "payload": payload,
                    "metadata": metadata or {},
                    "timestamp": ts,
                    "created_at": datetime.utcnow()
                })
            except Exception:
                pass
        return {
            "organization_id": organization_id,
            "session_id": session_id,
            "timestamp": ts,
            "user_id": user_id,
            "data_type": data_type,
            "payload": payload,
            "metadata": metadata or {},
        }

    async def get_session_events(
        self,
        organization_id: str,
        session_id: str,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        query = (
            "SELECT * FROM live_telemetry WHERE organization_id=%s AND session_id=%s"
        )
        rows = await CassandraClient.execute(
            query,
            {"organization_id": organization_id, "session_id": session_id},
        )
        results = []
        for row in rows:
            payload = {}
            if row.get("payload"):
                try:
                    payload = json.loads(row["payload"])
                except Exception:
                    payload = {}
            metadata = {}
            if row.get("metadata"):
                try:
                    metadata = json.loads(row["metadata"])
                except Exception:
                    metadata = {}
            results.append({
                "organization_id": row.get("organization_id"),
                "session_id": row.get("session_id"),
                "timestamp": row.get("timestamp"),
                "user_id": row.get("user_id"),
                "data_type": row.get("data_type"),
                "payload": payload,
                "metadata": metadata,
            })
        return results[:limit]


cassandra_live_telemetry_repository = CassandraLiveTelemetryRepository()
