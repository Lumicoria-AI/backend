from __future__ import annotations

from typing import Any, Dict, List, Optional
from datetime import datetime
import json
from bson import ObjectId

from .cassandra import CassandraClient
from ...core.config import settings


class CassandraWellbeingRepository:
    async def create_metric(
        self,
        organization_id: str,
        user_id: str,
        metric_type: str,
        value: float,
        source: str,
        metadata: Optional[Dict[str, Any]] = None,
        timestamp: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        ts = timestamp or datetime.utcnow()
        metadata_json = json.dumps(metadata or {})
        query = (
            "INSERT INTO wellbeing_metrics (organization_id, user_id, metric_type, timestamp, value, source, metadata) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s)"
        )
        await CassandraClient.execute(
            query,
            {
                "organization_id": organization_id,
                "user_id": user_id,
                "metric_type": metric_type,
                "timestamp": ts,
                "value": float(value),
                "source": source,
                "metadata": metadata_json,
            },
        )
        if settings.db.CASSANDRA_DUAL_WRITE:
            try:
                from backend.db.mongodb.mongodb import MongoDB
                collection = await MongoDB.get_collection("wellbeing_metrics")
                await collection.insert_one({
                    "user_id": ObjectId(user_id) if ObjectId.is_valid(user_id) else user_id,
                    "organization_id": ObjectId(organization_id) if ObjectId.is_valid(organization_id) else organization_id,
                    "metric_type": metric_type,
                    "value": float(value),
                    "metadata": metadata or {},
                    "source": source,
                    "timestamp": ts,
                    "created_at": datetime.utcnow()
                })
            except Exception:
                pass
        return {
            "id": f"{str(user_id)}:{ts.isoformat()}",
            "organization_id": str(organization_id),
            "user_id": str(user_id),
            "metric_type": metric_type,
            "value": value,
            "source": source,
            "metadata": metadata or {},
            "timestamp": ts,
        }

    async def get_user_metrics(
        self,
        organization_id: str,
        user_id: str,
        metric_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        if metric_type:
            query = (
                "SELECT * FROM wellbeing_metrics "
                "WHERE organization_id=%s AND user_id=%s AND metric_type=%s"
            )
            params = {
                "organization_id": organization_id,
                "user_id": user_id,
                "metric_type": metric_type,
            }
        else:
            # Not ideal in Cassandra; allow filtering for flexibility
            query = (
                "SELECT * FROM wellbeing_metrics "
                "WHERE organization_id=%s AND user_id=%s ALLOW FILTERING"
            )
            params = {
                "organization_id": organization_id,
                "user_id": user_id,
            }

        rows = await CassandraClient.execute(query, params)
        results = []
        for row in rows:
            ts = row.get("timestamp")
            if start_date and ts and ts < start_date:
                continue
            if end_date and ts and ts > end_date:
                continue
            metadata = {}
            if row.get("metadata"):
                try:
                    metadata = json.loads(row["metadata"])
                except Exception:
                    metadata = {}
            results.append({
                "organization_id": row.get("organization_id"),
                "user_id": row.get("user_id"),
                "metric_type": row.get("metric_type"),
                "value": row.get("value"),
                "source": row.get("source"),
                "metadata": metadata,
                "timestamp": row.get("timestamp"),
            })
        return results[:limit]

    async def get_metrics_summary(
        self,
        organization_id: str,
        user_id: str,
        metric_type: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 500,
    ) -> Dict[str, Any]:
        metrics = await self.get_user_metrics(
            organization_id=organization_id,
            user_id=user_id,
            metric_type=metric_type,
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )
        by_type: Dict[str, Dict[str, Any]] = {}
        for m in metrics:
            mtype = m.get("metric_type") or "unknown"
            entry = by_type.setdefault(mtype, {
                "count": 0,
                "sum": 0.0,
                "min": None,
                "max": None,
                "latest": None
            })
            value = float(m.get("value") or 0)
            entry["count"] += 1
            entry["sum"] += value
            entry["min"] = value if entry["min"] is None else min(entry["min"], value)
            entry["max"] = value if entry["max"] is None else max(entry["max"], value)
            if entry["latest"] is None or (m.get("timestamp") and m.get("timestamp") > entry["latest"]["timestamp"]):
                entry["latest"] = {"value": value, "timestamp": m.get("timestamp")}

        for entry in by_type.values():
            entry["avg"] = (entry["sum"] / entry["count"]) if entry["count"] else 0.0
            entry.pop("sum", None)

        return {
            "total_points": len(metrics),
            "by_metric_type": by_type,
            "time_range": {"start": start_date, "end": end_date},
        }


cassandra_wellbeing_repository = CassandraWellbeingRepository()
