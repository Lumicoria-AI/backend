"""Per-organization Legal Document analysis history.

We use MongoDB for persistence because legal analyses are document-shaped
(nested clause lists, risk objects with arrays of recommendations, etc.)
and benefit from JSON-native storage rather than rigid SQL columns.  All
rows are multi-tenant on `organization_id` and soft-deleted.

The collection layout is intentionally narrow: every analysis is one
document.  We do not split clauses into their own collection — they live
inside `result` as the agent returned them.  The frontend reopens an
analysis by id and reads the same shape it received when it was first
returned.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from ...db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)


COLLECTION = "legal_analyses"

VALID_MODES = (
    "clause_extraction",
    "risk_analysis",
    "version_comparison",
    "plain_language",
    "compliance_check",
)
VALID_STATUSES = ("ready", "error", "running")
_TIME_RANGE_TO_DAYS = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}


# ── Setup ───────────────────────────────────────────────────────────


_indexes_created = False


async def _get_collection():
    """Lazy collection accessor that creates the indexes on first use."""
    global _indexes_created
    coll = await MongoDB.get_collection(COLLECTION)
    if not _indexes_created:
        try:
            await coll.create_index([
                ("organization_id", ASCENDING),
                ("deleted_at", ASCENDING),
                ("created_at", DESCENDING),
            ])
            await coll.create_index([
                ("organization_id", ASCENDING),
                ("mode", ASCENDING),
                ("created_at", DESCENDING),
            ])
            await coll.create_index([
                ("organization_id", ASCENDING),
                ("user_id", ASCENDING),
                ("created_at", DESCENDING),
            ])
            await coll.create_index("deleted_at")
            _indexes_created = True
        except Exception as e:  # noqa: BLE001
            # Index creation failures are non-fatal but worth logging.
            logger.warning("legal_index_create_failed", error=str(e))
    return coll


def _iso_utc(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _serialize(row: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a raw Mongo document to the shape the API returns."""
    if not row:
        return {}
    return {
        "id": str(row.get("_id")),
        "organization_id": row.get("organization_id"),
        "user_id": row.get("user_id"),
        "mode": row.get("mode"),
        "status": row.get("status"),
        "title": row.get("title"),
        "content_preview": row.get("content_preview"),
        "source_kind": row.get("source_kind"),
        "source_ref": row.get("source_ref"),
        "model_provider": row.get("model_provider"),
        "model_name": row.get("model_name"),
        "parameters": row.get("parameters") or {},
        "result": row.get("result") or {},
        "metadata": row.get("metadata") or {},
        "processing_time_ms": row.get("processing_time_ms"),
        "error_message": row.get("error_message"),
        "created_at": _iso_utc(row.get("created_at")),
    }


# ── Create / finalize ───────────────────────────────────────────────


async def create_analysis(
    *,
    organization_id: str,
    user_id: str,
    mode: str,
    title: Optional[str] = None,
    content_preview: Optional[str] = None,
    source_kind: Optional[str] = None,
    source_ref: Optional[str] = None,
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Insert a `running` row; finalize_analysis updates the same id."""
    if mode not in VALID_MODES:
        raise ValueError(f"Unsupported legal analysis mode: {mode}")
    if not organization_id:
        raise ValueError("organization_id is required")

    doc = {
        "organization_id": organization_id,
        "user_id": user_id,
        "mode": mode,
        "status": "running",
        "title": title,
        "content_preview": content_preview,
        "source_kind": source_kind,
        "source_ref": source_ref,
        "model_provider": model_provider,
        "model_name": model_name,
        "parameters": parameters or {},
        "metadata": metadata or {},
        "result": {},
        "processing_time_ms": None,
        "error_message": None,
        "created_at": datetime.utcnow(),
        "deleted_at": None,
    }
    coll = await _get_collection()
    result = await coll.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _serialize(doc)


async def finalize_analysis(
    organization_id: str,
    analysis_id: str,
    *,
    status: str = "ready",
    result_payload: Optional[Dict[str, Any]] = None,
    processing_time_ms: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    """Stamp the run with its outcome.  Scoped to the org to prevent
    cross-tenant updates even if an id is leaked."""
    if status not in VALID_STATUSES:
        status = "ready"
    update: Dict[str, Any] = {"status": status}
    if result_payload is not None:
        update["result"] = result_payload
    if processing_time_ms is not None:
        update["processing_time_ms"] = int(processing_time_ms)
    if error_message is not None:
        update["error_message"] = error_message[:2000]
    update["updated_at"] = datetime.utcnow()

    try:
        oid = ObjectId(analysis_id)
    except Exception:
        return
    coll = await _get_collection()
    await coll.update_one(
        {"_id": oid, "organization_id": organization_id},
        {"$set": update},
    )


# ── Read ────────────────────────────────────────────────────────────


async def get_analysis(
    organization_id: str, analysis_id: str
) -> Optional[Dict[str, Any]]:
    try:
        oid = ObjectId(analysis_id)
    except Exception:
        return None
    coll = await _get_collection()
    row = await coll.find_one({
        "_id": oid,
        "organization_id": organization_id,
        "deleted_at": None,
    })
    return _serialize(row) if row else None


async def list_analyses(
    organization_id: str,
    *,
    mode: Optional[str] = None,
    time_range: Optional[str] = None,
    limit: int = 30,
    offset: int = 0,
) -> Dict[str, Any]:
    """Recent analyses for the history sidebar.  Results omit the full
    `result` body to keep the payload small; clients re-fetch by id."""
    limit = max(1, min(int(limit or 30), 100))
    offset = max(0, int(offset or 0))

    query: Dict[str, Any] = {
        "organization_id": organization_id,
        "deleted_at": None,
    }
    if mode and mode in VALID_MODES:
        query["mode"] = mode
    if time_range and time_range in _TIME_RANGE_TO_DAYS:
        cutoff = datetime.utcnow() - timedelta(days=_TIME_RANGE_TO_DAYS[time_range])
        query["created_at"] = {"$gte": cutoff}

    coll = await _get_collection()
    total = await coll.count_documents(query)
    cursor = (
        coll.find(query, projection={"result": 0})
        .sort("created_at", DESCENDING)
        .skip(offset)
        .limit(limit)
    )
    rows = [_serialize(r) async for r in cursor]
    return {"analyses": rows, "total": int(total), "limit": limit, "offset": offset}


# ── Delete ──────────────────────────────────────────────────────────


async def soft_delete_analysis(organization_id: str, analysis_id: str) -> bool:
    try:
        oid = ObjectId(analysis_id)
    except Exception:
        return False
    coll = await _get_collection()
    result = await coll.update_one(
        {
            "_id": oid,
            "organization_id": organization_id,
            "deleted_at": None,
        },
        {"$set": {"deleted_at": datetime.utcnow()}},
    )
    return (result.modified_count or 0) > 0


# ── Analytics ───────────────────────────────────────────────────────


async def get_analytics(
    organization_id: str, time_range: str = "7d"
) -> Dict[str, Any]:
    """Counts that drive the analytics card on the agent page.  Real
    data — no hardcoded placeholders."""
    days = _TIME_RANGE_TO_DAYS.get(time_range, 7)
    cutoff = datetime.utcnow() - timedelta(days=days)
    base_match = {
        "organization_id": organization_id,
        "deleted_at": None,
        "created_at": {"$gte": cutoff},
    }

    coll = await _get_collection()

    total = await coll.count_documents(base_match)

    # Per-mode counts.
    mode_cursor = coll.aggregate([
        {"$match": base_match},
        {"$group": {"_id": "$mode", "count": {"$sum": 1}}},
    ])
    by_mode = {m: 0 for m in VALID_MODES}
    async for row in mode_cursor:
        if row.get("_id"):
            by_mode[row["_id"]] = int(row["count"])

    # Per-model counts.
    model_cursor = coll.aggregate([
        {"$match": base_match},
        {"$group": {
            "_id": {"provider": "$model_provider", "model": "$model_name"},
            "count": {"$sum": 1},
        }},
    ])
    by_model: List[Dict[str, Any]] = []
    async for row in model_cursor:
        key = row.get("_id") or {}
        by_model.append({
            "provider": key.get("provider") or "unknown",
            "model": key.get("model") or "unknown",
            "count": int(row.get("count") or 0),
        })

    # Average processing time over successful runs.
    avg_cursor = coll.aggregate([
        {"$match": {**base_match, "status": "ready"}},
        {"$group": {"_id": None, "avg": {"$avg": "$processing_time_ms"}}},
    ])
    avg_ms = 0
    async for row in avg_cursor:
        avg_ms = int(row.get("avg") or 0)

    # Success / error split.
    success = await coll.count_documents({**base_match, "status": "ready"})
    errors = await coll.count_documents({**base_match, "status": "error"})

    return {
        "time_range": time_range,
        "total_analyses": int(total),
        "by_mode": by_mode,
        "by_model": by_model,
        "average_processing_time_ms": avg_ms,
        "success_count": int(success),
        "error_count": int(errors),
        "success_rate": (success / total) if total else 0.0,
    }
