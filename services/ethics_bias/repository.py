"""Per-organization Ethics & Bias analysis history.

We use MongoDB because every run produces nested, JSON-shaped output
(ethics issues, bias issues, guideline checks, citations) that maps
naturally to a single document — there's no benefit to splitting
across relational tables.  Every row is multi-tenant on
`organization_id` and soft-deleted.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING, DESCENDING

from ...db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)


COLLECTION = "ethics_bias_analyses"

VALID_ACTIONS = (
    "analyze",
    "check_guidelines",
    "generate_suggestions",
    "get_citations",
)
VALID_STATUSES = ("ready", "error", "running")
_TIME_RANGE_TO_DAYS = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}


_indexes_created = False


async def _get_collection():
    """Lazy collection accessor; creates indexes on first use."""
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
                ("action", ASCENDING),
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
            logger.warning("ethics_bias_index_create_failed", error=str(e))
    return coll


def _iso_utc(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def _serialize(row: Dict[str, Any]) -> Dict[str, Any]:
    if not row:
        return {}
    return {
        "id": str(row.get("_id")),
        "organization_id": row.get("organization_id"),
        "user_id": row.get("user_id"),
        "action": row.get("action"),
        "status": row.get("status"),
        "title": row.get("title"),
        "content_preview": row.get("content_preview"),
        "content_type": row.get("content_type"),
        "model_provider": row.get("model_provider"),
        "model_name": row.get("model_name"),
        "parameters": row.get("parameters") or {},
        "result": row.get("result") or {},
        "metadata": row.get("metadata") or {},
        "ethics_score": row.get("ethics_score"),
        "issue_count": row.get("issue_count") or 0,
        "processing_time_ms": row.get("processing_time_ms"),
        "error_message": row.get("error_message"),
        "created_at": _iso_utc(row.get("created_at")),
    }


# ── Create / finalize ───────────────────────────────────────────────


async def create_analysis(
    *,
    organization_id: str,
    user_id: str,
    action: str,
    title: Optional[str] = None,
    content_preview: Optional[str] = None,
    content_type: Optional[str] = None,
    model_provider: Optional[str] = None,
    model_name: Optional[str] = None,
    parameters: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Insert a `running` row.  Finalize updates the same id."""
    if action not in VALID_ACTIONS:
        raise ValueError(f"Unsupported ethics-bias action: {action}")
    if not organization_id:
        raise ValueError("organization_id is required")

    doc = {
        "organization_id": organization_id,
        "user_id": user_id,
        "action": action,
        "status": "running",
        "title": title,
        "content_preview": content_preview,
        "content_type": content_type,
        "model_provider": model_provider,
        "model_name": model_name,
        "parameters": parameters or {},
        "metadata": metadata or {},
        "result": {},
        "ethics_score": None,
        "issue_count": 0,
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
    ethics_score: Optional[int] = None,
    issue_count: Optional[int] = None,
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
    if ethics_score is not None:
        update["ethics_score"] = int(max(0, min(100, ethics_score)))
    if issue_count is not None:
        update["issue_count"] = int(issue_count)
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
    action: Optional[str] = None,
    time_range: Optional[str] = None,
    limit: int = 30,
    offset: int = 0,
) -> Dict[str, Any]:
    """Recent analyses for the history sidebar.  Omits the full result
    body to keep payloads small; clients re-fetch by id."""
    limit = max(1, min(int(limit or 30), 100))
    offset = max(0, int(offset or 0))

    query: Dict[str, Any] = {
        "organization_id": organization_id,
        "deleted_at": None,
    }
    if action and action in VALID_ACTIONS:
        query["action"] = action
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
    return {
        "analyses": rows,
        "total": int(total),
        "limit": limit,
        "offset": offset,
    }


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
    aggregations — no placeholders."""
    days = _TIME_RANGE_TO_DAYS.get(time_range, 7)
    cutoff = datetime.utcnow() - timedelta(days=days)
    base_match = {
        "organization_id": organization_id,
        "deleted_at": None,
        "created_at": {"$gte": cutoff},
    }

    coll = await _get_collection()

    total = await coll.count_documents(base_match)

    # Per-action counts.
    action_cursor = coll.aggregate([
        {"$match": base_match},
        {"$group": {"_id": "$action", "count": {"$sum": 1}}},
    ])
    by_action = {a: 0 for a in VALID_ACTIONS}
    async for row in action_cursor:
        if row.get("_id"):
            by_action[row["_id"]] = int(row["count"])

    # Average ethics score on successful analyses.
    avg_score_cursor = coll.aggregate([
        {"$match": {**base_match, "status": "ready", "ethics_score": {"$ne": None}}},
        {"$group": {"_id": None, "avg": {"$avg": "$ethics_score"}}},
    ])
    avg_ethics_score = 0
    async for row in avg_score_cursor:
        avg_ethics_score = int(round(row.get("avg") or 0))

    # Total issues found across analyses.
    issues_cursor = coll.aggregate([
        {"$match": {**base_match, "status": "ready"}},
        {"$group": {"_id": None, "total": {"$sum": "$issue_count"}}},
    ])
    total_issues = 0
    async for row in issues_cursor:
        total_issues = int(row.get("total") or 0)

    success = await coll.count_documents({**base_match, "status": "ready"})
    errors = await coll.count_documents({**base_match, "status": "error"})

    return {
        "time_range": time_range,
        "total_analyses": int(total),
        "by_action": by_action,
        "average_ethics_score": avg_ethics_score,
        "total_issues": total_issues,
        "success_count": int(success),
        "error_count": int(errors),
        "success_rate": (success / total) if total else 0.0,
    }
