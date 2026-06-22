"""Pull the user's currently-open Lumicoria tasks.

The brain uses these two ways:
  - Morning prioritise reads them to avoid proposing duplicates
    ("you already have a task for X").
  - Evening compose renders them as the "still open / what's left"
    list so the user can decide what slips to tomorrow.

We sort by priority then due date so the brain agent sees the most
urgent items first when its prompt window is tight.
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog
from bson import ObjectId

from ..state import BrainState, OpenTaskRef
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


_PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@traced_node("fetch_open_tasks")
async def fetch_open_tasks(state: BrainState) -> Dict[str, Any]:
    try:
        from backend.db.mongodb.mongodb import MongoDB
        db = await MongoDB.get_database()
        col = db.tasks
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_open_tasks.mongo_failed", error=str(exc))
        return {
            "open_tasks": [],
            "__payload_summary": {"count": 0, "error": "mongo_failed"},
            "__eval_score": 0.0,
        }

    try:
        uid_oid: Any = ObjectId(state.user_id)
    except Exception:
        uid_oid = state.user_id

    query = {
        "$or": [
            {"assigned_to": uid_oid},
            {"created_by": uid_oid},
        ],
        "status": {"$nin": ["completed", "cancelled"]},
        "$or_delete": [{"deleted_at": None}, {"deleted_at": {"$exists": False}}],
    }
    # Mongo doesn't allow two $or at top level — wrap second one as $and.
    query = {
        "$and": [
            {"$or": [{"assigned_to": uid_oid}, {"created_by": uid_oid}]},
            {"status": {"$nin": ["completed", "cancelled"]}},
            {"$or": [{"deleted_at": None}, {"deleted_at": {"$exists": False}}]},
        ]
    }

    open_tasks: List[OpenTaskRef] = []
    try:
        cursor = col.find(query).limit(50)
        async for t in cursor:
            open_tasks.append(
                OpenTaskRef(
                    task_id=str(t.get("_id") or t.get("id") or ""),
                    title=str(t.get("title") or "")[:200],
                    priority=t.get("priority"),
                    due_date=t.get("due_date"),
                    status=t.get("status"),
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_open_tasks.query_failed", error=str(exc))
        return {
            "open_tasks": [],
            "__payload_summary": {"count": 0, "error": "query_failed"},
            "__eval_score": 0.0,
            "__status": "fallback",
        }

    # Sort by priority then due date.
    open_tasks.sort(
        key=lambda t: (
            _PRIORITY_RANK.get((t.priority or "low").lower(), 3),
            t.due_date or _far_future(),
        )
    )

    return {
        "open_tasks": open_tasks,
        "__payload_summary": {"count": len(open_tasks)},
        "__eval_score": 1.0,
    }


def _far_future():
    from datetime import datetime
    return datetime(9999, 1, 1)
