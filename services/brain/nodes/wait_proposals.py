"""Poll Mongo until the top-priority tasks have a proposal — or timeout.

We don't block the whole run on every task — only the top 5 by
priority. Lower-priority items get whatever proposal status they
happen to have when the timeout hits; the digest still renders, and
those proposals trickle in over the next few minutes (the
notification system pings the user when each one is ready).

Poll cadence: every 3 s, max 90 s. Each pass reads only the watched
ids. Costs are negligible at brain-run scale.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List

import structlog

from ..state import BrainState
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


_PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}
_POLL_INTERVAL_SEC = 3.0
_POLL_DEADLINE_SEC = 90.0
_WATCH_TOP_N = 5


@traced_node("wait_proposals")
async def wait_proposals(state: BrainState) -> Dict[str, Any]:
    watched_ids = _pick_top_n(state)
    if not watched_ids:
        return {
            "__payload_summary": {"watched": 0, "ready": 0, "timed_out": 0},
            "__eval_score": 1.0,
        }

    try:
        from bson import ObjectId
        from backend.db.mongodb.mongodb import MongoDB
    except Exception as exc:  # noqa: BLE001
        logger.warning("wait_proposals.imports_failed", error=str(exc))
        return {
            "__payload_summary": {"watched": len(watched_ids), "ready": 0,
                                  "error": "imports_failed"},
            "__eval_score": 0.0,
            "__status": "fallback",
        }

    db = await MongoDB.get_database()
    oid_list = []
    for tid in watched_ids:
        try:
            oid_list.append(ObjectId(tid))
        except Exception:
            oid_list.append(tid)

    deadline = time.time() + _POLL_DEADLINE_SEC
    status_by_task: Dict[str, str] = dict(state.proposal_status_by_task)

    while time.time() < deadline:
        cursor = db.tasks.find(
            {"_id": {"$in": oid_list}},
            projection={"_id": 1, "agent_proposal": 1},
        )
        ready_count = 0
        async for t in cursor:
            tid = str(t.get("_id"))
            proposal = t.get("agent_proposal") or {}
            status = proposal.get("status") or status_by_task.get(tid, "queued")
            status_by_task[tid] = status
            if status not in ("queued", "running"):
                ready_count += 1
        if ready_count >= len(watched_ids):
            break
        await asyncio.sleep(_POLL_INTERVAL_SEC)

    ready = sum(
        1 for tid in watched_ids
        if status_by_task.get(tid) not in (None, "queued", "running")
    )
    timed_out = len(watched_ids) - ready

    return {
        "proposal_status_by_task": status_by_task,
        "__payload_summary": {
            "watched": len(watched_ids),
            "ready": ready,
            "timed_out": timed_out,
        },
        "__eval_score": ready / len(watched_ids) if watched_ids else 1.0,
    }


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _pick_top_n(state: BrainState) -> List[str]:
    """Pick the top N tasks to wait on by priority. We use the
    RankedAction list (still in state) to pick which created tasks
    matter most — the order matches the create_tasks node's iteration,
    so index N in ranked_actions ≈ index N in created_task_ids."""
    if not state.created_task_ids:
        return []
    if not state.ranked_actions:
        return list(state.created_task_ids[:_WATCH_TOP_N])

    # Pair (ranked_action, task_id) and sort by action priority.
    pairs = list(
        zip(state.ranked_actions[: len(state.created_task_ids)],
            state.created_task_ids)
    )
    pairs.sort(key=lambda p: _PRIORITY_RANK.get(p[0].priority, 3))
    return [tid for _, tid in pairs[:_WATCH_TOP_N]]
