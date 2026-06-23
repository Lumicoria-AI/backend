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

from ..evals import judge_proposal
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

    # Map task_id → action (so the judge knows what context to expect).
    action_by_task: Dict[str, Any] = {}
    if state.ranked_actions:
        pairs = list(
            zip(state.ranked_actions[: len(state.created_task_ids)],
                state.created_task_ids)
        )
        for action, tid in pairs:
            action_by_task[tid] = action

    deadline = time.time() + _POLL_DEADLINE_SEC
    status_by_task: Dict[str, str] = dict(state.proposal_status_by_task)
    proposal_content_by_task: Dict[str, str] = {}

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
                # Cache content so we can judge it after the loop.
                content = proposal.get("content") or proposal.get("draft") or ""
                if content:
                    proposal_content_by_task[tid] = content
        if ready_count >= len(watched_ids):
            break
        await asyncio.sleep(_POLL_INTERVAL_SEC)

    # ── LLM-as-judge each ready proposal — drop trashy drafts ───────
    judge_scores: Dict[str, float] = {}
    judge_reasons: Dict[str, str] = {}
    bad_proposal_ids: List[str] = []

    for tid, content in proposal_content_by_task.items():
        action = action_by_task.get(tid)
        passed, score, reason = judge_proposal(
            content,
            action_title=getattr(action, "title", "") if action else "",
            action_description=getattr(action, "description", "") if action else "",
        )
        judge_scores[tid] = score
        judge_reasons[tid] = reason
        if not passed:
            bad_proposal_ids.append(tid)
            # Mark as 'rejected_by_judge' so compose hides the preview.
            status_by_task[tid] = "rejected_by_judge"

    ready = sum(
        1 for tid in watched_ids
        if status_by_task.get(tid) not in (None, "queued", "running")
    )
    timed_out = len(watched_ids) - ready
    n_judged = len(judge_scores)
    n_judged_bad = len(bad_proposal_ids)
    mean_judge = (
        sum(judge_scores.values()) / n_judged if n_judged else 1.0
    )
    # Per-task readiness × per-proposal judge mean = combined node score.
    combined_score = (
        (ready / len(watched_ids)) * mean_judge if watched_ids else 1.0
    )

    if bad_proposal_ids:
        logger.info(
            "wait_proposals.judge_rejected",
            run_id=state.run_id,
            rejected=n_judged_bad,
            judged=n_judged,
            samples={tid[:8]: judge_reasons[tid][:80] for tid in bad_proposal_ids[:5]},
        )

    return {
        "proposal_status_by_task": status_by_task,
        "meta": {
            **(state.meta or {}),
            "judge_bad_proposal_ids": bad_proposal_ids,
            "judge_proposal_scores": {tid: round(s, 3) for tid, s in judge_scores.items()},
        },
        "__payload_summary": {
            "watched": len(watched_ids),
            "ready": ready,
            "timed_out": timed_out,
            "judged": n_judged,
            "judged_bad": n_judged_bad,
            "judge_mean": round(mean_judge, 3),
        },
        "__eval_score": round(combined_score, 3),
        **({"__status": "fallback"} if bad_proposal_ids and n_judged_bad >= max(1, n_judged // 2) else {}),
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
