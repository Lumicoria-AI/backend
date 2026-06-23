"""Brain Agent ranks the classified items into a prioritised task list.

This module exposes three LangGraph nodes that the graph wires up with
conditional edges so each attempt is its own trace row:

  ┌─ prioritise_attempt_1   (first BrainAgent call + judge)
  │         │
  │      passed? ──── yes ──→ create_tasks
  │         │ no
  │         ▼
  ├─ prioritise_attempt_2   (strict retry + judge — keeps better of the two)
  │         │
  │      passed? ──── yes ──→ create_tasks
  │         │ no
  │         ▼
  └─ prioritise_rule_fallback (no LLM — urgency-ordered heuristic)
            │
            ▼
       create_tasks

Each node:
  * is decorated with @traced_node so its row lands in brain_traces
    with its own duration_ms, status, eval_score, and payload_summary;
  * writes ``state.ranked_actions`` so downstream nodes see whatever
    the latest attempt produced;
  * writes ``state.meta["prioritise_attempt_N_passed"]`` so the
    conditional router can choose where to go next.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import structlog

from ..evals import judge_ranked_actions
from ..state import BrainState, ClassifiedEmail, EvalResult, RankedAction
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Conditional routers — pure functions, no side effects
# ─────────────────────────────────────────────────────────────────────


def after_attempt_1(state: BrainState) -> str:
    """Route from attempt 1: 'ok' if judge passed, 'retry' otherwise."""
    return "ok" if (state.meta or {}).get("prioritise_attempt_1_passed") else "retry"


def after_attempt_2(state: BrainState) -> str:
    """Route from attempt 2: 'ok' if judge passed, 'fallback' otherwise."""
    return "ok" if (state.meta or {}).get("prioritise_attempt_2_passed") else "fallback"


# ─────────────────────────────────────────────────────────────────────
# Attempt 1 — first BrainAgent call
# ─────────────────────────────────────────────────────────────────────


@traced_node("prioritise_attempt_1")
async def prioritise_attempt_1(state: BrainState) -> Dict[str, Any]:
    if _nothing_to_rank(state):
        return _empty_result(state)

    agent = _instantiate_agent_safe()
    if agent is None:
        # No agent available — leave attempt_1 flagged as failed so the
        # router skips attempt_2 and falls straight to rule_fallback.
        return {
            "ranked_actions": [],
            "meta": {**state.meta, "prioritise_attempt_1_passed": False, "prioritise_agent_unavailable": True},
            "__payload_summary": {"agent_available": False, "skipped": True},
            "__eval_score": 0.0,
            "__status": "fallback",
        }

    data = _build_input_bundle(state, strict=False)
    return await _run_and_pack(
        attempt_label="prioritise_attempt_1",
        state=state,
        agent=agent,
        data=data,
    )


# ─────────────────────────────────────────────────────────────────────
# Attempt 2 — strict retry. Picks whichever attempt scored higher.
# ─────────────────────────────────────────────────────────────────────


@traced_node("prioritise_attempt_2")
async def prioritise_attempt_2(state: BrainState) -> Dict[str, Any]:
    agent = _instantiate_agent_safe()
    if agent is None:
        return {
            "meta": {**state.meta, "prioritise_attempt_2_passed": False},
            "__payload_summary": {"agent_available": False},
            "__eval_score": 0.0,
            "__status": "fallback",
        }

    prior_reason = (state.meta or {}).get("prioritise_attempt_1_judge_reason", "")
    data = _build_input_bundle(state, strict=True, prior_failure=prior_reason)

    result = await _run_and_pack(
        attempt_label="prioritise_attempt_2",
        state=state,
        agent=agent,
        data=data,
    )

    # If attempt 1 was better than attempt 2, keep attempt 1's actions.
    attempt_1_score = float((state.meta or {}).get("prioritise_attempt_1_judge_score") or 0.0)
    attempt_2_score = float(result.get("meta", {}).get("prioritise_attempt_2_judge_score") or 0.0)

    if attempt_1_score > attempt_2_score and (state.meta or {}).get("prioritise_attempt_1_actions"):
        # Restore attempt 1's actions onto the state.
        kept_first = [
            RankedAction.model_validate(d)
            for d in state.meta["prioritise_attempt_1_actions"]
        ]
        result["ranked_actions"] = kept_first
        result["meta"] = {
            **result["meta"],
            "prioritise_kept_attempt": "first",
            "prioritise_attempt_2_passed": attempt_1_score >= 0.6,
        }
        result["__payload_summary"] = {
            **result.get("__payload_summary", {}),
            "kept_attempt": "first",
            "attempt_1_score": attempt_1_score,
            "attempt_2_score": attempt_2_score,
        }
    else:
        result["meta"] = {**result.get("meta", {}), "prioritise_kept_attempt": "retry"}
        result["__payload_summary"] = {
            **result.get("__payload_summary", {}),
            "kept_attempt": "retry",
            "attempt_1_score": attempt_1_score,
            "attempt_2_score": attempt_2_score,
        }

    return result


# ─────────────────────────────────────────────────────────────────────
# Rule fallback — no LLM. Final safety net.
# ─────────────────────────────────────────────────────────────────────


@traced_node("prioritise_rule_fallback")
async def prioritise_rule_fallback(state: BrainState) -> Dict[str, Any]:
    actions: List[RankedAction] = []

    by_urgency: List[ClassifiedEmail] = sorted(
        state.classified,
        key=lambda c: _URGENCY_RANK.get(c.urgency, 3),
    )[:8]

    for c in by_urgency:
        title = (c.summary or "Review email").strip()[:80]
        actions.append(
            RankedAction(
                title=title,
                description=(c.summary or "")[:400],
                priority=c.urgency,  # type: ignore[arg-type]
                assigned_to_agent=c.suggested_agent,
                confidence=max(0.2, c.confidence),
                evidence_message_ids=[c.message_id] if c.message_id else [],
            )
        )

    return {
        "ranked_actions": actions,
        "meta": {**state.meta, "prioritise_kept_attempt": "rule_fallback"},
        "__payload_summary": {
            "classified": len(state.classified),
            "ranked": len(actions),
            "mode": "rule_based_fallback",
        },
        "__eval_score": 0.3,
        "__status": "fallback",
    }


# ─────────────────────────────────────────────────────────────────────
# Helpers — shared across the three nodes
# ─────────────────────────────────────────────────────────────────────


_URGENCY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _nothing_to_rank(state: BrainState) -> bool:
    return (
        not state.classified
        and not state.huddle_recents
        and not state.events
        and not state.open_tasks
    )


def _empty_result(state: BrainState) -> Dict[str, Any]:
    return {
        "ranked_actions": [],
        "meta": {**state.meta, "prioritise_attempt_1_passed": True, "prioritise_kept_attempt": "empty"},
        "__payload_summary": {
            "classified": 0, "open_tasks": 0, "ranked": 0,
            "mode": state.mode,
        },
        "__eval_score": 1.0,
    }


def _instantiate_agent_safe():
    try:
        from backend.services.task_executor import instantiate_agent
        return instantiate_agent("brain")
    except Exception as exc:  # noqa: BLE001
        logger.warning("prioritise.brain_agent_instantiation_failed", error=str(exc))
        return None


def _build_input_bundle(
    state: BrainState,
    *,
    strict: bool = False,
    prior_failure: str = "",
) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        "classified_emails": [c.model_dump() for c in state.classified],
        "calendar_events": [e.model_dump() for e in state.events],
        "open_tasks": [t.model_dump() for t in state.open_tasks],
        "huddle_recents": [h.model_dump() for h in state.huddle_recents],
        "user_id": state.user_id,
        "organization_id": state.organization_id,
        "timezone": state.timezone,
        "mode": state.mode,
        "max_actions": 5 if strict else 8,
    }
    if strict:
        data["retry_strict"] = True
        data["min_confidence"] = 0.7
        data["require_evidence"] = True
        data["retry_hint"] = (
            "Previous attempt failed quality review: "
            + (prior_failure[:300] if prior_failure else "low quality")
            + ". Be specific. Lead with a verb. Cite the exact message_id, "
              "event_id, or file_id in evidence_*. Drop low-confidence "
              "items entirely — fewer high-quality actions beats many "
              "noisy ones."
        )
    return data


async def _run_and_pack(
    *,
    attempt_label: str,
    state: BrainState,
    agent,
    data: Dict[str, Any],
) -> Dict[str, Any]:
    """Call the agent, judge the output, and pack into a state update
    that includes the per-attempt flag the routers read."""
    avail_msg_ids: Set[str] = {c.message_id for c in state.classified if c.message_id}
    avail_evt_ids: Set[str] = {e.event_id for e in state.events if e.event_id}
    avail_file_ids: Set[str] = {f.file_id for f in state.drive_changes if f.file_id}

    try:
        result = await agent.process_async(data)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "prioritise.agent_call_failed", attempt=attempt_label, run_id=state.run_id,
        )
        flag_key = f"{attempt_label}_passed"
        return {
            "meta": {**state.meta, flag_key: False, f"{attempt_label}_error": str(exc)},
            "__payload_summary": {"error": str(exc)[:200], "attempt": attempt_label},
            "__eval_score": 0.0,
            "__status": "fail",
        }

    ranked_dicts = result.get("ranked_actions") or []
    overall_confidence = float(result.get("confidence") or 0.0)
    summary_line = result.get("summary_line") or ""
    reasoning = result.get("reasoning") or ""

    actions: List[RankedAction] = []
    for d in ranked_dicts:
        try:
            actions.append(RankedAction.model_validate(d))
        except Exception as e:  # noqa: BLE001
            logger.debug("prioritise.action_drop", error=str(e), item=d)

    kept, dropped, judge = judge_ranked_actions(
        actions,
        available_message_ids=avail_msg_ids,
        available_event_ids=avail_evt_ids,
        available_file_ids=avail_file_ids,
    )

    if dropped:
        logger.info(
            "prioritise.judge_dropped",
            run_id=state.run_id,
            attempt=attempt_label,
            kept=len(kept),
            dropped=len(dropped),
            mean=judge.score,
        )

    # Blend the agent's self-reported confidence with the judge's verdict.
    eval_score = (overall_confidence if overall_confidence > 0 else 0.6) * (
        0.4 + 0.6 * judge.score
    )

    passed_flag = f"{attempt_label}_passed"
    score_key = f"{attempt_label}_judge_score"
    reason_key = f"{attempt_label}_judge_reason"
    actions_dump_key = f"{attempt_label}_actions"

    new_meta = {
        **state.meta,
        passed_flag: judge.passed,
        score_key: judge.score,
        reason_key: judge.reason,
        actions_dump_key: [a.model_dump(mode="json") for a in kept],
        "brain_summary_line": summary_line,
        "brain_reasoning": reasoning,
        "judge_dropped_actions": [
            {"title": a.title, "score": s, "reason": r}
            for (a, s, r) in dropped
        ],
    }

    payload_summary = {
        "attempt": attempt_label,
        "classified": len(state.classified),
        "ranked_raw": len(actions),
        "ranked_kept": len(kept),
        "ranked_dropped": len(dropped),
        "judge_mean_score": judge.score,
        "judge_passed": judge.passed,
        "judge_reason": judge.reason[:200],
        "overall_confidence": round(overall_confidence, 3),
    }

    status_override: Optional[str] = None
    if not judge.passed:
        status_override = "fallback"

    update: Dict[str, Any] = {
        "ranked_actions": kept,
        "meta": new_meta,
        "__payload_summary": payload_summary,
        "__eval_score": round(eval_score, 3),
    }
    if status_override:
        update["__status"] = status_override
    return update
