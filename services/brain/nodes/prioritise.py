"""Brain Agent ranks the classified items into a prioritised task list.

Pipeline:
  1. Skip when there's nothing to rank (no classified emails + no
     huddle recents + no calendar deltas).
  2. Instantiate the BrainAgent via the same `instantiate_agent` path
     the task executor uses. The agent shares the LLM client pool.
  3. Pass the full input bundle — emails (already classified),
     calendar, open tasks, huddle recents — plus user_id + org_id +
     mode. The agent itself pulls RAG context per item internally.
  4. The agent returns ``ranked_actions`` (already validated via
     schema_eval) + a self-reported overall confidence.
  5. If confidence < 0.4 → ``__status="fallback"`` so the audit row
     records degradation; the create_tasks node still proceeds but
     the digest gets a "low confidence — review carefully" header.
  6. Fallback (BrainAgent unavailable / hard error): a rule-based
     ranker — top urgent classified emails become tasks with the
     suggested_agent the classifier gave us. No LLM required.
"""

from __future__ import annotations

from typing import Any, Dict, List

import structlog

from ..state import BrainState, ClassifiedEmail, RankedAction
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


@traced_node("prioritise")
async def prioritise(state: BrainState) -> Dict[str, Any]:
    # Skip when there's nothing to rank.
    if (
        not state.classified
        and not state.huddle_recents
        and not state.events
        and not state.open_tasks
    ):
        return {
            "ranked_actions": [],
            "__payload_summary": {
                "classified": 0, "open_tasks": 0, "ranked": 0,
                "mode": state.mode,
            },
            "__eval_score": 1.0,
        }

    # Try the BrainAgent first.
    try:
        from backend.services.task_executor import instantiate_agent
        agent = instantiate_agent("brain")
    except Exception as exc:  # noqa: BLE001
        logger.warning("prioritise.brain_agent_instantiation_failed", error=str(exc))
        agent = None

    if agent is None:
        return _rule_based_fallback(state)

    # Build the agent's input bundle.
    data = {
        "classified_emails": [c.model_dump() for c in state.classified],
        "calendar_events": [e.model_dump() for e in state.events],
        "open_tasks": [t.model_dump() for t in state.open_tasks],
        "huddle_recents": [h.model_dump() for h in state.huddle_recents],
        "user_id": state.user_id,
        "organization_id": state.organization_id,
        "timezone": state.timezone,
        "mode": state.mode,
        "max_actions": 8,
    }

    try:
        result = await agent.process_async(data)
    except Exception as exc:  # noqa: BLE001
        logger.exception(
            "prioritise.brain_agent_failed", run_id=state.run_id,
        )
        return _rule_based_fallback(state, error=str(exc))

    ranked_dicts = result.get("ranked_actions") or []
    overall_confidence = float(result.get("confidence") or 0.0)
    summary_line = result.get("summary_line") or ""
    reasoning = result.get("reasoning") or ""

    # Re-validate to RankedAction objects so the create_tasks node
    # consumes typed values, not raw dicts.
    actions: List[RankedAction] = []
    for d in ranked_dicts:
        try:
            actions.append(RankedAction.model_validate(d))
        except Exception as e:  # noqa: BLE001
            logger.debug("prioritise.action_drop", error=str(e), item=d)

    eval_score = overall_confidence if overall_confidence > 0 else 0.6
    payload_summary = {
        "classified": len(state.classified),
        "open_tasks": len(state.open_tasks),
        "ranked": len(actions),
        "overall_confidence": round(overall_confidence, 3),
        "agent_used": "brain",
        "mode": state.mode,
        "summary_line": summary_line[:200],
        "reasoning": reasoning[:200],
    }

    extras: Dict[str, Any] = {
        "ranked_actions": actions,
        # Stash the agent's narrative for compose to read.
        "meta": {**state.meta, "brain_summary_line": summary_line, "brain_reasoning": reasoning},
        "__payload_summary": payload_summary,
        "__eval_score": eval_score,
    }
    if overall_confidence < 0.4:
        extras["__status"] = "fallback"

    return extras


# ─────────────────────────────────────────────────────────────────────
# Fallback ranker — no LLM. Used when BrainAgent is unavailable.
# ─────────────────────────────────────────────────────────────────────


_URGENCY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def _rule_based_fallback(
    state: BrainState, *, error: str = "",
) -> Dict[str, Any]:
    """Pull the most-urgent classified emails directly into actions.
    No reasoning, no grounding, no confidence — just a safe baseline
    so the digest still goes out when the LLM is down."""
    actions: List[RankedAction] = []

    # Pick the top 8 urgent emails.
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
        "__payload_summary": {
            "classified": len(state.classified),
            "ranked": len(actions),
            "mode": "rule_based_fallback",
            "error": error[:200] if error else None,
        },
        "__eval_score": 0.3,
        "__status": "fallback",
    }
