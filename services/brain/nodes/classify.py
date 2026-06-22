"""LLM classify each email — action_required / scheduling / fyi / promo / spam.

Pipeline:
  1. Skip when state.emails is empty (the morning brain runs fine on a
     calendar-only day).
  2. Batch up to ``BATCH_SIZE`` emails per LLM call to amortise overhead.
  3. For each batch, ask the LLM for a JSON array of
     ``ClassifiedEmail`` items. Use the schema_eval to validate.
  4. Mean confidence < 0.4 across the run → ``__status="fallback"`` so
     the audit log records a degraded classify pass.
  5. Fallback (when the LLM is unavailable or schema repeatedly fails):
     a tiny regex heuristic that maps obvious patterns (noreply, Re:,
     Promo labels) into a coarse classification at confidence=0.3.

The node returns ``classified: list[ClassifiedEmail dicts]`` and the
prioritise node consumes that.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List

import structlog

from ..evals import check_confidence, check_schema
from ..state import BrainState, ClassifiedEmail, GmailMessageRef
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


BATCH_SIZE = 10
_SYSTEM_PROMPT = (
    "You are an email triage classifier for Lumicoria. For each email "
    "in the input list, output exactly one classification with: "
    "category (action_required | scheduling | informational | promotional | "
    "spam | unknown), urgency (critical | high | medium | low), confidence "
    "(0..1), summary (≤120 chars), suggested_agent (optional key from the "
    "agent directory). Return STRICT JSON only — an array of objects, one "
    "per input email, in the same order. No prose, no markdown fences."
)


@traced_node("classify")
async def classify(state: BrainState) -> Dict[str, Any]:
    if not state.emails:
        return {
            "classified": [],
            "__payload_summary": {"input_emails": 0, "classified": 0},
            "__eval_score": 1.0,
        }

    try:
        from backend.ai_models import get_llm_client
        client = get_llm_client()
    except Exception as exc:  # noqa: BLE001
        logger.warning("classify.llm_unavailable", error=str(exc))
        return _fallback_heuristic(state.emails)

    # Batch the emails.
    batches: List[List[GmailMessageRef]] = [
        state.emails[i : i + BATCH_SIZE]
        for i in range(0, len(state.emails), BATCH_SIZE)
    ]

    all_classified: List[ClassifiedEmail] = []
    failed_batches = 0

    for batch in batches:
        user_prompt = _render_user_prompt(batch)
        try:
            response = await client.generate(
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
            )
            raw = getattr(response, "content", None) or str(response)
        except Exception as exc:  # noqa: BLE001
            logger.warning("classify.batch_llm_failed", error=str(exc))
            failed_batches += 1
            continue

        # Validate the structured output.
        items, schema_eval = check_schema(raw, ClassifiedEmail, pass_floor=0.6)
        if not schema_eval.passed:
            logger.warning(
                "classify.batch_schema_failed",
                reason=schema_eval.reason,
                batch_size=len(batch),
            )
            failed_batches += 1
            # Per-batch fallback: simple heuristic.
            heur = _heuristic_for_batch(batch)
            all_classified.extend(heur)
            continue

        # Re-align: the LLM may have produced fewer items than the batch
        # — backfill message_ids in batch order so the prioritise node's
        # evidence pointers still match.
        for i, item in enumerate(items[: len(batch)]):
            if not item.message_id:
                item.message_id = batch[i].message_id
        all_classified.extend(items[: len(batch)])

    # Aggregate evals.
    confidence_eval = check_confidence(all_classified, floor=0.4)
    eval_score = confidence_eval.score
    fallback_triggered = failed_batches > 0

    return {
        "classified": all_classified,
        "__payload_summary": {
            "input_emails": len(state.emails),
            "classified": len(all_classified),
            "batches": len(batches),
            "failed_batches": failed_batches,
        },
        "__eval_score": eval_score,
        **({"__status": "fallback"} if fallback_triggered else {}),
    }


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _render_user_prompt(batch: List[GmailMessageRef]) -> str:
    lines = ["Emails to classify (output JSON array, one item per email, same order):", ""]
    for i, m in enumerate(batch):
        lines.append(
            f"[{i}] id={m.message_id}\n"
            f"    subject: {m.subject or ''}\n"
            f"    from: {m.from_addr or ''}\n"
            f"    labels: {m.label_ids}\n"
            f"    has_attachments: {m.has_attachments}\n"
            f"    snippet: {(m.snippet or '')[:300]}\n"
        )
    lines.append("")
    lines.append(
        "Output schema (one object per email, in order):\n"
        '  {"message_id": str, "category": str, "urgency": str, '
        '"confidence": float, "summary": str, '
        '"suggested_agent": str or null}'
    )
    return "\n".join(lines)


def _fallback_heuristic(emails: List[GmailMessageRef]) -> Dict[str, Any]:
    """Whole-node fallback when the LLM client itself is unavailable."""
    classified = _heuristic_for_batch(emails)
    return {
        "classified": classified,
        "__payload_summary": {
            "input_emails": len(emails),
            "classified": len(classified),
            "mode": "heuristic_fallback",
        },
        "__eval_score": 0.3,
        "__status": "fallback",
    }


def _heuristic_for_batch(batch: List[GmailMessageRef]) -> List[ClassifiedEmail]:
    """Tiny regex/keyword-based classifier. Used only when the LLM
    classifier itself dropped a batch. Conservative — labels things
    `unknown` rather than guessing categories the user can't trust."""
    out: List[ClassifiedEmail] = []
    for m in batch:
        subj = (m.subject or "").lower()
        from_addr = (m.from_addr or "").lower()
        labels = {str(lbl).upper() for lbl in (m.label_ids or [])}

        category = "unknown"
        urgency = "low"
        suggested_agent = None

        if (
            "CATEGORY_PROMOTIONS" in labels
            or "noreply" in from_addr
            or "no-reply" in from_addr
        ):
            category = "promotional"
        elif "CATEGORY_SOCIAL" in labels:
            category = "informational"
        elif subj.startswith("re:") or subj.startswith("fwd:"):
            category = "action_required"
            urgency = "medium"
            suggested_agent = "meeting" if "meet" in subj or "call" in subj else "document"
        elif any(k in subj for k in ("invoice", "receipt", "payment")):
            category = "informational"
            urgency = "medium"

        out.append(
            ClassifiedEmail(
                message_id=m.message_id,
                category=category,  # type: ignore[arg-type]
                urgency=urgency,  # type: ignore[arg-type]
                confidence=0.3,
                summary=(m.snippet or m.subject or "")[:120],
                suggested_agent=suggested_agent,
            )
        )
    return out
