"""Pre-send gate for the composed digest.

After ``compose`` builds the render dict, but before ``send`` hands it
to the email transport, we run this single eval. If it fails, the
digest is suppressed — the run still finishes, the run row records
``digest_sent=False`` with reason ``digest_quality_failed``, and the
in-app notification (which is lower-stakes than an email) still goes
out so the user knows the brain ran.

What gets checked:

  * **Safety**: scan every visible string (titles, descriptions, summary
    line, proposal previews) for SSN / CC / leaked-credentials regex.
    Any hit → fail.
  * **Refusal**: the LLM said "I'm sorry, I can't…" anywhere → fail.
  * **Confidence floor**: if every top action has confidence below
    DEGRADED_FLOOR → fail (the run is too uncertain to ask for action).
  * **Empty digest**: no actions, no events, no calendar, no recap →
    fail unless explicitly the "all clear" branch.
  * **Length sanity**: subject and summary_line not empty + not
    suspicious patterns (placeholder strings, raw JSON dumps).

Returns ``(passed: bool, score: float, reason: str)``. The caller
sets ``state.meta["digest_quality"] = {...}`` so the send node + audit
node can see what happened.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

import structlog

from ..metrics import record_judge
from .llm_judge import (
    _FENCED_JSON_RE,
    _PLACEHOLDER_RE,
    _REFUSAL_RE,
    _has_safety_violation,
)

logger = structlog.get_logger(__name__)


# Floors aligned with the action-level judge.
_CONFIDENCE_FLOOR = 0.4
_MIN_SUMMARY_CHARS = 8


def evaluate_digest(render: Dict[str, Any]) -> Tuple[bool, float, str]:
    """Return (passed, score, reason).

    ``passed`` False → the send node should NOT send the email. The
    run still records the audit row.
    """
    if not render:
        record_judge("digest", 0.0)
        return False, 0.0, "no_render"

    bits: List[str] = []
    score = 1.0

    # ── Collect every visible string for safety + refusal scanning ──
    visible_text = _collect_visible_text(render)

    if _has_safety_violation(visible_text):
        record_judge("digest", 0.0)
        return False, 0.0, "safety_violation"
    if _REFUSAL_RE.search(visible_text):
        record_judge("digest", 0.0)
        return False, 0.0, "llm_refusal"
    if _FENCED_JSON_RE.search(visible_text):
        record_judge("digest", 0.0)
        return False, 0.0, "fenced_json_in_user_text"
    if _PLACEHOLDER_RE.search(visible_text):
        score -= 0.4
        bits.append("placeholders_visible")

    # ── Subject + summary line non-empty ────────────────────────────
    subject = (render.get("subject") or "").strip()
    summary = (render.get("summary_line") or "").strip()
    if len(subject) < _MIN_SUMMARY_CHARS:
        score -= 0.3
        bits.append("subject_too_short")
    if len(summary) < _MIN_SUMMARY_CHARS:
        score -= 0.2
        bits.append("summary_too_short")

    # ── Action sanity ───────────────────────────────────────────────
    top_actions = render.get("top_actions") or []
    secondary = render.get("secondary_actions") or []
    events = render.get("calendar_today") or []
    completed = render.get("completed_today") or []
    slipped = render.get("slipped_tasks") or []
    open_tasks = render.get("open_tasks") or []

    n_actions = len(top_actions) + len(secondary)
    has_anything = bool(n_actions or events or completed or slipped or open_tasks)

    if not has_anything:
        # Hard fail unless the summary explicitly signals "all clear".
        if "all clear" not in summary.lower() and "nothing pressing" not in summary.lower():
            record_judge("digest", 0.0)
            return False, 0.0, "empty_digest"
        # Still send the "all clear" digest, but flag in the audit row.
        score -= 0.2
        bits.append("empty_but_signalled")

    # All-low-confidence guardrail.
    if top_actions:
        lc_count = sum(1 for a in top_actions if a.get("low_confidence"))
        if lc_count == len(top_actions):
            score -= 0.4
            bits.append("all_top_low_confidence")
        elif lc_count >= max(1, len(top_actions) // 2):
            score -= 0.2
            bits.append("many_low_confidence")

    # Every top action needs a usable title.
    for a in top_actions:
        title = (a.get("title") or "").strip()
        if not title or title.lower() in ("review email", "task", "todo"):
            score -= 0.15
            bits.append("generic_top_title")
            break

    # ── Action URLs sanity: at least one of approve/view ────────────
    if top_actions:
        any_actionable = any(
            (a.get("approve_url") or a.get("view_url")) for a in top_actions
        )
        if not any_actionable:
            score -= 0.2
            bits.append("no_actionable_links")

    score = max(0.0, min(1.0, score))
    passed = score >= _CONFIDENCE_FLOOR

    record_judge("digest", score)
    return passed, round(score, 3), "; ".join(bits) or "ok"


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _collect_visible_text(render: Dict[str, Any]) -> str:
    """Concatenate every string the user will see — so a single regex
    pass catches safety / refusal / format issues anywhere."""
    parts: List[str] = []
    parts.append(str(render.get("subject", "")))
    parts.append(str(render.get("summary_line", "")))
    parts.append(str(render.get("focus_message", "")))

    for key in ("top_actions", "secondary_actions"):
        for a in render.get(key) or []:
            parts.append(str(a.get("title", "")))
            parts.append(str(a.get("description", "")))
            parts.append(str(a.get("proposal_preview", "")))

    for ev in render.get("calendar_today") or []:
        parts.append(str(ev.get("summary", "")))

    for t in (render.get("open_tasks") or []) + (render.get("completed_today") or []) + (render.get("slipped_tasks") or []):
        parts.append(str(t.get("title", "")))

    return "\n".join(p for p in parts if p)
