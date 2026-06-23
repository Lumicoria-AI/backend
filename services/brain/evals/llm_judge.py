"""LLM-as-judge for agent outputs.

The brain pipeline produces three things a real human will see:
  1. **RankedAction** items from the Brain Agent (titles + descriptions).
  2. **Agent proposals** drafted by the specialist agents.
  3. **The composed digest** that gets emailed.

Without a judge, every LLM hiccup leaks into the user's inbox: generic
"Review email" titles, hallucinated evidence, leaked credit-card
numbers in subject lines, or refusal apologies as the digest summary.
This module is the last quality gate before any of that reaches a real
person.

Architecture — two layers, both mandatory:

  Layer 1 — heuristics (deterministic, ~1ms total)
    * Safety:     regex-based PII detection (SSN, CC, password=, raw token).
    * Specificity: title length, verb-led, non-generic.
    * Groundedness: evidence_message_ids ⊆ available_message_ids.
    * Format:     fields populated, priorities in vocabulary, no fenced JSON.

  Layer 2 — LLM judge (optional, opt-in via env / cost budget)
    * Single LLM call grading up to N top items on a 0..1 rubric.
    * Returns (item_id, score, reason) per item.
    * Disabled when ``BRAIN_LLM_JUDGE_ENABLED != "1"`` or no client.

Public API:

  judge_ranked_actions(actions, *, available_message_ids, available_event_ids,
                       available_file_ids, drop_below=0.4)
      → (kept_actions, dropped_actions, EvalResult)

  judge_proposal(content, *, action_title, action_description,
                 evidence_snippets)
      → (passed: bool, score: float, reason: str)

Both are best-effort: a judge failure should never break the run.
They return a fail-safe ("pass with warning") and the audit log
captures the degradation.
"""

from __future__ import annotations

import json
import re
from typing import Any, Iterable, List, Optional, Set, Tuple

import structlog

from ..metrics import record_judge
from ..state import EvalResult, RankedAction

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Tuning knobs
# ─────────────────────────────────────────────────────────────────────


# An action with score below this is dropped from the digest entirely.
DEFAULT_DROP_FLOOR = 0.4
# Mean judge score below this puts the whole run into "degraded" status.
DEGRADED_FLOOR = 0.6
# Top-N actions sent to the optional LLM judge layer.
LLM_JUDGE_TOP_N = 8

# Generic-title blacklist — the LLM falls back to these when it's
# uncertain. We never want them in front of a user.
_GENERIC_TITLES = {
    "review email", "check email", "follow up", "respond to email",
    "see attached", "tbd", "todo", "task", "untitled", "action item",
    "review", "respond", "follow-up", "check", "read", "no subject",
    "(no subject)", "review document", "review attachment",
}

# Specificity — a healthy task title starts with a verb.
_TASK_VERBS = {
    "review", "draft", "send", "respond", "reply", "schedule",
    "book", "call", "email", "approve", "reject", "sign", "pay",
    "submit", "file", "prepare", "follow", "confirm", "decline",
    "delegate", "assign", "create", "update", "share", "post",
    "publish", "deliver", "ship", "merge", "deploy", "fix",
    "investigate", "resolve", "close", "open", "read", "skim",
    "summarise", "summarize", "decide", "negotiate", "renew",
    "cancel", "process", "ack", "acknowledge", "write", "build",
    "plan", "ping", "nudge", "remind", "verify", "audit",
    "test", "complete", "finish", "wrap", "handoff", "kickoff",
}

# Safety — patterns that should never appear in titles or descriptions.
_SSN_RE = re.compile(r"\b\d{3}-\d{2}-\d{4}\b")
_CC_RE = re.compile(r"\b(?:\d[ -]?){13,19}\b")
_PASSWORD_LEAK_RE = re.compile(
    r"\b(?:password|passwd|pwd|secret|api[_ -]?key|token)\s*[=:]\s*[A-Za-z0-9!@#$%^&*_\-/+]{4,}",
    re.IGNORECASE,
)
_REFUSAL_RE = re.compile(
    r"\b(i (?:cannot|can'?t|won'?t|am unable)|i'?m (?:sorry|afraid)|"
    r"as an? (?:ai|llm|language model))\b",
    re.IGNORECASE,
)

# Format — common LLM glitches that leak through truncated outputs.
_FENCED_JSON_RE = re.compile(r"```")
_PLACEHOLDER_RE = re.compile(r"\{\{[^}]+\}\}|<[A-Z_]+>")


# ─────────────────────────────────────────────────────────────────────
# Ranked-action judge — the main entry point
# ─────────────────────────────────────────────────────────────────────


def judge_ranked_actions(
    actions: List[RankedAction],
    *,
    available_message_ids: Iterable[str] = (),
    available_event_ids: Iterable[str] = (),
    available_file_ids: Iterable[str] = (),
    drop_below: float = DEFAULT_DROP_FLOOR,
    use_llm: Optional[bool] = None,
) -> Tuple[List[RankedAction], List[Tuple[RankedAction, float, str]], EvalResult]:
    """Score each RankedAction and split into keep / drop.

    Returns:
        (kept_actions, dropped, eval_result)

        ``dropped`` is a list of (action, score, reason) tuples so the
        audit trail can show what got pulled and why.

        ``eval_result.score`` is the mean of all kept-action scores —
        the caller uses it for ``__eval_score`` on the trace row.
        ``eval_result.passed`` is True when no critical safety issue
        fired AND mean score ≥ DEGRADED_FLOOR.
    """
    if not actions:
        return [], [], EvalResult(score=1.0, passed=True, reason="no_actions", checked_fields=[])

    avail_msgs = set(available_message_ids or ())
    avail_evts = set(available_event_ids or ())
    avail_files = set(available_file_ids or ())

    # ── Layer 1: heuristic scoring per action ───────────────────────
    scored: List[Tuple[RankedAction, float, str]] = []
    safety_violations = 0
    for a in actions:
        score, reason, fatal = _score_action_heuristic(
            a, avail_msgs=avail_msgs, avail_evts=avail_evts, avail_files=avail_files,
        )
        if fatal:
            safety_violations += 1
        scored.append((a, score, reason))

    # ── Layer 2: optional LLM judge on top-N ────────────────────────
    if use_llm is None:
        use_llm = _llm_judge_enabled()
    if use_llm:
        try:
            llm_scores = _llm_judge_actions([s for s, _, _ in [(a, sc, rn) for a, sc, rn in scored]][:LLM_JUDGE_TOP_N])
            # Multiply heuristic by LLM score (both 0..1) so a fail in
            # either layer pulls the action below the drop floor.
            for idx, llm_score in enumerate(llm_scores):
                if idx >= len(scored):
                    break
                a, h_score, reason = scored[idx]
                blended = h_score * (0.5 + 0.5 * llm_score)
                reason = f"{reason}; llm={llm_score:.2f}"
                scored[idx] = (a, blended, reason)
        except Exception as exc:  # noqa: BLE001
            logger.warning("judge.llm_layer_failed", error=str(exc))

    # ── Partition ──────────────────────────────────────────────────
    kept: List[RankedAction] = []
    dropped: List[Tuple[RankedAction, float, str]] = []
    for a, score, reason in scored:
        record_judge("ranked_action", score)
        if score < drop_below:
            dropped.append((a, score, reason))
        else:
            kept.append(a)

    mean_score = sum(s for _, s, _ in scored) / len(scored)
    passed = safety_violations == 0 and mean_score >= DEGRADED_FLOOR
    reason_str = (
        f"kept={len(kept)} dropped={len(dropped)} "
        f"mean={mean_score:.3f} safety_hits={safety_violations}"
    )

    return kept, dropped, EvalResult(
        score=round(mean_score, 3),
        passed=passed,
        reason=reason_str,
        checked_fields=["title", "description", "evidence_message_ids", "confidence"],
    )


# ─────────────────────────────────────────────────────────────────────
# Proposal judge — called from wait_proposals
# ─────────────────────────────────────────────────────────────────────


def judge_proposal(
    content: str,
    *,
    action_title: str = "",
    action_description: str = "",
    evidence_snippets: Iterable[str] = (),
    use_llm: Optional[bool] = None,
) -> Tuple[bool, float, str]:
    """Score a single agent proposal draft.

    A failing proposal is hidden from the digest (the task still
    exists in-app, but the email doesn't show a preview that would
    embarrass the user).
    """
    if not content or not content.strip():
        record_judge("proposal", 0.0)
        return False, 0.0, "empty_content"

    text = content.strip()
    bits: List[str] = []
    score = 1.0

    # Safety — fatal.
    if _has_safety_violation(text):
        record_judge("proposal", 0.0)
        return False, 0.0, "safety_violation"

    # Refusal — fatal.
    if _REFUSAL_RE.search(text):
        record_judge("proposal", 0.0)
        return False, 0.0, "llm_refusal"

    # Length — too-short proposals are usually broken streams.
    n_words = len(text.split())
    if n_words < 12:
        score -= 0.3
        bits.append("very_short")
    elif n_words < 30:
        score -= 0.1
        bits.append("short")

    # Placeholders that the agent forgot to fill in.
    if _PLACEHOLDER_RE.search(text):
        score -= 0.3
        bits.append("placeholders")

    # Fenced JSON in a proposal is almost always an unintended dump.
    if _FENCED_JSON_RE.search(text):
        score -= 0.2
        bits.append("fenced_json")

    # Relevance — proposal should mention something from the action.
    relevance = _string_overlap(text, action_title + " " + action_description)
    if relevance < 0.05:
        score -= 0.2
        bits.append("low_action_overlap")

    score = max(0.0, min(1.0, score))

    if use_llm is None:
        use_llm = _llm_judge_enabled()
    if use_llm:
        try:
            llm_score = _llm_judge_single_proposal(
                content=text,
                action_title=action_title,
                evidence_snippets=list(evidence_snippets)[:3],
            )
            if llm_score is not None:
                score = score * (0.5 + 0.5 * llm_score)
                bits.append(f"llm={llm_score:.2f}")
        except Exception as exc:  # noqa: BLE001
            logger.warning("judge.llm_proposal_failed", error=str(exc))

    passed = score >= DEFAULT_DROP_FLOOR
    record_judge("proposal", score)
    return passed, round(score, 3), "; ".join(bits) or "ok"


# ─────────────────────────────────────────────────────────────────────
# Heuristic action scorer
# ─────────────────────────────────────────────────────────────────────


def _score_action_heuristic(
    a: RankedAction,
    *,
    avail_msgs: Set[str],
    avail_evts: Set[str],
    avail_files: Set[str],
) -> Tuple[float, str, bool]:
    """Return (score, reason, fatal_safety_flag)."""
    bits: List[str] = []
    score = 1.0

    title = (a.title or "").strip()
    desc = (a.description or "").strip()
    full_text = f"{title}\n{desc}"

    # Safety — fatal.
    if _has_safety_violation(full_text):
        return 0.0, "safety_violation", True
    if _REFUSAL_RE.search(full_text):
        return 0.0, "llm_refusal", True

    # Empty title — fatal.
    if not title:
        return 0.0, "empty_title", True

    # Generic title.
    title_lc = title.lower().strip(" .!?")
    if title_lc in _GENERIC_TITLES:
        score -= 0.5
        bits.append("generic_title")

    # Verb-led check — first word should resemble a verb.
    first_word = re.split(r"[\s,:]", title_lc, maxsplit=1)[0]
    if first_word not in _TASK_VERBS:
        # Don't penalise heavily — many fine titles open with a noun
        # (e.g. "Acme contract negotiation"). Mild signal only.
        score -= 0.1
        bits.append("no_verb_lead")

    # Title length — too short OR too long.
    title_len = len(title)
    if title_len < 6:
        score -= 0.3
        bits.append("title_too_short")
    elif title_len > 140:
        score -= 0.2
        bits.append("title_too_long")

    # Format problems.
    if _FENCED_JSON_RE.search(full_text):
        score -= 0.3
        bits.append("fenced_json")
    if _PLACEHOLDER_RE.search(full_text):
        score -= 0.3
        bits.append("placeholders")

    # Groundedness — evidence_ids must be subsets of what we actually
    # had in input. If a model invents IDs, we drop hard.
    if a.evidence_message_ids:
        bad_msgs = [m for m in a.evidence_message_ids if m not in avail_msgs]
        if bad_msgs:
            score -= 0.4
            bits.append(f"unknown_msgs={len(bad_msgs)}")
    if a.evidence_event_ids:
        bad_evts = [e for e in a.evidence_event_ids if e not in avail_evts]
        if bad_evts:
            score -= 0.3
            bits.append(f"unknown_evts={len(bad_evts)}")
    if a.evidence_file_ids:
        bad_files = [f for f in a.evidence_file_ids if f not in avail_files]
        if bad_files:
            score -= 0.3
            bits.append(f"unknown_files={len(bad_files)}")

    # An ungrounded action with no evidence at all is suspicious
    # (the Brain Agent should always cite something it read).
    if not (a.evidence_message_ids or a.evidence_event_ids or a.evidence_file_ids):
        score -= 0.2
        bits.append("no_evidence")

    # Confidence floor.
    if (a.confidence or 0.0) < 0.3:
        score -= 0.2
        bits.append("low_self_confidence")

    # Priority sanity — defensive; Pydantic already constrains this.
    if a.priority not in ("critical", "high", "medium", "low"):
        score -= 0.1
        bits.append("bad_priority")

    score = max(0.0, min(1.0, score))
    return score, "; ".join(bits) or "ok", False


# ─────────────────────────────────────────────────────────────────────
# Optional LLM judge layer
# ─────────────────────────────────────────────────────────────────────


def _llm_judge_enabled() -> bool:
    import os
    return os.environ.get("BRAIN_LLM_JUDGE_ENABLED", "0").strip() == "1"


_LLM_ACTION_JUDGE_PROMPT = """You are a strict reviewer. For each action below,
give a quality score 0..1 reflecting whether you'd be comfortable putting it
in front of a busy executive at 6am as a today-priority. Reward: specific
verb-led titles, clear evidence, well-judged priority. Penalise: vague titles
("Review email"), missing evidence, hallucinated names, leaked credentials,
refusals.

Output STRICT JSON only — an array of floats, one per action, in input order.
No prose, no fences."""


def _llm_judge_actions(actions: List[RankedAction]) -> List[float]:
    """Score up to N actions with a single LLM call. Returns 0..1 per item.

    Returns empty list on failure (caller treats as no LLM signal).
    """
    if not actions:
        return []
    try:
        from backend.ai_models import get_llm_client
        client = get_llm_client()
    except Exception:
        return []

    user_prompt_lines = ["Actions to score (output array of floats):", ""]
    for i, a in enumerate(actions):
        user_prompt_lines.append(
            f"[{i}] title: {a.title!r}\n"
            f"    desc: {(a.description or '')[:200]!r}\n"
            f"    priority: {a.priority}\n"
            f"    confidence: {a.confidence}\n"
            f"    evidence: msgs={a.evidence_message_ids} "
            f"evts={a.evidence_event_ids} files={a.evidence_file_ids}\n"
        )

    try:
        response = _sync_or_async_generate(
            client,
            messages=[
                {"role": "system", "content": _LLM_ACTION_JUDGE_PROMPT},
                {"role": "user", "content": "\n".join(user_prompt_lines)},
            ],
        )
        raw = getattr(response, "content", None) or str(response)
        parsed = json.loads(_strip_fences(raw))
        if isinstance(parsed, list):
            return [float(max(0.0, min(1.0, x))) for x in parsed[: len(actions)]]
        return []
    except Exception as exc:  # noqa: BLE001
        logger.debug("judge.llm_action_call_failed", error=str(exc))
        return []


_LLM_PROPOSAL_JUDGE_PROMPT = """You are a strict reviewer. Score the draft
proposal below 0..1 on whether it directly addresses the task title using the
provided evidence. Reward: relevant, grounded, ready to send with light edits.
Penalise: filler, hallucinated facts, off-topic, refusal apologies.

Output STRICT JSON only — a single float between 0 and 1. No prose, no fences."""


def _llm_judge_single_proposal(
    *,
    content: str,
    action_title: str,
    evidence_snippets: List[str],
) -> Optional[float]:
    try:
        from backend.ai_models import get_llm_client
        client = get_llm_client()
    except Exception:
        return None

    user = (
        f"Task title: {action_title}\n"
        f"Evidence snippets (truncated):\n"
        + "\n".join(f"- {s[:200]}" for s in evidence_snippets[:3])
        + f"\n\nProposal draft (truncated):\n{content[:1200]}"
    )
    try:
        response = _sync_or_async_generate(
            client,
            messages=[
                {"role": "system", "content": _LLM_PROPOSAL_JUDGE_PROMPT},
                {"role": "user", "content": user},
            ],
        )
        raw = getattr(response, "content", None) or str(response)
        return float(max(0.0, min(1.0, float(_strip_fences(raw).strip()))))
    except Exception as exc:  # noqa: BLE001
        logger.debug("judge.llm_proposal_call_failed", error=str(exc))
        return None


def _sync_or_async_generate(client: Any, *, messages: list) -> Any:
    """LLM client's ``generate`` may be sync or async. Handle both."""
    import asyncio
    import inspect
    fn = client.generate
    if inspect.iscoroutinefunction(fn):
        # We may be called from sync context (eval suite) or async (live judge).
        # The live caller awaits us — but here we're called from inside an
        # already-running loop in the brain pipeline, so use create_task.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(fn(messages=messages))
        # In an async context — caller of judge_* should have awaited us if
        # async. We don't await here; fall back to a quick block via new loop.
        future = asyncio.ensure_future(fn(messages=messages))
        # Best-effort sync wait without blocking the loop: not possible.
        # Caller should have used async path; bail.
        raise RuntimeError("llm_judge called from running loop without await")
    return fn(messages=messages)


# ─────────────────────────────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────────────────────────────


def _has_safety_violation(text: str) -> bool:
    if _SSN_RE.search(text):
        return True
    if _PASSWORD_LEAK_RE.search(text):
        return True
    cc = _CC_RE.search(text)
    if cc and _luhn_ok(re.sub(r"[ -]", "", cc.group(0))):
        return True
    return False


def _luhn_ok(num: str) -> bool:
    """Luhn check to weed out non-CC long digit strings (phone numbers etc)."""
    if not num.isdigit():
        return False
    s = 0
    for i, d in enumerate(reversed(num)):
        v = int(d)
        if i % 2 == 1:
            v *= 2
            if v > 9:
                v -= 9
        s += v
    return s % 10 == 0


def _string_overlap(a: str, b: str) -> float:
    """Crude Jaccard over word tokens. 0..1."""
    aw = {w.lower() for w in re.split(r"\W+", a) if len(w) > 2}
    bw = {w.lower() for w in re.split(r"\W+", b) if len(w) > 2}
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines).strip()
    return s
