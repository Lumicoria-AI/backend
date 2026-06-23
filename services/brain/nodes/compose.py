"""Build the DigestPayload from everything the graph collected.

The compose pipeline is three LangGraph nodes wired with conditional
edges so each retry attempt is its own trace row:

  ┌─ compose_primary           (full render + judge)
  │         │
  │      passed? ──── yes ──→ send
  │         │ no
  │         ▼
  ├─ compose_prune_promote     (drop low-conf top, promote secondary)
  │         │
  │      passed? ──── yes ──→ send
  │         │ no
  │         ▼
  └─ compose_minimal_safe      (strip LLM strings, keep calendar + CTA)
            │
            ▼
          send   (refuses to email if quality_passed is still False)

The render dict lives on ``state.meta["render"]``. Each node overwrites
it with its current attempt's output. The conditional routers below
read ``state.meta["digest_quality_passed"]`` to choose where to go.
"""

from __future__ import annotations

import urllib.parse
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from ..evals import evaluate_digest
from ..state import BrainState, DigestPayload
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


_PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


# ─────────────────────────────────────────────────────────────────────
# Conditional routers — pure functions, no side effects
# ─────────────────────────────────────────────────────────────────────


def after_primary(state: BrainState) -> str:
    return "ok" if (state.meta or {}).get("digest_quality_passed") else "retry"


def after_prune_promote(state: BrainState) -> str:
    return "ok" if (state.meta or {}).get("digest_quality_passed") else "retry"


# ─────────────────────────────────────────────────────────────────────
# Compose node 1 — primary full render
# ─────────────────────────────────────────────────────────────────────


@traced_node("compose_primary")
async def compose_primary(state: BrainState) -> Dict[str, Any]:
    # Pull settings for action-token URLs (frontend base + API base).
    try:
        from backend.core.config import settings
        web_base = (getattr(settings, "FRONTEND_URL", None) or "").rstrip("/")
        api_base = (getattr(settings, "API_BASE_URL", None) or "").rstrip("/")
        if not api_base:
            # Best-effort fall back to FRONTEND_URL/api/v1.
            api_base = f"{web_base}/api/v1" if web_base else ""
    except Exception:
        web_base, api_base = "", ""

    # Look up proposal text per task so the email shows the agent's draft.
    proposals_by_task = await _fetch_proposal_previews(state.created_task_ids)

    # Hide previews the judge rejected — the task still exists in-app
    # but the email never embarrasses the user with bad drafts.
    bad_proposal_ids = set((state.meta or {}).get("judge_bad_proposal_ids") or [])
    if bad_proposal_ids:
        proposals_by_task = {
            tid: content
            for tid, content in proposals_by_task.items()
            if tid not in bad_proposal_ids
        }

    # Pair ranked_actions with their created_task_ids in order.
    top_actions_dicts: List[Dict[str, Any]] = []
    secondary_actions_dicts: List[Dict[str, Any]] = []

    pairs = list(
        zip(
            state.ranked_actions,
            state.created_task_ids + [None] * max(0, len(state.ranked_actions) - len(state.created_task_ids)),
        )
    )
    pairs.sort(key=lambda p: _PRIORITY_RANK.get((p[0].priority or "medium").lower(), 3))

    for idx, (action, task_id) in enumerate(pairs):
        rendered = _render_action(
            action=action,
            task_id=task_id,
            proposal_preview=proposals_by_task.get(task_id or "", ""),
            user_id=state.user_id,
            web_base=web_base,
            api_base=api_base,
        )
        if idx < 5:
            top_actions_dicts.append(rendered)
        else:
            secondary_actions_dicts.append(rendered)

    # Calendar — only today + tomorrow start.
    calendar_today_dicts = [_render_event(ev) for ev in state.events[:8]]

    # Evening-only sections.
    completed_today: List[Dict[str, Any]] = []
    slipped_tasks: List[Dict[str, Any]] = []
    if state.mode == "evening":
        completed_today = await _fetch_completed_today(state.user_id)
        slipped_tasks = await _fetch_slipped_today(state.user_id)

    # Build the payload. The state.digest_payload field expects a
    # DigestPayload Pydantic — we round-trip through dicts so the
    # send node has everything in template-ready shape.
    user_name = await _resolve_user_name(state.user_id)
    summary_line = state.meta.get("brain_summary_line") or _default_summary_line(
        mode=state.mode,
        emails=len(state.emails),
        actions=len(state.ranked_actions),
        events=len(state.events),
    )

    counts = {
        "emails": len(state.emails),
        "tasks_created": len(state.created_task_ids),
        "drive_changes": len(state.drive_changes),
        "meetings": len(state.events),
        "open": len([t for t in state.open_tasks if t.status not in ("completed", "cancelled")]),
        "completed": len(completed_today),
    }

    # Persist the typed payload on state too (DigestPayload retains
    # only Pydantic-safe types; the send node reads from state.meta
    # for the rendered-dict variant below).
    payload = DigestPayload(
        mode=state.mode,
        user_name=user_name,
        summary_line=summary_line,
        top_actions=state.ranked_actions[:5],
        secondary_actions=state.ranked_actions[5:],
        calendar_today=state.events,
        completed_today=[],  # OpenTaskRef not Task — keep state lightweight
        open_tasks=state.open_tasks,
        counts=counts,
    )

    # Free-form render-ready dict for the send node + email template.
    render = {
        "subject": _subject(state.mode, user_name),
        "user_name": user_name,
        "summary_line": summary_line,
        "top_actions": top_actions_dicts,
        "secondary_actions": secondary_actions_dicts,
        "calendar_today": calendar_today_dicts,
        "completed_today": completed_today,
        "slipped_tasks": slipped_tasks,
        "open_tasks": [_render_open_task(t) for t in state.open_tasks[:8]],
        "counts": counts,
        "focus_message": _focus_message(state, completed_today, slipped_tasks),
        "generated_at": datetime.utcnow().strftime("%H:%M UTC"),
        "dashboard_url": f"{web_base}/tasks" if web_base else "https://lumicoria.ai/tasks",
        "prefs_url": f"{web_base}/brain/preferences" if web_base else "https://lumicoria.ai/brain/preferences",
    }

    # Single-pass quality gate. The graph's conditional edges decide
    # whether to advance to send or fall through to a recompose node.
    quality_passed, quality_score, quality_reason = evaluate_digest(render)

    return {
        "digest_payload": payload,
        "meta": {
            **state.meta,
            "render": render,
            "digest_quality_passed": quality_passed,
            "digest_quality_score": quality_score,
            "digest_quality_reason": quality_reason,
            "compose_user_name": user_name,
            "compose_counts": counts,
        },
        "__payload_summary": {
            "strategy": "primary",
            "top_actions": len(top_actions_dicts),
            "secondary_actions": len(secondary_actions_dicts),
            "calendar_items": len(calendar_today_dicts),
            "completed_today": len(completed_today),
            "slipped_tasks": len(slipped_tasks),
            "hidden_bad_proposals": len(bad_proposal_ids),
            "quality_passed": quality_passed,
            "quality_score": quality_score,
            "quality_reason": quality_reason[:200],
        },
        "__eval_score": quality_score,
        **({"__status": "fallback"} if not quality_passed else {}),
    }


# ─────────────────────────────────────────────────────────────────────
# Compose node 2 — prune + promote retry
# ─────────────────────────────────────────────────────────────────────


@traced_node("compose_prune_promote")
async def compose_prune_promote(state: BrainState) -> Dict[str, Any]:
    """Retry strategy A: drop low-confidence + generic-title top actions,
    promote good secondaries to fill the top slot."""
    base_render = (state.meta or {}).get("render") or {}
    if not base_render:
        return {
            "meta": {**state.meta, "digest_quality_passed": False, "digest_quality_reason": "no_base_render"},
            "__payload_summary": {"strategy": "prune_promote", "skipped": True},
            "__eval_score": 0.0,
            "__status": "fail",
        }

    new_render = _recompose_strategy_a(base_render)
    passed, score, reason = evaluate_digest(new_render)

    return {
        "meta": {
            **state.meta,
            "render": new_render,
            "digest_quality_passed": passed,
            "digest_quality_score": score,
            "digest_quality_reason": reason,
        },
        "__payload_summary": {
            "strategy": "prune_promote",
            "top_actions": len(new_render.get("top_actions") or []),
            "quality_passed": passed,
            "quality_score": score,
            "quality_reason": reason[:200],
        },
        "__eval_score": score,
        "__status": "retry" if passed else "fallback",
    }


# ─────────────────────────────────────────────────────────────────────
# Compose node 3 — minimal-safe last resort
# ─────────────────────────────────────────────────────────────────────


@traced_node("compose_minimal_safe")
async def compose_minimal_safe(state: BrainState) -> Dict[str, Any]:
    """Final fallback render. Strips every LLM-generated string —
    keeps calendar, open tasks, dashboard CTA, and a neutral
    "your brain ran" summary. Designed to always pass the safety gate."""
    base_render = (state.meta or {}).get("render") or {}
    user_name = (state.meta or {}).get("compose_user_name")
    counts = (state.meta or {}).get("compose_counts") or {}

    new_render = _recompose_strategy_b(
        base=base_render,
        user_name=user_name,
        mode=state.mode,
        counts=counts,
    )
    passed, score, reason = evaluate_digest(new_render)

    return {
        "meta": {
            **state.meta,
            "render": new_render,
            "digest_quality_passed": passed,
            "digest_quality_score": score,
            "digest_quality_reason": reason,
        },
        "__payload_summary": {
            "strategy": "minimal_safe",
            "quality_passed": passed,
            "quality_score": score,
            "quality_reason": reason[:200],
        },
        "__eval_score": score,
        "__status": "retry" if passed else "fail",
    }


# ─────────────────────────────────────────────────────────────────────
# Helpers — keep the render block easy to reason about
# ─────────────────────────────────────────────────────────────────────


def _subject(mode: str, user_name: Optional[str]) -> str:
    name_part = f", {user_name}" if user_name else ""
    if mode == "morning":
        return f"☀️ Your morning brief{name_part}"
    return f"🌙 Your evening review{name_part}"


def _default_summary_line(
    *, mode: str, emails: int, actions: int, events: int,
) -> str:
    if mode == "morning":
        if actions == 0 and emails == 0:
            return "All clear — no priorities surfaced from your inbox this morning."
        bits = []
        if actions:
            bits.append(f"{actions} priorit{'y' if actions == 1 else 'ies'}")
        if events:
            bits.append(f"{events} meeting{'s' if events != 1 else ''}")
        if emails:
            bits.append(f"{emails} email{'s' if emails != 1 else ''} reviewed")
        return "Today: " + " · ".join(bits) + "."
    # Evening
    return "Here is what shipped today and where to put your energy tomorrow."


def _focus_message(
    state: BrainState,
    completed_today: List[Dict[str, Any]],
    slipped_tasks: List[Dict[str, Any]],
) -> str:
    if state.mode == "morning":
        return ""
    if state.ranked_actions:
        top = state.ranked_actions[0]
        return (
            f"Lead with “{top.title}” first thing — it's "
            f"{top.priority} priority and your agents have already "
            "drafted what they can."
        )
    if slipped_tasks:
        return "A few items slipped today — tackle them before opening fresh threads tomorrow."
    if completed_today:
        return "Strong day. Reset, then come back to the open list with fresh eyes."
    return "Nothing pressing — use the morning to do focused, undirected work."


def _render_action(
    *,
    action,
    task_id: Optional[str],
    proposal_preview: str,
    user_id: str,
    web_base: str,
    api_base: str,
) -> Dict[str, Any]:
    """Render one RankedAction into the dict shape the email template
    expects — including signed action-token URLs."""
    approve_url = revise_url = reject_url = view_url = ""
    if task_id:
        view_url = f"{web_base}/tasks/{task_id}" if web_base else f"https://lumicoria.ai/tasks/{task_id}"
        approve_url = _signed_action_url(api_base, user_id, task_id, "approve_proposal")
        revise_url = _signed_action_url(api_base, user_id, task_id, "revise_proposal")
        reject_url = _signed_action_url(api_base, user_id, task_id, "reject_proposal")

    return {
        "title": action.title,
        "description": action.description,
        "priority": action.priority,
        "due_label": _due_label(action.due_date),
        "assigned_to_agent": action.assigned_to_agent,
        "low_confidence": float(action.confidence or 0.0) < 0.5,
        "proposal_preview": (proposal_preview or "")[:280] or None,
        "approve_url": approve_url,
        "revise_url": revise_url,
        "reject_url": reject_url,
        "view_url": view_url,
    }


def _signed_action_url(
    api_base: str, user_id: str, task_id: str, action: str,
) -> str:
    """Mint a signed JWT action token + wrap it in the /tasks/actions
    endpoint URL the email button hits."""
    try:
        from backend.services.task_action_tokens import (
            TaskAction, make_action_token,
        )
        token = make_action_token(
            user_id=user_id,
            task_id=task_id,
            action=TaskAction(action),
        )
    except Exception:
        return ""
    qs = urllib.parse.urlencode({"t": token})
    base = api_base or "https://lumicoria.ai/api/v1"
    return f"{base}/tasks/{task_id}/actions/{action}?{qs}"


def _render_event(ev) -> Dict[str, Any]:
    start = ev.start
    time_label = "TBD"
    if start:
        try:
            time_label = start.strftime("%a %H:%M")
        except Exception:
            time_label = start.isoformat()[:16]
    return {
        "event_id": ev.event_id,
        "summary": ev.summary or "(no title)",
        "time_label": time_label,
        "location": ev.location,
    }


def _render_open_task(t) -> Dict[str, Any]:
    return {
        "title": t.title,
        "priority": (t.priority or "medium").lower(),
        "due_label": _due_label(t.due_date),
    }


def _due_label(due: Optional[datetime]) -> Optional[str]:
    if due is None:
        return None
    now = datetime.utcnow()
    if due.tzinfo is not None:
        due = due.replace(tzinfo=None)
    delta = due - now
    if delta.total_seconds() < 0:
        return "overdue"
    days = int(delta.total_seconds() // 86400)
    if days == 0:
        return "due today"
    if days == 1:
        return "due tomorrow"
    if days < 7:
        return f"in {days}d"
    return due.strftime("%b %d")


# ─────────────────────────────────────────────────────────────────────
# Recompose strategies — invoked when the primary render fails the
# pre-send digest quality gate. Each strategy is deterministic and
# safe to apply repeatedly.
# ─────────────────────────────────────────────────────────────────────


def _recompose_strategy_a(render: Dict[str, Any]) -> Dict[str, Any]:
    """Prune + promote.

    The most common failure is "too many low-confidence top actions" or
    "generic top title." Strategy A:

      * Drop any top action flagged low_confidence.
      * Drop any top action with a generic title.
      * Promote the highest-priority secondary actions into the top
        until top has at least 1 or secondary is exhausted.
      * Tighten the summary line.
    """
    out = dict(render)

    GENERIC = {"review email", "task", "todo", "follow up", "respond"}

    def _keep(a: Dict[str, Any]) -> bool:
        if a.get("low_confidence"):
            return False
        title = (a.get("title") or "").strip().lower().strip(" .!?")
        if title in GENERIC or not title:
            return False
        return True

    top = [a for a in (render.get("top_actions") or []) if _keep(a)]
    secondary = list(render.get("secondary_actions") or [])

    # Promote good secondaries into top until we have at least one.
    while len(top) < 5 and secondary:
        promoted = secondary.pop(0)
        if _keep(promoted):
            top.append(promoted)

    out["top_actions"] = top
    out["secondary_actions"] = secondary

    # Tighten the summary line so it reflects what actually shipped.
    n_top = len(top)
    n_events = len(render.get("calendar_today") or [])
    if n_top == 0 and n_events == 0:
        out["summary_line"] = "All clear — nothing surfaced for today."
    elif n_top == 0:
        out["summary_line"] = f"{n_events} meeting{'s' if n_events != 1 else ''} on the calendar today."
    else:
        out["summary_line"] = (
            f"Today: {n_top} priorit{'y' if n_top == 1 else 'ies'}"
            + (f" · {n_events} meeting{'s' if n_events != 1 else ''}" if n_events else "")
            + "."
        )
    return out


def _recompose_strategy_b(
    *,
    base: Dict[str, Any],
    user_name: Optional[str],
    mode: str,
    counts: Dict[str, int],
) -> Dict[str, Any]:
    """Minimal-safe digest.

    Strips every LLM-generated string. Keeps only deterministic content
    (calendar items, open tasks, dashboard CTA, counts). Almost always
    passes the quality gate because the safety / refusal / placeholder
    regexes fire on the action block we just removed.
    """
    name = f", {user_name}" if user_name else ""
    if mode == "morning":
        subject = f"☀️ Your morning brief{name}"
        summary = (
            "Your brain ran this morning. We held the priority list back "
            "for review — open the dashboard for the full breakdown."
        )
        focus = ""
    else:
        subject = f"🌙 Your evening review{name}"
        summary = (
            "Your brain ran this evening. We held the recap back for "
            "review — open the dashboard for what shipped today."
        )
        focus = ""

    return {
        "subject": subject,
        "user_name": user_name,
        "summary_line": summary,
        "top_actions": [],
        "secondary_actions": [],
        "calendar_today": base.get("calendar_today") or [],
        "completed_today": base.get("completed_today") or [],
        "slipped_tasks": [],
        "open_tasks": base.get("open_tasks") or [],
        "counts": counts,
        "focus_message": focus,
        "generated_at": base.get("generated_at") or "",
        "dashboard_url": base.get("dashboard_url") or "https://lumicoria.ai/tasks",
        "prefs_url": base.get("prefs_url") or "https://lumicoria.ai/brain/preferences",
    }


# ─────────────────────────────────────────────────────────────────────
# Mongo lookups
# ─────────────────────────────────────────────────────────────────────


async def _resolve_user_name(user_id: str) -> Optional[str]:
    try:
        from bson import ObjectId
        from backend.db.mongodb.mongodb import MongoDB
        db = await MongoDB.get_database()
        try:
            oid: Any = ObjectId(user_id)
        except Exception:
            oid = user_id
        doc = await db.users.find_one(
            {"_id": oid},
            projection={"full_name": 1, "first_name": 1, "name": 1},
        )
        if not doc:
            return None
        full = doc.get("first_name") or (doc.get("full_name") or "").split(" ")[0]
        return full or None
    except Exception:
        return None


async def _fetch_proposal_previews(task_ids: List[str]) -> Dict[str, str]:
    """Read each task's agent_proposal.content for the email preview block."""
    if not task_ids:
        return {}
    try:
        from bson import ObjectId
        from backend.db.mongodb.mongodb import MongoDB
        db = await MongoDB.get_database()
        oids = []
        for tid in task_ids:
            try:
                oids.append(ObjectId(tid))
            except Exception:
                oids.append(tid)
        out: Dict[str, str] = {}
        cursor = db.tasks.find(
            {"_id": {"$in": oids}},
            projection={"_id": 1, "agent_proposal": 1},
        )
        async for t in cursor:
            tid = str(t.get("_id"))
            prop = t.get("agent_proposal") or {}
            content = prop.get("content") or ""
            if content:
                out[tid] = content
        return out
    except Exception:
        return {}


async def _fetch_completed_today(user_id: str) -> List[Dict[str, Any]]:
    try:
        from bson import ObjectId
        from backend.db.mongodb.mongodb import MongoDB
        db = await MongoDB.get_database()
        try:
            uid_oid: Any = ObjectId(user_id)
        except Exception:
            uid_oid = user_id
        cutoff = datetime.utcnow() - timedelta(hours=24)
        cursor = db.tasks.find(
            {
                "$or": [{"assigned_to": uid_oid}, {"created_by": uid_oid}],
                "status": "completed",
                "completed_at": {"$gte": cutoff},
            },
            projection={"title": 1, "priority": 1, "completed_at": 1},
        ).limit(20)
        items = []
        async for t in cursor:
            items.append({
                "title": t.get("title", ""),
                "priority": (t.get("priority") or "medium").lower(),
            })
        return items
    except Exception:
        return []


async def _fetch_slipped_today(user_id: str) -> List[Dict[str, Any]]:
    """Tasks that were due before now and aren't done — they slipped."""
    try:
        from bson import ObjectId
        from backend.db.mongodb.mongodb import MongoDB
        db = await MongoDB.get_database()
        try:
            uid_oid: Any = ObjectId(user_id)
        except Exception:
            uid_oid = user_id
        cutoff = datetime.utcnow() - timedelta(hours=24)
        cursor = db.tasks.find(
            {
                "$or": [{"assigned_to": uid_oid}, {"created_by": uid_oid}],
                "status": {"$nin": ["completed", "cancelled"]},
                "due_date": {"$gte": cutoff, "$lte": datetime.utcnow()},
            },
            projection={"title": 1, "priority": 1, "due_date": 1},
        ).limit(20)
        items = []
        async for t in cursor:
            items.append({
                "title": t.get("title", ""),
                "priority": (t.get("priority") or "medium").lower(),
            })
        return items
    except Exception:
        return []
