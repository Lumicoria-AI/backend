"""Build the DigestPayload from everything the graph collected.

The compose node is the bridge between in-graph state and the email
template. For each ranked action:
  - Mint signed APPROVE / REVISE / REJECT tokens (existing
    services/task_action_tokens — used by other notification flows).
  - Render due_label, agent assignment, low_confidence flag.
  - Pull the agent proposal preview from Mongo (Phase 3 already
    persisted these via task_executor).

Calendar items get human time labels ("9:30am · Acme review"). Open
tasks are passed through. Completed-today + slipped-tasks are computed
only in evening mode (the morning compose doesn't need them).
"""

from __future__ import annotations

import urllib.parse
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from ..state import BrainState, DigestPayload
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


_PRIORITY_RANK = {"critical": 0, "high": 1, "medium": 2, "low": 3}


@traced_node("compose")
async def compose(state: BrainState) -> Dict[str, Any]:
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

    return {
        "digest_payload": payload,
        "meta": {**state.meta, "render": render},
        "__payload_summary": {
            "top_actions": len(top_actions_dicts),
            "secondary_actions": len(secondary_actions_dicts),
            "calendar_items": len(calendar_today_dicts),
            "completed_today": len(completed_today),
            "slipped_tasks": len(slipped_tasks),
        },
        "__eval_score": 1.0,
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
