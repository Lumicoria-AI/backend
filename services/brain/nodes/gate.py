"""Gate node — decides whether the run should proceed.

Checks (in order, short-circuiting on first skip):

  1. Brain enabled in user.preferences.brain.enabled.
  2. Local-time window matches morning or evening, unless force=True
     (force is set by POST /brain/trigger so dev can fire any time).
  3. Dedupe — skip if a successful run already happened today.
  4. Google integration is active (the fetchers need it).

Sets ``skip_reason`` on the state when a check fails — the graph's
conditional edge after this node then routes straight to audit so we
still record the skip + reason for diagnosis.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

from bson import ObjectId

from .._google_client import resolve_google_client
from .._time import now_hour_matches_tz, now_in_tz
from ..state import BrainState
from ..tracing import traced_node


@traced_node("gate")
async def gate(state: BrainState) -> Dict[str, Any]:
    user_id = state.user_id
    mode = state.mode
    force = bool(state.meta.get("force"))

    # 1. Load preferences + last-sent timestamps from the user doc.
    try:
        from backend.db.mongodb.mongodb import MongoDB
        db = await MongoDB.get_database()
        try:
            oid: Any = ObjectId(user_id)
        except Exception:
            oid = user_id
        user_doc = await db.users.find_one(
            {"_id": oid},
            projection={
                "preferences": 1,
                "last_brain_morning_sent": 1,
                "last_brain_evening_sent": 1,
                "email": 1,
            },
        ) or {}
    except Exception:
        user_doc = {}

    prefs_root = user_doc.get("preferences") or {}
    brain_prefs = prefs_root.get("brain") or {}

    payload: Dict[str, Any] = {
        "mode": mode,
        "force": force,
        "tz": state.timezone,
    }

    # 2. Enabled gate (bypassed by force).
    if not force and not brain_prefs.get("enabled", False):
        payload["skip_reason"] = "brain_disabled"
        return {
            "skip_reason": "brain_disabled",
            "__payload_summary": payload,
            "__eval_score": 1.0,
        }

    # 3. Local-time window gate (bypassed by force).
    if not force:
        target_hour = int(
            brain_prefs.get(
                "morning_hour_local" if mode == "morning" else "evening_hour_local",
                6 if mode == "morning" else 22,
            )
        )
        if not now_hour_matches_tz(state.timezone, target_hour, tolerance_minutes=15):
            payload["skip_reason"] = "off_window"
            payload["target_hour"] = target_hour
            return {
                "skip_reason": "off_window",
                "__payload_summary": payload,
                "__eval_score": 1.0,
            }

    # 4. Dedupe — already sent today?
    last_sent_field = (
        "last_brain_morning_sent" if mode == "morning"
        else "last_brain_evening_sent"
    )
    last_sent = user_doc.get(last_sent_field)
    if not force and last_sent and _is_same_local_day(last_sent, state.timezone):
        payload["skip_reason"] = "already_sent_today"
        return {
            "skip_reason": "already_sent_today",
            "__payload_summary": payload,
            "__eval_score": 1.0,
        }

    # 5. Google integration check. Cache the resolved client in
    # state.meta so the fetchers don't redo the lookup + token refresh.
    google_client = await resolve_google_client(user_id)
    if google_client is None:
        payload["skip_reason"] = "no_google_integration"
        return {
            "skip_reason": "no_google_integration",
            "__payload_summary": payload,
            "__eval_score": 1.0,
        }

    # Stash the client + prefs for downstream nodes. state.meta is a
    # dict[str, Any] so it accepts non-serialisable handles like the
    # google client — fine because the graph runs in-process and we
    # don't checkpoint here.
    return {
        "user_email": user_doc.get("email") or state.user_email,
        "meta": {
            **state.meta,
            "google_client": google_client,
            "brain_prefs": brain_prefs,
            "last_sent_at": last_sent,
        },
        "__payload_summary": payload,
        "__eval_score": 1.0,
    }


def _is_same_local_day(ts: datetime, tz: str) -> bool:
    """Compare ts.date in user's TZ vs now.date in user's TZ."""
    try:
        from zoneinfo import ZoneInfo
        zone = ZoneInfo(tz)
        if ts.tzinfo is None:
            # Assume UTC.
            from datetime import timezone as _tz
            ts = ts.replace(tzinfo=_tz.utc)
        return ts.astimezone(zone).date() == now_in_tz(tz).date()
    except Exception:
        return False
