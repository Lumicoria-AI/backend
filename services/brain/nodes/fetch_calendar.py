"""Pull calendar events relevant to today + tomorrow.

Morning mode: events from `now` to end of tomorrow (so "today" + "next
24h" both render in the digest).
Evening mode: events from now to end of tomorrow (focus on what's
coming up after dinner planning).

Uses the existing ``client.get_events`` helper which already handles
the primary calendar + RFC-3339 time bounds.
"""

from __future__ import annotations

from datetime import datetime, time, timedelta
from typing import Any, Dict, List

import structlog

from ..state import BrainState, CalendarEventRef
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


@traced_node("fetch_calendar")
async def fetch_calendar(state: BrainState) -> Dict[str, Any]:
    client = state.meta.get("google_client")
    if client is None:
        return {
            "events": [],
            "__payload_summary": {"count": 0, "reason": "no_client"},
            "__eval_score": 1.0,
        }

    # Build a 36-hour forward window starting now. Cheap on the API
    # (Calendar list returns paged) but rich enough for both digests.
    now = datetime.utcnow().replace(microsecond=0)
    end = (now + timedelta(hours=36))
    time_min = now.isoformat() + "Z"
    time_max = end.isoformat() + "Z"

    try:
        raw = await client.get_events(
            calendar_id="primary",
            time_min=time_min,
            time_max=time_max,
            max_results=50,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_calendar.failed", error=str(exc))
        return {
            "events": [],
            "__payload_summary": {"count": 0, "error": "api_failed"},
            "__eval_score": 0.0,
            "__status": "fallback",
        }

    events: List[CalendarEventRef] = []
    for ev in raw or []:
        events.append(_to_event_ref(ev))

    return {
        "events": events,
        "__payload_summary": {
            "count": len(events),
            "window_hours": 36,
            "start": time_min,
            "end": time_max,
        },
        "__eval_score": 1.0,
    }


def _to_event_ref(ev: Dict[str, Any]) -> CalendarEventRef:
    start_dt = _parse_event_time(ev.get("start"))
    end_dt = _parse_event_time(ev.get("end"))
    attendees = [
        a.get("email") for a in (ev.get("attendees") or []) if a.get("email")
    ]
    return CalendarEventRef(
        event_id=str(ev.get("id") or ""),
        summary=(ev.get("summary") or "")[:300] or None,
        start=start_dt,
        end=end_dt,
        attendees=attendees[:20],
        location=(ev.get("location") or "")[:200] or None,
    )


def _parse_event_time(slot: Any) -> datetime | None:
    if not slot or not isinstance(slot, dict):
        return None
    val = slot.get("dateTime") or slot.get("date")
    if not val:
        return None
    try:
        dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
        return dt.replace(tzinfo=None) if dt.tzinfo else dt
    except Exception:
        return None
