"""Timezone-aware "is it the right hour for this user?" helper.

The brain (and the existing task reminder system) fan out hourly; each
per-user task gates locally on the user's preferred send time. Without
TZ awareness the gate fires UTC-only and every user gets the same
"6 AM" — which is 1 AM in California.

`now_hour_matches_tz(timezone, target_hour, *, tolerance_minutes=15)`
returns True iff *now* in the user's TZ is within ``tolerance_minutes``
of ``target_hour:00``. Used by both `tasks/task_reminder_tasks.py`
(when we migrate it) and the new brain fan-out.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

try:
    from zoneinfo import ZoneInfo
except ImportError:  # pragma: no cover — Python <3.9 fallback
    ZoneInfo = None  # type: ignore


def now_in_tz(timezone: str) -> datetime:
    """Return the current wall-clock time in the named TZ. Falls back
    to UTC if the TZ string is malformed (rather than raising) so a
    single bad user pref can't break the fan-out for everyone else."""
    if ZoneInfo is None:
        return datetime.utcnow()
    try:
        return datetime.now(ZoneInfo(timezone))
    except Exception:
        return datetime.utcnow()


def now_hour_matches_tz(
    timezone: str,
    target_hour: int,
    *,
    tolerance_minutes: int = 15,
) -> bool:
    """True iff ``target_hour:00`` local TZ is within ±tolerance_minutes
    of right now. The window absorbs Celery beat jitter and worker
    queue lag — beat ticks at :00, but a worker can pick up the task
    several minutes later. Keep tolerance ≤30 to avoid overlap between
    the morning (6 AM) and evening (10 PM) windows.
    """
    if not 0 <= target_hour <= 23:
        return False
    now = now_in_tz(timezone)
    minute_of_day = now.hour * 60 + now.minute
    target_minute = target_hour * 60
    delta = abs(minute_of_day - target_minute)
    # Handle wrap-around at midnight.
    delta = min(delta, 24 * 60 - delta)
    return delta <= tolerance_minutes


def offset_hours_from_utc(timezone: str) -> Optional[float]:
    """UTC offset in hours for the named TZ, or None if invalid.
    Mostly useful for logging — "user runs at 06:00 (UTC-7)"."""
    if ZoneInfo is None:
        return None
    try:
        tz = ZoneInfo(timezone)
        offset = datetime.now(tz).utcoffset()
        if offset is None:
            return None
        return offset.total_seconds() / 3600.0
    except Exception:
        return None
