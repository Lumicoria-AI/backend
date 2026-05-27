"""Redis-backed activity / break tracker.

Powers the live break countdown on the Coach page and the periodic
break-reminder Celery task.  We keep this in Redis (not Mongo) so
heartbeat writes stay cheap — the frontend pings every 30s and we
never want that to touch the primary database.

Keys (all in the default Redis DB):
  wellbeing:last_activity:{user_id}   epoch seconds, TTL 24h
  wellbeing:last_break:{user_id}      epoch seconds, TTL 24h
  wellbeing:mood_prompt_pending:{user_id}   "1" if frontend should show modal
  wellbeing:mood_prompt_cooldown:{user_id}  cooldown after a prompt fires
  wellbeing:break_reminder_cooldown:{user_id}  cooldown after a reminder fires
"""

from __future__ import annotations

import time
from typing import Optional

import structlog

# Reuse the existing redis client helper so we don't manage a second
# connection pool.  `_get_redis()` returns None when Redis is down,
# which is fine — we degrade gracefully.
from ...core.security import _get_redis  # type: ignore

logger = structlog.get_logger(__name__)


# TTLs in seconds.
_HEARTBEAT_TTL = 24 * 60 * 60       # 24h — survives a long break / overnight
_BREAK_TTL = 24 * 60 * 60
_PROMPT_PENDING_TTL = 30 * 60       # if the frontend doesn't ack, drop after 30m
_PROMPT_COOLDOWN_TTL = 90 * 60      # 90 min between mood prompts per user
_BREAK_REMINDER_COOLDOWN_TTL = 30 * 60  # 30 min between break reminder pushes


# ── Activity heartbeat ────────────────────────────────────────────


def mark_activity(user_id: str) -> None:
    """Record that the user is active right now.  Called by the
    frontend heartbeat endpoint every ~30s while the tab is focused."""
    if not user_id:
        return
    client = _get_redis()
    if client is None:
        return
    try:
        client.setex(f"wellbeing:last_activity:{user_id}", _HEARTBEAT_TTL, int(time.time()))
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_heartbeat_write_failed", error=str(e))


def last_activity(user_id: str) -> Optional[int]:
    """Return the unix-seconds timestamp of the user's last heartbeat,
    or None if Redis is down / no heartbeat yet."""
    if not user_id:
        return None
    client = _get_redis()
    if client is None:
        return None
    try:
        v = client.get(f"wellbeing:last_activity:{user_id}")
        return int(v) if v is not None else None
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_heartbeat_read_failed", error=str(e))
        return None


def seconds_since_activity(user_id: str) -> Optional[int]:
    ts = last_activity(user_id)
    if ts is None:
        return None
    return max(0, int(time.time()) - ts)


# ── Break tracking ────────────────────────────────────────────────


def mark_break(user_id: str) -> None:
    """Record that the user just took a break.  Called whenever an
    activity with break-like type is logged."""
    if not user_id:
        return
    client = _get_redis()
    if client is None:
        return
    try:
        client.setex(f"wellbeing:last_break:{user_id}", _BREAK_TTL, int(time.time()))
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_break_write_failed", error=str(e))


def last_break(user_id: str) -> Optional[int]:
    if not user_id:
        return None
    client = _get_redis()
    if client is None:
        return None
    try:
        v = client.get(f"wellbeing:last_break:{user_id}")
        return int(v) if v is not None else None
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_break_read_failed", error=str(e))
        return None


def seconds_since_break(user_id: str) -> Optional[int]:
    ts = last_break(user_id)
    if ts is None:
        return None
    return max(0, int(time.time()) - ts)


def seconds_until_next_break(user_id: str, interval_minutes: int) -> int:
    """How many seconds until the user's next scheduled break.

    Falls back to `interval_minutes * 60` when we have no heartbeat
    yet (treat the user as just-started).  Returns 0 when they are
    already overdue.
    """
    interval_seconds = max(60, int(interval_minutes) * 60)
    elapsed = seconds_since_break(user_id)
    if elapsed is None:
        return interval_seconds
    return max(0, interval_seconds - elapsed)


# ── Mood prompt scheduler ─────────────────────────────────────────


def queue_mood_prompt(user_id: str) -> bool:
    """Mark that a mood prompt should appear for this user the next
    time their frontend polls.  Returns True if a prompt was queued,
    False if the user is in cooldown or Redis is down."""
    if not user_id:
        return False
    client = _get_redis()
    if client is None:
        return False
    try:
        cooldown = client.get(f"wellbeing:mood_prompt_cooldown:{user_id}")
        if cooldown is not None:
            return False
        client.setex(
            f"wellbeing:mood_prompt_pending:{user_id}",
            _PROMPT_PENDING_TTL,
            "1",
        )
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_mood_queue_failed", error=str(e))
        return False


def pop_mood_prompt(user_id: str) -> bool:
    """Frontend calls this to ask 'should I show the mood modal now?'.
    If yes, we consume the pending flag and set the cooldown so we
    don't show another modal for 90 minutes."""
    if not user_id:
        return False
    client = _get_redis()
    if client is None:
        return False
    try:
        pending = client.get(f"wellbeing:mood_prompt_pending:{user_id}")
        if pending is None:
            return False
        # Consume and set cooldown atomically-ish.
        pipe = client.pipeline()
        pipe.delete(f"wellbeing:mood_prompt_pending:{user_id}")
        pipe.setex(
            f"wellbeing:mood_prompt_cooldown:{user_id}",
            _PROMPT_COOLDOWN_TTL,
            "1",
        )
        pipe.execute()
        return True
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_mood_pop_failed", error=str(e))
        return False


def snooze_mood_prompt(user_id: str, minutes: int = 90) -> None:
    """Frontend calls this on Snooze.  Extends the cooldown."""
    if not user_id:
        return
    client = _get_redis()
    if client is None:
        return
    try:
        client.delete(f"wellbeing:mood_prompt_pending:{user_id}")
        client.setex(
            f"wellbeing:mood_prompt_cooldown:{user_id}",
            max(60, int(minutes) * 60),
            "1",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_mood_snooze_failed", error=str(e))


# ── Break reminder cooldown ───────────────────────────────────────


def break_reminder_in_cooldown(user_id: str) -> bool:
    if not user_id:
        return True
    client = _get_redis()
    if client is None:
        return True
    try:
        return client.get(f"wellbeing:break_reminder_cooldown:{user_id}") is not None
    except Exception:
        return True


def mark_break_reminder_sent(user_id: str) -> None:
    if not user_id:
        return
    client = _get_redis()
    if client is None:
        return
    try:
        client.setex(
            f"wellbeing:break_reminder_cooldown:{user_id}",
            _BREAK_REMINDER_COOLDOWN_TTL,
            "1",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_break_reminder_cooldown_failed", error=str(e))


# ── Snooze the break reminder via UI ─────────────────────────────


def snooze_break(user_id: str, minutes: int = 15) -> None:
    """User clicked Snooze on the break reminder — extend the
    countdown by recording a fresh 'last break' at now - (interval -
    snooze)."""
    if not user_id:
        return
    # Push the last_break forward by N minutes so the countdown resets.
    client = _get_redis()
    if client is None:
        return
    try:
        snooze_seconds = max(60, int(minutes) * 60)
        client.setex(
            f"wellbeing:last_break:{user_id}",
            _BREAK_TTL,
            int(time.time()) - 0 + snooze_seconds,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_break_snooze_failed", error=str(e))
