"""
Signed action tokens for in-email task buttons (Phase 4).

The daily / critical / weekly reminder emails carry "Mark complete" and
"Mark started" buttons.  Clicking them must work without making the user
log in — so each button URL embeds a short-lived, scope-limited JWT that
authorises one specific action on one specific task.

Design:
  • HS256 signed with `settings.SECRET_KEY` — same secret the rest of the
    app already trusts, so we don't introduce a new key surface.
  • 7-day expiry by default (matches the cadence reminders are sent at).
  • Strict claims: `sub` (user_id), `tid` (task_id), `act` (action),
    `scope` = "task_action", `exp`, `iat`.
  • Allowed actions are a closed enum, validated on decode.
  • Decode raises `TaskActionTokenError` with a specific reason so the
    endpoint can render a clean "expired" / "already done" page instead
    of a stack trace.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, Optional

from jose import JWTError, jwt

from backend.core.config import settings

ALGORITHM = "HS256"
SCOPE = "task_action"
DEFAULT_EXPIRY_DAYS = 7


class TaskAction(str, Enum):
    COMPLETE = "complete"
    START = "start"
    SNOOZE = "snooze"


class TaskActionTokenError(Exception):
    """Decode failure — `reason` is one of: invalid, expired, wrong_scope, bad_action."""

    def __init__(self, reason: str, detail: str = ""):
        super().__init__(detail or reason)
        self.reason = reason
        self.detail = detail


def make_action_token(
    *,
    user_id: str,
    task_id: str,
    action: TaskAction | str,
    expires_in_days: int = DEFAULT_EXPIRY_DAYS,
) -> str:
    """Mint a signed JWT for one action on one task by one user."""
    if isinstance(action, TaskAction):
        action_value = action.value
    else:
        # Validate against the closed enum upfront.
        try:
            action_value = TaskAction(action).value
        except ValueError as e:
            raise ValueError(f"Unsupported task action: {action}") from e

    now = datetime.utcnow()
    payload: Dict[str, Any] = {
        "sub": str(user_id),
        "tid": str(task_id),
        "act": action_value,
        "scope": SCOPE,
        "iat": now,
        "exp": now + timedelta(days=expires_in_days),
    }
    return jwt.encode(payload, settings.SECRET_KEY, algorithm=ALGORITHM)


def decode_action_token(token: str) -> Dict[str, Any]:
    """Verify and decode a task-action JWT.

    Returns a dict with: `user_id`, `task_id`, `action`.
    Raises `TaskActionTokenError` with a specific `reason`.
    """
    if not token:
        raise TaskActionTokenError("invalid", "token is empty")

    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError as e:
        msg = str(e).lower()
        if "expir" in msg:
            raise TaskActionTokenError("expired", str(e)) from e
        raise TaskActionTokenError("invalid", str(e)) from e

    if payload.get("scope") != SCOPE:
        raise TaskActionTokenError("wrong_scope", f"scope={payload.get('scope')}")

    action_raw = payload.get("act")
    try:
        action = TaskAction(action_raw)
    except ValueError as e:
        raise TaskActionTokenError("bad_action", f"action={action_raw}") from e

    user_id = payload.get("sub")
    task_id = payload.get("tid")
    if not user_id or not task_id:
        raise TaskActionTokenError("invalid", "missing sub/tid claim")

    return {"user_id": str(user_id), "task_id": str(task_id), "action": action.value}


def action_url(
    *,
    base_url: str,
    user_id: str,
    task_id: str,
    action: TaskAction | str,
    expires_in_days: int = DEFAULT_EXPIRY_DAYS,
) -> str:
    """Build the full clickable URL for an in-email task button."""
    token = make_action_token(
        user_id=user_id,
        task_id=task_id,
        action=action,
        expires_in_days=expires_in_days,
    )
    base = (base_url or "").rstrip("/")
    return f"{base}/api/v1/tasks/action?token={token}"
