"""Deliver the digest via email + in-app + push.

Each channel is wrapped in its own try/except so one failing channel
doesn't block the others. The user's brain preferences gate per-channel
opt-out:

  prefs.send_email   default True
  prefs.send_in_app  default True
  prefs.send_push    default True (push only fires for critical-priority
                                   items)

On success we stamp ``user.last_brain_morning_sent`` (or
``_evening_sent``) so tomorrow's gate node can dedupe.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import structlog
from bson import ObjectId

from ..state import BrainState
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


@traced_node("send")
async def send(state: BrainState) -> Dict[str, Any]:
    render = (state.meta or {}).get("render") or {}
    if not render:
        return {
            "delivery_channels": [],
            "__payload_summary": {"sent": 0, "reason": "no_render"},
            "__eval_score": 0.0,
            "__status": "fallback",
        }

    # ── Digest quality gate — refuse to send a bad email ────────────
    quality_passed = (state.meta or {}).get("digest_quality_passed", True)
    quality_reason = (state.meta or {}).get("digest_quality_reason", "")
    if not quality_passed:
        # Still create an in-app notification so the user knows the
        # brain ran — but never an email or push for a low-quality run.
        try:
            await _send_in_app(
                user_id=state.user_id,
                mode=state.mode,
                render={
                    **render,
                    "summary_line": (
                        "We ran your brain this morning but held the digest "
                        "back for review — open the dashboard to see why."
                    ),
                },
            )
        except Exception:
            pass
        logger.warning(
            "send.digest_quality_blocked",
            run_id=state.run_id,
            reason=quality_reason,
        )
        return {
            "delivery_channels": [],
            "__payload_summary": {
                "sent": 0,
                "blocked": True,
                "reason": f"digest_quality_failed:{quality_reason[:100]}",
            },
            "__eval_score": 0.0,
            "__status": "fallback",
        }

    prefs = (state.meta or {}).get("brain_prefs") or {}
    user_email = state.user_email
    user_id = state.user_id
    mode = state.mode

    channels: List[str] = []

    # ── Email ───────────────────────────────────────────────────────
    if prefs.get("send_email", True) and user_email:
        ok = await _send_email(
            to_email=user_email,
            template_name=(
                "morning_brain_digest" if mode == "morning"
                else "evening_brain_review"
            ),
            data=render,
        )
        if ok:
            channels.append("email")

    # ── In-app notification ─────────────────────────────────────────
    if prefs.get("send_in_app", True):
        ok = await _send_in_app(
            user_id=user_id,
            mode=mode,
            render=render,
        )
        if ok:
            channels.append("in_app")

    # ── Push (only for critical actions) ────────────────────────────
    has_critical = any(
        (a.get("priority") or "").lower() == "critical"
        for a in render.get("top_actions") or []
    )
    if prefs.get("send_push", True) and has_critical:
        ok = await _send_push(
            user_id=user_id,
            mode=mode,
            render=render,
        )
        if ok:
            channels.append("push")

    # ── Stamp last_sent on success so the next morning's gate dedupes
    if channels:
        await _stamp_last_sent(user_id, mode)

    return {
        "delivery_channels": channels,
        "__payload_summary": {
            "channels": channels,
            "sent": len(channels),
            "had_critical": has_critical,
        },
        "__eval_score": 1.0 if channels else 0.0,
        **({"__status": "fallback"} if not channels else {}),
    }


# ─────────────────────────────────────────────────────────────────────
# Channel implementations
# ─────────────────────────────────────────────────────────────────────


async def _send_email(
    *,
    to_email: str,
    template_name: str,
    data: Dict[str, Any],
) -> bool:
    try:
        from backend.services.notification_service import notification_service
        from backend.db.mongodb.models.notification import NotificationPriority
        sent = await notification_service.send_email_notification(
            to_email=to_email,
            template_name=template_name,
            template_data=data,
            priority=NotificationPriority.NORMAL,
        )
        return bool(sent)
    except Exception as exc:  # noqa: BLE001
        logger.warning("send.email_failed", error=str(exc))
        return False


async def _send_in_app(
    *,
    user_id: str,
    mode: str,
    render: Dict[str, Any],
) -> bool:
    try:
        from backend.services.notification_service import notification_service
        from backend.db.mongodb.models.notification import (
            NotificationPriority, NotificationType,
        )
        title = render.get("subject") or (
            "Your morning brief" if mode == "morning"
            else "Your evening review"
        )
        content = render.get("summary_line") or ""
        await notification_service.create_in_app_notification(
            user_id=user_id,
            title=title,
            content=content,
            notification_type=NotificationType.TASK,
            priority=NotificationPriority.NORMAL,
            metadata={
                "category": f"brain.{mode}",
                "run_id": render.get("run_id"),
                "counts": render.get("counts") or {},
            },
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.warning("send.in_app_failed", error=str(exc))
        return False


async def _send_push(
    *,
    user_id: str,
    mode: str,
    render: Dict[str, Any],
) -> bool:
    try:
        from backend.services.push_notification_service import (
            push_notification_service,
        )
        top = (render.get("top_actions") or [{}])[0]
        await push_notification_service.send_to_user(
            user_id=user_id,
            title=("Critical" if mode == "morning" else "Day wrap-up"),
            body=top.get("title", "Your brain has updates.")[:80],
            data={
                "type": f"brain.{mode}",
                "dashboard_url": render.get("dashboard_url"),
            },
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("send.push_failed", error=str(exc))
        return False


async def _stamp_last_sent(user_id: str, mode: str) -> None:
    """Best-effort write — failure here just means the next run may
    re-send if its window also overlaps."""
    try:
        from backend.db.mongodb.mongodb import MongoDB
        db = await MongoDB.get_database()
        try:
            uid_oid: Any = ObjectId(user_id)
        except Exception:
            uid_oid = user_id
        field = "last_brain_morning_sent" if mode == "morning" else "last_brain_evening_sent"
        await db.users.update_one(
            {"_id": uid_oid},
            {"$set": {field: datetime.utcnow()}},
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("send.stamp_failed", error=str(exc))
