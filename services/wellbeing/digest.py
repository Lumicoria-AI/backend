"""Weekly Well-being Digest assembler.

Builds a single per-user payload that powers the `weekly_digest.html`
email template.  Pulled from:

  - `wellbeing_repository.get_user_metrics` (mood / energy / stress / sleep)
  - `wellbeing_repository.get_recent_activities`
  - `services.wellbeing.productivity.compute_productivity`
  - `services.wellbeing.orchestrator.weekly_reflection` (AI summary)

The Celery task `send_weekly_digest_for_all_users` walks the user
list, builds one payload each, renders the template, and sends via
the existing `EmailService`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from . import orchestrator
from .productivity import compute_productivity

logger = structlog.get_logger(__name__)


async def build_user_digest(
    *,
    organization_id: str,
    user_id: str,
    email: str,
    name: Optional[str] = None,
) -> Dict[str, Any]:
    """Assemble the full digest payload for one user."""
    now = datetime.utcnow()
    week_start = now - timedelta(days=7)

    metrics_summary: Dict[str, Dict[str, Any]] = {}
    activities: List[Dict[str, Any]] = []

    # ── Metrics ────────────────────────────────────────────────
    try:
        from ...db.mongodb.repositories.wellbeing_repository import (
            wellbeing_repository,
        )

        recent = await wellbeing_repository.get_user_metrics(
            user_id=user_id,
            organization_id=organization_id,
            start_date=week_start,
            end_date=now,
        )
        # Group by metric_type, compute simple stats.
        for row in recent or []:
            mtype = row.get("metric_type") or "other"
            try:
                value = float(row.get("value") or 0)
            except Exception:
                continue
            bucket = metrics_summary.setdefault(
                mtype, {"count": 0, "sum": 0.0, "min": value, "max": value, "last": value}
            )
            bucket["count"] += 1
            bucket["sum"] += value
            bucket["min"] = min(bucket["min"], value)
            bucket["max"] = max(bucket["max"], value)
            bucket["last"] = value
        for mtype, bucket in metrics_summary.items():
            if bucket["count"]:
                bucket["avg"] = round(bucket["sum"] / bucket["count"], 2)
            else:
                bucket["avg"] = 0.0

        # Activities count.
        try:
            activities = await wellbeing_repository.get_recent_activities(
                user_id=user_id,
                organization_id=organization_id,
                limit=20,
            )
        except Exception:  # noqa: BLE001
            activities = []
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_digest_metrics_failed", error=str(e))

    # ── Productivity ───────────────────────────────────────────
    try:
        productivity = await compute_productivity(
            organization_id=organization_id, user_id=user_id
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_digest_productivity_failed", error=str(e))
        productivity = {}

    # ── AI reflection ──────────────────────────────────────────
    try:
        reflection = await orchestrator.weekly_reflection(
            organization_id=organization_id,
            user_id=user_id,
            user_data={
                "metrics_summary": metrics_summary,
                "productivity": productivity,
                "activities_this_week": len(activities),
            },
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_digest_reflection_failed", error=str(e))
        reflection = {
            "summary": "",
            "highlights": [],
            "concerns": [],
            "focus_for_next_week": [],
            "encouragement": "",
        }

    return {
        "user_id": user_id,
        "user_name": name or "there",
        "user_email": email,
        "week_start": week_start.date().isoformat(),
        "week_end": now.date().isoformat(),
        "metrics_summary": metrics_summary,
        "activities_count": len(activities),
        "productivity": productivity,
        "reflection": reflection,
        "generated_at": now.isoformat() + "Z",
    }


async def send_user_digest(payload: Dict[str, Any]) -> bool:
    """Render the email template and send via the platform's email
    service.  Returns True on success, False otherwise (logged)."""
    try:
        from ..email_service import get_email_service

        email_service = await get_email_service()
        result = await email_service.send(
            to=payload["user_email"],
            subject="Your weekly wellness digest",
            template_name="weekly_digest",
            template_data=payload,
            tags=["wellbeing", "weekly_digest"],
        )
        return bool(getattr(result, "success", False) or getattr(result, "message_id", None))
    except Exception as e:  # noqa: BLE001
        logger.error(
            "wellbeing_digest_send_failed",
            user_id=payload.get("user_id"),
            error=str(e),
        )
        return False
