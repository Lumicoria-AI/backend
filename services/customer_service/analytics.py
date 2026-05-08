"""Real customer-service analytics — replaces the mock dict in
`backend/api/v1/endpoints/customer_service.py`.

All aggregations run against `support_tickets`, `ticket_replies`, and
`response_templates`.  Every query is parameterized via SQLAlchemy.

Response shape is identical to the prior mock so the frontend doesn't
need to move:
    {
      "time_range": "...",
      "total_requests": int,
      "average_response_time": float,    # seconds
      "satisfaction_rate": float,        # 0..1
      "common_issues": [{"issue": str, "count": int}, ...],
      "template_usage": {name: count, ...},
      "feedback_trends": {"positive": float, "neutral": float, "negative": float},
    }
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

from sqlalchemy import and_, asc, case, desc, func, select

from ...db.postgres import get_async_sessionmaker
from ...db.postgres_models import (
    ResponseTemplateSQL,
    SupportTicketSQL,
    TicketReplySQL,
)


_TIME_RANGE_TO_DAYS = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}


def _cutoff(time_range: str) -> datetime:
    days = _TIME_RANGE_TO_DAYS.get(time_range, 7)
    return datetime.utcnow() - timedelta(days=days)


def _empty_payload(time_range: str) -> Dict[str, Any]:
    return {
        "time_range": time_range,
        "total_requests": 0,
        "average_response_time": 0.0,
        "satisfaction_rate": 0.0,
        "common_issues": [],
        "template_usage": {},
        "feedback_trends": {"positive": 0.0, "neutral": 0.0, "negative": 0.0},
    }


async def get_analytics(
    organization_id: str,
    time_range: str = "7d",
) -> Dict[str, Any]:
    """Aggregate ticket + reply + template data into the analytics dict.

    Empty orgs return zeroed metrics with the same shape — the frontend
    detects empty state via the per-section conditionals it already has.
    """
    if not organization_id:
        return _empty_payload(time_range)
    if time_range not in _TIME_RANGE_TO_DAYS:
        time_range = "7d"

    cutoff = _cutoff(time_range)
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:

        # ── total_requests ────────────────────────────────────────────
        total = await session.scalar(
            select(func.count())
            .select_from(SupportTicketSQL)
            .where(
                SupportTicketSQL.organization_id == organization_id,
                SupportTicketSQL.deleted_at.is_(None),
                SupportTicketSQL.created_at >= cutoff,
            )
        ) or 0

        # ── average_response_time ─────────────────────────────────────
        # Per ticket: seconds between ticket.created_at and the FIRST
        # operator/agent_ai reply.  Average across the window.
        first_reply_subq = (
            select(
                TicketReplySQL.ticket_id.label("ticket_id"),
                func.min(TicketReplySQL.created_at).label("first_reply_at"),
            )
            .where(
                TicketReplySQL.organization_id == organization_id,
                TicketReplySQL.author_type.in_(("operator", "agent_ai")),
                TicketReplySQL.deleted_at.is_(None),
            )
            .group_by(TicketReplySQL.ticket_id)
            .subquery()
        )
        # EPOCH FROM (a - b) — seconds difference, Postgres-native.
        delta_seconds = func.extract(
            "epoch",
            first_reply_subq.c.first_reply_at - SupportTicketSQL.created_at,
        )
        avg_resp = await session.scalar(
            select(func.avg(delta_seconds))
            .select_from(SupportTicketSQL)
            .join(first_reply_subq, first_reply_subq.c.ticket_id == SupportTicketSQL.id)
            .where(
                SupportTicketSQL.organization_id == organization_id,
                SupportTicketSQL.deleted_at.is_(None),
                SupportTicketSQL.created_at >= cutoff,
            )
        )
        average_response_time = float(avg_resp) if avg_resp is not None else 0.0

        # ── satisfaction_rate ────────────────────────────────────────
        # Resolved tickets with sentiment_score > 0 / total resolved.
        resolved_total = await session.scalar(
            select(func.count())
            .select_from(SupportTicketSQL)
            .where(
                SupportTicketSQL.organization_id == organization_id,
                SupportTicketSQL.deleted_at.is_(None),
                SupportTicketSQL.status.in_(("Resolved", "Closed")),
                SupportTicketSQL.created_at >= cutoff,
            )
        ) or 0
        positive_resolved = 0
        if resolved_total:
            positive_resolved = await session.scalar(
                select(func.count())
                .select_from(SupportTicketSQL)
                .where(
                    SupportTicketSQL.organization_id == organization_id,
                    SupportTicketSQL.deleted_at.is_(None),
                    SupportTicketSQL.status.in_(("Resolved", "Closed")),
                    SupportTicketSQL.created_at >= cutoff,
                    SupportTicketSQL.sentiment_score.is_not(None),
                    SupportTicketSQL.sentiment_score >= 0,
                )
            ) or 0
        satisfaction_rate = (positive_resolved / resolved_total) if resolved_total else 0.0

        # ── common_issues ────────────────────────────────────────────
        common_rows = (
            await session.execute(
                select(
                    SupportTicketSQL.category,
                    func.count(SupportTicketSQL.id).label("c"),
                )
                .where(
                    SupportTicketSQL.organization_id == organization_id,
                    SupportTicketSQL.deleted_at.is_(None),
                    SupportTicketSQL.created_at >= cutoff,
                    SupportTicketSQL.category.is_not(None),
                )
                .group_by(SupportTicketSQL.category)
                .order_by(desc("c"))
                .limit(5)
            )
        ).all()
        common_issues = [
            {"issue": cat or "uncategorized", "count": int(count)}
            for cat, count in common_rows
        ]

        # ── template_usage ───────────────────────────────────────────
        template_rows = (
            await session.execute(
                select(ResponseTemplateSQL.name, ResponseTemplateSQL.usage_count)
                .where(
                    ResponseTemplateSQL.organization_id == organization_id,
                    ResponseTemplateSQL.deleted_at.is_(None),
                    ResponseTemplateSQL.usage_count > 0,
                )
                .order_by(desc(ResponseTemplateSQL.usage_count))
                .limit(10)
            )
        ).all()
        template_usage = {name: int(count) for name, count in template_rows}

        # ── feedback_trends ──────────────────────────────────────────
        # Bucketed counts off sentiment_score (-100..100).
        sent_bucket = case(
            (SupportTicketSQL.sentiment_score >= 25, "positive"),
            (SupportTicketSQL.sentiment_score <= -25, "negative"),
            else_="neutral",
        )
        sent_rows = (
            await session.execute(
                select(sent_bucket.label("bucket"), func.count())
                .where(
                    SupportTicketSQL.organization_id == organization_id,
                    SupportTicketSQL.deleted_at.is_(None),
                    SupportTicketSQL.created_at >= cutoff,
                    SupportTicketSQL.sentiment_score.is_not(None),
                )
                .group_by("bucket")
            )
        ).all()
        bucket_totals = {row[0]: int(row[1]) for row in sent_rows}
        sent_total = sum(bucket_totals.values()) or 0
        if sent_total:
            feedback_trends = {
                "positive": round(bucket_totals.get("positive", 0) / sent_total, 3),
                "neutral": round(bucket_totals.get("neutral", 0) / sent_total, 3),
                "negative": round(bucket_totals.get("negative", 0) / sent_total, 3),
            }
        else:
            feedback_trends = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}

    return {
        "time_range": time_range,
        "total_requests": int(total),
        "average_response_time": round(average_response_time, 2),
        "satisfaction_rate": round(satisfaction_rate, 3),
        "common_issues": common_issues,
        "template_usage": template_usage,
        "feedback_trends": feedback_trends,
    }
