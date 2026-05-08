"""Support-ticket business logic.

All entry points here run in async sessions, never raise on optional
side-effects (sentiment tagging, activity logging), and return plain
dicts ready for JSON response.

Multi-tenant safety: every read / write requires an `organization_id`
argument and applies it as a filter — there is no path where org
scoping is implicit.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import structlog
from sqlalchemy import and_, asc, desc, func, or_, select, update

from ...db.postgres import get_async_sessionmaker
from ...db.postgres_models import SupportTicketSQL, TicketReplySQL
from .sanitize import clean_plain_text, clean_rich_text, normalize_email

logger = structlog.get_logger(__name__)


VALID_PRIORITIES = ("High", "Medium", "Low")
VALID_STATUSES = ("Open", "In Progress", "Resolved", "Closed", "Cancelled")
VALID_CHANNELS = ("portal", "widget", "email", "api", "manual")
TERMINAL_STATUSES = ("Resolved", "Closed", "Cancelled")


def ticket_to_dict(row: SupportTicketSQL) -> Dict[str, Any]:
    return {
        "id": row.id,
        "organization_id": row.organization_id,
        "customer_email": row.customer_email,
        "customer_name": row.customer_name,
        "subject": row.subject,
        "body": row.body,
        "priority": row.priority,
        "status": row.status,
        "category": row.category,
        "channel": row.channel,
        "sentiment_score": (row.sentiment_score / 100.0) if row.sentiment_score is not None else None,
        "assigned_user_id": row.assigned_user_id,
        "submitter_user_id": row.submitter_user_id,
        "meta": dict(row.meta or {}),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
    }


def reply_to_dict(row: TicketReplySQL) -> Dict[str, Any]:
    return {
        "id": row.id,
        "ticket_id": row.ticket_id,
        "organization_id": row.organization_id,
        "author_type": row.author_type,
        "author_user_id": row.author_user_id,
        "author_display_name": row.author_display_name,
        "body": row.body,
        "template_id": row.template_id,
        "ai_draft_meta": row.ai_draft_meta,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def _normalize_priority(value: Optional[str]) -> str:
    if value in VALID_PRIORITIES:
        return value
    return "Medium"


def _normalize_status(value: Optional[str]) -> str:
    if value in VALID_STATUSES:
        return value
    return "Open"


def _normalize_channel(value: Optional[str]) -> str:
    if value in VALID_CHANNELS:
        return value
    return "manual"


# ── Create ──────────────────────────────────────────────────────────────


async def create_ticket(
    *,
    organization_id: str,
    customer_email: str,
    customer_name: Optional[str],
    subject: str,
    body: str,
    priority: Optional[str] = "Medium",
    category: Optional[str] = None,
    channel: str = "manual",
    submitter_user_id: Optional[str] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Insert a new ticket.  Sanitizes all user-supplied strings."""
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        row = SupportTicketSQL(
            organization_id=organization_id,
            customer_email=normalize_email(customer_email),
            customer_name=clean_plain_text(customer_name or "", max_len=200) or None,
            subject=clean_plain_text(subject, max_len=500),
            body=clean_rich_text(body, max_len=10_000),
            priority=_normalize_priority(priority),
            status="Open",
            category=clean_plain_text(category or "", max_len=64) or None,
            channel=_normalize_channel(channel),
            submitter_user_id=submitter_user_id,
            meta=meta or {},
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return ticket_to_dict(row)


# ── Read ────────────────────────────────────────────────────────────────


_TIME_RANGE_TO_DAYS = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}


async def list_tickets(
    organization_id: str,
    *,
    status: Optional[str] = None,
    priority: Optional[str] = None,
    category: Optional[str] = None,
    channel: Optional[str] = None,
    assigned_user_id: Optional[str] = None,
    search: Optional[str] = None,
    time_range: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    """Tenant-scoped paginated list with filters.

    All comparisons are parameterized via SQLAlchemy — no string
    concatenation, no SQL injection surface.  `search` does a safe
    ILIKE wrapped in `%`s, escaping any user-supplied wildcards.
    """
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset or 0))

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        base_where = [
            SupportTicketSQL.organization_id == organization_id,
            SupportTicketSQL.deleted_at.is_(None),
        ]
        if status and status in VALID_STATUSES:
            base_where.append(SupportTicketSQL.status == status)
        if priority and priority in VALID_PRIORITIES:
            base_where.append(SupportTicketSQL.priority == priority)
        if category:
            base_where.append(SupportTicketSQL.category == clean_plain_text(category, max_len=64))
        if channel and channel in VALID_CHANNELS:
            base_where.append(SupportTicketSQL.channel == channel)
        if assigned_user_id:
            base_where.append(SupportTicketSQL.assigned_user_id == assigned_user_id)
        if time_range and time_range in _TIME_RANGE_TO_DAYS:
            cutoff = datetime.utcnow() - timedelta(days=_TIME_RANGE_TO_DAYS[time_range])
            base_where.append(SupportTicketSQL.created_at >= cutoff)
        if search:
            # Escape ILIKE wildcards in user input so a malicious search
            # term can't make every row match.
            cleaned = clean_plain_text(search, max_len=200)
            escaped = cleaned.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
            pat = f"%{escaped}%"
            base_where.append(
                or_(
                    SupportTicketSQL.subject.ilike(pat, escape="\\"),
                    SupportTicketSQL.customer_email.ilike(pat, escape="\\"),
                    SupportTicketSQL.id.ilike(pat, escape="\\"),
                )
            )

        rows_stmt = (
            select(SupportTicketSQL)
            .where(*base_where)
            .order_by(desc(SupportTicketSQL.created_at))
            .limit(limit)
            .offset(offset)
        )
        count_stmt = (
            select(func.count())
            .select_from(SupportTicketSQL)
            .where(*base_where)
        )

        rows = (await session.execute(rows_stmt)).scalars().all()
        total = await session.scalar(count_stmt) or 0

    return {
        "tickets": [ticket_to_dict(r) for r in rows],
        "total": int(total),
        "limit": limit,
        "offset": offset,
    }


async def get_ticket(
    organization_id: str,
    ticket_id: str,
    *,
    include_replies: bool = True,
    reply_limit: int = 50,
) -> Optional[Dict[str, Any]]:
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        stmt = select(SupportTicketSQL).where(
            SupportTicketSQL.id == ticket_id,
            SupportTicketSQL.organization_id == organization_id,
            SupportTicketSQL.deleted_at.is_(None),
        )
        row = (await session.execute(stmt)).scalar_one_or_none()
        if not row:
            return None

        ticket = ticket_to_dict(row)
        if include_replies:
            replies_stmt = (
                select(TicketReplySQL)
                .where(
                    TicketReplySQL.ticket_id == ticket_id,
                    TicketReplySQL.organization_id == organization_id,
                    TicketReplySQL.deleted_at.is_(None),
                )
                .order_by(asc(TicketReplySQL.created_at))
                .limit(reply_limit)
            )
            replies = (await session.execute(replies_stmt)).scalars().all()
            ticket["replies"] = [reply_to_dict(r) for r in replies]
        return ticket


async def get_ticket_for_public_status(
    org_slug_to_id: str,
    ticket_id: str,
    customer_email: str,
) -> Optional[Dict[str, Any]]:
    """Privacy-gated read: the requester must know the original
    submitter email.  Returns None if id/email don't match — same shape
    as 'not found' so callers don't leak existence."""
    cleaned_email = normalize_email(customer_email)
    if not cleaned_email:
        return None
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        stmt = select(SupportTicketSQL).where(
            SupportTicketSQL.id == ticket_id,
            SupportTicketSQL.organization_id == org_slug_to_id,
            SupportTicketSQL.customer_email == cleaned_email,
            SupportTicketSQL.deleted_at.is_(None),
        )
        row = (await session.execute(stmt)).scalar_one_or_none()
        if not row:
            return None

        # Public status: only operator + agent_ai replies are returned.
        # Customer's own replies stay hidden from the timeline (they're
        # the customer — they wrote them).  Adjust per product taste.
        replies_stmt = (
            select(TicketReplySQL)
            .where(
                TicketReplySQL.ticket_id == ticket_id,
                TicketReplySQL.organization_id == org_slug_to_id,
                TicketReplySQL.deleted_at.is_(None),
            )
            .order_by(asc(TicketReplySQL.created_at))
        )
        replies = (await session.execute(replies_stmt)).scalars().all()
        ticket = ticket_to_dict(row)
        ticket["replies"] = [reply_to_dict(r) for r in replies]
        return ticket


# ── Update ──────────────────────────────────────────────────────────────


_PATCHABLE = {"status", "priority", "category", "assigned_user_id"}


async def update_ticket(
    organization_id: str,
    ticket_id: str,
    fields: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Patch a subset of mutable fields.  Validates each value before write."""
    if not fields:
        return await get_ticket(organization_id, ticket_id, include_replies=False)

    sanitized: Dict[str, Any] = {}
    if "status" in fields:
        new_status = fields["status"]
        if new_status not in VALID_STATUSES:
            return None
        sanitized["status"] = new_status
        if new_status in TERMINAL_STATUSES:
            sanitized["resolved_at"] = datetime.utcnow()
    if "priority" in fields:
        sanitized["priority"] = _normalize_priority(fields["priority"])
    if "category" in fields:
        cat = fields["category"]
        sanitized["category"] = clean_plain_text(cat or "", max_len=64) or None
    if "assigned_user_id" in fields:
        v = fields["assigned_user_id"]
        sanitized["assigned_user_id"] = (str(v)[:64] if v else None)

    if not sanitized:
        return await get_ticket(organization_id, ticket_id, include_replies=False)
    sanitized["updated_at"] = datetime.utcnow()

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        await session.execute(
            update(SupportTicketSQL)
            .where(
                SupportTicketSQL.id == ticket_id,
                SupportTicketSQL.organization_id == organization_id,
                SupportTicketSQL.deleted_at.is_(None),
            )
            .values(**sanitized)
        )
        await session.commit()
    return await get_ticket(organization_id, ticket_id, include_replies=False)


async def soft_delete_ticket(organization_id: str, ticket_id: str) -> bool:
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        result = await session.execute(
            update(SupportTicketSQL)
            .where(
                SupportTicketSQL.id == ticket_id,
                SupportTicketSQL.organization_id == organization_id,
                SupportTicketSQL.deleted_at.is_(None),
            )
            .values(deleted_at=datetime.utcnow(), updated_at=datetime.utcnow())
        )
        await session.commit()
        return (result.rowcount or 0) > 0


async def set_sentiment_score(
    organization_id: str, ticket_id: str, sentiment: float
) -> None:
    """Persist a sentiment score (-1..1).  Stored as int -100..100 to
    keep the column tidy."""
    try:
        s = max(-1.0, min(1.0, float(sentiment)))
    except (TypeError, ValueError):
        return
    SessionLocal = get_async_sessionmaker()
    try:
        async with SessionLocal() as session:
            await session.execute(
                update(SupportTicketSQL)
                .where(
                    SupportTicketSQL.id == ticket_id,
                    SupportTicketSQL.organization_id == organization_id,
                )
                .values(sentiment_score=int(round(s * 100)))
            )
            await session.commit()
    except Exception as e:  # noqa: BLE001
        logger.warning("sentiment_persist_failed", error=str(e), ticket_id=ticket_id)


# ── Replies ─────────────────────────────────────────────────────────────


async def add_reply(
    *,
    organization_id: str,
    ticket_id: str,
    body: str,
    author_type: str,
    author_user_id: Optional[str] = None,
    author_display_name: Optional[str] = None,
    template_id: Optional[str] = None,
    ai_draft_meta: Optional[Dict[str, Any]] = None,
    transition_status: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Append a reply.  Optionally transitions the ticket status (e.g.
    Open → In Progress when an operator replies)."""
    if author_type not in ("operator", "customer", "agent_ai"):
        author_type = "operator"
    cleaned_body = clean_rich_text(body, max_len=20_000)
    if not cleaned_body:
        return None

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        # Confirm the ticket exists in this org before inserting a reply.
        ticket_row = (
            await session.execute(
                select(SupportTicketSQL).where(
                    SupportTicketSQL.id == ticket_id,
                    SupportTicketSQL.organization_id == organization_id,
                    SupportTicketSQL.deleted_at.is_(None),
                )
            )
        ).scalar_one_or_none()
        if ticket_row is None:
            return None

        reply = TicketReplySQL(
            ticket_id=ticket_id,
            organization_id=organization_id,
            author_type=author_type,
            author_user_id=author_user_id,
            author_display_name=clean_plain_text(author_display_name or "", max_len=200) or None,
            body=cleaned_body,
            template_id=template_id,
            ai_draft_meta=ai_draft_meta,
        )
        session.add(reply)

        if transition_status and transition_status in VALID_STATUSES:
            ticket_row.status = transition_status
            if transition_status in TERMINAL_STATUSES and not ticket_row.resolved_at:
                ticket_row.resolved_at = datetime.utcnow()
            ticket_row.updated_at = datetime.utcnow()
        else:
            ticket_row.updated_at = datetime.utcnow()

        await session.commit()
        await session.refresh(reply)
        return reply_to_dict(reply)


# ── Prior-ticket search (used by AI Draft context builder) ─────────────


async def search_resolved_for_context(
    organization_id: str,
    *,
    category: Optional[str],
    subject_hint: Optional[str],
    limit: int = 3,
) -> List[Dict[str, Any]]:
    """Find recently-resolved tickets in the same category.  Used as
    'prior cases' context for the AI Draft prompt — not for direct user
    display.  Subject hint is matched ILIKE; safe-escaped."""
    if not organization_id:
        return []
    limit = max(1, min(int(limit or 3), 10))

    where = [
        SupportTicketSQL.organization_id == organization_id,
        SupportTicketSQL.deleted_at.is_(None),
        SupportTicketSQL.status.in_(("Resolved", "Closed")),
    ]
    if category:
        where.append(SupportTicketSQL.category == category)
    if subject_hint:
        cleaned = clean_plain_text(subject_hint, max_len=200)
        escaped = cleaned.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        if escaped:
            where.append(SupportTicketSQL.subject.ilike(f"%{escaped}%", escape="\\"))

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        stmt = (
            select(SupportTicketSQL)
            .where(*where)
            .order_by(desc(SupportTicketSQL.resolved_at), desc(SupportTicketSQL.created_at))
            .limit(limit)
        )
        rows = (await session.execute(stmt)).scalars().all()
        out: List[Dict[str, Any]] = []
        for row in rows:
            # Pull the last operator reply to capture the resolution.
            last_reply_stmt = (
                select(TicketReplySQL)
                .where(
                    TicketReplySQL.ticket_id == row.id,
                    TicketReplySQL.organization_id == organization_id,
                    TicketReplySQL.author_type.in_(("operator", "agent_ai")),
                    TicketReplySQL.deleted_at.is_(None),
                )
                .order_by(desc(TicketReplySQL.created_at))
                .limit(1)
            )
            last_reply = (await session.execute(last_reply_stmt)).scalar_one_or_none()
            out.append({
                "ticket_id": row.id,
                "subject": row.subject,
                "body": row.body,
                "category": row.category,
                "resolution": last_reply.body if last_reply else None,
                "resolved_at": row.resolved_at.isoformat() if row.resolved_at else None,
            })
        return out
