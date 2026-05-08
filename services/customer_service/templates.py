"""Response template service: CRUD helpers + 5-default seeder.

Templates are tenant-scoped.  On the first read of /customer-service/templates
for any organization that has zero `is_default=true` rows, we insert the
five canonical templates below in a single transaction.  Subsequent reads
are no-ops thanks to the `(organization_id, name)` partial unique index
configured in `backend/db/postgres.py`.

The seeded set is deliberately small and useful — every template has
`{{customer_name}}` and `{{ticket_id}}` placeholders the operator can
fill in by hand or that AI Draft can interpolate from ticket context.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import and_, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ...db.postgres import get_async_sessionmaker
from ...db.postgres_models import ResponseTemplateSQL
from .sanitize import clean_plain_text, clean_rich_text

logger = structlog.get_logger(__name__)


# ── Five canonical templates ────────────────────────────────────────────


DEFAULT_TEMPLATES: List[Dict[str, Any]] = [
    {
        "name": "Acknowledge Inquiry",
        "category": "general_inquiry",
        "tone": "professional_friendly",
        "description": "Acknowledge & buy time",
        "body": (
            "Hi {{customer_name}},\n\n"
            "Thanks for reaching out — your request (ticket {{ticket_id}}) "
            "has landed with our team. I'm looking into it now and will "
            "follow up with an update as soon as I have more details.\n\n"
            "If anything new comes up on your end, feel free to reply to "
            "this thread.\n\n"
            "Best,\nSupport"
        ),
        "variables": ["customer_name", "ticket_id"],
    },
    {
        "name": "Request More Details",
        "category": "general_inquiry",
        "tone": "professional_friendly",
        "description": "Ask the customer to clarify",
        "body": (
            "Hi {{customer_name}},\n\n"
            "Thanks for letting us know about this. To help me get you a "
            "fast and accurate answer on ticket {{ticket_id}}, could you "
            "share a few more details:\n\n"
            "- When did this start happening?\n"
            "- What were you doing right before it occurred?\n"
            "- Any error message text or screenshots you can share?\n"
            "- The browser / OS / device you were using.\n\n"
            "As soon as I have that, I'll dig in.\n\n"
            "Best,\nSupport"
        ),
        "variables": ["customer_name", "ticket_id"],
    },
    {
        "name": "Technical Resolution",
        "category": "technical_support",
        "tone": "professional_friendly",
        "description": "Resolved — technical fix",
        "body": (
            "Hi {{customer_name}},\n\n"
            "Good news — ticket {{ticket_id}} is resolved. Here's what I "
            "did and what you can do on your end:\n\n"
            "What I changed:\n"
            "- [brief summary of the fix]\n\n"
            "What you can do:\n"
            "- Refresh the page / restart the app\n"
            "- Verify the issue is gone\n\n"
            "If you see anything else off, just reply here and we'll "
            "reopen the ticket. Otherwise, have a great rest of your day.\n\n"
            "Best,\nSupport"
        ),
        "variables": ["customer_name", "ticket_id"],
    },
    {
        "name": "Billing Inquiry Response",
        "category": "billing_issue",
        "tone": "professional",
        "description": "Billing — explain charges",
        "body": (
            "Hi {{customer_name}},\n\n"
            "Thanks for the note about your billing — I've pulled up "
            "ticket {{ticket_id}} and reviewed the charges in question.\n\n"
            "Here's what I found:\n"
            "- [breakdown of the charge or credit]\n"
            "- [any pro-rations, taxes, or plan changes that apply]\n\n"
            "If anything looks off or you'd like me to walk through "
            "this on a call, just reply and we'll get it sorted.\n\n"
            "Best,\nSupport"
        ),
        "variables": ["customer_name", "ticket_id"],
    },
    {
        "name": "Escalation Notice",
        "category": "escalation",
        "tone": "empathetic",
        "description": "Escalate to specialist",
        "body": (
            "Hi {{customer_name}},\n\n"
            "Thanks for your patience on ticket {{ticket_id}}. Because "
            "this needs a closer look from a specialist, I've escalated "
            "it to our [team name] team — they'll reach out within the "
            "next business day.\n\n"
            "I'll stay copied on the thread, so feel free to keep replying "
            "here. We'll keep you posted as soon as we have an update.\n\n"
            "Best,\nSupport"
        ),
        "variables": ["customer_name", "ticket_id"],
    },
]


# ── Public helpers ──────────────────────────────────────────────────────


async def seed_default_templates_if_empty(organization_id: str) -> int:
    """Ensure the five canonical templates exist for an organization.

    Idempotent: safe to call on every list query.  Inserts ON CONFLICT
    DO NOTHING using the (organization_id, name) partial unique index.
    Returns the number of rows actually inserted.
    """
    if not organization_id:
        return 0

    SessionLocal = get_async_sessionmaker()
    inserted = 0
    async with SessionLocal() as session:
        # Quick existence check before any insert work — keeps the hot
        # path of "templates already seeded" close to a single SELECT.
        existing_count = await session.scalar(
            select(func.count())
            .select_from(ResponseTemplateSQL)
            .where(
                ResponseTemplateSQL.organization_id == organization_id,
                ResponseTemplateSQL.is_default.is_(True),
                ResponseTemplateSQL.deleted_at.is_(None),
            )
        )
        if (existing_count or 0) >= len(DEFAULT_TEMPLATES):
            return 0

        # Batch insert with ON CONFLICT DO NOTHING — survives a race
        # where two requests try to seed the same org simultaneously.
        rows = [
            {
                "organization_id": organization_id,
                "name": tpl["name"],
                "category": tpl["category"],
                "tone": tpl["tone"],
                "body": clean_rich_text(tpl["body"]),
                "description": clean_plain_text(tpl["description"], max_len=200),
                "variables": tpl["variables"],
                "is_default": True,
                "created_by_agent": False,
                "usage_count": 0,
            }
            for tpl in DEFAULT_TEMPLATES
        ]
        stmt = pg_insert(ResponseTemplateSQL).values(rows)
        stmt = stmt.on_conflict_do_nothing(index_elements=["organization_id", "name"])
        result = await session.execute(stmt)
        inserted = result.rowcount or 0
        await session.commit()

    if inserted:
        logger.info(
            "default_templates_seeded",
            organization_id=organization_id,
            inserted=inserted,
        )
    return inserted


def to_dict(row: ResponseTemplateSQL) -> Dict[str, Any]:
    return {
        "id": row.id,
        "organization_id": row.organization_id,
        "name": row.name,
        "category": row.category,
        "tone": row.tone,
        "body": row.body,
        "description": row.description,
        "variables": list(row.variables or []),
        "usage_count": row.usage_count,
        "is_default": row.is_default,
        "created_by_user_id": row.created_by_user_id,
        "created_by_agent": row.created_by_agent,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


async def list_templates(
    organization_id: str,
    *,
    category: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """List active templates for an org, seeding defaults on first read."""
    if not organization_id:
        return []
    await seed_default_templates_if_empty(organization_id)

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        stmt = select(ResponseTemplateSQL).where(
            ResponseTemplateSQL.organization_id == organization_id,
            ResponseTemplateSQL.deleted_at.is_(None),
        )
        if category:
            stmt = stmt.where(ResponseTemplateSQL.category == category)
        stmt = stmt.order_by(
            ResponseTemplateSQL.is_default.desc(),
            ResponseTemplateSQL.usage_count.desc(),
            ResponseTemplateSQL.created_at.desc(),
        )
        rows = (await session.execute(stmt)).scalars().all()
        return [to_dict(r) for r in rows]


async def get_template(
    organization_id: str, template_id: str
) -> Optional[Dict[str, Any]]:
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        stmt = select(ResponseTemplateSQL).where(
            ResponseTemplateSQL.id == template_id,
            ResponseTemplateSQL.organization_id == organization_id,
            ResponseTemplateSQL.deleted_at.is_(None),
        )
        row = (await session.execute(stmt)).scalar_one_or_none()
        return to_dict(row) if row else None


async def create_template(
    *,
    organization_id: str,
    name: str,
    category: str,
    body: str,
    tone: Optional[str] = None,
    description: Optional[str] = None,
    variables: Optional[List[str]] = None,
    created_by_user_id: Optional[str] = None,
    created_by_agent: bool = False,
) -> Dict[str, Any]:
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        row = ResponseTemplateSQL(
            organization_id=organization_id,
            name=clean_plain_text(name, max_len=200),
            category=clean_plain_text(category, max_len=64),
            tone=clean_plain_text(tone or "", max_len=32) or None,
            body=clean_rich_text(body),
            description=clean_plain_text(description or "", max_len=500) or None,
            variables=list(variables or []),
            created_by_user_id=created_by_user_id,
            created_by_agent=created_by_agent,
            is_default=False,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return to_dict(row)


async def update_template(
    *,
    organization_id: str,
    template_id: str,
    fields: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Update mutable fields. `is_default` is immutable from this API."""
    if not fields:
        return await get_template(organization_id, template_id)

    sanitized: Dict[str, Any] = {}
    if "name" in fields:
        sanitized["name"] = clean_plain_text(fields["name"], max_len=200)
    if "category" in fields:
        sanitized["category"] = clean_plain_text(fields["category"], max_len=64)
    if "tone" in fields:
        sanitized["tone"] = clean_plain_text(fields["tone"] or "", max_len=32) or None
    if "body" in fields:
        sanitized["body"] = clean_rich_text(fields["body"])
    if "description" in fields:
        sanitized["description"] = clean_plain_text(fields["description"] or "", max_len=500) or None
    if "variables" in fields:
        sanitized["variables"] = list(fields["variables"] or [])

    if not sanitized:
        return await get_template(organization_id, template_id)
    sanitized["updated_at"] = datetime.utcnow()

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        await session.execute(
            update(ResponseTemplateSQL)
            .where(
                ResponseTemplateSQL.id == template_id,
                ResponseTemplateSQL.organization_id == organization_id,
                ResponseTemplateSQL.deleted_at.is_(None),
            )
            .values(**sanitized)
        )
        await session.commit()
    return await get_template(organization_id, template_id)


async def soft_delete_template(
    organization_id: str, template_id: str
) -> bool:
    """Soft-delete.  Default seeded templates are protected — they can be
    deleted, but the seeder will recreate them on next read.  That's
    intentional: a tenant who deletes "Acknowledge Inquiry" and wants
    it back just clears its `deleted_at` indirectly via the seeder."""
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        result = await session.execute(
            update(ResponseTemplateSQL)
            .where(
                ResponseTemplateSQL.id == template_id,
                ResponseTemplateSQL.organization_id == organization_id,
                ResponseTemplateSQL.deleted_at.is_(None),
            )
            .values(deleted_at=datetime.utcnow())
        )
        await session.commit()
        return (result.rowcount or 0) > 0


async def increment_usage(organization_id: str, template_id: str) -> None:
    """Bump usage_count when an operator clicks a quick-reply or AI Draft
    selects this template.  Best-effort — failures don't break the reply."""
    if not template_id:
        return
    SessionLocal = get_async_sessionmaker()
    try:
        async with SessionLocal() as session:
            await session.execute(
                update(ResponseTemplateSQL)
                .where(
                    ResponseTemplateSQL.id == template_id,
                    ResponseTemplateSQL.organization_id == organization_id,
                    ResponseTemplateSQL.deleted_at.is_(None),
                )
                .values(usage_count=ResponseTemplateSQL.usage_count + 1)
            )
            await session.commit()
    except Exception as e:  # noqa: BLE001
        logger.warning("template_usage_inc_failed", error=str(e), template_id=template_id)


async def find_best_match_for_category(
    organization_id: str, category: Optional[str]
) -> Optional[Dict[str, Any]]:
    """Pick the most-used template in the matching category.  Used by
    the AI Draft context builder to ground the LLM with a canonical
    structure when an obvious template fits."""
    if not (organization_id and category):
        return None
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        stmt = (
            select(ResponseTemplateSQL)
            .where(
                ResponseTemplateSQL.organization_id == organization_id,
                ResponseTemplateSQL.category == category,
                ResponseTemplateSQL.deleted_at.is_(None),
            )
            .order_by(
                ResponseTemplateSQL.usage_count.desc(),
                ResponseTemplateSQL.is_default.desc(),
                ResponseTemplateSQL.created_at.asc(),
            )
            .limit(1)
        )
        row = (await session.execute(stmt)).scalar_one_or_none()
        return to_dict(row) if row else None
