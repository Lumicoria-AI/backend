"""Help-center article service for the public support portal.

Operators write & publish here; anonymous visitors at
`/portal/{slug}/help` only see published rows.

Auto-slugs are derived from the title.  Slugs are unique per org via
the partial unique index in `init_postgres`; on conflict we suffix `-2`,
`-3`, ... until we find a free one.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import and_, asc, desc, func, select, update

from ...db.postgres import get_async_sessionmaker
from ...db.postgres_models import SupportArticleSQL
from .sanitize import clean_plain_text, clean_rich_text

logger = structlog.get_logger(__name__)


_SLUG_CHARS = re.compile(r"[^a-z0-9]+")


def slugify(value: str) -> str:
    """Title → URL-safe slug (lowercase, hyphens, trim).  Collapse runs
    of non-alphanumeric chars to a single hyphen."""
    if not value:
        return "article"
    s = _SLUG_CHARS.sub("-", value.lower()).strip("-")
    return (s or "article")[:100]


def to_dict(row: SupportArticleSQL, *, public: bool = False) -> Dict[str, Any]:
    base: Dict[str, Any] = {
        "id": row.id,
        "organization_id": row.organization_id,
        "slug": row.slug,
        "title": row.title,
        "summary": row.summary,
        "body": row.body,
        "category": row.category,
        "tags": list(row.tags or []),
        "published": row.published,
        "featured": row.featured,
        "view_count": row.view_count,
        "helpful_count": row.helpful_count,
        "not_helpful_count": row.not_helpful_count,
        "rag_document_id": row.rag_document_id,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
        "published_at": row.published_at.isoformat() if row.published_at else None,
    }
    if public:
        # Strip operator-only fields before returning to anonymous clients.
        for k in ("organization_id", "rag_document_id", "not_helpful_count"):
            base.pop(k, None)
    return base


async def _resolve_unique_slug(
    organization_id: str, base_slug: str, *, exclude_id: Optional[str] = None
) -> str:
    """Find an unused slug for this org, suffixing -2, -3, ... if needed."""
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        for i in range(1, 50):
            candidate = base_slug if i == 1 else f"{base_slug}-{i}"
            stmt = select(SupportArticleSQL.id).where(
                SupportArticleSQL.organization_id == organization_id,
                SupportArticleSQL.slug == candidate,
                SupportArticleSQL.deleted_at.is_(None),
            )
            if exclude_id:
                stmt = stmt.where(SupportArticleSQL.id != exclude_id)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return candidate
    # Extreme fallback — append a UUID prefix.
    import uuid as _uuid
    return f"{base_slug}-{_uuid.uuid4().hex[:6]}"


# ── Operator-side ──────────────────────────────────────────────────────


async def create_article(
    *,
    organization_id: str,
    title: str,
    body: str,
    summary: Optional[str] = None,
    category: Optional[str] = None,
    tags: Optional[List[str]] = None,
    published: bool = False,
    featured: bool = False,
    created_by_user_id: Optional[str] = None,
) -> Dict[str, Any]:
    base_slug = slugify(title)
    slug = await _resolve_unique_slug(organization_id, base_slug)

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        row = SupportArticleSQL(
            organization_id=organization_id,
            slug=slug,
            title=clean_plain_text(title, max_len=300),
            summary=clean_plain_text(summary or "", max_len=500) or None,
            body=clean_rich_text(body, max_len=50_000),
            category=clean_plain_text(category or "", max_len=64) or None,
            tags=list(tags or []),
            published=bool(published),
            featured=bool(featured),
            created_by_user_id=created_by_user_id,
            published_at=(datetime.utcnow() if published else None),
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return to_dict(row)


async def list_articles_admin(
    organization_id: str,
    *,
    published: Optional[bool] = None,
    category: Optional[str] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    limit = max(1, min(int(limit or 100), 200))
    offset = max(0, int(offset or 0))

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        where = [
            SupportArticleSQL.organization_id == organization_id,
            SupportArticleSQL.deleted_at.is_(None),
        ]
        if published is not None:
            where.append(SupportArticleSQL.published == bool(published))
        if category:
            where.append(SupportArticleSQL.category == clean_plain_text(category, max_len=64))

        rows_stmt = (
            select(SupportArticleSQL)
            .where(*where)
            .order_by(desc(SupportArticleSQL.updated_at))
            .limit(limit)
            .offset(offset)
        )
        count_stmt = select(func.count()).select_from(SupportArticleSQL).where(*where)
        rows = (await session.execute(rows_stmt)).scalars().all()
        total = await session.scalar(count_stmt) or 0
    return {
        "articles": [to_dict(r) for r in rows],
        "total": int(total),
        "limit": limit,
        "offset": offset,
    }


async def get_article_admin(
    organization_id: str, article_id: str
) -> Optional[Dict[str, Any]]:
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(SupportArticleSQL).where(
                    SupportArticleSQL.id == article_id,
                    SupportArticleSQL.organization_id == organization_id,
                    SupportArticleSQL.deleted_at.is_(None),
                )
            )
        ).scalar_one_or_none()
        return to_dict(row) if row else None


async def update_article(
    organization_id: str, article_id: str, fields: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    if not fields:
        return await get_article_admin(organization_id, article_id)

    sanitized: Dict[str, Any] = {}
    if "title" in fields:
        sanitized["title"] = clean_plain_text(fields["title"], max_len=300)
    if "summary" in fields:
        sanitized["summary"] = clean_plain_text(fields["summary"] or "", max_len=500) or None
    if "body" in fields:
        sanitized["body"] = clean_rich_text(fields["body"], max_len=50_000)
    if "category" in fields:
        sanitized["category"] = clean_plain_text(fields["category"] or "", max_len=64) or None
    if "tags" in fields:
        sanitized["tags"] = list(fields["tags"] or [])
    if "featured" in fields:
        sanitized["featured"] = bool(fields["featured"])
    if "published" in fields:
        new_pub = bool(fields["published"])
        sanitized["published"] = new_pub
        if new_pub:
            # Set published_at on first publish; preserve on subsequent updates.
            existing = await get_article_admin(organization_id, article_id)
            if existing and not existing.get("published_at"):
                sanitized["published_at"] = datetime.utcnow()
        else:
            sanitized["published_at"] = None

    # Slug change driven by title — only update slug if title changed AND
    # the caller didn't pass a `slug` themselves.
    if "slug" in fields and fields["slug"]:
        from re import match
        candidate = slugify(str(fields["slug"]))
        sanitized["slug"] = await _resolve_unique_slug(
            organization_id, candidate, exclude_id=article_id,
        )

    if not sanitized:
        return await get_article_admin(organization_id, article_id)
    sanitized["updated_at"] = datetime.utcnow()

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        await session.execute(
            update(SupportArticleSQL)
            .where(
                SupportArticleSQL.id == article_id,
                SupportArticleSQL.organization_id == organization_id,
                SupportArticleSQL.deleted_at.is_(None),
            )
            .values(**sanitized)
        )
        await session.commit()
    return await get_article_admin(organization_id, article_id)


async def soft_delete_article(organization_id: str, article_id: str) -> bool:
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        result = await session.execute(
            update(SupportArticleSQL)
            .where(
                SupportArticleSQL.id == article_id,
                SupportArticleSQL.organization_id == organization_id,
                SupportArticleSQL.deleted_at.is_(None),
            )
            .values(deleted_at=datetime.utcnow(), updated_at=datetime.utcnow())
        )
        await session.commit()
        return (result.rowcount or 0) > 0


# ── Public-side ────────────────────────────────────────────────────────


async def list_articles_public(
    organization_id: str,
    *,
    category: Optional[str] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """List published articles for the help center."""
    limit = max(1, min(int(limit or 50), 100))
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        where = [
            SupportArticleSQL.organization_id == organization_id,
            SupportArticleSQL.deleted_at.is_(None),
            SupportArticleSQL.published.is_(True),
        ]
        if category:
            where.append(SupportArticleSQL.category == clean_plain_text(category, max_len=64))
        rows = (
            await session.execute(
                select(SupportArticleSQL)
                .where(*where)
                .order_by(
                    desc(SupportArticleSQL.featured),
                    desc(SupportArticleSQL.updated_at),
                )
                .limit(limit)
            )
        ).scalars().all()
        return [to_dict(r, public=True) for r in rows]


async def get_article_public(
    organization_id: str, slug: str
) -> Optional[Dict[str, Any]]:
    if not slug:
        return None
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(SupportArticleSQL).where(
                    SupportArticleSQL.organization_id == organization_id,
                    SupportArticleSQL.slug == slug,
                    SupportArticleSQL.deleted_at.is_(None),
                    SupportArticleSQL.published.is_(True),
                )
            )
        ).scalar_one_or_none()
        if not row:
            return None
        # Bump view_count fire-and-forget (best-effort).
        try:
            await session.execute(
                update(SupportArticleSQL)
                .where(SupportArticleSQL.id == row.id)
                .values(view_count=SupportArticleSQL.view_count + 1)
            )
            await session.commit()
        except Exception:  # noqa: BLE001
            pass
        return to_dict(row, public=True)


async def vote_article_public(
    organization_id: str, slug: str, helpful: bool
) -> bool:
    """Thumbs-up / thumbs-down counter on the public article."""
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        col = SupportArticleSQL.helpful_count if helpful else SupportArticleSQL.not_helpful_count
        result = await session.execute(
            update(SupportArticleSQL)
            .where(
                SupportArticleSQL.organization_id == organization_id,
                SupportArticleSQL.slug == slug,
                SupportArticleSQL.published.is_(True),
                SupportArticleSQL.deleted_at.is_(None),
            )
            .values({col: col + 1})
        )
        await session.commit()
        return (result.rowcount or 0) > 0


# ── RAG sync helper ────────────────────────────────────────────────────


async def link_rag_document(
    organization_id: str, article_id: str, rag_document_id: str
) -> None:
    """Pin the RAG document id back onto the article so future edits
    can sync to the same document instead of duplicating."""
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        await session.execute(
            update(SupportArticleSQL)
            .where(
                SupportArticleSQL.id == article_id,
                SupportArticleSQL.organization_id == organization_id,
            )
            .values(rag_document_id=rag_document_id, updated_at=datetime.utcnow())
        )
        await session.commit()
