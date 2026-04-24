"""
RAG Document Registry — thin CRUD layer over RAGDocumentSQL.

The registry is the authoritative source of truth for documents ingested
via the RAG / chat pipeline.  Chunks + embeddings are still stored in
Weaviate; the file itself lives in MinIO (+ R2).  This module keeps the
`document_id ↔ s3_key` mapping durable and queryable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import func, or_, select, update as sql_update

from ..db.postgres import get_async_sessionmaker
from ..db.postgres_models import RAGDocumentSQL

logger = structlog.get_logger(__name__)


def _to_dict(row: RAGDocumentSQL) -> Dict[str, Any]:
    return {
        "document_id": row.id,
        "user_id": row.user_id,
        "organization_id": row.organization_id,
        "s3_key": row.s3_key,
        "filename": row.filename,
        "original_filename": row.original_filename,
        "title": row.title,
        "mime_type": row.mime_type,
        "source": row.source,
        "source_url": row.source_url,
        "size_bytes": row.size_bytes,
        "chunk_count": row.chunk_count,
        "status": row.status,
        "error_message": row.error_message,
        "tags": list(row.tags or []),
        "content_sha256": getattr(row, "content_sha256", None),
        "aliased_document_id": getattr(row, "aliased_document_id", None),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


async def create(
    *,
    document_id: str,
    user_id: str,
    s3_key: str,
    filename: str,
    organization_id: Optional[str] = None,
    original_filename: Optional[str] = None,
    title: Optional[str] = None,
    mime_type: Optional[str] = None,
    source: str = "upload",
    source_url: Optional[str] = None,
    conversation_id: Optional[str] = None,
    size_bytes: int = 0,
    tags: Optional[List[str]] = None,
    status: str = "processing",
    meta: Optional[Dict[str, Any]] = None,
    content_sha256: Optional[str] = None,
    aliased_document_id: Optional[str] = None,
    chunk_count: int = 0,
) -> str:
    """Insert a new RAG document row and return the document_id."""
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        row = RAGDocumentSQL(
            id=document_id,
            user_id=user_id,
            organization_id=organization_id,
            s3_key=s3_key,
            filename=filename,
            original_filename=original_filename,
            title=title,
            mime_type=mime_type,
            source=source,
            source_url=source_url,
            conversation_id=conversation_id,
            size_bytes=size_bytes,
            chunk_count=chunk_count,
            tags=tags or [],
            status=status,
            meta=meta or {},
            content_sha256=content_sha256,
            aliased_document_id=aliased_document_id,
        )
        session.add(row)
        await session.commit()
    return document_id


async def find_by_content_sha256(
    user_id: str,
    content_sha256: str,
) -> Optional[Dict[str, Any]]:
    """Find the canonical (non-aliased, non-deleted, ready) document for a
    given content hash under this user.  Returns None if no prior copy exists.

    Aliased rows are skipped so we always alias back to the original.
    """
    if not content_sha256:
        return None
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        stmt = (
            select(RAGDocumentSQL)
            .where(
                RAGDocumentSQL.user_id == user_id,
                RAGDocumentSQL.content_sha256 == content_sha256,
                RAGDocumentSQL.aliased_document_id.is_(None),
                RAGDocumentSQL.deleted_at.is_(None),
                RAGDocumentSQL.status == "ready",
            )
            .order_by(RAGDocumentSQL.created_at.asc())
            .limit(1)
        )
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        return _to_dict(row) if row else None


async def get(document_id: str, user_id: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """Fetch a single document by id (optionally scoped to a user)."""
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        stmt = select(RAGDocumentSQL).where(
            RAGDocumentSQL.id == document_id,
            RAGDocumentSQL.deleted_at.is_(None),
        )
        if user_id:
            stmt = stmt.where(RAGDocumentSQL.user_id == user_id)
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        return _to_dict(row) if row else None


async def list_documents(
    user_id: str,
    organization_id: Optional[str] = None,
    source_types: Optional[List[str]] = None,
    tags: Optional[List[str]] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    """List a user's documents with optional filters.  Returns dict compatible
    with the legacy Weaviate-backed response shape."""
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        base_where = [
            RAGDocumentSQL.user_id == user_id,
            RAGDocumentSQL.deleted_at.is_(None),
        ]
        if organization_id:
            base_where.append(
                or_(
                    RAGDocumentSQL.organization_id == organization_id,
                    RAGDocumentSQL.organization_id.is_(None),
                )
            )
        if source_types:
            base_where.append(RAGDocumentSQL.source.in_(source_types))
        if tags:
            base_where.append(RAGDocumentSQL.tags.overlap(tags))

        rows_stmt = (
            select(RAGDocumentSQL)
            .where(*base_where)
            .order_by(RAGDocumentSQL.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        count_stmt = select(func.count()).select_from(RAGDocumentSQL).where(*base_where)

        rows_result = await session.execute(rows_stmt)
        rows = rows_result.scalars().all()
        total = await session.scalar(count_stmt) or 0

    documents = [_to_dict(r) for r in rows]
    return {
        "documents": documents,
        "total": total,
        "unique_count": len(documents),
        "limit": limit,
        "offset": offset,
    }


async def update(document_id: str, **fields: Any) -> None:
    """Update arbitrary fields on a document row."""
    if not fields:
        return
    fields["updated_at"] = datetime.utcnow()
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        await session.execute(
            sql_update(RAGDocumentSQL)
            .where(RAGDocumentSQL.id == document_id)
            .values(**fields)
        )
        await session.commit()


async def upsert(
    *,
    document_id: str,
    user_id: str,
    s3_key: str,
    filename: str,
    organization_id: Optional[str] = None,
    original_filename: Optional[str] = None,
    title: Optional[str] = None,
    mime_type: Optional[str] = None,
    source: str = "upload",
    source_url: Optional[str] = None,
    conversation_id: Optional[str] = None,
    size_bytes: int = 0,
    chunk_count: Optional[int] = None,
    status: str = "processing",
    tags: Optional[List[str]] = None,
    meta: Optional[Dict[str, Any]] = None,
) -> str:
    """Insert or update a RAG document row keyed by document_id.

    If the row already exists, only the mutable fields (s3_key, title, status,
    chunk_count, size_bytes, tags, meta, updated_at) are refreshed.  Immutable
    identity fields (user_id, source, conversation_id) are left alone.
    """
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        stmt = select(RAGDocumentSQL).where(RAGDocumentSQL.id == document_id)
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()

        if row is None:
            row = RAGDocumentSQL(
                id=document_id,
                user_id=user_id,
                organization_id=organization_id,
                s3_key=s3_key,
                filename=filename,
                original_filename=original_filename,
                title=title,
                mime_type=mime_type,
                source=source,
                source_url=source_url,
                conversation_id=conversation_id,
                size_bytes=size_bytes,
                chunk_count=chunk_count or 0,
                status=status,
                tags=tags or [],
                meta=meta or {},
            )
            session.add(row)
        else:
            row.s3_key = s3_key
            row.filename = filename
            if original_filename is not None:
                row.original_filename = original_filename
            if title is not None:
                row.title = title
            if mime_type is not None:
                row.mime_type = mime_type
            if source_url is not None:
                row.source_url = source_url
            if organization_id is not None:
                row.organization_id = organization_id
            row.size_bytes = size_bytes
            if chunk_count is not None:
                row.chunk_count = chunk_count
            row.status = status
            if tags is not None:
                row.tags = tags
            if meta is not None:
                row.meta = meta
            row.deleted_at = None
            row.updated_at = datetime.utcnow()

        await session.commit()
    return document_id


async def backfill_chat_history_from_weaviate(user_id: str) -> int:
    """Find chat_history documents that live in Weaviate but have no Postgres row,
    and insert minimal registry rows for them.

    This is a self-healing migration for conversations indexed by the old
    `add_chat_context` path (before the Postgres registry existed).  It runs
    on demand when the user queries the Chat History tab, so no separate
    migration script is required.  Returns the number of rows created.
    """
    from ..core.config import settings as _settings  # avoid circular import at module load
    if not _settings.db.VECTOR_STORE_ENABLED:
        return 0

    try:
        from ..db.vector_stores import get_vector_store
        vector_store = get_vector_store()
        chunks = await vector_store.get_documents(
            filters={"user_id": user_id, "source": "chat_history"},
            limit=2000,
        )
    except Exception as e:
        logger.warning("chat_history backfill: Weaviate scan failed", error=str(e))
        return 0

    # Group chunks by document_id
    by_doc: Dict[str, Dict[str, Any]] = {}
    for c in chunks:
        meta = c.get("metadata", {}) or {}
        doc_id = meta.get("document_id")
        if not doc_id or not doc_id.startswith("chat_"):
            continue
        entry = by_doc.setdefault(doc_id, {
            "chunk_count": 0,
            "metadata": meta,
            "first_content": "",
            "earliest_created_at": meta.get("created_at"),
        })
        entry["chunk_count"] += 1
        if not entry["first_content"]:
            entry["first_content"] = c.get("content", "") or ""
        ts = meta.get("created_at")
        if ts and (not entry["earliest_created_at"] or ts < entry["earliest_created_at"]):
            entry["earliest_created_at"] = ts

    if not by_doc:
        return 0

    # Which of these already have a Postgres row?
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        existing_stmt = select(RAGDocumentSQL.id).where(
            RAGDocumentSQL.user_id == user_id,
            RAGDocumentSQL.id.in_(list(by_doc.keys())),
        )
        existing_result = await session.execute(existing_stmt)
        existing_ids = {r[0] for r in existing_result}

    created = 0
    for doc_id, info in by_doc.items():
        if doc_id in existing_ids:
            continue
        conversation_id = doc_id[len("chat_"):]
        meta = info["metadata"]

        # Try to derive a nice title from MongoDB; fall back to first chunk.
        title: Optional[str] = meta.get("title")
        try:
            from ..agents import memory as chat_memory
            conv = await chat_memory.get_full_conversation(conversation_id)
            if conv:
                messages = conv.get("messages", []) or []
                first_user = next(
                    (m.get("content", "") for m in messages if m.get("role") == "user"),
                    "",
                )
                if first_user:
                    turn_count = sum(1 for m in messages if m.get("role") == "user")
                    short = (first_user[:80] + "…") if len(first_user) > 80 else first_user
                    title = f"{short} ({turn_count} turn{'s' if turn_count != 1 else ''})"
        except Exception:
            pass

        if not title:
            snippet = (info["first_content"] or "").strip().replace("\n", " ")
            title = (snippet[:80] + "…") if len(snippet) > 80 else (snippet or f"Conversation {conversation_id[:8]}")

        try:
            await create(
                document_id=doc_id,
                user_id=user_id,
                organization_id=meta.get("organization_id"),
                s3_key=f"rag/{user_id}/{doc_id}.md",  # placeholder — real upload happens on next turn
                filename=f"{doc_id}.md",
                title=title,
                mime_type="text/markdown",
                source="chat_history",
                conversation_id=conversation_id,
                status="ready",  # chunks already exist in Weaviate
                size_bytes=0,
                meta={"backfilled": True},
            )
            created += 1
        except Exception as e:
            logger.warning("chat_history backfill: row create failed", doc_id=doc_id, error=str(e))

    if created:
        logger.info("chat_history backfill complete", user_id=user_id, created=created)
    return created


async def soft_delete(document_id: str, user_id: str) -> Optional[str]:
    """Mark a row as deleted and return its s3_key (for MinIO cleanup)."""
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        stmt = select(RAGDocumentSQL).where(
            RAGDocumentSQL.id == document_id,
            RAGDocumentSQL.user_id == user_id,
            RAGDocumentSQL.deleted_at.is_(None),
        )
        result = await session.execute(stmt)
        row = result.scalar_one_or_none()
        if not row:
            return None
        s3_key = row.s3_key
        row.deleted_at = datetime.utcnow()
        await session.commit()
        return s3_key
