"""
PostgreSQL / SQLAlchemy session management for Lumicoria.ai

This module provides both async and sync SQLAlchemy session factories and
startup/shutdown helpers. It is optional and only used when POSTGRES_ENABLED
or SQLALCHEMY_DATABASE_URI is configured.
"""

from __future__ import annotations

from typing import AsyncGenerator, Generator, Optional
from urllib.parse import urlparse
import asyncio
import structlog

from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import sessionmaker

from backend.core.config import settings
from backend.db.base_class import Base

logger = structlog.get_logger(__name__)

_async_engine = None
_async_sessionmaker: Optional[async_sessionmaker[AsyncSession]] = None
_sync_engine = None
_sync_sessionmaker: Optional[sessionmaker] = None


def _build_sync_uri() -> Optional[str]:
    return settings.SQLALCHEMY_DATABASE_URI


def _build_async_uri() -> Optional[str]:
    uri = settings.SQLALCHEMY_DATABASE_URI
    if not uri:
        return None
    parsed = urlparse(uri)
    if parsed.scheme.startswith("postgresql+asyncpg"):
        return uri
    if parsed.scheme.startswith("postgresql"):
        return uri.replace("postgresql://", "postgresql+asyncpg://", 1)
    return uri


def _get_sync_engine():
    global _sync_engine
    if _sync_engine is None:
        uri = _build_sync_uri()
        if not uri:
            raise RuntimeError("SQLALCHEMY_DATABASE_URI is not configured")
        _sync_engine = create_engine(
            uri,
            pool_pre_ping=True,
            echo=settings.SQLALCHEMY_ECHO,
            pool_size=settings.SQLALCHEMY_POOL_SIZE,
            max_overflow=settings.SQLALCHEMY_MAX_OVERFLOW,
        )
    return _sync_engine


def _get_async_engine():
    global _async_engine
    if _async_engine is None:
        uri = _build_async_uri()
        if not uri:
            raise RuntimeError("SQLALCHEMY_DATABASE_URI is not configured")
        _async_engine = create_async_engine(
            uri,
            pool_pre_ping=True,
            echo=settings.SQLALCHEMY_ECHO,
            pool_size=settings.SQLALCHEMY_POOL_SIZE,
            max_overflow=settings.SQLALCHEMY_MAX_OVERFLOW,
        )
    return _async_engine


def get_sync_sessionmaker() -> sessionmaker:
    global _sync_sessionmaker
    if _sync_sessionmaker is None:
        _sync_sessionmaker = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=_get_sync_engine(),
        )
    return _sync_sessionmaker


def get_async_sessionmaker() -> async_sessionmaker[AsyncSession]:
    global _async_sessionmaker
    if _async_sessionmaker is None:
        _async_sessionmaker = async_sessionmaker(
            bind=_get_async_engine(),
            autoflush=False,
            expire_on_commit=False,
        )
    return _async_sessionmaker


def get_db() -> Generator:
    """Sync DB dependency (use only in sync paths)."""
    SessionLocal = get_sync_sessionmaker()
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Async DB dependency for FastAPI routes."""
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        yield session


async def get_optional_async_db() -> AsyncGenerator[Optional[AsyncSession], None]:
    """Optional async DB dependency; yields None when Postgres is disabled."""
    if not (settings.POSTGRES_ENABLED and settings.SQLALCHEMY_DATABASE_URI):
        yield None
        return
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        yield session


async def init_postgres() -> None:
    """Initialize Postgres connection, validate connectivity, and create tables."""
    if not (settings.POSTGRES_ENABLED and settings.SQLALCHEMY_DATABASE_URI):
        logger.info("Postgres disabled or not configured; skipping init")
        return
    try:
        engine = _get_async_engine()
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        logger.info("Postgres connection verified")

        # Auto-create tables from SQLAlchemy models if they don't exist
        # Import models so Base.metadata knows about them
        import backend.db.postgres_models  # noqa: F401
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("Postgres tables created/verified successfully")

        # Lightweight in-place schema patches for columns added after a table
        # was first created.  `create_all` never alters existing tables, so we
        # apply idempotent `ADD COLUMN IF NOT EXISTS` statements here.
        async with engine.begin() as conn:
            await conn.execute(text(
                "ALTER TABLE rag_documents "
                "ADD COLUMN IF NOT EXISTS conversation_id VARCHAR(64)"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_rag_documents_conversation_id "
                "ON rag_documents (conversation_id)"
            ))
            # Widen primary key to fit prefixed IDs like "chat_{uuid}" (41 chars).
            # ALTER COLUMN TYPE is a no-op if the column is already wide enough.
            await conn.execute(text(
                "ALTER TABLE rag_documents "
                "ALTER COLUMN id TYPE VARCHAR(64)"
            ))
            # Dedup columns + indexes.
            await conn.execute(text(
                "ALTER TABLE rag_documents "
                "ADD COLUMN IF NOT EXISTS content_sha256 VARCHAR(64)"
            ))
            await conn.execute(text(
                "ALTER TABLE rag_documents "
                "ADD COLUMN IF NOT EXISTS aliased_document_id VARCHAR(64)"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_rag_documents_content_sha256 "
                "ON rag_documents (user_id, content_sha256)"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_rag_documents_aliased_document_id "
                "ON rag_documents (aliased_document_id)"
            ))

            # ── Customer service: composite indexes on the new tables ─
            # `Base.metadata.create_all` made the tables; these indexes
            # accelerate the most common operator queries.
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_support_tickets_org_status "
                "ON support_tickets (organization_id, status) "
                "WHERE deleted_at IS NULL"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_support_tickets_org_created "
                "ON support_tickets (organization_id, created_at DESC) "
                "WHERE deleted_at IS NULL"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_ticket_replies_ticket_created "
                "ON ticket_replies (ticket_id, created_at)"
            ))
            await conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_response_templates_org_name "
                "ON response_templates (organization_id, name) "
                "WHERE deleted_at IS NULL"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_response_templates_org_category "
                "ON response_templates (organization_id, category) "
                "WHERE deleted_at IS NULL"
            ))

            # ── Support help-center articles ─────────────────────────
            await conn.execute(text(
                "CREATE UNIQUE INDEX IF NOT EXISTS uq_support_articles_org_slug "
                "ON support_articles (organization_id, slug) "
                "WHERE deleted_at IS NULL"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_support_articles_published "
                "ON support_articles (organization_id, published, featured DESC, updated_at DESC) "
                "WHERE deleted_at IS NULL"
            ))

            # ── Data analysis runs ───────────────────────────────────
            await conn.execute(text(
                "ALTER TABLE data_analysis_runs "
                "ADD COLUMN IF NOT EXISTS statistical_results JSONB"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_data_analysis_runs_org_status "
                "ON data_analysis_runs (organization_id, status) "
                "WHERE deleted_at IS NULL"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_data_analysis_runs_org_created "
                "ON data_analysis_runs (organization_id, created_at DESC) "
                "WHERE deleted_at IS NULL"
            ))
            await conn.execute(text(
                "CREATE INDEX IF NOT EXISTS ix_data_analysis_runs_user_created "
                "ON data_analysis_runs (user_id, created_at DESC) "
                "WHERE deleted_at IS NULL"
            ))
        logger.info("Postgres in-place schema patches applied")
    except Exception as e:
        logger.error("Failed to initialize Postgres", error=str(e))
        raise


async def close_postgres() -> None:
    """Dispose Postgres engines."""
    global _async_engine, _sync_engine
    if _async_engine is not None:
        await _async_engine.dispose()
        _async_engine = None
    if _sync_engine is not None:
        _sync_engine.dispose()
        _sync_engine = None
    logger.info("Postgres connections closed")


async def check_postgres() -> bool:
    """Health check for Postgres."""
    if not (settings.POSTGRES_ENABLED and settings.SQLALCHEMY_DATABASE_URI):
        return False
    try:
        engine = _get_async_engine()
        async with engine.begin() as conn:
            await conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
