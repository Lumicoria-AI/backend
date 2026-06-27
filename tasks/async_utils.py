"""Shared asyncio helpers for Celery worker processes.

Celery prefork workers are long-lived processes, while Motor / asyncpg /
HTTP clients bind internal futures and pools to the event loop that first
touches them.  Mixing ``asyncio.run()`` with module-level async clients
leaves cached clients pointing at closed loops and causes intermittent
``Event loop is closed`` failures.

Use one persistent loop per worker process for all async Celery tasks, and
reset loop-bound clients whenever a new loop is created.
"""

from __future__ import annotations

import asyncio
from typing import Any, Awaitable, Optional, TypeVar

import structlog

logger = structlog.get_logger(__name__)

T = TypeVar("T")

_loop: Optional[asyncio.AbstractEventLoop] = None
_services_initialized = False


def _reset_loop_bound_clients() -> None:
    """Drop process-wide async clients that may be bound to an old loop."""
    try:
        from backend.db.mongodb.mongodb import MongoDB

        MongoDB.reset_for_new_loop()
    except Exception:
        pass


def get_worker_loop() -> asyncio.AbstractEventLoop:
    """Return the persistent event loop for this Celery worker process."""
    global _loop, _services_initialized
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        _reset_loop_bound_clients()
        _services_initialized = False
    return _loop


async def ensure_worker_services() -> None:
    """Warm process-level services used by RAG ingest.

    Safe to call at the start of every RAG task; the expensive pieces are
    guarded by process-wide singletons.
    """
    global _services_initialized
    if _services_initialized:
        return

    try:
        from backend.services.storage_service import storage_service

        await storage_service.initialize()
    except Exception as exc:
        logger.warning("worker_storage_init_failed", error=str(exc))

    try:
        from backend.services.document_processor import document_processor

        await document_processor.initialize()
    except Exception as exc:
        logger.warning("worker_document_processor_init_failed", error=str(exc))

    try:
        from backend.core.config import settings

        if settings.db.VECTOR_STORE_ENABLED:
            from backend.db.vector_stores import get_vector_store

            vector_store = get_vector_store()
            if getattr(vector_store, "client", None) is None and hasattr(vector_store, "connect"):
                await vector_store.connect()
    except Exception as exc:
        logger.warning("worker_vector_store_init_failed", error=str(exc))

    try:
        from backend.core.config import settings

        embedding_provider = (
            getattr(settings, "DEFAULT_EMBEDDING_PROVIDER", None)
            or getattr(settings, "DEFAULT_LLM_PROVIDER", None)
        )
        if (
            embedding_provider == "local"
            and getattr(settings, "LOCAL_EMBEDDING_WARMUP_ON_STARTUP", True)
        ):
            from backend.ai_models import get_embedding_client

            client = get_embedding_client(provider="local")
            if hasattr(client, "warmup"):
                await client.warmup()
    except Exception as exc:
        logger.warning("worker_embedding_warmup_failed", error=str(exc))

    _services_initialized = True


def run_worker_coro(coro: Awaitable[T], *, ensure_services: bool = False) -> T:
    """Run ``coro`` on this worker process's persistent event loop."""

    async def _wrapped() -> T:
        if ensure_services:
            await ensure_worker_services()
        return await coro

    return get_worker_loop().run_until_complete(_wrapped())


def shutdown_worker_loop() -> None:
    """Close the persistent worker loop during Celery process shutdown."""
    global _loop, _services_initialized
    if _loop is not None and not _loop.is_closed():
        try:
            _loop.run_until_complete(_loop.shutdown_asyncgens())
        except Exception:
            pass
        try:
            _loop.close()
        except Exception:
            pass
    _loop = None
    _services_initialized = False
    _reset_loop_bound_clients()


try:
    from celery.signals import worker_process_shutdown

    @worker_process_shutdown.connect
    def _close_worker_loop(**_kwargs: Any) -> None:
        shutdown_worker_loop()
except ImportError:
    pass
