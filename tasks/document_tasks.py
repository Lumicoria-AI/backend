"""Celery tasks for RAG document ingestion.

Three entry points — one per upload modality:
    ingest_file(document_id, s3_key, ...)   # binary upload, already in S3
    ingest_url(document_id, url, s3_key, ...)  # URL snapshot
    ingest_text(document_id, text, s3_key, ...)  # direct text / notes

Each task is a sync Celery shell around an async pipeline. The task:
  1. Publishes progress events (start / download / parse / chunk / embed / store).
  2. Calls `document_processor.process_file|process_text|process_url`.
  3. Updates the `rag_documents` registry row with chunk_count + status.
  4. On exception: marks status=error and emits a final progress event.

All tasks run with `task_acks_late=True` so a crashed worker re-queues the
job, and `task_reject_on_worker_lost=True` so an OOM kill doesn't leave a
document stuck in "processing".
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog

from .celery_app import celery_app

logger = structlog.get_logger(__name__)


# ── Per-worker-process event loop + lazy service init ─────────────────
#
# Celery worker processes don't run FastAPI's lifespan, so the storage
# service singleton never gets `initialize()`'d.  Also, creating a fresh
# event loop per task invocation breaks async SQLAlchemy + asyncpg
# because their pools get bound to whichever loop first touched them —
# subsequent tasks from a new loop see "Future attached to a different
# loop" and "Event loop is closed" errors.
#
# Fix: one long-lived event loop per worker process (reused across task
# invocations), and initialize services lazily on first task in that
# process.  The event loop is torn down on worker_process_shutdown.

_loop: Optional[asyncio.AbstractEventLoop] = None
_services_initialized = False


def _get_loop() -> asyncio.AbstractEventLoop:
    """Return this worker process's persistent event loop, creating it
    on first use.  Reusing the same loop keeps DB + HTTP connection pools
    valid across successive tasks."""
    global _loop
    if _loop is None or _loop.is_closed():
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
    return _loop


async def _ensure_services() -> None:
    """Run the one-time async initialization that FastAPI's lifespan
    normally performs.  Idempotent — safe to call at the start of every
    task."""
    global _services_initialized
    if _services_initialized:
        return
    try:
        from backend.services.storage_service import storage_service
        await storage_service.initialize()
    except Exception as e:
        logger.warning("worker_storage_init_failed", error=str(e))
    _services_initialized = True


def _run_async(coro):
    """Run an async coroutine from a sync Celery task on the worker's
    persistent event loop, after ensuring singletons are initialized."""
    async def _wrapped():
        await _ensure_services()
        return await coro
    return _get_loop().run_until_complete(_wrapped())


# Tear down the event loop cleanly when the worker process exits.
# Registered via Celery signal so it runs in the child process.
try:
    from celery.signals import worker_process_shutdown

    @worker_process_shutdown.connect
    def _close_worker_loop(**_kwargs):
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
except ImportError:
    # celery not importable at module load (e.g. running as a plain script) —
    # signal registration is best-effort.
    pass


async def _mark_error(
    document_id: str,
    error: str,
    code: str = "ingest_failed",
    source: Optional[str] = None,
    mime: Optional[str] = None,
) -> None:
    from backend.services import rag_document_registry as rag_registry
    from backend.services.ingest import progress as _progress
    from backend.services.ingest import metrics as _metrics

    # Structured errors: "<code>: <detail>".  Frontend parses the prefix to
    # present a user-facing category (extraction_failed / embedding_quota / ...).
    message = f"{code}: {error}" if code and not error.startswith(f"{code}:") else error
    try:
        await rag_registry.update(
            document_id, status="error", error_message=message
        )
    except Exception as e:
        logger.error("rag_registry_update_failed", document_id=document_id, error=str(e))
    _progress.stage(document_id, "error", message=message, code=code)
    _metrics.record_status(source, mime, "error")


async def _mark_ready(
    document_id: str,
    chunk_count: int,
    source: Optional[str] = None,
    mime: Optional[str] = None,
) -> None:
    from backend.services import rag_document_registry as rag_registry
    from backend.services.ingest import metrics as _metrics
    try:
        await rag_registry.update(
            document_id, chunk_count=chunk_count, status="ready"
        )
    except Exception as e:
        logger.error("rag_registry_update_failed", document_id=document_id, error=str(e))
    _metrics.record_chunks(source, mime, chunk_count)
    _metrics.record_status(source, mime, "ready")


async def _is_cancelled(document_id: str) -> bool:
    """Check registry status for a cancel request.  Tasks call this between
    stages so users can abort long-running ingests."""
    from backend.services import rag_document_registry as rag_registry
    try:
        doc = await rag_registry.get(document_id)
        return bool(doc and doc.get("status") == "cancelled")
    except Exception:
        return False


# ── File ingestion ─────────────────────────────────────────────────────


@celery_app.task(
    bind=True,
    name="backend.tasks.document_tasks.ingest_file",
    max_retries=2,
    default_retry_delay=30,
)
def ingest_file(
    self,
    document_id: str,
    user_id: str,
    s3_key: str,
    stored_filename: str,
    original_filename: str,
    content_type: str,
    organization_id: Optional[str] = None,
    title: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Download an uploaded file from S3 → parse → chunk → embed → store."""
    from backend.services.storage_service import storage_service
    from backend.services.document_processor import document_processor
    from backend.services.ingest import progress as _progress
    from datetime import datetime

    tags = tags or []

    async def _run() -> Dict[str, Any]:
        if await _is_cancelled(document_id):
            _progress.stage(document_id, "cancelled")
            return {"status": "cancelled", "document_id": document_id}

        _progress.stage(document_id, "downloading")
        try:
            content = await storage_service.download_file(s3_key)
        except Exception as e:
            await _mark_error(document_id, str(e), code="storage_download_failed")
            raise RuntimeError(f"S3 download failed: {e}") from e

        tmp_dir = Path(tempfile.gettempdir()) / "rag_ingest"
        tmp_dir.mkdir(exist_ok=True, parents=True)
        tmp_path = tmp_dir / stored_filename
        tmp_path.write_bytes(content)

        metadata: Dict[str, Any] = {
            "document_id": document_id,
            "user_id": user_id,
            "source": "upload",
            "s3_key": s3_key,
            "filename": stored_filename,
            "original_filename": original_filename,
            "title": title or original_filename,
            "mime_type": content_type,
            "tags": tags,
            "created_at": datetime.utcnow().isoformat(),
        }
        if organization_id:
            metadata["organization_id"] = organization_id

        if await _is_cancelled(document_id):
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            _progress.stage(document_id, "cancelled")
            return {"status": "cancelled", "document_id": document_id}

        # Generate preview artifact (DOCX/PPTX/XLSX) alongside main ingest.
        # Best-effort — failure never blocks chunking.
        try:
            from backend.services.ingest.preview import render_preview, preview_artifact_key
            rendered = render_preview(str(tmp_path), content_type)
            if rendered is not None:
                artifact_bytes, artifact_ct = rendered
                artifact_key = preview_artifact_key(s3_key, content_type)
                if artifact_key:
                    await storage_service.upload_file(artifact_bytes, artifact_key, artifact_ct)
                    logger.info("preview_artifact_stored",
                                document_id=document_id, key=artifact_key,
                                bytes=len(artifact_bytes))
        except Exception as e:
            logger.warning("preview_artifact_failed",
                           document_id=document_id, error=str(e))

        _progress.stage(document_id, "parsing")
        try:
            result = await document_processor.process_file(str(tmp_path), metadata)
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

        if result.status == "error":
            await _mark_error(document_id, result.error or "processing_failed",
                              code="extraction_failed")
            return {"status": "error", "document_id": document_id, "error": result.error}

        await _mark_ready(document_id, result.chunk_count,
                          source=metadata.get("source"),
                          mime=metadata.get("mime_type"))
        return {"status": "ready", "document_id": document_id, "chunks": result.chunk_count}

    try:
        return _run_async(_run())
    except Exception as e:
        logger.error("ingest_file_failed", document_id=document_id, error=str(e))
        try:
            _run_async(_mark_error(document_id, str(e)))
        except Exception:
            pass
        # Retry transient failures (network, S3 hiccups). Structural errors
        # re-raise on final attempt.
        if self.request.retries < (self.max_retries or 0):
            raise self.retry(exc=e, countdown=30 * (2 ** self.request.retries))
        return {"status": "error", "document_id": document_id, "error": str(e)}


# ── URL ingestion ──────────────────────────────────────────────────────


@celery_app.task(
    bind=True,
    name="backend.tasks.document_tasks.ingest_url",
    max_retries=2,
    default_retry_delay=30,
)
def ingest_url(
    self,
    document_id: str,
    user_id: str,
    url: str,
    s3_key: str,
    stored_filename: str,
    organization_id: Optional[str] = None,
    title: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Fetch URL → snapshot to S3 → parse (trafilatura) → chunk → embed → store."""
    from backend.services.storage_service import storage_service
    from backend.services.document_processor import document_processor
    from backend.services import rag_document_registry as rag_registry
    from backend.services.ingest import progress as _progress
    from datetime import datetime
    import httpx

    tags = tags or []

    async def _run() -> Dict[str, Any]:
        if await _is_cancelled(document_id):
            _progress.stage(document_id, "cancelled")
            return {"status": "cancelled", "document_id": document_id}

        _progress.stage(document_id, "fetching")
        try:
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; Lumicoria.ai/1.0)"},
                )
                resp.raise_for_status()
                html = resp.text
        except Exception as e:
            await _mark_error(document_id, str(e), code="url_fetch_failed")
            return {"status": "error", "document_id": document_id, "error": str(e)}

        html_bytes = html.encode("utf-8")
        try:
            await storage_service.upload_file(html_bytes, s3_key, "text/html; charset=utf-8")
        except Exception as e:
            logger.warning("url_snapshot_upload_failed", error=str(e), s3_key=s3_key)

        try:
            await rag_registry.update(document_id, size_bytes=len(html_bytes))
        except Exception:
            pass

        metadata: Dict[str, Any] = {
            "document_id": document_id,
            "user_id": user_id,
            "source": "web",
            "url": url,
            "source_url": url,
            "s3_key": s3_key,
            "filename": stored_filename,
            "title": title or url,
            "mime_type": "text/html",
            "tags": tags,
            "created_at": datetime.utcnow().isoformat(),
        }
        if organization_id:
            metadata["organization_id"] = organization_id

        if await _is_cancelled(document_id):
            _progress.stage(document_id, "cancelled")
            return {"status": "cancelled", "document_id": document_id}

        _progress.stage(document_id, "parsing")
        result = await document_processor.process_url(url, metadata)

        if result.status == "error":
            await _mark_error(document_id, result.error or "processing_failed",
                              code="extraction_failed")
            return {"status": "error", "document_id": document_id, "error": result.error}

        await _mark_ready(document_id, result.chunk_count,
                          source=metadata.get("source"),
                          mime=metadata.get("mime_type"))
        return {"status": "ready", "document_id": document_id, "chunks": result.chunk_count}

    try:
        return _run_async(_run())
    except Exception as e:
        logger.error("ingest_url_failed", document_id=document_id, error=str(e))
        try:
            _run_async(_mark_error(document_id, str(e)))
        except Exception:
            pass
        if self.request.retries < (self.max_retries or 0):
            raise self.retry(exc=e, countdown=30 * (2 ** self.request.retries))
        return {"status": "error", "document_id": document_id, "error": str(e)}


# ── Text / note ingestion ──────────────────────────────────────────────


@celery_app.task(
    bind=True,
    name="backend.tasks.document_tasks.ingest_text",
    max_retries=1,
    default_retry_delay=10,
)
def ingest_text(
    self,
    document_id: str,
    user_id: str,
    text: str,
    s3_key: str,
    stored_filename: str,
    source: str = "manual_entry",
    organization_id: Optional[str] = None,
    title: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Parse a text/markdown note and store its chunks."""
    from backend.services.document_processor import document_processor
    from backend.services.ingest import progress as _progress
    from datetime import datetime

    tags = tags or []

    async def _run() -> Dict[str, Any]:
        if await _is_cancelled(document_id):
            _progress.stage(document_id, "cancelled")
            return {"status": "cancelled", "document_id": document_id}

        metadata: Dict[str, Any] = {
            "document_id": document_id,
            "user_id": user_id,
            "source": source,
            "s3_key": s3_key,
            "filename": stored_filename,
            "title": title or f"Note {document_id[:8]}",
            "mime_type": "text/markdown",
            "tags": tags,
            "created_at": datetime.utcnow().isoformat(),
        }
        if organization_id:
            metadata["organization_id"] = organization_id

        _progress.stage(document_id, "parsing")
        result = await document_processor.process_text(text, metadata)

        if result.status == "error":
            await _mark_error(document_id, result.error or "processing_failed",
                              code="extraction_failed")
            return {"status": "error", "document_id": document_id, "error": result.error}

        await _mark_ready(document_id, result.chunk_count,
                          source=metadata.get("source"),
                          mime=metadata.get("mime_type"))
        return {"status": "ready", "document_id": document_id, "chunks": result.chunk_count}

    try:
        return _run_async(_run())
    except Exception as e:
        logger.error("ingest_text_failed", document_id=document_id, error=str(e))
        try:
            _run_async(_mark_error(document_id, str(e)))
        except Exception:
            pass
        return {"status": "error", "document_id": document_id, "error": str(e)}
