"""
Re-embed every RAG document with the currently-configured embedding
provider (typically 'local' / FastEmbed after switching from Gemini).

Why this is needed
------------------
Embeddings are model-specific: a vector produced by gemini-embedding-001
cannot be meaningfully compared to one produced by BAAI/bge-base-en-v1.5,
even when both are 768-dimensional.  Switching the embedding provider
without re-embedding silently degrades retrieval quality — old chunks
become noise.  This script rebuilds the vector space for every Postgres-
registered document.

What it does (per document)
---------------------------
1. Load the row from `rag_documents` (Postgres).
2. Download the original payload from MinIO using `s3_key`.
3. Delete every chunk belonging to `document_id` from the vector store.
4. Re-invoke the appropriate `document_processor.process_*` method so
   the file is re-chunked and re-embedded with the new provider.
5. Update `rag_documents.chunk_count` + `status` with the new totals.

Concurrency model
-----------------
* An outer `asyncio.Semaphore` caps the number of documents being
  processed in flight — the bottleneck is CPU (ONNX inference) and disk
  I/O, so 4-8 concurrent documents saturates a typical 8-core box.
* Each document embeds its entire chunk list in ONE call to the local
  provider, which fans out across CPU cores via FastEmbed's internal
  batching (`LOCAL_EMBEDDING_BATCH_SIZE`).

Usage
-----
    # Dry run — list what would be reindexed
    python -m backend.scripts.reindex_embeddings --dry-run

    # Full run, all users, 6 documents concurrently
    python -m backend.scripts.reindex_embeddings --concurrency 6

    # Scope to one user
    python -m backend.scripts.reindex_embeddings --user-id abc123

    # Scope to one source type (e.g. only chat_history)
    python -m backend.scripts.reindex_embeddings --source chat_history
"""

from __future__ import annotations

import argparse
import asyncio
import mimetypes
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import select

# Allow the script to run either via `python -m backend.scripts.reindex_embeddings`
# or directly via `python backend/scripts/reindex_embeddings.py`
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.core.config import settings  # noqa: E402
from backend.db.postgres import (  # noqa: E402
    close_postgres,
    get_async_sessionmaker,
    init_postgres,
)
from backend.db.postgres_models import RAGDocumentSQL  # noqa: E402
from backend.db.vector_stores import get_vector_store  # noqa: E402
from backend.services import rag_document_registry as rag_registry  # noqa: E402
from backend.services.document_processor import document_processor  # noqa: E402
from backend.services.storage_service import storage_service  # noqa: E402

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════


async def _fetch_targets(
    user_id: Optional[str],
    source: Optional[str],
    limit: Optional[int],
) -> List[RAGDocumentSQL]:
    """Pull every eligible document row from Postgres."""
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        stmt = select(RAGDocumentSQL).where(RAGDocumentSQL.deleted_at.is_(None))
        if user_id:
            stmt = stmt.where(RAGDocumentSQL.user_id == user_id)
        if source:
            stmt = stmt.where(RAGDocumentSQL.source == source)
        stmt = stmt.order_by(RAGDocumentSQL.created_at.desc())
        if limit:
            stmt = stmt.limit(limit)
        result = await session.execute(stmt)
        return list(result.scalars().all())


def _infer_mime(row: RAGDocumentSQL) -> str:
    if row.mime_type:
        return row.mime_type
    guess, _ = mimetypes.guess_type(row.filename or row.s3_key)
    return guess or "application/octet-stream"


async def _download_from_minio(s3_key: str) -> Optional[bytes]:
    """Fetch the raw bytes behind a rag document; None if the object is gone."""
    try:
        data = await storage_service.download_file(s3_key)
        return data
    except Exception as exc:
        logger.warning("reindex_minio_download_failed", key=s3_key, error=str(exc))
        return None


async def _delete_old_chunks(user_id: str, document_id: str) -> None:
    """Purge every vector for this document so we don't end up with two
    overlapping embedding spaces in the same collection."""
    if not settings.db.VECTOR_STORE_ENABLED:
        return
    try:
        vs = get_vector_store()
        await vs.delete_documents(
            filters={"user_id": user_id, "document_id": document_id}
        )
    except Exception as exc:
        logger.warning(
            "reindex_delete_chunks_failed",
            document_id=document_id,
            error=str(exc),
        )


async def _reprocess(row: RAGDocumentSQL, payload: bytes) -> Dict[str, Any]:
    """Hand the payload back to document_processor so it re-chunks + re-embeds."""
    mime = _infer_mime(row)
    base_metadata = {
        "user_id": row.user_id,
        "organization_id": row.organization_id,
        "source": row.source or "upload",
        "document_id": row.id,
        "title": row.title,
        "mime_type": mime,
        "s3_key": row.s3_key,
        "filename": row.filename,
        "original_filename": row.original_filename,
        "tags": list(row.tags or []),
    }
    if row.source_url:
        base_metadata["url"] = row.source_url
    if row.conversation_id:
        base_metadata["conversation_id"] = row.conversation_id

    # Route by MIME: text-y payloads go through process_text (fast, no disk
    # dance); everything else is written to a tempfile for process_file.
    is_textlike = (
        mime.startswith("text/")
        or mime == "application/json"
        or mime == "application/xml"
    )

    if is_textlike:
        try:
            text = payload.decode("utf-8", errors="replace")
        except Exception:
            text = ""
        result = await document_processor.process_text(text=text, metadata=base_metadata)
    else:
        suffix = Path(row.filename or row.s3_key).suffix or ""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(payload)
            tmp_path = tmp.name
        try:
            result = await document_processor.process_file(
                file_path=tmp_path, metadata=base_metadata
            )
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    return {
        "status": result.status,
        "chunk_count": result.chunk_count,
        "error": result.error,
    }


async def _process_one(
    row: RAGDocumentSQL,
    dry_run: bool,
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    """Run the full delete → re-embed → registry-update pipeline for one doc."""
    async with semaphore:
        t0 = time.perf_counter()
        doc_id = row.id
        logger.info(
            "reindex_document_start",
            document_id=doc_id,
            user_id=row.user_id,
            source=row.source,
            title=row.title,
        )

        if dry_run:
            return {
                "document_id": doc_id,
                "status": "dry_run",
                "chunk_count": row.chunk_count or 0,
            }

        # 1. Download the original payload from MinIO
        payload = await _download_from_minio(row.s3_key)
        if payload is None:
            # Common case: legacy chat_history rows that were backfilled with
            # placeholder s3_keys — nothing on disk yet.  Skip.
            await rag_registry.update(doc_id, status="missing_payload")
            return {
                "document_id": doc_id,
                "status": "skipped",
                "error": "payload missing in MinIO",
            }

        # 2. Delete old Weaviate vectors for this document
        await _delete_old_chunks(row.user_id, doc_id)

        # 3. Mark row as processing so the UI reflects work-in-progress
        await rag_registry.update(doc_id, status="processing", error_message=None)

        # 4. Re-embed via document_processor (one big call to local provider)
        try:
            result = await _reprocess(row, payload)
        except Exception as exc:
            logger.error(
                "reindex_document_error",
                document_id=doc_id,
                error=str(exc),
            )
            await rag_registry.update(
                doc_id,
                status="error",
                error_message=str(exc),
            )
            return {
                "document_id": doc_id,
                "status": "error",
                "error": str(exc),
            }

        # 5. Update registry with new chunk_count + final status
        await rag_registry.update(
            doc_id,
            chunk_count=result.get("chunk_count", 0),
            status="ready" if result.get("status") == "success" else "error",
            error_message=result.get("error"),
            size_bytes=len(payload),
        )

        elapsed = time.perf_counter() - t0
        logger.info(
            "reindex_document_done",
            document_id=doc_id,
            chunks=result.get("chunk_count"),
            status=result.get("status"),
            seconds=round(elapsed, 2),
        )
        return {
            "document_id": doc_id,
            "status": result.get("status"),
            "chunk_count": result.get("chunk_count", 0),
            "seconds": round(elapsed, 2),
        }


# ═══════════════════════════════════════════════════════════════════════════
# Entrypoint
# ═══════════════════════════════════════════════════════════════════════════


async def _main(
    user_id: Optional[str],
    source: Optional[str],
    limit: Optional[int],
    concurrency: int,
    dry_run: bool,
) -> int:
    provider = (
        settings.DEFAULT_EMBEDDING_PROVIDER
        or settings.DEFAULT_LLM_PROVIDER
    )
    print(f"[reindex] embedding provider = {provider}")
    print(f"[reindex] vector store dim    = {settings.db.VECTOR_STORE_DIMENSION}")
    print(f"[reindex] concurrency         = {concurrency}")
    print(f"[reindex] dry run             = {dry_run}")

    await init_postgres()

    # Warm the local model once up front so individual docs don't race on the
    # first-call cold start.
    if provider == "local":
        try:
            from backend.ai_models import get_embedding_client
            client = get_embedding_client(provider="local")
            if hasattr(client, "warmup"):
                await client.warmup()
        except Exception as exc:
            print(f"[reindex] warmup failed: {exc}")

    targets = await _fetch_targets(user_id=user_id, source=source, limit=limit)
    print(f"[reindex] {len(targets)} document(s) queued")
    if not targets:
        await close_postgres()
        return 0

    semaphore = asyncio.Semaphore(concurrency)
    t0 = time.perf_counter()
    results = await asyncio.gather(
        *[_process_one(row, dry_run, semaphore) for row in targets],
        return_exceptions=True,
    )

    ok = sum(1 for r in results if isinstance(r, dict) and r.get("status") in ("success", "dry_run", "ready"))
    err = sum(1 for r in results if isinstance(r, Exception) or (isinstance(r, dict) and r.get("status") == "error"))
    skipped = sum(1 for r in results if isinstance(r, dict) and r.get("status") == "skipped")
    elapsed = time.perf_counter() - t0

    print(
        f"[reindex] done in {elapsed:.1f}s — ok={ok} skipped={skipped} errors={err}"
    )
    await close_postgres()
    return 0 if err == 0 else 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--user-id", help="Only reindex this user's documents")
    parser.add_argument(
        "--source",
        help="Only reindex this source type (upload|web|chat_history|manual_entry|drive)",
    )
    parser.add_argument("--limit", type=int, help="Max documents to process")
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.getenv("REINDEX_CONCURRENCY", "4")),
        help="Documents processed in parallel (default 4)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List targets without touching MinIO / Weaviate",
    )
    args = parser.parse_args()

    exit_code = asyncio.run(
        _main(
            user_id=args.user_id,
            source=args.source,
            limit=args.limit,
            concurrency=args.concurrency,
            dry_run=args.dry_run,
        )
    )
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
