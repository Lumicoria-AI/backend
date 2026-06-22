"""Ingest emails + attachments + Drive deltas into Weaviate.

Pipeline:
  - Email body  → ``document_processor.process_text`` with
                  source="gmail", message_id, thread_id, user_id, org_id.
  - Attachment  → fetch bytes from MinIO via the key we stored in the
                  previous node, write to a temp file, pass through
                  ``document_processor.process_file`` (uses the parser
                  factory — PDF/DOCX/etc. all handled).
  - Drive file  → ``document_processor.process_google_drive`` (already
                  real after Phase 1).

A 4-wide semaphore caps parallel ingest so we don't saturate the
embedding client at the same time the rest of the brain is calling
LLMs.

Coverage eval: at least 80% of inputs should ingest. Below that → mark
the run degraded so the digest can warn "context is partial."
"""

from __future__ import annotations

import asyncio
import os
import tempfile
from typing import Any, Dict, List

import structlog

from ..evals import check_coverage
from ..state import BrainState
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


_CONCURRENCY = 4


@traced_node("ingest_to_rag")
async def ingest_to_rag(state: BrainState) -> Dict[str, Any]:
    blobs = state.meta.get("attachment_blobs") or []
    has_email = bool(state.emails)
    has_drive = bool(state.drive_changes)

    if not has_email and not has_drive and not blobs:
        return {
            "ingested_doc_ids": [],
            "__payload_summary": {
                "inputs": 0,
                "ingested": 0,
                "skipped": 0,
                "failed": 0,
            },
            "__eval_score": 1.0,
        }

    try:
        from backend.services.document_processor import document_processor
    except Exception as exc:  # noqa: BLE001
        logger.warning("ingest_to_rag.import_failed", error=str(exc))
        return {
            "ingested_doc_ids": [],
            "__payload_summary": {"error": "import_failed"},
            "__eval_score": 0.0,
            "__status": "fallback",
        }

    sem = asyncio.Semaphore(_CONCURRENCY)
    ingested_ids: List[str] = []
    failed = 0

    async def _ingest_email(message_ref) -> None:
        async with sem:
            text = _email_body_text(message_ref)
            if not text:
                return
            try:
                result = await document_processor.process_text(
                    text=text,
                    metadata={
                        "user_id": state.user_id,
                        "organization_id": state.organization_id,
                        "source": "gmail",
                        "message_id": message_ref.message_id,
                        "thread_id": message_ref.thread_id,
                        "title": message_ref.subject or "(no subject)",
                        "filename": f"gmail-{message_ref.message_id}.md",
                        "mime_type": "text/markdown",
                    },
                )
                if result.status == "completed" and result.document_id:
                    ingested_ids.append(result.document_id)
                else:
                    nonlocal_inc("failed")
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "ingest_to_rag.email_failed",
                    message_id=message_ref.message_id, error=str(exc),
                )
                nonlocal_inc("failed")

    async def _ingest_attachment(blob: Dict[str, Any]) -> None:
        async with sem:
            try:
                from backend.services.storage_service import storage_service
                # CMK-unseal on the way back in. The same org_id we used
                # on upload scopes the KEK lookup. download_decrypted
                # falls through cleanly on plaintext blobs (no header)
                # so legacy uploads also work.
                org_id_for_cmk = state.organization_id or state.user_id
                data = await storage_service.download_decrypted(
                    key=blob["minio_key"],
                    organization_id=org_id_for_cmk,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "ingest_to_rag.blob_download_failed",
                    key=blob.get("minio_key"), error=str(exc),
                )
                nonlocal_inc("failed")
                return

            if not data:
                nonlocal_inc("failed")
                return

            # Write to temp file so process_file's parser factory works.
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=_suffix_from_filename(blob.get("filename", "")),
                    delete=False,
                ) as tmp:
                    tmp.write(data)
                    tmp_path = tmp.name
                result = await document_processor.process_file(
                    file_path=tmp_path,
                    metadata={
                        "user_id": state.user_id,
                        "organization_id": state.organization_id,
                        "source": "gmail_attachment",
                        "message_id": blob.get("message_id"),
                        "attachment_id": blob.get("attachment_id"),
                        "filename": blob.get("filename"),
                        "mime_type": blob.get("mime_type"),
                        "title": blob.get("filename"),
                        "minio_key": blob.get("minio_key"),
                    },
                )
                if result.status == "completed" and result.document_id:
                    ingested_ids.append(result.document_id)
                else:
                    nonlocal_inc("failed")
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "ingest_to_rag.attachment_failed",
                    key=blob.get("minio_key"), error=str(exc),
                )
                nonlocal_inc("failed")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

    async def _ingest_drive(ref) -> None:
        async with sem:
            if ref.removed:
                return
            try:
                client = state.meta.get("google_client")
                result = await document_processor.process_google_drive(
                    drive_file_id=ref.file_id,
                    metadata={
                        "user_id": state.user_id,
                        "organization_id": state.organization_id,
                        "source": "drive",
                        "title": ref.name or ref.file_id,
                    },
                    drive_client=client,
                )
                if result.status == "completed" and result.document_id:
                    ingested_ids.append(result.document_id)
                else:
                    nonlocal_inc("failed")
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "ingest_to_rag.drive_failed",
                    file_id=ref.file_id, error=str(exc),
                )
                nonlocal_inc("failed")

    # `failed` is captured by closure — Python `nonlocal` quirk
    # requires a mutable container.
    counter = {"failed": 0}
    def nonlocal_inc(_):
        counter["failed"] += 1

    tasks = (
        [_ingest_email(m) for m in state.emails]
        + [_ingest_attachment(b) for b in blobs]
        + [_ingest_drive(r) for r in state.drive_changes if not r.removed]
    )

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=False)

    inputs = len(state.emails) + len(blobs) + sum(1 for r in state.drive_changes if not r.removed)
    coverage = check_coverage(inputs, len(ingested_ids), floor=0.8, field="ingest")

    return {
        "ingested_doc_ids": ingested_ids,
        "__payload_summary": {
            "inputs": inputs,
            "ingested": len(ingested_ids),
            "failed": counter["failed"],
            "coverage_score": coverage.score,
        },
        "__eval_score": coverage.score,
        **({"__status": "fallback"} if not coverage.passed else {}),
    }


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _email_body_text(m) -> str:
    """Compose a minimal markdown body from the message ref. The full
    body lives on the original Gmail payload — for Phase 5 we work
    from subject + snippet which is enough to embed for similarity
    search. Phase 6 swaps in real body fetch + MIME parsing."""
    parts: List[str] = []
    if m.subject:
        parts.append(f"# {m.subject}")
    if m.from_addr:
        parts.append(f"From: {m.from_addr}")
    if m.snippet:
        parts.append("")
        parts.append(m.snippet)
    return "\n".join(parts).strip()


def _suffix_from_filename(name: str) -> str:
    ext = os.path.splitext(name or "")[1].lower()
    return ext or ".bin"
