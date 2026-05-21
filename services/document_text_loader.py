"""Resolve an uploaded document id to its full extracted plain text.

The Legal Document Agent (and any other agent that wants the raw text of
an already-uploaded document) calls into this module instead of decoding
PDF bytes as UTF-8 — which produces `%PDF-1.3 %����...` garbage and gets
rejected by every LLM.

Pipeline:
  1. Look up the document row from the RAG registry (tenant-scoped).
  2. Download the original bytes from MinIO via `storage_service`.
  3. Write the bytes to a NamedTemporaryFile (the ingest parsers expect
     a file path; PyMuPDFParser specifically refuses raw bytes).
  4. Pick the right parser via `ingest.get_parser(mime_type, metadata)` —
     the same factory the ingestion pipeline uses, so PDF → PyMuPDF (or
     Docling), DOCX → Docling, plain text → PlainTextParser, etc.
  5. Run the parser, join every block's text in document order, and
     return the joined string.

We extract from the raw bytes synchronously rather than waiting on the
background ingestion that feeds Weaviate.  This makes the legal flow
robust to:
  - Documents whose chunks haven't been embedded yet
  - Ingestion failures unrelated to text extraction (embedding outages,
    Weaviate connectivity, etc.)
  - The race where the user runs an analysis the moment the upload
    toast appears
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import structlog

from .ingest import get_parser
from .legal_document.sanitize import MAX_DOCUMENT_CHARS, clean_text
from .rag_document_registry import get as registry_get
from .storage_service import storage_service

logger = structlog.get_logger(__name__)


class DocumentNotFoundError(Exception):
    """The document does not exist or is not visible to the caller."""


class DocumentTextLoadError(Exception):
    """Storage / parser failure while loading the document."""


def _suffix_for(filename: Optional[str], mime_type: Optional[str]) -> str:
    """Pick a sensible filename suffix so PyMuPDF / Docling / etc. see a
    real extension on the temp file.  Falls back to `.bin` for unknown
    mime types."""
    if filename:
        ext = Path(filename).suffix
        if ext:
            return ext
    mime = (mime_type or "").lower()
    return {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
        "application/msword": ".doc",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
        "text/plain": ".txt",
        "text/markdown": ".md",
        "text/x-markdown": ".md",
        "text/html": ".html",
        "application/xhtml+xml": ".html",
        "text/csv": ".csv",
        "image/png": ".png",
        "image/jpeg": ".jpg",
    }.get(mime, ".bin")


def _join_blocks(parsed) -> str:
    """Join a ParsedDocument's blocks into a single plain-text string.
    Skips empty / whitespace-only blocks; preserves document order."""
    if parsed is None or not getattr(parsed, "blocks", None):
        return ""
    parts = []
    for block in parsed.blocks:
        text = (getattr(block, "text", "") or "").strip()
        if not text:
            continue
        parts.append(text)
    return "\n\n".join(parts)


async def load_extracted_text(
    document_id: str,
    user_id: str,
    organization_id: Optional[str] = None,
) -> str:
    """Resolve a RAG document id to its full extracted plain text.

    Tenant-scoped: raises `DocumentNotFoundError` if the document does
    not exist or belongs to a different organization.

    Returns a plain-text string (already passed through
    `legal_document.sanitize.clean_text`).
    """
    if not document_id:
        raise DocumentNotFoundError("document_id is required")

    doc = await registry_get(document_id, user_id=user_id)
    if not doc:
        raise DocumentNotFoundError("Document not found")

    # Belt-and-braces tenant check.  The registry already scopes by
    # user_id, but if a user belongs to an org we ensure the document's
    # org matches the caller's scope.
    if organization_id:
        doc_org = doc.get("organization_id")
        if doc_org and doc_org != organization_id and doc.get("user_id") != organization_id:
            raise DocumentNotFoundError("Document not found")

    s3_key = doc.get("s3_key")
    if not s3_key:
        raise DocumentTextLoadError("Document has no storage key")

    try:
        raw = await storage_service.download_file(s3_key)
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "document_text_loader_download_failed",
            document_id=document_id,
            error=str(e),
        )
        raise DocumentTextLoadError(f"Could not fetch document: {e}")

    mime_type = doc.get("mime_type") or "application/octet-stream"
    suffix = _suffix_for(doc.get("original_filename") or doc.get("filename"), mime_type)

    # Parser metadata.  The ingest parsers use these for cache keys,
    # logging, and content-type-specific routing.
    parser_metadata: Dict[str, Any] = {
        "document_id": document_id,
        "user_id": user_id,
        "organization_id": organization_id,
        "mime_type": mime_type,
        "filename": doc.get("original_filename") or doc.get("filename") or f"{document_id}{suffix}",
        "title": doc.get("title") or doc.get("original_filename") or doc.get("filename"),
        "source": doc.get("source") or "upload",
    }

    parser = get_parser(mime_type, parser_metadata)

    # The PDF parser (PyMuPDF) refuses raw bytes; image parsers may
    # need to seek; everyone is happy with a real path.  Write to a
    # tempfile with a sensible extension so the parsers can detect
    # format from the suffix when they want to.
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix=f"legal_doc_{document_id[:8]}_"
        ) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            parsed = await parser.parse(tmp_path, parser_metadata)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "document_text_loader_parse_failed",
                document_id=document_id,
                mime_type=mime_type,
                parser=getattr(parser, "name", "unknown"),
                error=str(e),
            )
            # Last-resort fallback: decode as text.  Better than nothing
            # for content the parser registry couldn't handle.
            return clean_text(
                raw.decode("utf-8", errors="replace"),
                max_len=MAX_DOCUMENT_CHARS,
            )
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass

    text = _join_blocks(parsed)
    if not text:
        # Parser returned no blocks — fall back to a plain decode so the
        # LLM at least sees *something* rather than an empty string.
        text = raw.decode("utf-8", errors="replace")

    return clean_text(text, max_len=MAX_DOCUMENT_CHARS)
