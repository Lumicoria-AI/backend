"""Prometheus metrics for the RAG ingest pipeline.

All metrics live in a single process-wide registry (the default one used by
`backend/main.py`'s `/metrics` endpoint).  Using stable label cardinalities:

    source       ∈ {upload, web, manual_entry, chat_history, drive}
    mime         ∈ bucketed by family (pdf, docx, pptx, xlsx, image, html, text, code, other)
    stage        ∈ {download, parse, chunk, embed, store, total}
    parser       ∈ {pymupdf, docling, trafilatura, plaintext, image, code}
    status       ∈ {ready, error, cancelled}

Callers should use `record_stage()` / `record_embed()` — keeps the metric
construction details in one place.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Optional

try:
    from prometheus_client import Counter, Histogram
    HAS_PROMETHEUS = True
except ImportError:
    HAS_PROMETHEUS = False
    Counter = None  # type: ignore
    Histogram = None  # type: ignore


_MIME_FAMILY = {
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
    "application/msword": "docx",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation": "pptx",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": "xlsx",
    "application/vnd.ms-excel": "xlsx",
    "text/html": "html",
    "application/xhtml+xml": "html",
    "text/markdown": "text",
    "text/x-markdown": "text",
    "text/plain": "text",
}


def mime_family(mime: Optional[str]) -> str:
    """Collapse arbitrary mime-types into a small, bounded label set so
    Prometheus doesn't explode on exotic user uploads."""
    if not mime:
        return "other"
    mime = mime.lower()
    if mime in _MIME_FAMILY:
        return _MIME_FAMILY[mime]
    if mime.startswith("image/"):
        return "image"
    if mime.startswith("text/x-") or mime.startswith("application/x-"):
        return "code"
    if mime.startswith("text/"):
        return "text"
    return "other"


if HAS_PROMETHEUS:
    CHUNKS_PRODUCED = Counter(
        "lumicoria_ingest_chunks_produced_total",
        "Chunks emitted by the ingest pipeline",
        ["source", "mime"],
    )
    INGEST_DURATION = Histogram(
        "lumicoria_ingest_duration_seconds",
        "Duration of each ingest stage",
        ["mime", "stage"],
        buckets=(0.1, 0.5, 1, 2.5, 5, 10, 30, 60, 120, 300, 600),
    )
    EMBED_LATENCY = Histogram(
        "lumicoria_ingest_embed_latency_seconds",
        "Time spent waiting on the embedding provider per batch",
        ["provider"],
        buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30),
    )
    OCR_FALLBACK = Counter(
        "lumicoria_ingest_ocr_fallback_total",
        "Pages that required Tesseract OCR because PyMuPDF found no text",
        ["mime"],
    )
    INGEST_STATUS = Counter(
        "lumicoria_ingest_status_total",
        "Terminal status of ingest runs",
        ["source", "mime", "status"],
    )
    DEDUP_HITS = Counter(
        "lumicoria_ingest_dedup_hits_total",
        "Uploads that were aliased to an existing document by content hash",
        ["source"],
    )


def record_chunks(source: Optional[str], mime: Optional[str], count: int) -> None:
    if HAS_PROMETHEUS and count:
        CHUNKS_PRODUCED.labels(source=source or "unknown", mime=mime_family(mime)).inc(count)


@contextmanager
def record_stage(mime: Optional[str], stage: str):
    """Time a pipeline stage. Use: `with record_stage(mime, "parse"): ...`."""
    start = time.perf_counter()
    try:
        yield
    finally:
        if HAS_PROMETHEUS:
            INGEST_DURATION.labels(mime=mime_family(mime), stage=stage).observe(
                time.perf_counter() - start
            )


@contextmanager
def record_embed(provider: str):
    start = time.perf_counter()
    try:
        yield
    finally:
        if HAS_PROMETHEUS:
            EMBED_LATENCY.labels(provider=provider or "unknown").observe(
                time.perf_counter() - start
            )


def record_ocr_fallback(mime: Optional[str], pages: int = 1) -> None:
    if HAS_PROMETHEUS and pages:
        OCR_FALLBACK.labels(mime=mime_family(mime)).inc(pages)


def record_status(source: Optional[str], mime: Optional[str], status: str) -> None:
    if HAS_PROMETHEUS:
        INGEST_STATUS.labels(
            source=source or "unknown",
            mime=mime_family(mime),
            status=status,
        ).inc()


def record_dedup_hit(source: Optional[str]) -> None:
    if HAS_PROMETHEUS:
        DEDUP_HITS.labels(source=source or "unknown").inc()
