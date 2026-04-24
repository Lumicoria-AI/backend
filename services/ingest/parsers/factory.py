"""Parser selection.

Routing rules:
  1. URL / HTML       → TrafilaturaParser (if installed, else fallback strip).
  2. PDF              → PyMuPDFParser (fast) unless INGEST_PARSER_DEFAULT="docling"
                         or metadata["force_docling"] is truthy.
  3. DOCX/PPTX/XLSX   → DoclingParser (falls back gracefully if missing).
  4. Images           → ImageParser (OCR + optional caption) if deps; else Docling.
  5. text/markdown/code    → PlainTextParser (language-aware chunking).
  6. Anything else    → PlainTextParser as a safe default.
"""

from __future__ import annotations

from typing import Dict

from ....core.config import settings
from ..base import DocumentParser
from .docling_parser import DoclingParser, HAS_DOCLING
from .image_parser import HAS_PIL, ImageParser
from .plaintext_parser import PlainTextParser
from .pymupdf_parser import HAS_PYMUPDF, PyMuPDFParser
from .trafilatura_parser import TrafilaturaParser


_plaintext = PlainTextParser()
_trafilatura = TrafilaturaParser()
_pymupdf = PyMuPDFParser() if HAS_PYMUPDF else None
_docling = DoclingParser() if HAS_DOCLING else None
_image = ImageParser() if HAS_PIL else None


_RICH_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/msword",
}


def get_parser(mime_type: str, metadata: Dict | None = None) -> DocumentParser:
    metadata = metadata or {}
    mime_type = (mime_type or "").lower()

    if mime_type in {"text/html", "application/xhtml+xml"}:
        return _trafilatura

    if mime_type == "application/pdf":
        force_docling = metadata.get("force_docling") or getattr(settings, "INGEST_PARSER_DEFAULT", "fast") == "docling"
        if force_docling and _docling is not None:
            return _docling
        if _pymupdf is not None:
            return _pymupdf
        if _docling is not None:
            return _docling
        return _plaintext

    if mime_type.startswith("image/"):
        if _image is not None:
            return _image
        if _docling is not None:
            return _docling
        return _plaintext

    if mime_type in _RICH_MIMES:
        if _docling is not None:
            return _docling
        return _plaintext

    return _plaintext
