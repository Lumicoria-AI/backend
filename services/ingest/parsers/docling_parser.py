"""IBM Docling parser — one code path for PDF/DOCX/PPTX/XLSX/HTML/images.

Heavy (~500 MB of ONNX models); loaded lazily and gated behind HAS_DOCLING
so the rest of the app still works when docling is missing.

Docling's DocumentConverter emits typed elements; we map them to
`ParsedBlock` so the downstream chunker stays format-agnostic.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import structlog

from ..base import ParsedBlock, ParsedDocument

logger = structlog.get_logger(__name__)

try:
    from docling.document_converter import DocumentConverter
    HAS_DOCLING = True
except ImportError:
    DocumentConverter = None  # type: ignore[misc,assignment]
    HAS_DOCLING = False


_SUPPORTED_MIMES = {
    "application/pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/msword",
    "text/html",
    "image/png", "image/jpeg", "image/tiff",
}


_converter: Optional["DocumentConverter"] = None


def _get_converter() -> "DocumentConverter":
    global _converter
    if _converter is None:
        if not HAS_DOCLING:
            raise RuntimeError("docling is not installed")
        _converter = DocumentConverter()
    return _converter


def _label_to_block_type(label: str) -> str:
    """Map a Docling element label onto our BlockType vocabulary."""
    label = (label or "").lower()
    if label in {"title", "section_header", "sectionheader", "subheader"}:
        return "heading"
    if label in {"list_item", "listitem"}:
        return "list"
    if label == "table":
        return "table"
    if label in {"code", "formula"}:
        return "code"
    if label in {"caption", "figure_caption"}:
        return "caption"
    if label in {"picture", "figure"}:
        return "figure"
    return "paragraph"


def _bbox_from_prov(prov: Any) -> Tuple[Optional[int], Optional[Tuple[float, float, float, float]], Optional[float], Optional[float]]:
    """Best-effort bbox extraction from a Docling provenance record."""
    try:
        first = prov[0] if prov else None
        if first is None:
            return None, None, None, None
        page_number = getattr(first, "page_no", None) or getattr(first, "page", None)
        box = getattr(first, "bbox", None)
        if box is None:
            return page_number, None, None, None
        # Docling bbox: l, t, r, b (in document points).
        l, t, r, b = (
            getattr(box, "l", None),
            getattr(box, "t", None),
            getattr(box, "r", None),
            getattr(box, "b", None),
        )
        if None in (l, t, r, b):
            return page_number, None, None, None
        return page_number, (float(l), float(t), float(r), float(b)), None, None
    except Exception:
        return None, None, None, None


def _extract_table_rows(table_elem: Any) -> Optional[List[List[str]]]:
    """Extract a rectangular list[list[str]] from a Docling TableItem."""
    try:
        data = getattr(table_elem, "data", None)
        if data is None:
            return None
        # Docling 2.x: data.table_cells is a list; grid_shape gives (rows, cols).
        grid = getattr(data, "grid", None)
        if grid is not None:
            return [[(cell.text if cell else "") for cell in row] for row in grid]
        cells = getattr(data, "table_cells", None)
        num_rows = getattr(data, "num_rows", 0)
        num_cols = getattr(data, "num_cols", 0)
        if cells and num_rows and num_cols:
            rows = [[""] * num_cols for _ in range(num_rows)]
            for cell in cells:
                r = getattr(cell, "start_row_offset_idx", 0)
                c = getattr(cell, "start_col_offset_idx", 0)
                if 0 <= r < num_rows and 0 <= c < num_cols:
                    rows[r][c] = getattr(cell, "text", "") or ""
            return rows
    except Exception:
        return None
    return None


def _table_to_markdown(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:] if len(rows) > 1 else []
    md = ["| " + " | ".join(c.replace("|", "\\|") for c in header) + " |"]
    md.append("| " + " | ".join("---" for _ in header) + " |")
    for row in body:
        md.append("| " + " | ".join(c.replace("|", "\\|") for c in row) + " |")
    return "\n".join(md)


def _convert_sync(file_path: str) -> ParsedDocument:
    """Blocking conversion — must be called via asyncio.to_thread."""
    converter = _get_converter()
    result = converter.convert(file_path)
    doc = result.document

    blocks: List[ParsedBlock] = []
    order = 0

    # Docling exposes `iterate_items()` → (item, level) pairs in reading order.
    iterator = getattr(doc, "iterate_items", None)
    if iterator is None:
        # Older API — dump everything as one markdown block.
        md = getattr(doc, "export_to_markdown", lambda: "")()
        if md:
            blocks.append(ParsedBlock(type="paragraph", text=md, order=0))
        return ParsedDocument(blocks=blocks, metadata={}, source_type="docling")

    for item, _level in iterator():
        label = getattr(item, "label", "") or item.__class__.__name__.lower()
        block_type = _label_to_block_type(str(label))

        prov = getattr(item, "prov", None)
        page_number, bbox, pw, ph = _bbox_from_prov(prov)

        if block_type == "table":
            rows = _extract_table_rows(item)
            if rows:
                text = _table_to_markdown(rows)
                blocks.append(ParsedBlock(
                    type="table", text=text, table_rows=rows,
                    page_number=page_number, bbox=bbox,
                    page_width=pw, page_height=ph, order=order,
                ))
                order += 1
            continue

        text = getattr(item, "text", "") or ""
        text = text.strip()
        if not text:
            continue

        heading_level = None
        if block_type == "heading":
            heading_level = int(getattr(item, "level", 1) or 1)

        blocks.append(ParsedBlock(
            type=block_type,
            text=text,
            page_number=page_number,
            bbox=bbox,
            page_width=pw,
            page_height=ph,
            heading_level=heading_level,
            order=order,
        ))
        order += 1

    return ParsedDocument(
        blocks=blocks,
        metadata={},
        source_type="docling",
        title=None,
    )


class DoclingParser:
    name = "docling"

    def supports(self, mime_type: str) -> bool:
        return HAS_DOCLING and mime_type in _SUPPORTED_MIMES

    async def parse(
        self, source: Union[str, bytes], metadata: Dict[str, Any]
    ) -> ParsedDocument:
        if not HAS_DOCLING:
            raise RuntimeError("docling is not installed")
        if not isinstance(source, str) or not Path(source).exists():
            raise ValueError("DoclingParser requires a file path on disk")

        parsed = await asyncio.to_thread(_convert_sync, source)

        merged = dict(metadata)
        merged.update(parsed.metadata)
        parsed.metadata = merged
        parsed.title = parsed.title or metadata.get("title")

        # Override source_type with the caller's mime so downstream policy
        # (prose vs table-heavy) can pick a chunking strategy.
        mime = metadata.get("mime_type", "")
        if "pdf" in mime:
            parsed.source_type = "pdf"
        elif "wordprocessingml" in mime:
            parsed.source_type = "docx"
        elif "presentationml" in mime:
            parsed.source_type = "pptx"
        elif "spreadsheetml" in mime:
            parsed.source_type = "xlsx"
        elif "html" in mime:
            parsed.source_type = "html"
        elif mime.startswith("image/"):
            parsed.source_type = "image"

        logger.info("docling_parsed", blocks=len(parsed.blocks), source_type=parsed.source_type)
        return parsed
