"""Fast-path PDF parser. Uses PyMuPDF to extract typed blocks with bbox
metadata in parallel across a shared ProcessPoolExecutor.

Scanned pages (very little extractable text) are reported via
`ParsedDocument.metadata['scanned_page_ratio']` so the factory / caller can
decide whether to fall back to Docling's OCR pipeline.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Dict, List, Optional, Union

import structlog

from ...ingest.base import ParsedBlock, ParsedDocument
from ....core.config import settings

logger = structlog.get_logger(__name__)

try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

try:
    import pytesseract  # type: ignore
    from PIL import Image as _PILImage  # type: ignore
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


def _ocr_page(file_path: str, page_number: int) -> Dict[str, Any]:
    """Worker: render a single page to PNG, Tesseract it, return one block."""
    import io
    import fitz
    doc = fitz.open(file_path)
    try:
        page = doc[page_number - 1]
        # 200 DPI — good balance between accuracy and speed.
        pix = page.get_pixmap(dpi=200)
        png_bytes = pix.tobytes("png")
        try:
            from PIL import Image  # type: ignore
            import pytesseract  # type: ignore
            img = Image.open(io.BytesIO(png_bytes))
            text = (pytesseract.image_to_string(img) or "").strip()
        except Exception:
            text = ""
        return {
            "page_number": page_number,
            "text": text,
            "page_width": page.rect.width,
            "page_height": page.rect.height,
            "bbox": (0, 0, page.rect.width, page.rect.height),
        }
    finally:
        doc.close()


def _extract_pdf_page_range(
    file_path: str, start_page: int, end_page: int
) -> List[Dict[str, Any]]:
    """Worker: extract positional blocks from pages [start, end). Returns
    plain dicts — cheap to pickle back to the parent process."""
    import fitz  # re-import in worker

    out: List[Dict[str, Any]] = []
    doc = fitz.open(file_path)
    try:
        for page_idx in range(start_page, min(end_page, len(doc))):
            page = doc[page_idx]
            page_dict = page.get_text("dict", sort=True)
            page_width = page_dict.get("width", page.rect.width)
            page_height = page_dict.get("height", page.rect.height)
            page_char_count = 0

            for block_idx, block in enumerate(page_dict.get("blocks", [])):
                if block.get("type", 0) != 0:
                    continue  # image block

                parts = [
                    "".join(span.get("text", "") for span in line.get("spans", []))
                    for line in block.get("lines", [])
                ]
                block_text = "\n".join(parts).strip()
                if not block_text:
                    continue

                page_char_count += len(block_text)
                bbox = block.get("bbox", [0, 0, page_width, page_height])

                # Heuristic: short blocks with large font size ≈ heading.
                heading_level: Optional[int] = None
                block_type = "paragraph"
                if len(block_text) < 120 and "\n" not in block_text:
                    max_size = 0.0
                    for line in block.get("lines", []):
                        for span in line.get("spans", []):
                            max_size = max(max_size, span.get("size", 0.0))
                    if max_size >= 14:
                        block_type = "heading"
                        heading_level = 1 if max_size >= 20 else (2 if max_size >= 16 else 3)

                out.append({
                    "type": block_type,
                    "text": block_text,
                    "page_number": page_idx + 1,
                    "bbox": tuple(bbox),
                    "page_width": page_width,
                    "page_height": page_height,
                    "heading_level": heading_level,
                    "page_char_count": page_char_count,  # last-write-wins per page
                })
    finally:
        doc.close()
    return out


_pool: Optional[ProcessPoolExecutor] = None


def _get_pool() -> Optional[ProcessPoolExecutor]:
    global _pool
    workers = max(1, int(getattr(settings, "INGEST_PROCESS_POOL_WORKERS", 4)))
    if workers <= 1:
        return None
    if _pool is None:
        _pool = ProcessPoolExecutor(max_workers=workers)
    return _pool


class PyMuPDFParser:
    name = "pymupdf"

    def supports(self, mime_type: str) -> bool:
        return mime_type == "application/pdf" and HAS_PYMUPDF

    async def parse(
        self, source: Union[str, bytes], metadata: Dict[str, Any]
    ) -> ParsedDocument:
        if not HAS_PYMUPDF:
            raise RuntimeError("PyMuPDF is not installed")
        if not isinstance(source, str):
            raise TypeError("PyMuPDFParser requires a file path")

        file_path = source

        doc = fitz.open(file_path)
        total_pages = len(doc)
        doc.close()

        pages_per_worker = max(1, int(getattr(settings, "INGEST_PDF_PAGES_PER_WORKER", 25)))
        pool = _get_pool()

        if pool is None or total_pages <= pages_per_worker:
            raw = await asyncio.to_thread(_extract_pdf_page_range, file_path, 0, total_pages)
        else:
            loop = asyncio.get_running_loop()
            ranges = [
                (s, min(s + pages_per_worker, total_pages))
                for s in range(0, total_pages, pages_per_worker)
            ]
            results = await asyncio.gather(*[
                loop.run_in_executor(pool, _extract_pdf_page_range, file_path, s, e)
                for s, e in ranges
            ])
            raw = [b for chunk in results for b in chunk]

        # Per-page text counts for scanned-page detection.
        page_chars: Dict[int, int] = {}
        for b in raw:
            p = b["page_number"]
            page_chars[p] = page_chars.get(p, 0) + len(b["text"])

        min_chars = int(getattr(settings, "INGEST_OCR_MIN_CHARS_PER_PAGE", 50))
        scanned_page_nums = [
            p for p in range(1, total_pages + 1) if page_chars.get(p, 0) < min_chars
        ]
        scanned_ratio = len(scanned_page_nums) / total_pages if total_pages else 0.0

        # OCR fallback: fill in scanned pages via Tesseract.  Disabled when
        # Tesseract isn't installed — the scanned_page_ratio metadata still
        # lets the caller decide to hand off to Docling's OCR pipeline.
        ocr_blocks: List[Dict[str, Any]] = []
        if scanned_page_nums and HAS_TESSERACT and getattr(settings, "INGEST_OCR_ENABLED", True):
            if pool is None:
                for p in scanned_page_nums:
                    try:
                        ocr_blocks.append(await asyncio.to_thread(_ocr_page, file_path, p))
                    except Exception as e:
                        logger.warning("pdf_ocr_failed", page=p, error=str(e))
            else:
                loop = asyncio.get_running_loop()
                try:
                    futures = [
                        loop.run_in_executor(pool, _ocr_page, file_path, p)
                        for p in scanned_page_nums
                    ]
                    ocr_blocks = [b for b in await asyncio.gather(*futures, return_exceptions=False)]
                except Exception as e:
                    logger.warning("pdf_ocr_batch_failed", error=str(e))
            ocr_blocks = [b for b in ocr_blocks if b and b.get("text")]

        blocks: List[ParsedBlock] = []
        for i, b in enumerate(raw):
            blocks.append(ParsedBlock(
                type=b["type"],
                text=b["text"],
                page_number=b["page_number"],
                bbox=b["bbox"],
                page_width=b["page_width"],
                page_height=b["page_height"],
                heading_level=b.get("heading_level"),
                order=i,
            ))
        for j, b in enumerate(ocr_blocks):
            blocks.append(ParsedBlock(
                type="paragraph",
                text=b["text"],
                page_number=b["page_number"],
                bbox=b["bbox"],
                page_width=b["page_width"],
                page_height=b["page_height"],
                order=len(blocks) + j,
                extra={"origin": "ocr"},
            ))
        blocks.sort(key=lambda bb: (bb.page_number or 0, bb.order))

        merged = dict(metadata)
        merged.setdefault("page_count", total_pages)
        merged["scanned_page_ratio"] = scanned_ratio
        merged["ocr_pages"] = len(ocr_blocks)

        logger.info(
            "pdf_parsed",
            pages=total_pages,
            blocks=len(blocks),
            scanned_ratio=scanned_ratio,
            ocr_pages=len(ocr_blocks),
            parallel=pool is not None and total_pages > pages_per_worker,
        )
        return ParsedDocument(
            blocks=blocks,
            metadata=merged,
            source_type="pdf",
            title=metadata.get("title"),
        )
