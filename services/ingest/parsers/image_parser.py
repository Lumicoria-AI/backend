"""Image ingestion: Tesseract OCR + optional multimodal caption.

Two blocks are emitted per image so both paths are retrievable:
  1. `paragraph` with the OCR text   (if Tesseract finds any).
  2. `figure` with a short caption    (if a multimodal LLM is available).

Both are optional. If neither Tesseract nor a multimodal model is usable,
the parser returns a placeholder `figure` block so the document still has
an entry — preserving the s3_key/preview linkage.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict, Union

import structlog

from ..base import ParsedBlock, ParsedDocument

logger = structlog.get_logger(__name__)

try:
    from PIL import Image  # type: ignore
    HAS_PIL = True
except ImportError:
    Image = None  # type: ignore
    HAS_PIL = False

try:
    import pytesseract  # type: ignore
    HAS_TESSERACT = True
except ImportError:
    pytesseract = None  # type: ignore
    HAS_TESSERACT = False


def _load_image(source: Union[str, bytes]):
    if not HAS_PIL:
        return None
    if isinstance(source, bytes):
        return Image.open(io.BytesIO(source))
    path = Path(source)
    if path.exists():
        return Image.open(path)
    return None


def _ocr(img) -> str:
    if not (HAS_TESSERACT and img is not None):
        return ""
    try:
        return (pytesseract.image_to_string(img) or "").strip()
    except Exception as e:
        logger.warning("image_ocr_failed", error=str(e))
        return ""


async def _caption_via_gemini(img_bytes: bytes, mime_type: str) -> str:
    """Best-effort caption via Gemini Vision.  Returns "" on any failure so
    ingestion never fails because captioning is unavailable."""
    try:
        from ....ai_models.client_factory import get_llm_client  # type: ignore
    except Exception:
        return ""
    try:
        import base64
        client = get_llm_client("gemini")
        if client is None:
            return ""
        b64 = base64.b64encode(img_bytes).decode("ascii")
        prompt = (
            "Describe this image in 1-2 sentences for search indexing. "
            "Focus on subjects, text visible, and distinguishing details."
        )
        # Prefer a generic `generate` surface if the client exposes it;
        # otherwise fall back to a no-op.
        if hasattr(client, "generate_content"):
            resp = await client.generate_content(  # type: ignore[attr-defined]
                prompt=prompt,
                image_base64=b64,
                image_mime=mime_type,
            )
            return str(resp or "").strip()
    except Exception as e:
        logger.warning("image_caption_failed", error=str(e))
    return ""


class ImageParser:
    name = "image"

    def supports(self, mime_type: str) -> bool:
        return (mime_type or "").lower().startswith("image/")

    async def parse(
        self, source: Union[str, bytes], metadata: Dict[str, Any]
    ) -> ParsedDocument:
        mime_type = metadata.get("mime_type", "image/png")
        img = _load_image(source)

        img_bytes: bytes = b""
        if isinstance(source, bytes):
            img_bytes = source
        else:
            try:
                p = Path(source)
                if p.exists():
                    img_bytes = p.read_bytes()
            except Exception:
                pass

        ocr_text = _ocr(img)
        caption = ""
        if metadata.get("enable_caption", True):
            caption = await _caption_via_gemini(img_bytes, mime_type)

        blocks = []
        order = 0
        if ocr_text:
            blocks.append(ParsedBlock(
                type="paragraph", text=ocr_text, order=order,
                extra={"origin": "ocr"},
            ))
            order += 1
        if caption:
            blocks.append(ParsedBlock(
                type="figure", text=caption, order=order,
                extra={"origin": "caption"},
            ))
            order += 1
        if not blocks:
            blocks.append(ParsedBlock(
                type="figure",
                text=metadata.get("title") or metadata.get("filename", "image"),
                order=0,
                extra={"origin": "placeholder"},
            ))

        return ParsedDocument(
            blocks=blocks,
            metadata=metadata,
            source_type="image",
            title=metadata.get("title") or metadata.get("filename"),
        )
