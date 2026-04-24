"""URL parser — trafilatura extracts main content and discards nav / ads /
script / style. Falls back to a regex tag-strip if trafilatura is missing."""

from __future__ import annotations

import re
from typing import Any, Dict, Union

import httpx
import structlog

from ..base import ParsedBlock, ParsedDocument
from .plaintext_parser import _parse_markdown

logger = structlog.get_logger(__name__)

try:
    import trafilatura
    HAS_TRAFILATURA = True
except ImportError:
    HAS_TRAFILATURA = False


_TITLE_RE = re.compile(r"<title>(.*?)</title>", re.IGNORECASE | re.DOTALL)


def _fallback_strip(html: str) -> str:
    text = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<style[^>]*>.*?</style>", " ", text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


class TrafilaturaParser:
    name = "trafilatura"

    def supports(self, mime_type: str) -> bool:
        return mime_type in {"text/html", "application/xhtml+xml"}

    async def parse(
        self, source: Union[str, bytes], metadata: Dict[str, Any]
    ) -> ParsedDocument:
        url = metadata.get("url")

        if isinstance(source, bytes):
            html = source.decode("utf-8", errors="replace")
        elif url and source == url:
            # Caller passed the URL; fetch it ourselves.
            async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
                resp = await client.get(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (compatible; Lumicoria.ai/1.0)"},
                )
                resp.raise_for_status()
                html = resp.text
        else:
            html = source

        title = metadata.get("title")
        if not title:
            match = _TITLE_RE.search(html)
            title = (match.group(1).strip() if match else None) or (url or "Web content")

        if HAS_TRAFILATURA:
            md = trafilatura.extract(
                html,
                output_format="markdown",
                include_tables=True,
                include_links=False,
                include_images=False,
                favor_precision=True,
            )
            if md and md.strip():
                blocks = _parse_markdown(md)
            else:
                blocks = [ParsedBlock(type="paragraph", text=_fallback_strip(html), order=0)]
        else:
            blocks = [ParsedBlock(type="paragraph", text=_fallback_strip(html), order=0)]

        merged = dict(metadata)
        merged.setdefault("title", title)

        return ParsedDocument(
            blocks=blocks,
            metadata=merged,
            source_type="html",
            title=title,
        )
