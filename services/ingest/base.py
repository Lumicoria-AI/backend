"""Parser-agnostic data model.

Every parser in this package emits a `ParsedDocument` — an ordered list of
typed `ParsedBlock`s. The chunker consumes that list and never has to know
which parser produced it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, Union


BlockType = str
# Canonical values (not a strict enum so parsers can extend without churn):
#   "heading"    — section title / h1..h6
#   "paragraph"  — body prose
#   "list"       — bulleted / numbered list (already joined to one string)
#   "table"      — tabular data; `table_rows` carries the structure
#   "code"       — verbatim code block
#   "caption"    — figure / table caption
#   "figure"     — image or chart reference (text is alt/caption)
#   "quote"      — block quote


@dataclass
class ParsedBlock:
    """One structural unit from a document — cheap to pickle."""
    type: BlockType
    text: str
    # Positional metadata (PDF/DOCX only).
    page_number: Optional[int] = None
    bbox: Optional[Tuple[float, float, float, float]] = None
    page_width: Optional[float] = None
    page_height: Optional[float] = None
    # Type-specific extras.
    heading_level: Optional[int] = None  # 1..6 for "heading"
    language: Optional[str] = None       # e.g. "python" for "code"
    table_rows: Optional[List[List[str]]] = None  # for "table" (first row = header)
    # Block-level sequence number within the document — 0-indexed.
    order: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    def is_splittable(self) -> bool:
        """Tables, code, and lists must stay intact through chunking."""
        return self.type in {"heading", "paragraph", "caption", "quote"}


@dataclass
class ParsedDocument:
    blocks: List[ParsedBlock]
    # Caller-supplied metadata (document_id, user_id, s3_key, etc.) —
    # propagated onto every resulting chunk.
    metadata: Dict[str, Any] = field(default_factory=dict)
    # Rough mime / source classification used by the chunker to pick a policy.
    source_type: str = "text"  # "pdf" | "docx" | "pptx" | "xlsx" | "html" | "url" | "text" | "chat" | "code"
    title: Optional[str] = None

    def total_chars(self) -> int:
        return sum(len(b.text) for b in self.blocks)


class DocumentParser(Protocol):
    """Minimal interface implemented by every parser in parsers/.

    Parsers are async so URL-fetching parsers can stream without blocking.
    File-based parsers typically offload heavy work via `asyncio.to_thread`.
    """

    name: str

    def supports(self, mime_type: str) -> bool: ...

    async def parse(
        self,
        source: Union[str, bytes],
        metadata: Dict[str, Any],
    ) -> ParsedDocument: ...
