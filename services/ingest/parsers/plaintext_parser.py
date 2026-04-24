"""Parser for text, markdown, and source code.

- text/plain → one `paragraph` block per paragraph (split on blank lines).
- text/markdown → headings, code fences, and paragraphs detected naively.
- code mime types → one `code` block per file; the chunker splits it safely
  with language-aware separators.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Union

from ..base import ParsedBlock, ParsedDocument


_CODE_EXTENSIONS = {
    ".py": "python", ".js": "javascript", ".jsx": "javascript",
    ".ts": "typescript", ".tsx": "typescript", ".java": "java",
    ".go": "go", ".rs": "rust", ".rb": "ruby", ".php": "php",
    ".c": "c", ".h": "c", ".cpp": "cpp", ".hpp": "cpp", ".cc": "cpp",
    ".cs": "csharp", ".swift": "swift", ".kt": "kotlin", ".scala": "scala",
    ".sh": "bash", ".bash": "bash", ".zsh": "bash",
    ".sql": "sql", ".yaml": "yaml", ".yml": "yaml", ".json": "json",
    ".toml": "toml", ".xml": "xml",
}


def _detect_code_language(filename: str, mime_type: str) -> str:
    ext = Path(filename).suffix.lower() if filename else ""
    if ext in _CODE_EXTENSIONS:
        return _CODE_EXTENSIONS[ext]
    # text/x-python, application/x-ruby, etc.
    if "/" in mime_type:
        sub = mime_type.split("/", 1)[1].lstrip("x-")
        return sub
    return "text"


class PlainTextParser:
    name = "plaintext"

    def supports(self, mime_type: str) -> bool:
        return mime_type in {
            "text/plain", "text/markdown", "text/x-markdown",
        } or mime_type.startswith("text/x-") or mime_type.startswith("application/x-")

    async def parse(
        self, source: Union[str, bytes], metadata: Dict[str, Any]
    ) -> ParsedDocument:
        mime_type = metadata.get("mime_type", "text/plain")
        filename = metadata.get("filename", "")

        if isinstance(source, bytes):
            text = source.decode("utf-8", errors="replace")
        else:
            # Path or raw text. If it's an existing file, read it.
            path = Path(source)
            if path.exists() and path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
            else:
                text = source

        # Code file short-circuit: one block, chunker splits language-aware.
        if (
            metadata.get("is_code")
            or Path(filename).suffix.lower() in _CODE_EXTENSIONS
            or mime_type.startswith("text/x-") and mime_type != "text/x-markdown"
        ):
            lang = _detect_code_language(filename, mime_type)
            return ParsedDocument(
                blocks=[ParsedBlock(type="code", text=text, language=lang, order=0)],
                metadata=metadata,
                source_type="code",
                title=metadata.get("title") or filename,
            )

        blocks: List[ParsedBlock] = []

        if mime_type in {"text/markdown", "text/x-markdown"}:
            blocks = _parse_markdown(text)
        else:
            # Plain text: paragraph-split on blank lines.
            for i, para in enumerate(re.split(r"\n{2,}", text)):
                para = para.strip()
                if para:
                    blocks.append(ParsedBlock(type="paragraph", text=para, order=i))

        if not blocks and text.strip():
            blocks = [ParsedBlock(type="paragraph", text=text.strip(), order=0)]

        return ParsedDocument(
            blocks=blocks,
            metadata=metadata,
            source_type="markdown" if "markdown" in mime_type else "text",
            title=metadata.get("title"),
        )


_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^```([\w-]*)\s*$")


def _parse_markdown(text: str) -> List[ParsedBlock]:
    blocks: List[ParsedBlock] = []
    lines = text.splitlines()
    i = 0
    order = 0

    def push(kind: str, buf: List[str], **extras) -> None:
        nonlocal order
        content = "\n".join(buf).strip()
        if content:
            blocks.append(ParsedBlock(type=kind, text=content, order=order, **extras))
            order += 1

    while i < len(lines):
        line = lines[i]
        fence = _FENCE_RE.match(line.strip())
        if fence:
            lang = fence.group(1) or None
            buf: List[str] = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                buf.append(lines[i])
                i += 1
            i += 1  # skip closing fence
            push("code", buf, language=lang)
            continue

        heading = _HEADING_RE.match(line)
        if heading:
            push("heading", [heading.group(2)], heading_level=len(heading.group(1)))
            i += 1
            continue

        # List run: consecutive lines starting with -, *, or N.
        if re.match(r"^\s*[-*+]\s+", line) or re.match(r"^\s*\d+\.\s+", line):
            buf = []
            while i < len(lines) and (
                re.match(r"^\s*[-*+]\s+", lines[i]) or re.match(r"^\s*\d+\.\s+", lines[i])
            ):
                buf.append(lines[i])
                i += 1
            push("list", buf)
            continue

        # Paragraph: consecutive non-blank lines.
        if line.strip():
            buf = []
            while i < len(lines) and lines[i].strip():
                buf.append(lines[i])
                i += 1
            push("paragraph", buf)
            continue

        i += 1

    return blocks
