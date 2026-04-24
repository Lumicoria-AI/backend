"""Structure-aware token-sized chunker.

Rules:
  - Token counts come from tiktoken's cl100k_base (falls back to word count
    if tiktoken isn't installed).
  - Tables, code, lists, and figures are emitted as single chunks whenever
    they fit; oversize tables split by row groups, oversize code by newline
    runs, oversize lists by items.
  - Prose blocks are packed greedily up to the policy target with token
    overlap between adjacent chunks.
  - A heading immediately preceding prose is prepended to the next chunk so
    semantic context survives.
  - Every chunk carries `prev_context` / `next_context` (first ~80 tokens of
    neighbours) and `content_sha256` for exact-dedup.

Public entry point: `chunk_document(doc, user_metadata=None)`.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import structlog

from ...core.config import settings
from .base import ParsedBlock, ParsedDocument

logger = structlog.get_logger(__name__)


try:
    import tiktoken
    _ENC = tiktoken.get_encoding("cl100k_base")
    HAS_TIKTOKEN = True
except Exception:  # noqa: BLE001 — tiktoken download can fail offline
    _ENC = None
    HAS_TIKTOKEN = False


def _count_tokens(text: str) -> int:
    if _ENC is not None:
        return len(_ENC.encode(text, disallowed_special=()))
    # 1 token ≈ 4 chars is a reasonable fallback for English.
    return max(1, len(text) // 4)


def _encode(text: str) -> List[int]:
    if _ENC is not None:
        return _ENC.encode(text, disallowed_special=())
    return list(text.encode("utf-8"))


def _decode(tokens: List[int]) -> str:
    if _ENC is not None:
        return _ENC.decode(tokens)
    return bytes(tokens).decode("utf-8", errors="replace")


@dataclass
class Chunk:
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class _Policy:
    target_tokens: int
    overlap_tokens: int
    max_table_rows: int = 20
    max_code_tokens: int = 1000


def _policy_for(source_type: str) -> _Policy:
    overlap = int(getattr(settings, "INGEST_CHUNK_OVERLAP_TOKENS", 50))
    if source_type in {"chat", "chat_history"}:
        return _Policy(int(getattr(settings, "INGEST_CHUNK_TOKENS_CHAT", 1500)), overlap)
    if source_type == "code":
        return _Policy(int(getattr(settings, "INGEST_CHUNK_TOKENS_CODE", 1000)), overlap=0, max_code_tokens=int(getattr(settings, "INGEST_CHUNK_TOKENS_CODE", 1000)))
    return _Policy(int(getattr(settings, "INGEST_CHUNK_TOKENS_PROSE", 512)), overlap)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _context_snippet(text: str, max_tokens: int = 80) -> str:
    tokens = _encode(text)
    if len(tokens) <= max_tokens:
        return text
    return _decode(tokens[:max_tokens])


def _pack_prose(
    buffer: List[ParsedBlock], policy: _Policy
) -> List[Dict[str, Any]]:
    """Greedy-pack a run of splittable blocks (paragraphs/headings/captions)
    into chunks of ~target_tokens with token overlap."""
    out: List[Dict[str, Any]] = []
    if not buffer:
        return out

    # Flatten the run into a single token stream while remembering the
    # block-origin of every token so we can carry positional metadata onto
    # the resulting chunks.
    all_tokens: List[int] = []
    # token_index -> (page_number, bbox, pw, ph, order)
    origin: List[Optional[ParsedBlock]] = []

    pending_heading: Optional[ParsedBlock] = None
    for blk in buffer:
        if blk.type == "heading":
            pending_heading = blk
            continue
        prefix = ""
        if pending_heading is not None:
            prefix = pending_heading.text + "\n\n"
            pending_heading = None
        text = prefix + blk.text
        toks = _encode(text)
        all_tokens.extend(toks)
        origin.extend([blk] * len(toks))

    if not all_tokens:
        return out

    step = max(1, policy.target_tokens - policy.overlap_tokens)
    i = 0
    while i < len(all_tokens):
        window = all_tokens[i : i + policy.target_tokens]
        if not window:
            break
        text = _decode(window).strip()
        if not text:
            i += step
            continue
        anchor = origin[i] if i < len(origin) else None
        last = origin[min(i + len(window) - 1, len(origin) - 1)] if origin else None
        out.append({
            "text": text,
            "page_number": anchor.page_number if anchor else None,
            "bbox": list(anchor.bbox) if anchor and anchor.bbox else None,
            "page_width": anchor.page_width if anchor else None,
            "page_height": anchor.page_height if anchor else None,
            "block_index": anchor.order if anchor else None,
            "page_end": last.page_number if last else None,
        })
        if i + policy.target_tokens >= len(all_tokens):
            break
        i += step
    return out


def _split_table(blk: ParsedBlock, policy: _Policy) -> List[Dict[str, Any]]:
    """Chunk a table by row groups; repeat the header on every group."""
    rows = blk.table_rows or []
    if not rows:
        return [{"text": blk.text, "page_number": blk.page_number,
                 "bbox": list(blk.bbox) if blk.bbox else None,
                 "page_width": blk.page_width, "page_height": blk.page_height,
                 "block_index": blk.order}]
    header, body = rows[0], rows[1:]
    group_size = max(1, policy.max_table_rows - 1)  # leave room for header
    chunks: List[Dict[str, Any]] = []
    for i in range(0, max(1, len(body)), group_size):
        group = [header, *body[i : i + group_size]] if body else [header]
        md = _rows_to_markdown(group)
        chunks.append({
            "text": md,
            "page_number": blk.page_number,
            "bbox": list(blk.bbox) if blk.bbox else None,
            "page_width": blk.page_width,
            "page_height": blk.page_height,
            "block_index": blk.order,
        })
    return chunks or [{"text": blk.text, "page_number": blk.page_number}]


def _rows_to_markdown(rows: List[List[str]]) -> str:
    if not rows:
        return ""
    header = rows[0]
    body = rows[1:]
    out = ["| " + " | ".join(c.replace("|", "\\|") for c in header) + " |"]
    out.append("| " + " | ".join("---" for _ in header) + " |")
    for row in body:
        padded = row + [""] * (len(header) - len(row))
        out.append("| " + " | ".join(c.replace("|", "\\|") for c in padded[: len(header)]) + " |")
    return "\n".join(out)


_LANG_MAP = {
    "python": "PYTHON", "javascript": "JS", "typescript": "TS",
    "js": "JS", "ts": "TS", "jsx": "JS", "tsx": "TS",
    "java": "JAVA", "go": "GO", "rust": "RUST", "rs": "RUST",
    "cpp": "CPP", "c": "CPP", "csharp": "CSHARP", "cs": "CSHARP",
    "ruby": "RUBY", "rb": "RUBY", "php": "PHP",
    "swift": "SWIFT", "kotlin": "KOTLIN",
    "scala": "SCALA", "html": "HTML", "markdown": "MARKDOWN",
    "md": "MARKDOWN", "sol": "SOL", "proto": "PROTO",
}


def _lang_aware_split(text: str, language: str, max_tokens: int) -> List[str]:
    """Use langchain's per-language separator list for semantic splits
    (def/class/fn boundaries).  Falls back to newline split if unavailable."""
    try:
        from langchain_text_splitters import Language, RecursiveCharacterTextSplitter
    except ImportError:
        return text.split("\n\n")

    lang_key = _LANG_MAP.get((language or "").lower())
    if not lang_key:
        return text.split("\n\n")
    try:
        lang_enum = getattr(Language, lang_key)
    except AttributeError:
        return text.split("\n\n")

    # Approx chars-per-token = 4 for code; langchain's splitter is char-based.
    chars = max_tokens * 4
    splitter = RecursiveCharacterTextSplitter.from_language(
        language=lang_enum, chunk_size=chars, chunk_overlap=0
    )
    return splitter.split_text(text)


def _split_code(blk: ParsedBlock, policy: _Policy) -> List[Dict[str, Any]]:
    """Split oversized code on semantic boundaries (def/class/fn) when the
    language is known.  Falls back to blank-line splitting otherwise.
    Never splits mid-line."""
    tokens = _count_tokens(blk.text)
    if tokens <= policy.max_code_tokens:
        return [{"text": blk.text, "block_index": blk.order}]

    if blk.language:
        pieces = _lang_aware_split(blk.text, blk.language, policy.max_code_tokens)
        if len(pieces) > 1:
            return [{"text": p, "block_index": blk.order} for p in pieces if p.strip()]

    out: List[Dict[str, Any]] = []
    chunks = blk.text.split("\n\n")
    buf: List[str] = []
    buf_tokens = 0
    for piece in chunks:
        t = _count_tokens(piece)
        if buf_tokens + t > policy.max_code_tokens and buf:
            out.append({"text": "\n\n".join(buf), "block_index": blk.order})
            buf, buf_tokens = [], 0
        buf.append(piece)
        buf_tokens += t
    if buf:
        out.append({"text": "\n\n".join(buf), "block_index": blk.order})
    return out


def _split_list(blk: ParsedBlock, policy: _Policy) -> List[Dict[str, Any]]:
    """Split oversized lists by item boundaries — never mid-item."""
    tokens = _count_tokens(blk.text)
    if tokens <= policy.target_tokens:
        return [{"text": blk.text, "page_number": blk.page_number,
                 "bbox": list(blk.bbox) if blk.bbox else None,
                 "page_width": blk.page_width, "page_height": blk.page_height,
                 "block_index": blk.order}]

    items = blk.text.split("\n")
    out: List[Dict[str, Any]] = []
    buf: List[str] = []
    buf_tokens = 0
    for item in items:
        t = _count_tokens(item)
        if buf_tokens + t > policy.target_tokens and buf:
            out.append({"text": "\n".join(buf), "block_index": blk.order,
                        "page_number": blk.page_number})
            buf, buf_tokens = [], 0
        buf.append(item)
        buf_tokens += t
    if buf:
        out.append({"text": "\n".join(buf), "block_index": blk.order,
                    "page_number": blk.page_number})
    return out


def chunk_document(
    doc: ParsedDocument,
    user_metadata: Optional[Dict[str, Any]] = None,
) -> List[Chunk]:
    """Chunk a ParsedDocument honoring structural boundaries."""
    user_metadata = user_metadata or {}
    policy = _policy_for(doc.source_type)

    # First pass: emit raw chunk dicts in reading order, grouping adjacent
    # splittable blocks so they pack together.
    raw: List[Dict[str, Any]] = []
    pending: List[ParsedBlock] = []

    def flush_pending() -> None:
        if pending:
            raw.extend(_pack_prose(pending, policy))
            pending.clear()

    for blk in doc.blocks:
        if blk.is_splittable():
            pending.append(blk)
            continue
        flush_pending()
        if blk.type == "table":
            raw.extend(_split_table(blk, policy))
        elif blk.type == "code":
            raw.extend(_split_code(blk, policy))
        elif blk.type == "list":
            raw.extend(_split_list(blk, policy))
        else:
            raw.append({
                "text": blk.text,
                "page_number": blk.page_number,
                "bbox": list(blk.bbox) if blk.bbox else None,
                "page_width": blk.page_width,
                "page_height": blk.page_height,
                "block_index": blk.order,
            })
    flush_pending()

    # Attach doc-level metadata + prev/next context + sha + chunk_id.
    base = dict(doc.metadata)
    base.update(user_metadata)
    dedup_enabled = bool(getattr(settings, "INGEST_CHUNK_DEDUP_ENABLED", True))
    seen_sha: set = set()

    chunks: List[Chunk] = []
    for idx, item in enumerate(raw):
        text = (item.get("text") or "").strip()
        if not text:
            continue
        sha = _sha(text)
        if dedup_enabled and sha in seen_sha:
            continue
        seen_sha.add(sha)

        prev_ctx = _context_snippet(raw[idx - 1]["text"]) if idx > 0 and raw[idx - 1].get("text") else ""
        next_ctx = _context_snippet(raw[idx + 1]["text"]) if idx + 1 < len(raw) and raw[idx + 1].get("text") else ""

        meta = dict(base)
        # Positional fields (keep exact names so citations still work).
        for key in ("page_number", "bbox", "page_width", "page_height", "block_index"):
            if item.get(key) is not None:
                meta[key] = item[key]
        meta["chunk_id"] = idx
        meta["content_sha256"] = sha
        meta["prev_context"] = prev_ctx
        meta["next_context"] = next_ctx
        meta.setdefault("mime_type", base.get("mime_type"))
        # Preserve existing char-offset tracking contract.
        meta.setdefault("start_char", 0)
        meta.setdefault("end_char", len(text))

        chunks.append(Chunk(text=text, metadata=meta))

    logger.info(
        "chunk_document",
        source_type=doc.source_type,
        blocks=len(doc.blocks),
        chunks=len(chunks),
        policy_tokens=policy.target_tokens,
        tiktoken=HAS_TIKTOKEN,
    )
    return chunks
