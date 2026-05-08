"""HTML / text sanitization for user-supplied customer-service content.

Centralized so portal / widget / operator inputs all hit the same allow-list.
We never trust the database to be HTML-safe — every render path that emits
user content also escapes again on output, but stripping at write-time
keeps the row sane for search + analytics.
"""

from __future__ import annotations

from typing import Iterable

try:
    import bleach  # type: ignore
    HAS_BLEACH = True
except ImportError:  # pragma: no cover — bleach is in requirements.txt
    HAS_BLEACH = False


# Conservative allow-list: keep formatting that operators actually use,
# strip everything that could carry XSS (script, iframe, on* attributes).
_ALLOWED_TAGS: frozenset = frozenset({
    "p", "br", "strong", "b", "em", "i", "u", "s",
    "ul", "ol", "li", "a", "blockquote", "code", "pre",
    "h3", "h4", "h5", "h6", "hr",
})
_ALLOWED_ATTRS = {
    "a": ["href", "title", "rel"],
    "*": [],
}
_ALLOWED_PROTOCOLS = ["http", "https", "mailto"]


def clean_rich_text(value: str | None, *, max_len: int = 50_000) -> str:
    """Strip dangerous HTML, cap length, normalize whitespace.

    For free-form ticket bodies and reply bodies.  Length capping happens
    BEFORE bleach so we can't be tricked into spending CPU sanitizing a
    multi-megabyte payload.
    """
    if not value:
        return ""
    text = str(value)
    if len(text) > max_len:
        text = text[:max_len]
    if HAS_BLEACH:
        text = bleach.clean(
            text,
            tags=_ALLOWED_TAGS,
            attributes=_ALLOWED_ATTRS,
            protocols=_ALLOWED_PROTOCOLS,
            strip=True,
        )
    return text.strip()


def clean_plain_text(value: str | None, *, max_len: int = 500) -> str:
    """Strip ALL HTML — used for subject, name, etc. where formatting
    is never desirable."""
    if not value:
        return ""
    text = str(value)
    if len(text) > max_len:
        text = text[:max_len]
    if HAS_BLEACH:
        text = bleach.clean(text, tags=[], strip=True)
    return text.strip()


def normalize_email(value: str | None) -> str:
    """Lowercase + trim. Validation happens upstream via Pydantic EmailStr."""
    return (value or "").strip().lower()


def sanitize_string_list(values: Iterable[str], *, max_each: int = 64) -> list[str]:
    """For org_branding.public_categories etc. — strip + cap each entry,
    drop empties."""
    out: list[str] = []
    for v in values or []:
        s = clean_plain_text(v, max_len=max_each)
        if s:
            out.append(s)
    return out
