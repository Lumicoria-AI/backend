"""Input sanitisation for the Legal Document Agent.

Mirrors the same approach used by customer_service and knowledge_graph:
strip control characters, cap length, allow only safe HTML-style markup
where relevant.  Defensive defaults — every user-supplied string flows
through here before reaching the agent or the database.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

import bleach


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


# Legal documents can be long.  We cap at 200k characters per call to
# avoid runaway prompt costs while still comfortably handling typical
# contracts and policies.
MAX_DOCUMENT_CHARS = 200_000
MAX_LABEL_CHARS = 500
MAX_PARAM_VALUE_CHARS = 2_000


def clean_text(value: Any, max_len: int = MAX_DOCUMENT_CHARS) -> str:
    """Strip control characters, run through bleach, and cap length."""
    if value is None:
        return ""
    text = str(value)
    text = _CONTROL_CHARS.sub("", text)
    text = bleach.clean(text, tags=[], attributes={}, strip=True)
    if len(text) > max_len:
        text = text[:max_len]
    return text.strip()


def clean_label(value: Any, max_len: int = MAX_LABEL_CHARS) -> Optional[str]:
    """Single-line label cleanup.  Returns None for empty/whitespace."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    text = _CONTROL_CHARS.sub("", text)
    text = re.sub(r"\s+", " ", text)
    if len(text) > max_len:
        text = text[: max_len - 1].rstrip() + "…"
    return text or None


def clean_parameters(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively clean a parameters dict.  Keeps booleans, numbers,
    None, and short strings; drops anything weird."""
    if not params or not isinstance(params, dict):
        return {}
    out: Dict[str, Any] = {}
    for key, value in params.items():
        if not isinstance(key, str) or len(key) > 64:
            continue
        clean_key = re.sub(r"[^a-zA-Z0-9_]", "", key)[:64]
        if not clean_key:
            continue
        out[clean_key] = _clean_param_value(value)
    return out


def _clean_param_value(value: Any) -> Any:
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, str):
        return clean_text(value, max_len=MAX_PARAM_VALUE_CHARS)
    if isinstance(value, list):
        return [_clean_param_value(v) for v in value[:50]]
    if isinstance(value, dict):
        return clean_parameters(value)
    return None


def make_title(mode: str, content: str) -> str:
    """Best-effort short title for the history list.  Pulls the first
    sentence or first 60 characters, with the mode as a prefix."""
    first = re.split(r"[.\n]", content.strip(), maxsplit=1)[0]
    first = first.strip()
    if len(first) > 80:
        first = first[:79].rstrip() + "…"
    label = mode.replace("_", " ").title()
    if not first:
        return label
    return f"{label}: {first}"


def make_preview(content: str, max_len: int = 500) -> str:
    """Short preview of the input for the history view."""
    if not content:
        return ""
    text = content.strip()
    if len(text) <= max_len:
        return text
    return text[: max_len - 1].rstrip() + "…"
