"""Input sanitisation for the Ethics & Bias Agent.

Same conventions as the Legal Document service:
  - strip control characters
  - run bleach over user text so HTML payloads do not reach the LLM
  - enforce length caps so a single submission can never exceed the
    model's context budget by accident

Used by both the orchestrator (before the LLM call) and the router
(before persisting input previews).
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

import bleach


# Ethics & Bias content can be long (full policies, contracts, code
# blocks, etc.) but we still need a hard cap.  150k chars is enough
# for a serious policy document while remaining safe for the LLM
# providers we route to.
MAX_CONTENT_CHARS = 150_000
MAX_LABEL_CHARS = 500
MAX_PARAM_VALUE_CHARS = 2_000

_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(value: Any, max_len: int = MAX_CONTENT_CHARS) -> str:
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
    """Single-line label cleanup.  Returns None for empty / whitespace."""
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


def clean_string_list(items: Any, max_items: int = 50, max_each: int = 200) -> List[str]:
    """Sanitise a list of short labels (guideline names, focus tags, etc.)."""
    if not items or not isinstance(items, list):
        return []
    out: List[str] = []
    for item in items[:max_items]:
        cleaned = clean_label(item, max_len=max_each)
        if cleaned:
            out.append(cleaned)
    return out


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


def make_title(action: str, content: str) -> str:
    """Best-effort short title for the history list.  Pulls the first
    sentence or first 60 characters, with the action as a prefix."""
    label = action.replace("_", " ").title()
    if not content:
        return label
    first = re.split(r"[.\n]", content.strip(), maxsplit=1)[0].strip()
    if len(first) > 80:
        first = first[:79].rstrip() + "…"
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


def coerce_jsonable(value: Any) -> Any:
    """Strip NaN / Infinity / non-JSON-safe types so the result can be
    persisted to Mongo and round-tripped to the frontend without
    surprises."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return [coerce_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): coerce_jsonable(v) for k, v in value.items()}
    # Fallback: stringify so persistence never blows up.
    try:
        return str(value)
    except Exception:
        return None
