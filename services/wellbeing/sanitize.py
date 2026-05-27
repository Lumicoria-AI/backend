"""Input sanitisation for the Well-being module.

Same conventions as the other service packages: strip control
characters, run bleach over user-supplied strings, cap lengths.
Used by the orchestrator (before LLM calls) and the router (before
persisting previews / titles).
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Optional

import bleach


MAX_MESSAGE_CHARS = 8_000        # one chat turn
MAX_LABEL_CHARS = 500
MAX_PARAM_VALUE_CHARS = 2_000


_CONTROL_CHARS = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(value: Any, max_len: int = MAX_MESSAGE_CHARS) -> str:
    """Strip control characters, run through bleach, cap length."""
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


def clean_parameters(params: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Recursively clean a parameters dict."""
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


def coerce_jsonable(value: Any) -> Any:
    """Make any value safe for JSON serialisation.

    Handles:
      - None / bool / int / float (NaN and Infinity → None)
      - str (passthrough)
      - list / tuple / set (recurse, set becomes list)
      - dict (recurse with str keys)
      - datetime / date (ISO string)
      - bson.ObjectId (str)
      - anything else (best-effort str)
    """
    from datetime import datetime, date

    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set)):
        return [coerce_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    try:
        from bson import ObjectId

        if isinstance(value, ObjectId):
            return str(value)
    except Exception:  # noqa: BLE001
        pass
    try:
        return str(value)
    except Exception:  # noqa: BLE001
        return None
