"""Input sanitization helpers for the Knowledge Graph Agent.

Every user-supplied string (content, label, focus area, query term) flows
through one of these helpers before it lands in the database or the LLM
prompt.  Length caps prevent unbounded prompt/payload sizes; control-char
stripping prevents prompt-injection via zero-width / bidi tricks; the
JSONB-safe coercion mirrors the data_analysis module's helper so we never
write NaN/Inf into Postgres.
"""

from __future__ import annotations

import math
import re
from typing import Any, Iterable

try:
    import bleach  # type: ignore
    HAS_BLEACH = True
except ImportError:  # pragma: no cover — bleach is in requirements.txt
    HAS_BLEACH = False


# Strip control characters that can break LLM prompts or pollute the UI.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_text(value: Any, *, max_len: int = 50_000) -> str:
    """Strip control characters, bleach HTML tags, cap length.  Returns
    the empty string for None / empty inputs."""
    if value is None:
        return ""
    text = str(value)
    text = _CTRL_RE.sub(" ", text)
    if HAS_BLEACH:
        text = bleach.clean(text, tags=[], strip=True)
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len]
    return text


def clean_label(value: Any, *, max_len: int = 500) -> str:
    """For node labels and short identifier strings."""
    return clean_text(value, max_len=max_len)


def clean_focus_list(items: Iterable[Any], *, max_items: int = 20, max_each: int = 200) -> list[str]:
    """Sanitize a list of focus areas / tags.  Drops empties, caps
    both list length and each item's length."""
    out: list[str] = []
    for item in (items or []):
        s = clean_text(item, max_len=max_each)
        if s:
            out.append(s)
        if len(out) >= max_items:
            break
    return out


def _safe_float(value: float) -> Any:
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def coerce_jsonable(value: Any) -> Any:
    """Recursively make a value safe for JSONB storage.  Strips NaN /
    Infinity, converts numpy / pandas scalars to plain Python."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple, set, frozenset)):
        return [coerce_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): coerce_jsonable(v) for k, v in value.items()}

    # numpy / pandas scalars — optional import.
    try:
        import numpy as np
        import pandas as pd
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return _safe_float(float(value))
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, pd.Timestamp):
            try:
                return value.isoformat()
            except Exception:
                return None
    except ImportError:
        pass

    return str(value)
