"""Small input sanitization helpers for the Data Analysis Agent.

Intentionally minimal: most of the user input here is structured data
(CSV, XLSX, JSON) which pandas validates on parse, and the LLM facing
strings only need length + control char trimming.
"""

from __future__ import annotations

import re
from typing import Any


_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def clean_question(value: str | None, *, max_len: int = 1000) -> str:
    """Strip control characters and cap length.  Leaves printable
    Unicode alone so non English questions still work."""
    if not value:
        return ""
    text = str(value)
    text = _CTRL_RE.sub(" ", text)
    text = text.strip()
    if len(text) > max_len:
        text = text[:max_len]
    return text


import math


def _safe_float(value: float) -> Any:
    """Return None for NaN / Infinity (Postgres JSONB rejects both),
    else the float."""
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def coerce_jsonable(value: Any) -> Any:
    """Make a pandas / numpy value safe for JSONB storage.  Recursively
    converts numpy scalars, NaN, Infinity, and Timestamps to plain
    Python that asyncpg's JSONB encoder accepts."""
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        return value

    if value is None:
        return None
    if isinstance(value, bool):  # bool first; bool is a subclass of int
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return _safe_float(value)
    if isinstance(value, str):
        return value
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return _safe_float(float(value))
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, (pd.Timestamp,)):
        try:
            return value.isoformat()
        except Exception:  # NaT etc.
            return None
    if isinstance(value, (list, tuple)):
        return [coerce_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): coerce_jsonable(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [coerce_jsonable(v) for v in value]
    return str(value)
