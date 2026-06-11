"""
Lumicoria AI — Mongo → JSON-safe serializer.

`stringify_oids(doc)` walks a dict / list recursively and replaces every
ObjectId with its string form, swaps `_id` for `id`, and leaves other
JSON-friendly types alone.  Use it on any endpoint that returns raw
collection rows whose nested fields aren't guaranteed to be Pydantic-
sanitised — most notably `activity_logs`, where `details` can contain
arbitrary ObjectId values.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any, Dict, List

from bson import ObjectId


def stringify_oids(value: Any) -> Any:
    """Return a JSON-safe copy of `value`.

    - Dict: walks every key, rewrites `_id` → `id`, recurses on values.
    - List / tuple: recurses on items.
    - ObjectId: returns `str(value)`.
    - datetime / date: left as-is (FastAPI's JSON encoder handles those).
    - Everything else: returned untouched.
    """
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, dict):
        out: Dict[str, Any] = {}
        for k, v in value.items():
            if k == "_id":
                out["id"] = stringify_oids(v)
            else:
                out[k] = stringify_oids(v)
        return out
    if isinstance(value, list):
        return [stringify_oids(v) for v in value]
    if isinstance(value, tuple):
        return tuple(stringify_oids(v) for v in value)
    return value


def stringify_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convenience for `[stringify_oids(r) for r in rows]`."""
    return [stringify_oids(r) for r in rows]
