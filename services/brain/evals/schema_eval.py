"""Schema-validation eval.

Given a raw LLM output (string, dict, or list) and a target Pydantic
model class, returns an EvalResult that's:

  passed=True, score=1.0   if every input element validates.
  passed=True, score=0.X   if some elements validate (X = ratio).
  passed=False, score=0.0  if nothing validates.

Used by the classify + prioritise nodes — both ask the LLM for a JSON
list of structured items. Mal-typed items get dropped; if too many
drop, the node retries with a stricter prompt; if a retry still under-
scores, it falls back to a heuristic.
"""

from __future__ import annotations

import json
from typing import Any, List, Tuple, Type, TypeVar

from pydantic import BaseModel, ValidationError

from ..state import EvalResult

_T = TypeVar("_T", bound=BaseModel)


def check_schema(
    raw: Any,
    item_model: Type[_T],
    *,
    pass_floor: float = 0.7,
) -> Tuple[List[_T], EvalResult]:
    """Validate ``raw`` against ``item_model`` (one model class per element).

    Args:
        raw: LLM output. Accepted shapes:
            - JSON string of an array
            - Python list of dicts
            - Single dict (treated as a 1-element list)
        item_model: Pydantic class each element must conform to.
        pass_floor: Minimum ratio of valid:total to mark passed=True.

    Returns:
        (valid_items, eval_result)
    """
    items_raw: List[Any] = _to_list(raw)
    total = len(items_raw)

    if total == 0:
        return [], EvalResult(
            score=0.0,
            passed=False,
            reason="empty_output",
            checked_fields=["__root__"],
        )

    valid: List[_T] = []
    errors: List[str] = []
    for i, item in enumerate(items_raw):
        try:
            valid.append(item_model.model_validate(item))
        except ValidationError as e:
            errors.append(f"item[{i}]: {e.error_count()} errors")
        except Exception as e:  # noqa: BLE001
            errors.append(f"item[{i}]: {type(e).__name__}: {e}")

    score = len(valid) / total if total else 0.0
    passed = score >= pass_floor

    reason_bits = [f"{len(valid)}/{total} valid"]
    if errors:
        reason_bits.append(f"errors={errors[:3]}")
    return valid, EvalResult(
        score=round(score, 3),
        passed=passed,
        reason="; ".join(reason_bits),
        checked_fields=list(item_model.model_fields.keys()),
    )


def _to_list(raw: Any) -> List[Any]:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        return [raw]
    if isinstance(raw, str):
        text = raw.strip()
        if not text:
            return []
        # Some LLMs wrap JSON in ```json fences — strip them.
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            parsed = json.loads(text)
        except Exception:
            return []
        if isinstance(parsed, list):
            return parsed
        if isinstance(parsed, dict):
            # LLM sometimes wraps the list in {"items": [...]}.
            for key in ("items", "results", "data", "tasks", "actions"):
                if key in parsed and isinstance(parsed[key], list):
                    return parsed[key]
            return [parsed]
        return []
    return []
