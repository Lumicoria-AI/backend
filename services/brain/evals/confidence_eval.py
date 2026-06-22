"""Confidence-floor eval.

The Brain Agent and classify node each have items that carry a
``confidence`` field. This eval takes a list of items + a floor and
returns:

  passed=True   if mean confidence ≥ floor.
  passed=False  otherwise.

Score is the mean confidence (0–1). The node uses this to decide
whether to retry or fallback.
"""

from __future__ import annotations

from typing import Any, Iterable

from ..state import EvalResult


def check_confidence(
    items: Iterable[Any],
    *,
    floor: float = 0.6,
    attr: str = "confidence",
) -> EvalResult:
    """Return mean confidence + pass/fail vs. floor.

    Works on Pydantic models, dataclasses, or plain dicts — anything
    with the ``attr`` attribute or key.
    """
    scores: list[float] = []
    for item in items:
        score = _get(item, attr)
        if score is None:
            continue
        try:
            scores.append(float(score))
        except Exception:
            continue

    if not scores:
        return EvalResult(
            score=0.0,
            passed=False,
            reason="no_confidence_values",
            checked_fields=[attr],
        )

    mean_score = sum(scores) / len(scores)
    return EvalResult(
        score=round(mean_score, 3),
        passed=mean_score >= floor,
        reason=f"mean={mean_score:.3f} floor={floor}",
        checked_fields=[attr],
    )


def _get(item: Any, attr: str) -> Any:
    if hasattr(item, attr):
        return getattr(item, attr)
    if isinstance(item, dict):
        return item.get(attr)
    return None
