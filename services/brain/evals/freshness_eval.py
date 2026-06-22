"""Freshness eval — what % of fetched items are newer than the last run.

A high freshness score means the brain is doing useful work; a 0%
score means we're re-ingesting yesterday's already-seen content,
likely because the last_run_at watermark on the user isn't being
updated. Below 50% → log a fallback for the run.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Optional

from ..state import EvalResult


def check_freshness(
    items: Iterable[object],
    *,
    since: Optional[datetime],
    timestamp_attr: str = "received_at",
    field: str = "items",
) -> EvalResult:
    """passed iff ≥50% of items have ``timestamp_attr`` > ``since``.

    ``since=None`` is treated as passing with score=1.0 — the user
    hasn't had a brain run yet so everything counts as fresh.
    """
    item_list = list(items)
    if not item_list:
        return EvalResult(
            score=1.0,
            passed=True,
            reason="no_items",
            checked_fields=[field, timestamp_attr],
        )
    if since is None:
        return EvalResult(
            score=1.0,
            passed=True,
            reason="no_baseline",
            checked_fields=[field, timestamp_attr],
        )

    fresh = 0
    total = 0
    for item in item_list:
        ts = _get_ts(item, timestamp_attr)
        if ts is None:
            continue
        total += 1
        if ts > since:
            fresh += 1

    if total == 0:
        return EvalResult(
            score=0.0,
            passed=False,
            reason="no_timestamps",
            checked_fields=[field, timestamp_attr],
        )

    ratio = fresh / total
    return EvalResult(
        score=round(ratio, 3),
        passed=ratio >= 0.5,
        reason=f"{fresh}/{total} fresh = {ratio:.2%}",
        checked_fields=[field, timestamp_attr],
    )


def _get_ts(item: object, attr: str) -> Optional[datetime]:
    val = None
    if hasattr(item, attr):
        val = getattr(item, attr)
    elif isinstance(item, dict):
        val = item.get(attr)
    if val is None:
        return None
    if isinstance(val, datetime):
        return val.replace(tzinfo=None) if val.tzinfo else val
    if isinstance(val, str):
        try:
            ts = datetime.fromisoformat(val.replace("Z", "+00:00"))
            return ts.replace(tzinfo=None) if ts.tzinfo else ts
        except Exception:
            return None
    return None
