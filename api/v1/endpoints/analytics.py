"""
Phase 9 — Dashboard analytics endpoint.

Returns the full Dashboard payload in one round-trip:

    GET /api/v1/analytics/dashboard?range=30d
    {
      "time_range": "30d",
      "productivity": {"score": 78, "band": "strong", "components": {...}},
      "tasks": {"total": ..., "completed": ..., "series": [...], ...},
      "agents": {"total_runs": ..., "leaderboard": [...], "series": [...], ...},
      "documents": {"total": ..., "by_type": [...], "series": [...], ...},
      "proposals": {"pending_review": ..., "approved": ..., "pending": [...]},
      "activity": [{...}, ...]
    }
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends, HTTPException, Query
import structlog

from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.services.analytics_service import build_dashboard

logger = structlog.get_logger(__name__)
router = APIRouter()


def _get_org_id(user: User) -> str:
    """Fallback: when the user is operating "implicitly" with no org row,
    use their user_id as the org id (same convention everything else in the
    codebase uses)."""
    return (
        getattr(user, "organization_id", None)
        or (getattr(user, "organization_ids", None) or [None])[0]
        or str(user.id)
    )


@router.get("/dashboard", response_model=None)
async def get_dashboard(
    range: str = Query("30d", description="1d | 7d | 30d | 90d | 1y"),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """One-shot dashboard aggregator.  Cheap by design — every panel reads
    from the same payload."""
    if range not in {"1d", "7d", "30d", "90d", "1y"}:
        raise HTTPException(
            status_code=400,
            detail="range must be one of: 1d, 7d, 30d, 90d, 1y",
        )

    try:
        return await build_dashboard(
            organization_id=_get_org_id(current_user),
            user_id=str(current_user.id),
            time_range=range,
        )
    except Exception as e:  # noqa: BLE001
        logger.error("dashboard_build_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Dashboard build failed: {str(e)[:200]}")
