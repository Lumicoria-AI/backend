from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from datetime import datetime, timezone
import csv
import io

from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.db.mongodb.repositories.activity_repository import activity_repository
from backend.db.cassandra.activity_repository import cassandra_activity_repository
from backend.core.config import settings
from backend.services.activity_logger import log_activity

router = APIRouter()

class ActivityEntry(BaseModel):
    id: str
    user_id: str
    organization_id: str
    activity_type: str # e.g., 'document.uploaded', 'agent.executed', 'task.created'
    timestamp: datetime
    details: Dict[str, Any] # Specific details about the activity
    related_resource_type: Optional[str] = None # e.g., 'DOCUMENT', 'AGENT', 'TASK'
    related_resource_id: Optional[str] = None

class ActivitySummaryResponse(BaseModel):
    total_events: int
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    time_range: Dict[str, Optional[datetime]]

class InternalLogRequest(BaseModel):
    activity_type: str
    details: Dict[str, Any] = {}
    related_resource_type: Optional[str] = None
    related_resource_id: Optional[str] = None
    agent_id: Optional[str] = None
    agent_name: Optional[str] = None
    severity: str = "info"

@router.get("/recent", response_model=List[ActivityEntry])
async def get_recent_activity(
    limit: int = Query(10, ge=1, le=50),
    skip: int = Query(0, ge=0),
    activity_type: Optional[str] = None,
    agent_id: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get a list of recent activities for the user or organization.
    Optionally filter by activity_type or agent_id.
    """
    if settings.db.CASSANDRA_ENABLED:
        activities = await cassandra_activity_repository.get_recent_activity(
            organization_id=current_user.organization_id,
            user_id=current_user.id,
            activity_type=activity_type,
            limit=limit,
            skip=skip
        )
    else:
        activities = await activity_repository.get_recent_activity(
            organization_id=current_user.organization_id,
            user_id=current_user.id,
            activity_type=activity_type,
            limit=limit,
            skip=skip
        )

    # Post-filter by agent_id if requested (in-memory, since the repo
    # doesn't natively support agent_id filtering yet)
    if agent_id:
        activities = [
            a for a in activities
            if getattr(a, "details", {}).get("agent_id") == agent_id
        ]

    return activities

@router.get("/by-agent/{agent_id}", response_model=List[ActivityEntry])
async def get_activity_by_agent(
    agent_id: str,
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get activities for a specific agent.
    """
    activities = await activity_repository.get_recent_activity(
        organization_id=current_user.organization_id,
        user_id=current_user.id,
        limit=200,  # fetch more to filter
        skip=0
    )

    # Filter by agent_id in details
    agent_activities = [
        a for a in activities
        if getattr(a, "details", {}).get("agent_id") == agent_id
    ]

    return agent_activities[skip:skip + limit]

@router.get("/summary", response_model=ActivitySummaryResponse)
async def get_activity_summary(
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get aggregate summary of activity logs.
    """
    if settings.db.CASSANDRA_ENABLED:
        summary = await cassandra_activity_repository.get_activity_summary(
            organization_id=current_user.organization_id,
            user_id=current_user.id,
            start_date=start_date,
            end_date=end_date
        )
        return summary

    activities = await activity_repository.get_recent_activity(
        organization_id=current_user.organization_id,
        user_id=current_user.id,
        limit=500,
        skip=0
    )
    by_type: Dict[str, int] = {}
    by_severity: Dict[str, int] = {}
    for entry in activities:
        ts = entry.timestamp
        if start_date and ts < start_date:
            continue
        if end_date and ts > end_date:
            continue
        by_type[entry.activity_type] = by_type.get(entry.activity_type, 0) + 1
        severity = getattr(entry, "severity", "info")
        by_severity[severity] = by_severity.get(severity, 0) + 1

    return {
        "total_events": sum(by_type.values()),
        "by_type": by_type,
        "by_severity": by_severity,
        "time_range": {"start": start_date, "end": end_date},
    }

# --- User-scoped audit log (Profile → Audit) ---


@router.get("/me/audit", response_model=List[ActivityEntry])
async def me_audit(
    limit: int = Query(200, ge=1, le=1000),
    skip: int = Query(0, ge=0),
    activity_type: Optional[str] = None,
    severity: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Return a chronological audit feed of the signed-in user's actions
    across this workspace.  Powers the /audit page.  Filters apply
    post-fetch when the underlying repo doesn't support them natively."""
    activities = await activity_repository.get_recent_activity(
        organization_id=current_user.organization_id,
        user_id=current_user.id,
        activity_type=activity_type,
        limit=max(limit + skip, 200),
        skip=0,
    )
    out: List[Any] = []
    for entry in activities:
        ts = entry.timestamp
        if start_date and ts < start_date:
            continue
        if end_date and ts > end_date:
            continue
        if severity and getattr(entry, "severity", "info") != severity:
            continue
        out.append(entry)
    return out[skip:skip + limit]


@router.get("/me/audit/export")
async def me_audit_export(
    activity_type: Optional[str] = None,
    severity: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_active_user),
):
    """Stream the signed-in user's audit history as a CSV.  Includes the
    same filters as /me/audit so the export matches what the user sees."""
    activities = await activity_repository.get_recent_activity(
        organization_id=current_user.organization_id,
        user_id=current_user.id,
        activity_type=activity_type,
        limit=5000,
        skip=0,
    )

    def _rows():
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "timestamp", "activity_type", "severity",
            "related_resource_type", "related_resource_id",
            "ip_address", "user_agent", "details",
        ])
        yield buf.getvalue()
        buf.seek(0); buf.truncate(0)
        for entry in activities:
            ts = entry.timestamp
            if start_date and ts < start_date:
                continue
            if end_date and ts > end_date:
                continue
            sev = getattr(entry, "severity", "info")
            if severity and sev != severity:
                continue
            details = getattr(entry, "details", {}) or {}
            writer.writerow([
                ts.isoformat() if ts else "",
                entry.activity_type,
                sev,
                getattr(entry, "related_resource_type", "") or "",
                getattr(entry, "related_resource_id", "") or "",
                details.get("ip_address", "") if isinstance(details, dict) else "",
                details.get("user_agent", "") if isinstance(details, dict) else "",
                str(details) if details else "",
            ])
            yield buf.getvalue()
            buf.seek(0); buf.truncate(0)

    filename = f"lumicoria-audit-{datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}.csv"
    return StreamingResponse(
        _rows(),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


# --- Internal debug endpoint (keep for development) ---

@router.post("/internal/log", status_code=status.HTTP_201_CREATED)
async def log_activity_internal(
    body: InternalLogRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    (Internal / Debug) Manually log an activity entry.
    Useful for testing the activity pipeline during development.
    """
    await log_activity(
        user_id=str(current_user.id),
        organization_id=current_user.organization_id,
        activity_type=body.activity_type,
        details=body.details,
        related_resource_type=body.related_resource_type,
        related_resource_id=body.related_resource_id,
        agent_id=body.agent_id,
        agent_name=body.agent_name,
        severity=body.severity,
    )
    return {"message": "Activity logged successfully"}
