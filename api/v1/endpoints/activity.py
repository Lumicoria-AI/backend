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


def _resolve_org_id(user: User, organization_id: Optional[str]) -> Optional[str]:
    """Pick an org id for the audit query.  Prefer the explicit query
    param (so the frontend can scope), then fall back to whichever org
    id field the User model carries — `organization_id`,
    `primary_organization_id`, or the first item in
    `organization_ids`.  Returns None when the user belongs to no org."""
    if organization_id:
        return organization_id
    for key in ("organization_id", "primary_organization_id", "default_organization_id"):
        v = getattr(user, key, None)
        if v:
            return str(v)
    orgs = getattr(user, "organization_ids", None) or []
    if orgs:
        return str(orgs[0])
    return None


def _as_naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Coerce a datetime to a tz-naive UTC instant.

    The frontend sends ISO strings ending in `Z` which FastAPI parses
    into tz-aware datetimes.  Mongo timestamps are typically stored
    naive (assumed UTC).  Comparing the two raises:
        TypeError: can't compare offset-naive and offset-aware datetimes
    This helper drops the tzinfo (after converting to UTC if needed) so
    both sides of a `<` / `>` comparison are naive UTC.
    """
    if dt is None:
        return None
    if dt.tzinfo is not None:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt


def _entry_to_jsonable(entry: Any) -> Dict[str, Any]:
    """Coerce a stored ActivityLogEntry into a JSON-safe dict.

    The Pydantic model uses ObjectId for id / user_id / organization_id
    (and `details` may carry arbitrary ObjectIds), which FastAPI's
    Pydantic validator rejects for str-typed response models.  Convert
    every ObjectId-shaped value to a string here so the endpoint can
    return the rows without a response_model gate.
    """
    if hasattr(entry, "model_dump"):
        d = entry.model_dump(mode="json")
    elif hasattr(entry, "dict"):
        d = entry.dict()
    else:
        d = dict(entry) if isinstance(entry, dict) else {}
    from backend.db.serializers import stringify_oids
    return stringify_oids(d)


@router.get("/me/audit")
async def me_audit(
    organization_id: Optional[str] = Query(None),
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
    org_id = _resolve_org_id(current_user, organization_id)
    if not org_id:
        # User belongs to no org — return an empty feed rather than 500.
        return []
    activities = await activity_repository.get_recent_activity(
        organization_id=org_id,
        user_id=current_user.id,
        activity_type=activity_type,
        limit=max(limit + skip, 200),
        skip=0,
    )
    start_naive = _as_naive_utc(start_date)
    end_naive = _as_naive_utc(end_date)
    out: List[Dict[str, Any]] = []
    for entry in activities:
        ts = _as_naive_utc(entry.timestamp)
        if start_naive and ts and ts < start_naive:
            continue
        if end_naive and ts and ts > end_naive:
            continue
        if severity and getattr(entry, "severity", "info") != severity:
            continue
        out.append(_entry_to_jsonable(entry))
    return out[skip:skip + limit]


@router.get("/me/audit/export")
async def me_audit_export(
    organization_id: Optional[str] = Query(None),
    activity_type: Optional[str] = None,
    severity: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_active_user),
):
    """Stream the signed-in user's audit history as a CSV.  Includes the
    same filters as /me/audit so the export matches what the user sees."""
    org_id = _resolve_org_id(current_user, organization_id)
    if not org_id:
        activities: List[Any] = []
    else:
        activities = await activity_repository.get_recent_activity(
            organization_id=org_id,
            user_id=current_user.id,
            activity_type=activity_type,
            limit=5000,
            skip=0,
        )

    start_naive = _as_naive_utc(start_date)
    end_naive = _as_naive_utc(end_date)

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
            ts = _as_naive_utc(entry.timestamp)
            if start_naive and ts and ts < start_naive:
                continue
            if end_naive and ts and ts > end_naive:
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
