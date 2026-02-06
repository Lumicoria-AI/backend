from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from datetime import datetime

from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.db.mongodb.repositories.activity_repository import activity_repository
from backend.db.cassandra.activity_repository import cassandra_activity_repository
from backend.core.config import settings

router = APIRouter()

class ActivityEntry(BaseModel):
    id: str
    user_id: str
    organization_id: str
    activity_type: str # e.g., 'document_uploaded', 'agent_executed', 'task_created', 'break_suggested'
    timestamp: datetime
    details: Dict[str, Any] # Specific details about the activity
    related_resource_type: Optional[str] # e.g., 'DOCUMENT', 'AGENT', 'TASK'
    related_resource_id: Optional[str]

class ActivitySummaryResponse(BaseModel):
    total_events: int
    by_type: Dict[str, int]
    by_severity: Dict[str, int]
    time_range: Dict[str, Optional[datetime]]

@router.get("/recent", response_model=List[ActivityEntry])
async def get_recent_activity(
    limit: int = Query(10, ge=1, le=50),
    skip: int = Query(0, ge=0),
    activity_type: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get a list of recent activities for the user or organization.
    """
    # In a real implementation, this would involve querying a dedicated activity log
    # or aggregating recent entries from various repositories (documents, tasks, agent runs, wellbeing events)
    # For now, returning dummy data
    # print(f"Fetching recent activity for user {current_user.id} in organization {current_user.organization_id}")

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
            user_id=current_user.id, # Assuming the feed is primarily for the current user
            activity_type=activity_type,
            limit=limit,
            skip=skip
        )
    return activities

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

# You might also want endpoints for creating activity logs from different parts of the application
# For example, an internal endpoint called by other services/repositories
# @router.post("/internal/log", status_code=status.HTTP_201_CREATED)
# async def log_activity_internal(
#     activity_type: str,
#     details: Dict[str, Any],
#     organization_id: str = Body(...), # Assuming organization_id is passed in the body for internal calls
#     user_id: str = Body(...), # Assuming user_id is passed in the body
#     related_resource_type: Optional[str] = Body(None),
#     related_resource_id: Optional[str] = Body(None),
# ) -> Any:
#     """
#     (Internal) Log a new activity entry.
#     """
#     try:
#         await activity_repository.create_log_entry(
#             organization_id=organization_id,
#             user_id=user_id,
#             activity_type=activity_type,
#             details=details,
#             related_resource_type=related_resource_type,
#             related_resource_id=related_resource_id
#         )
#         return {"message": "Activity logged successfully"}
#     except Exception as e:
#         raise HTTPException(
#             status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
#             detail=str(e)
#         )

# You would also need to ensure that other parts of the backend (repositories, services)
# call activity_repository.create_log_entry whenever a relevant event occurs
# (e.g., document upload, agent execution, task creation, goal completion, etc.) 
