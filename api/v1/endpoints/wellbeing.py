from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from enum import Enum

from api.deps import get_current_active_user
from db.mongodb.repositories.wellbeing_repository import wellbeing_repository
from db.mongodb.repositories.permission_repository import permission_repository
from models.user import User
from models.wellbeing import (
    WellbeingMetric,
    WellbeingMetricCreate,
    WellbeingGoal,
    WellbeingGoalCreate,
    WellbeingRecommendation,
    BreakType,
    ActivityType
)

router = APIRouter()

class WellbeingMetricResponse(BaseModel):
    id: str
    user_id: str
    organization_id: str
    metric_type: str
    value: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]]
    source: str

class WellbeingGoalResponse(BaseModel):
    id: str
    user_id: str
    organization_id: str
    goal_type: str
    target_value: float
    current_value: float
    start_date: datetime
    end_date: datetime
    status: str
    progress: float
    metadata: Optional[Dict[str, Any]]

class BreakRecommendationResponse(BaseModel):
    break_type: BreakType
    duration_minutes: int
    reason: str
    suggested_activities: List[str]
    metadata: Optional[Dict[str, Any]]

@router.post("/metrics", response_model=WellbeingMetricResponse)
async def record_wellbeing_metric(
    metric_in: WellbeingMetricCreate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Record a new wellbeing metric.
    """
    try:
        metric = await wellbeing_repository.create_metric(
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            metric_type=metric_in.metric_type,
            value=metric_in.value,
            metadata=metric_in.metadata,
            source=metric_in.source
        )
        return metric
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/metrics", response_model=List[WellbeingMetricResponse])
async def get_wellbeing_metrics(
    metric_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get wellbeing metrics for the current user.
    """
    metrics = await wellbeing_repository.get_user_metrics(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        metric_type=metric_type,
        start_date=start_date,
        end_date=end_date
    )
    return metrics

@router.post("/goals", response_model=WellbeingGoalResponse)
async def create_wellbeing_goal(
    goal_in: WellbeingGoalCreate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Create a new wellbeing goal.
    """
    try:
        goal = await wellbeing_repository.create_goal(
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            goal_type=goal_in.goal_type,
            target_value=goal_in.target_value,
            start_date=goal_in.start_date,
            end_date=goal_in.end_date,
            metadata=goal_in.metadata
        )
        return goal
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/goals", response_model=List[WellbeingGoalResponse])
async def get_wellbeing_goals(
    status: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get wellbeing goals for the current user.
    """
    goals = await wellbeing_repository.get_user_goals(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        status=status
    )
    return goals

@router.get("/goals/{goal_id}", response_model=WellbeingGoalResponse)
async def get_wellbeing_goal(
    goal_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get a specific wellbeing goal.
    """
    goal = await wellbeing_repository.get_goal_by_id(
        goal_id=goal_id,
        organization_id=current_user.organization_id
    )
    if not goal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Goal not found"
        )
    return goal

@router.put("/goals/{goal_id}", response_model=WellbeingGoalResponse)
async def update_wellbeing_goal(
    goal_id: str,
    update_data: Dict[str, Any],
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Update a wellbeing goal.
    """
    goal = await wellbeing_repository.update_goal(
        goal_id=goal_id,
        organization_id=current_user.organization_id,
        update_data=update_data
    )
    if not goal:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Goal not found"
        )
    return goal

@router.get("/recommendations", response_model=List[WellbeingRecommendation])
async def get_wellbeing_recommendations(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get personalized wellbeing recommendations.
    """
    recommendations = await wellbeing_repository.get_recommendations(
        user_id=current_user.id,
        organization_id=current_user.organization_id
    )
    return recommendations

@router.get("/break-recommendation", response_model=BreakRecommendationResponse)
async def get_break_recommendation(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get a personalized break recommendation based on user activity and metrics.
    """
    recommendation = await wellbeing_repository.get_break_recommendation(
        user_id=current_user.id,
        organization_id=current_user.organization_id
    )
    return recommendation

@router.post("/activity")
async def record_activity(
    activity_type: ActivityType,
    duration_minutes: int,
    metadata: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Record a wellbeing activity (break, exercise, etc.).
    """
    try:
        activity = await wellbeing_repository.record_activity(
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            activity_type=activity_type,
            duration_minutes=duration_minutes,
            metadata=metadata
        )
        return activity
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/analytics", response_model=Dict[str, Any])
async def get_wellbeing_analytics(
    time_range: str = Query("7d", regex="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get wellbeing analytics and insights.
    """
    analytics = await wellbeing_repository.get_wellbeing_analytics(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        time_range=time_range
    )
    return analytics

@router.get("/organization-analytics", response_model=Dict[str, Any])
async def get_organization_wellbeing_analytics(
    time_range: str = Query("7d", regex="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get organization-wide wellbeing analytics (requires admin permission).
    """
    # Check if user has permission to view organization analytics
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="WELLBEING",
        resource_id="*",
        permission_type="VIEW_ANALYTICS"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to view organization wellbeing analytics"
        )

    analytics = await wellbeing_repository.get_organization_analytics(
        organization_id=current_user.organization_id,
        time_range=time_range
    )
    return analytics

@router.get("/status", response_model=Dict[str, float])
async def get_wellbeing_status(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, float]:
    """
    Get the latest well-being status metrics for the current user.
    """
    metrics_to_fetch = ["Stress Level", "Focus Time", "Hydration", "Mental Energy"]
    latest_metrics = await wellbeing_repository.get_latest_user_metrics(
        user_id=current_user.id,
        metrics=metrics_to_fetch
    )
    return latest_metrics

# Create a singleton instance
wellbeing_repository = WellbeingRepository() 