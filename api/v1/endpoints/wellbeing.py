from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime, timedelta, timezone
from enum import Enum
import structlog

from backend.api.deps import get_current_active_user, get_current_user_id
from backend.db.mongodb.repositories.wellbeing_repository import WellbeingRepository, wellbeing_repository
from backend.db.mongodb.repositories.wellbeing_goal_repository import wellbeing_goal_repository
from backend.db.cassandra.wellbeing_repository import cassandra_wellbeing_repository
from backend.core.config import settings
from backend.db.mongodb.repositories.permission_repository import permission_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.models.wellbeing import (
    WellbeingMetric,
    WellbeingMetricCreate,
    WellbeingGoal,
    WellbeingGoalCreate,
    WellbeingRecommendation,
    BreakType,
    ActivityType,
    WellbeingStats,
    WellbeingRecord
)

# Import AgentService for wellbeing recommendations
from backend.agents.agent_service import AgentService
from backend.api.v1.endpoints.agents import get_agent_service

# Configure logger
logger = structlog.get_logger(__name__)

router = APIRouter()


def _org_id(current_user) -> str:
    """Defensive resolver for the user's organization_id.

    Falls back to the user's own id when the field is missing so
    every endpoint stays usable for single-tenant accounts.  The
    organization_id field is preserved across the model — this
    helper only changes how we read it.
    """
    return str(getattr(current_user, "organization_id", None) or current_user.id)


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

class WellbeingMetricsSummaryResponse(BaseModel):
    total_points: int
    by_metric_type: Dict[str, Dict[str, Any]]
    time_range: Dict[str, Optional[datetime]]

@router.post("/metrics", response_model=WellbeingMetricResponse)
async def record_wellbeing_metric(
    metric_in: WellbeingMetricCreate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Record a new wellbeing metric.
    """
    try:
        if settings.db.CASSANDRA_ENABLED:
            metric = await cassandra_wellbeing_repository.create_metric(
                organization_id=_org_id(current_user),
                user_id=current_user.id,
                metric_type=metric_in.metric_type,
                value=metric_in.value,
                metadata=metric_in.metadata,
                source=metric_in.source
            )
            if settings.db.CASSANDRA_DUAL_WRITE:
                try:
                    await wellbeing_repository.create_metric(
                        user_id=current_user.id,
                        organization_id=_org_id(current_user),
                        metric_type=metric_in.metric_type,
                        value=metric_in.value,
                        metadata=metric_in.metadata,
                        source=metric_in.source
                    )
                except Exception:
                    pass
            await log_activity(
                user_id=str(current_user.id),
                organization_id=_org_id(current_user),
                activity_type="wellbeing.metric_recorded",
                details={"metric_type": metric_in.metric_type, "source": metric_in.source},
                related_resource_type="AGENT",
                agent_name="Wellbeing Agent",
            )
            return metric
        # Fallback to MongoDB implementation if Cassandra is disabled
        metric = await wellbeing_repository.create_metric(
            user_id=current_user.id,
            organization_id=_org_id(current_user),
            metric_type=metric_in.metric_type,
            value=metric_in.value,
            metadata=metric_in.metadata,
            source=metric_in.source
        )
        await log_activity(
            user_id=str(current_user.id),
            organization_id=_org_id(current_user),
            activity_type="wellbeing.metric_recorded",
            details={"metric_type": metric_in.metric_type, "source": metric_in.source},
            related_resource_type="AGENT",
            agent_name="Wellbeing Agent",
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
    if settings.db.CASSANDRA_ENABLED:
        metrics = await cassandra_wellbeing_repository.get_user_metrics(
            organization_id=_org_id(current_user),
            user_id=current_user.id,
            metric_type=metric_type,
            start_date=start_date,
            end_date=end_date
        )
    else:
        metrics = await wellbeing_repository.get_user_metrics(
            user_id=current_user.id,
            organization_id=_org_id(current_user),
            metric_type=metric_type,
            start_date=start_date,
            end_date=end_date
        )
    return metrics

@router.get("/metrics/summary", response_model=WellbeingMetricsSummaryResponse)
async def get_wellbeing_metrics_summary(
    metric_type: Optional[str] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get aggregate summary statistics for wellbeing metrics.
    """
    if settings.db.CASSANDRA_ENABLED:
        summary = await cassandra_wellbeing_repository.get_metrics_summary(
            organization_id=_org_id(current_user),
            user_id=current_user.id,
            metric_type=metric_type,
            start_date=start_date,
            end_date=end_date
        )
        return summary

    metrics = await wellbeing_repository.get_user_metrics(
        user_id=current_user.id,
        organization_id=_org_id(current_user),
        metric_type=metric_type,
        start_date=start_date,
        end_date=end_date,
        limit=1000
    )
    by_type: Dict[str, Dict[str, Any]] = {}
    for m in metrics:
        mtype = m.get("metric_type") or "unknown"
        entry = by_type.setdefault(mtype, {
            "count": 0,
            "sum": 0.0,
            "min": None,
            "max": None,
            "latest": None
        })
        value = float(m.get("value") or 0)
        entry["count"] += 1
        entry["sum"] += value
        entry["min"] = value if entry["min"] is None else min(entry["min"], value)
        entry["max"] = value if entry["max"] is None else max(entry["max"], value)
        if entry["latest"] is None or (m.get("timestamp") and m.get("timestamp") > entry["latest"]["timestamp"]):
            entry["latest"] = {"value": value, "timestamp": m.get("timestamp")}

    for entry in by_type.values():
        entry["avg"] = (entry["sum"] / entry["count"]) if entry["count"] else 0.0
        entry.pop("sum", None)

    return {
        "total_points": len(metrics),
        "by_metric_type": by_type,
        "time_range": {"start": start_date, "end": end_date},
    }

@router.post("/goals", response_model=WellbeingGoalResponse)
async def create_wellbeing_goal(
    goal_in: WellbeingGoalCreate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Create a new wellbeing goal.
    """
    try:
        goal = await wellbeing_goal_repository.create_goal(
            user_id=current_user.id,
            organization_id=_org_id(current_user),
            goal_type=goal_in.goal_type,
            target_value=goal_in.target_value,
            start_date=goal_in.start_date,
            end_date=goal_in.end_date,
            metadata=goal_in.metadata
        )
        await log_activity(
            user_id=str(current_user.id),
            organization_id=_org_id(current_user),
            activity_type="wellbeing.goal_created",
            details={"goal_type": goal_in.goal_type, "target_value": goal_in.target_value},
            related_resource_type="AGENT",
            agent_name="Wellbeing Agent",
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
    goals = await wellbeing_goal_repository.get_user_goals(
        user_id=current_user.id,
        organization_id=_org_id(current_user),
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
    goal = await wellbeing_goal_repository.get_goal_by_id(
        goal_id=goal_id,
        organization_id=_org_id(current_user)
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
    goal = await wellbeing_goal_repository.update_goal(
        goal_id=goal_id,
        organization_id=_org_id(current_user),
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
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Get personalized wellbeing recommendations using the Perplexity-powered agent.
    """
    try:
        # First get user metrics and data from repository
        if settings.db.CASSANDRA_ENABLED:
            metrics = await cassandra_wellbeing_repository.get_user_metrics(
                organization_id=_org_id(current_user),
                user_id=current_user.id,
                limit=50
            )
            latest_metrics = {}
            for metric in metrics:
                mtype = metric.get("metric_type")
                if mtype and mtype not in latest_metrics:
                    latest_metrics[mtype] = metric.get("value")
            user_data = {
                "latest_metrics": latest_metrics,
                "activity_log": [],
                "screen_time": 0,
                "breaks_taken": 0,
                "focus_sessions": 0
            }
        else:
            user_data = await wellbeing_repository.get_user_wellbeing_data(
                user_id=current_user.id,
                organization_id=_org_id(current_user)
            )
        
        # Get recent metrics
        recent_metrics = await wellbeing_repository.get_user_metrics(
            user_id=current_user.id,
            organization_id=_org_id(current_user),
            start_date=datetime.utcnow() - timedelta(days=7)
        )
        
        # Get active goals
        active_goals = await wellbeing_repository.get_user_goals(
            user_id=current_user.id,
            organization_id=_org_id(current_user),
            status="active"
        )
        
        # Prepare data for the agent
        agent_input = {
            "user_id": current_user.id,
            "metrics": user_data.get("latest_metrics", {}),
            "activity_log": user_data.get("activity_log", []),
            "screen_time": user_data.get("screen_time", 0),
            "breaks": user_data.get("breaks_taken", 0),
            "focus_sessions": user_data.get("focus_sessions", 0),
            "goals": [goal.dict() for goal in active_goals],
            "metrics_history": [metric.dict() for metric in recent_metrics]
        }
        
        # Process asynchronously with the wellbeing agent
        try:
            wellbeing_result = await agent_service.execute_agent_async("wellbeing", agent_input)
        except ValueError:
            # Fall back to synchronous processing if async not available
            wellbeing_result = agent_service.execute_agent("wellbeing", agent_input)
        
        # Convert agent response to wellbeing recommendations
        recommendations = []
        wellbeing_advice = wellbeing_result.get("wellbeing_advice", {})
        
        # Process break recommendations
        if "break_recommendations" in wellbeing_advice:
            for i, rec in enumerate(wellbeing_advice["break_recommendations"]):
                recommendations.append(WellbeingRecommendation(
                    id=f"br-{i}",
                    user_id=current_user.id,
                    organization_id=_org_id(current_user),
                    recommendation_type="break",
                    content=rec,
                    priority=i + 1,
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(days=1),
                    metadata={"source": "perplexity"}
                ))
        
        # Process focus techniques
        if "focus_techniques" in wellbeing_advice:
            for i, rec in enumerate(wellbeing_advice["focus_techniques"]):
                recommendations.append(WellbeingRecommendation(
                    id=f"ft-{i}",
                    user_id=current_user.id,
                    organization_id=_org_id(current_user),
                    recommendation_type="focus",
                    content=rec,
                    priority=i + 1,
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(days=2),
                    metadata={"source": "perplexity"}
                ))
        
        # Process stress management
        if "stress_management" in wellbeing_advice:
            for i, rec in enumerate(wellbeing_advice["stress_management"]):
                recommendations.append(WellbeingRecommendation(
                    id=f"sm-{i}",
                    user_id=current_user.id,
                    organization_id=_org_id(current_user),
                    recommendation_type="stress",
                    content=rec,
                    priority=i + 1,
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(days=3),
                    metadata={"source": "perplexity"}
                ))
        
        # Process physical health
        if "physical_health" in wellbeing_advice:
            for i, rec in enumerate(wellbeing_advice["physical_health"]):
                recommendations.append(WellbeingRecommendation(
                    id=f"ph-{i}",
                    user_id=current_user.id,
                    organization_id=_org_id(current_user),
                    recommendation_type="physical",
                    content=rec,
                    priority=i + 1,
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(days=2),
                    metadata={"source": "perplexity"}
                ))
        
        # Process general recommendations
        if "general_recommendations" in wellbeing_advice:
            for i, rec in enumerate(wellbeing_advice["general_recommendations"]):
                recommendations.append(WellbeingRecommendation(
                    id=f"gr-{i}",
                    user_id=current_user.id,
                    organization_id=_org_id(current_user),
                    recommendation_type="general",
                    content=rec,
                    priority=i + 1,
                    created_at=datetime.utcnow(),
                    expires_at=datetime.utcnow() + timedelta(days=2),
                    metadata={"source": "perplexity"}
                ))
                
        # Save recommendations to database
        for recommendation in recommendations:
            await wellbeing_repository.save_recommendation(recommendation)
        
        return recommendations
    except Exception as e:
        # Fallback to repository-based recommendations if agent fails
        logger.error(f"Error getting recommendations from agent: {str(e)}")
        recommendations = await wellbeing_repository.get_recommendations(
            user_id=current_user.id,
            organization_id=_org_id(current_user)
        )
        return recommendations

@router.get("/break-recommendation", response_model=BreakRecommendationResponse)
async def get_break_recommendation(
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Get a personalized break recommendation based on user activity and metrics using the Perplexity agent.
    """
    try:
        # Get user metrics and recent activity
        latest_metrics = await wellbeing_repository.get_latest_user_metrics(
            user_id=current_user.id,
            metrics=["Stress Level", "Focus Time", "Screen Time", "Activity Level"]
        )
        
        recent_activity = await wellbeing_repository.get_recent_activities(
            user_id=current_user.id,
            organization_id=_org_id(current_user),
            limit=5
        )
        
        # Calculate active screen time
        screen_time = latest_metrics.get("Screen Time", 0)
        last_break_time = await wellbeing_repository.get_last_break_time(
            user_id=current_user.id,
            organization_id=_org_id(current_user)
        )
        
        # Prepare data for wellbeing agent
        agent_input = {
            "user_id": current_user.id,
            "metrics": latest_metrics,
            "recent_activity": recent_activity,
            "screen_time": screen_time,
            "last_break_time": last_break_time.isoformat() if last_break_time else None,
            "time_now": datetime.utcnow().isoformat(),
            "request_type": "break_recommendation"
        }
        
        # Process with wellbeing agent
        try:
            wellbeing_result = await agent_service.execute_agent_async("wellbeing", agent_input)
        except ValueError:
            # Fall back to synchronous processing if async not available
            wellbeing_result = agent_service.execute_agent("wellbeing", agent_input)
        
        # Extract break recommendation from agent response
        wellbeing_advice = wellbeing_result.get("wellbeing_advice", {})
        
        # Try to find a specific break recommendation
        break_recommendations = wellbeing_advice.get("break_recommendations", [])
        if break_recommendations:
            # Use the first recommendation for the response
            rec_text = break_recommendations[0]
            
            # Parse the recommendation to extract details
            break_type = BreakType.MICRO_BREAK  # Default
            duration = 5  # Default duration in minutes
            
            # Check for break type indicators
            if any(term in rec_text.lower() for term in ["walk", "stretch", "physical", "exercise"]):
                break_type = BreakType.PHYSICAL
                duration = 10
            elif any(term in rec_text.lower() for term in ["meditate", "breathe", "mindful", "mental"]):
                break_type = BreakType.MINDFULNESS
                duration = 5
            elif any(term in rec_text.lower() for term in ["longer", "extended", "rest", "refresh"]):
                break_type = BreakType.EXTENDED
                duration = 15
                
            # Extract suggested activities
            activities = []
            if len(break_recommendations) > 1:
                activities = break_recommendations[1:]
            elif "general_recommendations" in wellbeing_advice:
                activities = wellbeing_advice["general_recommendations"][:2]
                
            # Create and save the recommendation
            break_rec = BreakRecommendationResponse(
                break_type=break_type,
                duration_minutes=duration,
                reason=rec_text,
                suggested_activities=activities,
                metadata={"source": "perplexity", "generated_at": datetime.utcnow().isoformat()}
            )
            
            # Log the recommendation
            await wellbeing_repository.log_break_recommendation(
                user_id=current_user.id,
                organization_id=_org_id(current_user),
                recommendation=break_rec.dict()
            )
            
            return break_rec
            
        # If no specific break recommendations, fall back to repository-based recommendation
        return await wellbeing_repository.get_break_recommendation(
            user_id=current_user.id,
            organization_id=_org_id(current_user)
        )
            
    except Exception as e:
        logger.error(f"Error generating break recommendation: {str(e)}")
        # Fall back to repository-based recommendation if agent fails
        return await wellbeing_repository.get_break_recommendation(
            user_id=current_user.id,
            organization_id=_org_id(current_user)
        )

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
            organization_id=_org_id(current_user),
            activity_type=activity_type,
            duration_minutes=duration_minutes,
            metadata=metadata
        )
        await log_activity(
            user_id=str(current_user.id),
            organization_id=_org_id(current_user),
            activity_type="wellbeing.activity_recorded",
            details={"activity_type": str(activity_type), "duration_minutes": duration_minutes},
            related_resource_type="AGENT",
            agent_name="Wellbeing Agent",
        )
        return activity
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/analytics", response_model=Dict[str, Any])
async def get_wellbeing_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get wellbeing analytics and insights.
    """
    analytics = await wellbeing_repository.get_wellbeing_analytics(
        user_id=current_user.id,
        organization_id=_org_id(current_user),
        time_range=time_range
    )
    return analytics

@router.get("/organization-analytics", response_model=Dict[str, Any])
async def get_organization_wellbeing_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get organization-wide wellbeing analytics (requires admin permission).
    """
    # Check if user has permission to view organization analytics
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=_org_id(current_user),
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
        organization_id=_org_id(current_user),
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

@router.get("/stats")
async def get_wellbeing_stats(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get wellbeing statistics for the specified time range."""
    days_map = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}
    days = days_map.get(time_range, 7)
    since = datetime.utcnow() - timedelta(days=days)

    metrics = await wellbeing_repository.get_user_metrics(
        user_id=current_user.id,
        organization_id=_org_id(current_user),
        start_date=since,
        limit=1000,
    )

    values = {"mood": [], "energy": [], "stress": [], "sleep": []}
    for m in metrics:
        mt = (m.get("metric_type") or "").lower()
        val = float(m.get("value", 0))
        if mt in values:
            values[mt].append(val)

    def avg(lst):
        return round(sum(lst) / len(lst), 2) if lst else 0.0

    return {
        "user_id": str(current_user.id),
        "period_start": since.isoformat(),
        "period_end": datetime.utcnow().isoformat(),
        "average_mood": avg(values["mood"]),
        "average_energy": avg(values["energy"]),
        "average_stress": avg(values["stress"]),
        "average_sleep": avg(values["sleep"]),
        "total_records": len(metrics),
        "mood_trend": values["mood"][-7:],
        "energy_trend": values["energy"][-7:],
        "stress_trend": values["stress"][-7:],
        "sleep_trend": values["sleep"][-7:],
    }


@router.get("/history")
async def get_wellbeing_history(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get wellbeing history for the specified time range."""
    days_map = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}
    days = days_map.get(time_range, 7)
    since = datetime.utcnow() - timedelta(days=days)

    metrics = await wellbeing_repository.get_user_metrics(
        user_id=current_user.id,
        organization_id=_org_id(current_user),
        start_date=since,
        limit=500,
    )

    # Group metrics by day into WellbeingRecord-like objects
    from collections import defaultdict
    daily: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "mood_scores": [], "energy_levels": [], "stress_levels": [], "sleep_hours": [],
        "activities": [], "notes": [],
    })

    for m in metrics:
        ts = m.get("timestamp")
        day_key = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)[:10]
        mt = (m.get("metric_type") or "").lower()
        val = float(m.get("value", 0))
        if mt in ("mood", "mood_score"):
            daily[day_key]["mood_scores"].append(val)
        elif mt in ("energy", "energy_level"):
            daily[day_key]["energy_levels"].append(val)
        elif mt in ("stress", "stress_level"):
            daily[day_key]["stress_levels"].append(val)
        elif mt in ("sleep", "sleep_hours"):
            daily[day_key]["sleep_hours"].append(val)
        note = (m.get("metadata") or {}).get("notes")
        if note:
            daily[day_key]["notes"].append(note)

    def avg(lst):
        return round(sum(lst) / len(lst), 1) if lst else 5

    records = []
    for day_key in sorted(daily.keys()):
        d = daily[day_key]
        records.append({
            "user_id": str(current_user.id),
            "timestamp": f"{day_key}T00:00:00Z",
            "mood_score": int(avg(d["mood_scores"])),
            "energy_level": int(avg(d["energy_levels"])),
            "stress_level": int(avg(d["stress_levels"])),
            "sleep_hours": round(avg(d["sleep_hours"]), 1),
            "notes": "; ".join(d["notes"]) if d["notes"] else None,
            "activities": list(set(d["activities"])),
            "tags": [],
        })

    return records


# ════════════════════════════════════════════════════════════════════
#  Coach / live additions (production push)
#
#  These endpoints are NEW.  They live below the existing routes so
#  the original 14 endpoints stay byte-identical.  They are scoped
#  by `organization_id` and activity-logged the same way the rest of
#  the file does.
# ════════════════════════════════════════════════════════════════════

from backend.services.wellbeing import orchestrator as _wb_orchestrator
from backend.services.wellbeing import productivity as _wb_productivity
from backend.services.wellbeing import session_tracker as _wb_session
from backend.services.wellbeing import digest as _wb_digest
from backend.services.wellbeing.sanitize import (
    clean_text as _wb_clean_text,
    coerce_jsonable as _wb_coerce_jsonable,
)


def _user_scope(current_user) -> tuple[str, str]:
    """Return (user_id_str, org_id_str) with the same defensive
    `getattr` fallback we use in the other routers."""
    user_id = str(current_user.id)
    org_id = getattr(current_user, "organization_id", None) or user_id
    return user_id, str(org_id)


# ── Heartbeat ─────────────────────────────────────────────────────


@router.post("/heartbeat")
async def wellbeing_heartbeat(
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Frontend pings this every ~30s while the tab is active.  We
    record 'last activity' in Redis so the live break countdown and
    the periodic break-reminder task have a real signal."""
    user_id, _org_id = _user_scope(current_user)
    _wb_session.mark_activity(user_id)
    return {
        "ok": True,
        "now": int(datetime.utcnow().timestamp()),
    }


# ── Productivity aggregator ──────────────────────────────────────


@router.get("/productivity")
async def get_productivity(
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Return focus / task / streak stats for the Coach page."""
    user_id, org_id = _user_scope(current_user)
    result = await _wb_productivity.compute_productivity(
        organization_id=org_id, user_id=user_id
    )
    return _wb_coerce_jsonable(result)


# ── Coach state (one-shot read for the Coach page) ──────────────


@router.get("/coach-state")
async def get_coach_state(
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Bundle everything the Coach page needs in one round-trip.

    Returns:
      - latest metric per type
      - productivity aggregator output
      - countdown to next break (from session_tracker + user settings)
      - today's timeline (activity log)
      - top recommendations (most recent)
      - weekly score series (Mon..Sun)
    """
    user_id, org_id = _user_scope(current_user)

    # User settings — read for the countdown.
    try:
        from backend.db.mongodb.repositories.user_repository import (
            UserRepository,
        )

        settings_obj = await UserRepository().get_user_settings(user_id)
    except Exception:  # noqa: BLE001
        settings_obj = None
    break_interval = (
        getattr(settings_obj, "break_interval_minutes", 60) if settings_obj else 60
    )
    break_duration = (
        getattr(settings_obj, "break_duration_minutes", 5) if settings_obj else 5
    )

    # Productivity + latest metrics + recent activities in parallel.
    import asyncio as _asyncio

    productivity_task = _asyncio.create_task(
        _wb_productivity.compute_productivity(
            organization_id=org_id, user_id=user_id
        )
    )

    async def _latest_metrics():
        """Return a bundle with both ``metrics_summary`` (avg / count /
        trend per metric type — what the Coach UI reads for the tile
        values) and ``latest_metrics`` (last value per type, kept for
        the agent prompt context)."""
        try:
            analytics = await wellbeing_repository.get_wellbeing_analytics(
                user_id=user_id, organization_id=org_id, time_range="7d"
            ) or {}
            data = await wellbeing_repository.get_user_wellbeing_data(
                user_id=user_id, organization_id=org_id
            ) or {}
            return {
                "metrics_summary": analytics.get("metrics_summary") or {},
                "latest_metrics": data.get("latest_metrics") or {},
                "recent_activities": analytics.get("recent_activities") or [],
                "total_metrics": analytics.get("total_metrics", 0),
                "total_activities": analytics.get("total_activities", 0),
            }
        except Exception:  # noqa: BLE001
            return {"metrics_summary": {}, "latest_metrics": {}}

    async def _recent_activities():
        try:
            return await wellbeing_repository.get_recent_activities(
                user_id=user_id, organization_id=org_id, limit=20
            )
        except Exception:  # noqa: BLE001
            return []

    async def _recent_recommendations():
        try:
            return await wellbeing_repository.get_recommendations(
                user_id=user_id, organization_id=org_id, limit=8
            )
        except Exception:  # noqa: BLE001
            return []

    metrics_task = _asyncio.create_task(_latest_metrics())
    activities_task = _asyncio.create_task(_recent_activities())
    recs_task = _asyncio.create_task(_recent_recommendations())

    productivity = await productivity_task
    latest_metrics = await metrics_task
    activities = await activities_task
    recommendations = await recs_task

    seconds_until_break = _wb_session.seconds_until_next_break(
        user_id, int(break_interval)
    )
    seconds_since_break = _wb_session.seconds_since_break(user_id) or 0

    # Weekly score series: average mood per day across Mon..Sun of
    # the current week.  Pulled directly off the metrics collection
    # so we don't depend on an aggregator field that doesn't exist.
    today = datetime.now(timezone.utc).date()
    monday = today - timedelta(days=today.weekday())  # Monday this week
    week_series: List[Optional[float]] = [None] * 7
    try:
        recent_metrics = await wellbeing_repository.get_user_metrics(
            user_id=user_id,
            organization_id=org_id,
            metric_type="mood",
            start_date=datetime.combine(monday, datetime.min.time()) - timedelta(days=1),
            limit=1000,
        )
        # Fallback: if no mood data this week, accept any metric type so
        # the chart isn't blank when the user has only logged stress /
        # energy / sleep so far.
        if not recent_metrics:
            recent_metrics = await wellbeing_repository.get_user_metrics(
                user_id=user_id,
                organization_id=org_id,
                start_date=datetime.combine(monday, datetime.min.time()) - timedelta(days=1),
                limit=1000,
            )

        buckets: Dict[int, List[float]] = {i: [] for i in range(7)}
        for m in recent_metrics or []:
            ts = m.get("timestamp") or m.get("created_at")
            if not ts:
                continue
            if isinstance(ts, str):
                try:
                    ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except Exception:  # noqa: BLE001
                    continue
            if hasattr(ts, "date"):
                ts_date = ts.date()
            else:
                continue
            day_idx = (ts_date - monday).days
            if 0 <= day_idx < 7:
                try:
                    buckets[day_idx].append(float(m.get("value", 0)))
                except (TypeError, ValueError):
                    continue
        for i in range(7):
            if buckets[i]:
                week_series[i] = round(sum(buckets[i]) / len(buckets[i]), 1)
        logger.info(
            "coach_state.week_series",
            user_id=user_id,
            monday=str(monday),
            metrics_count=len(recent_metrics or []),
            series=week_series,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("coach_state.week_series_failed", err=str(exc))
        week_series = [None] * 7

    return _wb_coerce_jsonable({
        "metrics": latest_metrics,
        "productivity": productivity,
        "break_timer": {
            "interval_minutes": int(break_interval),
            "duration_minutes": int(break_duration),
            "seconds_until_break": seconds_until_break,
            "seconds_since_break": seconds_since_break,
        },
        "today_timeline": activities,
        "recommendations": recommendations,
        "week_series": week_series,
        "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    })


# ── Today's timeline (chronological activity log) ───────────────


@router.get("/history/timeline")
async def get_today_timeline(
    current_user: User = Depends(get_current_active_user),
) -> List[Dict[str, Any]]:
    """Return today's activity log in chronological order, for the
    timeline panel on the Coach page."""
    user_id, org_id = _user_scope(current_user)
    try:
        activities = await wellbeing_repository.get_recent_activities(
            user_id=user_id, organization_id=org_id, limit=40
        )
    except Exception:  # noqa: BLE001
        activities = []
    # Sort ascending by timestamp so the UI can render top-to-bottom.
    def _ts(a: Dict[str, Any]) -> str:
        return str(a.get("timestamp") or a.get("created_at") or "")
    activities = sorted(activities, key=_ts)
    return _wb_coerce_jsonable(activities)


# ── Chat with the coach ─────────────────────────────────────────


class CoachChatRequest(BaseModel):
    message: str = Field(..., max_length=4000)
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)


@router.post("/chat")
async def chat_with_coach(
    payload: CoachChatRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Conversational turn with the Well-being Coach (Gemini-locked)."""
    user_id, org_id = _user_scope(current_user)

    # Gather a small `user_data` slice so the coach knows the context.
    try:
        productivity_slice = await _wb_productivity.compute_productivity(
            organization_id=org_id, user_id=user_id
        )
    except Exception:  # noqa: BLE001
        productivity_slice = {}

    result = await _wb_orchestrator.chat_with_coach(
        organization_id=org_id,
        user_id=user_id,
        message=payload.message,
        history=payload.history or [],
        user_data={"productivity": productivity_slice},
    )

    try:
        await log_activity(
            user_id=user_id,
            organization_id=org_id,
            activity_type="wellbeing.coach_chat",
            details={"message_preview": _wb_clean_text(payload.message, max_len=200)},
            related_resource_type="AGENT",
            agent_name="Wellbeing Coach",
        )
    except Exception:  # noqa: BLE001
        pass
    return result


# ── Mood prompt scheduling (cross-app) ──────────────────────────


@router.get("/mood-prompts/poll")
async def poll_mood_prompt(
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Frontend polls this every couple of minutes.  Returns
    `{"prompt": true}` if a mood-log modal should be shown now.
    Consuming the prompt sets a 90-minute cooldown server-side."""
    user_id, _org_id = _user_scope(current_user)
    show = _wb_session.pop_mood_prompt(user_id)
    return {"prompt": bool(show)}


class MoodPromptDismissRequest(BaseModel):
    snooze_minutes: int = Field(90, ge=10, le=24 * 60)


@router.post("/mood-prompts/dismiss")
async def dismiss_mood_prompt(
    payload: MoodPromptDismissRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """User clicked Snooze or Not now — extend the cooldown."""
    user_id, _org_id = _user_scope(current_user)
    _wb_session.snooze_mood_prompt(user_id, minutes=payload.snooze_minutes)
    return {"ok": True, "snoozed_minutes": payload.snooze_minutes}


# ── Break snooze ────────────────────────────────────────────────


class BreakSnoozeRequest(BaseModel):
    minutes: int = Field(15, ge=1, le=120)


@router.post("/break/snooze")
async def snooze_break(
    payload: BreakSnoozeRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Push the next-break countdown forward by N minutes."""
    user_id, org_id = _user_scope(current_user)
    _wb_session.snooze_break(user_id, minutes=payload.minutes)
    try:
        await log_activity(
            user_id=user_id,
            organization_id=org_id,
            activity_type="wellbeing.break_snoozed",
            details={"minutes": payload.minutes},
            agent_name="Wellbeing Coach",
        )
    except Exception:  # noqa: BLE001
        pass
    return {"ok": True, "snoozed_minutes": payload.minutes}


# ── Weekly digest preview ───────────────────────────────────────


@router.get("/weekly-digest/preview")
async def preview_weekly_digest(
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Build the user's weekly digest payload without sending an
    email — useful for both preview UI and end-to-end testing."""
    user_id, org_id = _user_scope(current_user)
    email = getattr(current_user, "email", None) or ""
    name = getattr(current_user, "full_name", None)
    return await _wb_digest.build_user_digest(
        organization_id=org_id,
        user_id=user_id,
        email=email,
        name=name,
    )
