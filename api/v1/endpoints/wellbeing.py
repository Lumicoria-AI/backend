from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from enum import Enum
import structlog

from backend.api.deps import get_current_active_user, get_current_user_id
from backend.db.mongodb.repositories.wellbeing_repository import WellbeingRepository, wellbeing_repository
from backend.db.mongodb.repositories.permission_repository import permission_repository
from backend.models.user import User
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
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Get personalized wellbeing recommendations using the Perplexity-powered agent.
    """
    try:
        # First get user metrics and data from repository
        user_data = await wellbeing_repository.get_user_wellbeing_data(
            user_id=current_user.id,
            organization_id=current_user.organization_id
        )
        
        # Get recent metrics
        recent_metrics = await wellbeing_repository.get_user_metrics(
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            start_date=datetime.utcnow() - timedelta(days=7)
        )
        
        # Get active goals
        active_goals = await wellbeing_repository.get_user_goals(
            user_id=current_user.id,
            organization_id=current_user.organization_id,
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
                    organization_id=current_user.organization_id,
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
                    organization_id=current_user.organization_id,
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
                    organization_id=current_user.organization_id,
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
                    organization_id=current_user.organization_id,
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
                    organization_id=current_user.organization_id,
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
            organization_id=current_user.organization_id
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
            organization_id=current_user.organization_id,
            limit=5
        )
        
        # Calculate active screen time
        screen_time = latest_metrics.get("Screen Time", 0)
        last_break_time = await wellbeing_repository.get_last_break_time(
            user_id=current_user.id,
            organization_id=current_user.organization_id
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
                organization_id=current_user.organization_id,
                recommendation=break_rec.dict()
            )
            
            return break_rec
            
        # If no specific break recommendations, fall back to repository-based recommendation
        return await wellbeing_repository.get_break_recommendation(
            user_id=current_user.id,
            organization_id=current_user.organization_id
        )
            
    except Exception as e:
        logger.error(f"Error generating break recommendation: {str(e)}")
        # Fall back to repository-based recommendation if agent fails
        return await wellbeing_repository.get_break_recommendation(
            user_id=current_user.id,
            organization_id=current_user.organization_id
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
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
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
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
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

@router.get("/stats", response_model=WellbeingStats)
async def get_wellbeing_stats(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    user_id: str = Depends(get_current_user_id)
):
    """Get wellbeing statistics for the specified time range."""
    # ... existing code ...

@router.get("/history", response_model=List[WellbeingRecord])
async def get_wellbeing_history(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    user_id: str = Depends(get_current_user_id)
):
    """Get wellbeing history for the specified time range."""
    # ... existing code ...

# Create a singleton instance
wellbeing_repository = WellbeingRepository() 