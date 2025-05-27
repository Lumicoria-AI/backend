from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from pydantic import BaseModel, Field
from datetime import datetime

from api.deps import get_current_active_user
from db.mongodb.repositories.agent_universe_repository import agent_universe_repository
from db.mongodb.repositories.permission_repository import permission_repository
from models.user import User
from agents.agent_service import AgentService
from agents.social_media_agent import SocialMediaAgent, SocialMediaMode

router = APIRouter()

# Request and Response Models
class SocialMediaRequest(BaseModel):
    """Base model for social media analysis requests."""
    data: Dict[str, Any] = Field(..., description="Social media data to analyze")
    mode: str = Field("content_analysis", description="Analysis mode")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for analysis")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Analysis-specific parameters")
    model: Optional[str] = Field(None, description="AI model to use (defaults to sonar-large-online)")

class SocialMediaResponse(BaseModel):
    """Base model for social media analysis responses."""
    results: Dict[str, Any]
    metadata: Dict[str, Any]

class ContentAnalysisRequest(SocialMediaRequest):
    """Model for content analysis requests."""
    include_metrics: bool = Field(True, description="Whether to include detailed metrics")
    include_recommendations: bool = Field(True, description="Whether to include content recommendations")

class TrendAnalysisRequest(SocialMediaRequest):
    """Model for trend analysis requests."""
    timeframe: str = Field("7d", description="Timeframe for trend analysis")
    min_occurrences: int = Field(5, description="Minimum occurrences for trend consideration")
    include_growth: bool = Field(True, description="Whether to include trend growth metrics")

class SentimentAnalysisRequest(SocialMediaRequest):
    """Model for sentiment analysis requests."""
    include_emotions: bool = Field(True, description="Whether to include emotion analysis")
    include_aspects: bool = Field(True, description="Whether to include aspect-based sentiment")
    min_confidence: float = Field(0.7, description="Minimum confidence for sentiment analysis")

class ContentGenerationRequest(SocialMediaRequest):
    """Model for content generation requests."""
    content_types: List[str] = Field(["posts", "captions"], description="Types of content to generate")
    tone: str = Field("professional", description="Tone of the generated content")
    platforms: List[str] = Field(["twitter", "facebook", "instagram"], description="Target platforms")

class EngagementAnalysisRequest(SocialMediaRequest):
    """Model for engagement analysis requests."""
    metrics: List[str] = Field(["likes", "comments", "shares"], description="Metrics to analyze")
    comparison_period: str = Field("previous", description="Period to compare against")
    include_benchmarks: bool = Field(True, description="Whether to include industry benchmarks")

class SchedulingRequest(SocialMediaRequest):
    """Model for scheduling requests."""
    platforms: List[str] = Field(["twitter", "facebook", "instagram"], description="Platforms to optimize")
    content_mix: bool = Field(True, description="Whether to include content mix recommendations")
    time_optimization: bool = Field(True, description="Whether to optimize posting times")

# Helper function to get agent service
def get_agent_service() -> AgentService:
    """Get or create an instance of AgentService."""
    config = {
        "model": "sonar-large-online",
        "model_config": {
            "model": "sonar-large-online",
            "temperature": 0.7,
            "max_tokens": 2048
        }
    }
    return AgentService(config)

@router.post("/analyze", response_model=SocialMediaResponse)
async def process_social_media(
    request: SocialMediaRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Process a social media analysis request using the Social Media Agent.
    
    This endpoint handles various types of analysis tasks including:
    - Content analysis
    - Trend analysis
    - Sentiment analysis
    - Content generation
    - Engagement analysis
    - Schedule optimization
    """
    try:
        # Check permissions
        has_permission = await permission_repository.check_permission(
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            resource_type="AGENT",
            resource_id="social_media",
            permission_type="EXECUTE"
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to use the social media agent"
            )

        # Create social media agent
        agent_config = {
            "model": request.model or "sonar-large-online",
            "model_config": {
                "model": request.model or "sonar-large-online",
                "temperature": 0.7,
                "max_tokens": 2048
            }
        }
        
        social_media_agent = SocialMediaAgent(agent_config)
        
        # Process the request
        result = await social_media_agent.process_async({
            "data": request.data,
            "mode": request.mode,
            "context": request.context or {},
            "parameters": request.parameters or {}
        })
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )
        
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing social media analysis: {str(e)}"
        )

@router.post("/analyze/content", response_model=SocialMediaResponse)
async def analyze_content(
    request: ContentAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Analyze social media content.
    
    This endpoint provides:
    - Content performance metrics
    - Content insights
    - Content recommendations
    """
    try:
        # Set request type for content analysis
        request.mode = SocialMediaMode.CONTENT_ANALYSIS.value
        
        # Add content analysis parameters
        parameters = request.parameters or {}
        parameters.update({
            "include_metrics": request.include_metrics,
            "include_recommendations": request.include_recommendations
        })
        
        # Process using the main endpoint
        return await process_social_media(
            SocialMediaRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing content: {str(e)}"
        )

@router.post("/analyze/trends", response_model=SocialMediaResponse)
async def analyze_trends(
    request: TrendAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Analyze social media trends.
    
    This endpoint provides:
    - Trend identification
    - Growth analysis
    - Trend recommendations
    """
    try:
        # Set request type for trend analysis
        request.mode = SocialMediaMode.TREND_ANALYSIS.value
        
        # Add trend analysis parameters
        parameters = request.parameters or {}
        parameters.update({
            "timeframe": request.timeframe,
            "min_occurrences": request.min_occurrences,
            "include_growth": request.include_growth
        })
        
        # Process using the main endpoint
        return await process_social_media(
            SocialMediaRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing trends: {str(e)}"
        )

@router.post("/analyze/sentiment", response_model=SocialMediaResponse)
async def analyze_sentiment(
    request: SentimentAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Analyze sentiment and brand mentions.
    
    This endpoint provides:
    - Sentiment analysis
    - Brand mention tracking
    - Sentiment insights
    """
    try:
        # Set request type for sentiment analysis
        request.mode = SocialMediaMode.SENTIMENT_ANALYSIS.value
        
        # Add sentiment analysis parameters
        parameters = request.parameters or {}
        parameters.update({
            "include_emotions": request.include_emotions,
            "include_aspects": request.include_aspects,
            "min_confidence": request.min_confidence
        })
        
        # Process using the main endpoint
        return await process_social_media(
            SocialMediaRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing sentiment: {str(e)}"
        )

@router.post("/generate/content", response_model=SocialMediaResponse)
async def generate_content(
    request: ContentGenerationRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Generate social media content.
    
    This endpoint provides:
    - Content recommendations
    - Platform-specific content
    - Tone-appropriate messaging
    """
    try:
        # Set request type for content generation
        request.mode = SocialMediaMode.CONTENT_GENERATION.value
        
        # Add content generation parameters
        parameters = request.parameters or {}
        parameters.update({
            "content_types": request.content_types,
            "tone": request.tone,
            "platforms": request.platforms
        })
        
        # Process using the main endpoint
        return await process_social_media(
            SocialMediaRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating content: {str(e)}"
        )

@router.post("/analyze/engagement", response_model=SocialMediaResponse)
async def analyze_engagement(
    request: EngagementAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Analyze engagement metrics.
    
    This endpoint provides:
    - Engagement analysis
    - Performance metrics
    - Engagement recommendations
    """
    try:
        # Set request type for engagement analysis
        request.mode = SocialMediaMode.ENGAGEMENT_ANALYSIS.value
        
        # Add engagement analysis parameters
        parameters = request.parameters or {}
        parameters.update({
            "metrics": request.metrics,
            "comparison_period": request.comparison_period,
            "include_benchmarks": request.include_benchmarks
        })
        
        # Process using the main endpoint
        return await process_social_media(
            SocialMediaRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing engagement: {str(e)}"
        )

@router.post("/optimize/schedule", response_model=SocialMediaResponse)
async def optimize_schedule(
    request: SchedulingRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Optimize content scheduling.
    
    This endpoint provides:
    - Optimal posting times
    - Content mix recommendations
    - Platform-specific schedules
    """
    try:
        # Set request type for scheduling
        request.mode = SocialMediaMode.SCHEDULING.value
        
        # Add scheduling parameters
        parameters = request.parameters or {}
        parameters.update({
            "platforms": request.platforms,
            "content_mix": request.content_mix,
            "time_optimization": request.time_optimization
        })
        
        # Process using the main endpoint
        return await process_social_media(
            SocialMediaRequest(
                data=request.data,
                mode=request.mode,
                context=request.context,
                parameters=parameters,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error optimizing schedule: {str(e)}"
        )

@router.get("/analytics", response_model=Dict[str, Any])
async def get_social_media_analytics(
    time_range: str = Query("7d", regex="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get social media analytics.
    
    This endpoint provides analytics about:
    - Content performance
    - Engagement metrics
    - Platform usage
    - User engagement
    """
    # This would typically fetch from a database
    analytics = {
        "time_range": time_range,
        "total_posts": 1500,
        "average_engagement_rate": 0.045,
        "platform_usage": {
            "twitter": 600,
            "facebook": 400,
            "instagram": 300,
            "linkedin": 200
        },
        "content_types": {
            "text": 800,
            "image": 400,
            "video": 200,
            "link": 100
        },
        "engagement_metrics": {
            "likes": 45000,
            "comments": 15000,
            "shares": 8000,
            "clicks": 12000
        },
        "sentiment_distribution": {
            "positive": 0.65,
            "neutral": 0.25,
            "negative": 0.10
        },
        "top_performing_content": [
            {
                "platform": "twitter",
                "content_type": "text",
                "engagement_rate": 0.08,
                "sentiment": "positive"
            },
            {
                "platform": "instagram",
                "content_type": "image",
                "engagement_rate": 0.12,
                "sentiment": "positive"
            }
        ]
    }
    
    return analytics 