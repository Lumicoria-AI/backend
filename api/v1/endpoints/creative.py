from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Body
from pydantic import BaseModel, Field
from datetime import datetime
import structlog

from backend.api.deps import get_current_active_user
from backend.agents.agent_service import AgentService
from backend.agents.creative_agent import CreativeAgent
from backend.models.user import User
from backend.services.activity_logger import log_activity

# Configure logger
logger = structlog.get_logger(__name__)

router = APIRouter()

class CreativeRequest(BaseModel):
    """Base request model for creative agent."""
    content_type: str = Field(..., description="Type of creative content to generate: marketing, storytelling, poetry, scriptwriting, product_description, social_media, or blog_post")
    topic: str = Field(..., description="The main topic or subject for the creative content")
    guidelines: Optional[str] = Field(None, description="Specific guidelines or requirements for the content")
    audience: Optional[str] = Field("general audience", description="Target audience for the content")
    tone: Optional[str] = Field("professional", description="Tone of the content: professional, casual, formal, enthusiastic, etc.")
    length: Optional[str] = Field("medium", description="Length of the content: short, medium, or long")
    model: Optional[str] = Field(None, description="AI model to use")
    temperature: Optional[float] = Field(0.8, description="Temperature for model generation")

class CreativeResponse(BaseModel):
    """Response model for creative agent."""
    content: Dict[str, Any]
    raw_content: Optional[str] = None
    processed_at: str
    model_used: str
    content_type: str
    metadata: Dict[str, Any]
    citations: Optional[List[Dict[str, Any]]] = None

@router.post("/generate", response_model=CreativeResponse)
async def generate_creative_content(
    request: CreativeRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Generate creative content using Perplexity AI.
    
    This endpoint uses the Perplexity-powered creative agent to generate various types
    of creative content including marketing copy, stories, poems, scripts,
    product descriptions, and other creative text forms.
    """
    try:
        # Create creative agent
        creative_agent_config = {
            "type": "creative",
            "model_config": {
                "model": request.model or "sonar-large-online"
            },
        }
        
        creative_agent = CreativeAgent(creative_agent_config)
        
        # Process creative request
        creative_data = {
            "content_type": request.content_type,
            "topic": request.topic,
            "guidelines": request.guidelines,
            "audience": request.audience,
            "tone": request.tone,
            "length": request.length
        }
        
        # Process asynchronously for better performance
        result = await creative_agent.process_async(creative_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=current_user.organization_id,
            activity_type="creative.content_generated",
            details={"content_type": request.content_type, "topic": request.topic[:100]},
            related_resource_type="AGENT",
            agent_name="Creative Agent",
        )
        return result
    except Exception as e:
        logger.error("Error generating creative content", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Creative content generation failed: {str(e)}"
        )

@router.post("/marketing", response_model=CreativeResponse)
async def generate_marketing_content(
    request: CreativeRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Generate marketing copy using Perplexity AI.
    
    This endpoint specializes in creating marketing content including headlines,
    taglines, product descriptions, email copy, and ad copy.
    """
    try:
        # Set content type to marketing
        request.content_type = "marketing"
        
        # Create creative agent
        creative_agent_config = {
            "type": "creative",
            "model_config": {
                "model": request.model or "sonar-large-online"
            },
        }
        
        creative_agent = CreativeAgent(creative_agent_config)
        
        # Process creative request
        creative_data = {
            "content_type": request.content_type,
            "topic": request.topic,
            "guidelines": request.guidelines,
            "audience": request.audience,
            "tone": request.tone,
            "length": request.length
        }
        
        # Process asynchronously for better performance
        result = await creative_agent.process_async(creative_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=current_user.organization_id,
            activity_type="creative.content_generated",
            details={"content_type": "marketing", "topic": request.topic[:100]},
            related_resource_type="AGENT",
            agent_name="Creative Agent",
        )
        return result
    except Exception as e:
        logger.error("Error generating marketing content", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Marketing content generation failed: {str(e)}"
        )

@router.post("/story", response_model=CreativeResponse)
async def generate_story(
    request: CreativeRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Generate storytelling content using Perplexity AI.
    
    This endpoint specializes in creating narrative content including short stories,
    character sketches, plot outlines, and scene descriptions.
    """
    try:
        # Set content type to storytelling
        request.content_type = "storytelling"
        
        # Create creative agent
        creative_agent_config = {
            "type": "creative",
            "model_config": {
                "model": request.model or "sonar-large-online"
            },
        }
        
        creative_agent = CreativeAgent(creative_agent_config)
        
        # Process creative request
        creative_data = {
            "content_type": request.content_type,
            "topic": request.topic,
            "guidelines": request.guidelines,
            "audience": request.audience,
            "tone": request.tone,
            "length": request.length
        }
        
        # Process asynchronously for better performance
        result = await creative_agent.process_async(creative_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=current_user.organization_id,
            activity_type="creative.content_generated",
            details={"content_type": "storytelling", "topic": request.topic[:100]},
            related_resource_type="AGENT",
            agent_name="Creative Agent",
        )
        return result
    except Exception as e:
        logger.error("Error generating story content", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Story generation failed: {str(e)}"
        )

@router.post("/blog", response_model=CreativeResponse)
async def generate_blog_post(
    request: CreativeRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Generate blog post content using Perplexity AI.
    
    This endpoint specializes in creating structured blog posts with introductions,
    headings, sections, and conclusions.
    """
    try:
        # Set content type to blog_post
        request.content_type = "blog_post"
        
        # Create creative agent
        creative_agent_config = {
            "type": "creative",
            "model_config": {
                "model": request.model or "sonar-large-online"
            },
        }
        
        creative_agent = CreativeAgent(creative_agent_config)
        
        # Process creative request
        creative_data = {
            "content_type": request.content_type,
            "topic": request.topic,
            "guidelines": request.guidelines,
            "audience": request.audience,
            "tone": request.tone,
            "length": request.length
        }
        
        # Process asynchronously for better performance
        result = await creative_agent.process_async(creative_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=current_user.organization_id,
            activity_type="creative.content_generated",
            details={"content_type": "blog_post", "topic": request.topic[:100]},
            related_resource_type="AGENT",
            agent_name="Creative Agent",
        )
        return result
    except Exception as e:
        logger.error("Error generating blog post", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Blog post generation failed: {str(e)}"
        )

@router.post("/social-media", response_model=CreativeResponse)
async def generate_social_media_content(
    request: CreativeRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Generate social media content using Perplexity AI.
    
    This endpoint specializes in creating engaging social media posts with
    appropriate hashtags and formatting for various platforms.
    """
    try:
        # Set content type to social_media
        request.content_type = "social_media"
        
        # Create creative agent
        creative_agent_config = {
            "type": "creative",
            "model_config": {
                "model": request.model or "sonar-large-online"
            },
        }
        
        creative_agent = CreativeAgent(creative_agent_config)
        
        # Process creative request
        creative_data = {
            "content_type": request.content_type,
            "topic": request.topic,
            "guidelines": request.guidelines,
            "audience": request.audience,
            "tone": request.tone,
            "length": request.length
        }
        
        # Process asynchronously for better performance
        result = await creative_agent.process_async(creative_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=current_user.organization_id,
            activity_type="creative.content_generated",
            details={"content_type": "social_media", "topic": request.topic[:100]},
            related_resource_type="AGENT",
            agent_name="Creative Agent",
        )
        return result
    except Exception as e:
        logger.error("Error generating social media content", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Social media content generation failed: {str(e)}"
        )
