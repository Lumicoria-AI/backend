from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form, Query, Body
from pydantic import BaseModel, Field
from datetime import datetime
import structlog
import io
import base64

from api.deps import get_current_active_user
from agents.agent_service import AgentService
from models.user import User

# Configure logger
logger = structlog.get_logger(__name__)

router = APIRouter()

class VisionAnalysisOptions(BaseModel):
    """Vision analysis configuration options."""
    prompt: Optional[str] = Field(None, description="Custom prompt for image analysis")
    analysis_tasks: Optional[List[str]] = Field(None, description="Specific analysis tasks to perform")
    detailed: Optional[bool] = Field(True, description="Whether to return detailed analysis")
    max_tokens: Optional[int] = Field(1024, description="Maximum tokens for the response")
    temperature: Optional[float] = Field(0.7, description="Temperature for model generation")

class VisionAnalysisResponse(BaseModel):
    """Response model for vision analysis."""
    description: str
    structured_analysis: Dict[str, Any]
    processed_at: str
    model_used: str
    citations: Optional[List[Dict[str, Any]]] = None

class ImageURLRequest(BaseModel):
    """Request model for image URL analysis."""
    url: str = Field(..., description="URL of the image to analyze")
    options: Optional[VisionAnalysisOptions] = None

@router.post("/analyze", response_model=VisionAnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    options: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Analyze an uploaded image using Perplexity AI.
    
    This endpoint uses the Perplexity-powered vision agent to analyze images,
    extract text, identify objects, and provide detailed descriptions.
    """
    try:
        # Read image data
        image_content = await file.read()
        
        # Parse options if provided
        analysis_options = {}
        if options:
            try:
                import json
                analysis_options = json.loads(options)
            except json.JSONDecodeError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid options JSON format"
                )
        
        # Create vision agent (in real implementation, this would come from agent_service)
        vision_agent_config = {
            "type": "vision",
            "model_config": {
                "model": "sonar-large-online"
            },
            "vision_tasks": ["text_extraction", "object_detection", "scene_analysis", "content_description"]
        }
        
        from agents.vision_agent import VisionAgent
        vision_agent = VisionAgent(vision_agent_config)
        
        # Process image
        vision_data = {
            "image_content": image_content,
            "prompt": analysis_options.get("prompt", "Analyze the provided image and describe its content in detail."),
            "analysis_tasks": analysis_options.get("analysis_tasks", ["text_extraction", "object_detection", "scene_analysis"]),
            "max_tokens": analysis_options.get("max_tokens", 1024),
            "temperature": analysis_options.get("temperature", 0.7),
        }
        
        # Process asynchronously for better performance
        result = await vision_agent.process_async(vision_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return result
    except Exception as e:
        await logger.error("Error analyzing image", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image analysis failed: {str(e)}"
        )

@router.post("/analyze-url", response_model=VisionAnalysisResponse)
async def analyze_image_url(
    request: ImageURLRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(lambda: None)  # This will be injected properly in production
) -> Any:
    """
    Analyze an image from a URL using Perplexity AI.
    
    This endpoint uses the Perplexity-powered vision agent to analyze images from URLs,
    extract text, identify objects, and provide detailed descriptions.
    """
    try:
        # Create vision agent (in real implementation, this would come from agent_service)
        vision_agent_config = {
            "type": "vision",
            "model_config": {
                "model": "sonar-large-online"
            },
            "vision_tasks": ["text_extraction", "object_detection", "scene_analysis", "content_description"]
        }
        
        from agents.vision_agent import VisionAgent
        vision_agent = VisionAgent(vision_agent_config)
        
        # Get options
        options = request.options.dict() if request.options else {}
        
        # Process image
        vision_data = {
            "image_url": request.url,
            "prompt": options.get("prompt", "Analyze the provided image and describe its content in detail."),
            "analysis_tasks": options.get("analysis_tasks", ["text_extraction", "object_detection", "scene_analysis"]),
            "max_tokens": options.get("max_tokens", 1024),
            "temperature": options.get("temperature", 0.7),
        }
        
        # Process asynchronously for better performance
        result = await vision_agent.process_async(vision_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        return result
    except Exception as e:
        await logger.error("Error analyzing image URL", error=str(e), url=request.url)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image URL analysis failed: {str(e)}"
        )

@router.post("/ocr", response_model=Dict[str, Any])
async def extract_text_from_image(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Extract text from an uploaded image using Perplexity AI.
    
    This endpoint focuses specifically on Optical Character Recognition (OCR)
    to extract text content from images.
    """
    try:
        # Read image data
        image_content = await file.read()
        
        # Create vision agent specifically for OCR
        vision_agent_config = {
            "type": "vision",
            "model_config": {
                "model": "sonar-large-online"
            },
        }
        
        from agents.vision_agent import VisionAgent
        vision_agent = VisionAgent(vision_agent_config)
        
        # Process image with OCR-specific prompt
        vision_data = {
            "image_content": image_content,
            "prompt": "Extract and transcribe ALL text visible in this image. Include any numbers, letters, words, and sentences. Maintain the original formatting as much as possible, including line breaks and paragraph structure.",
            "max_tokens": 1024,
        }
        
        # Process asynchronously for better performance
        result = await vision_agent.process_async(vision_data)
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=result["error"]
            )
        
        # Format OCR-specific response
        ocr_result = {
            "extracted_text": result["description"],
            "detected_text": result["structured_analysis"]["detected_text"],
            "confidence": "high",  # Placeholder, would be provided by a real OCR service
            "processed_at": result["processed_at"]
        }
        
        return ocr_result
    except Exception as e:
        await logger.error("Error extracting text from image", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Text extraction failed: {str(e)}"
        )
