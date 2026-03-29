from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.agent_universe_repository import agent_universe_repository
from backend.db.mongodb.repositories.permission_repository import permission_repository
from backend.models.user import User
from backend.agents.agent_service import AgentService
from backend.agents.translation_agent import TranslationAgent, TranslationMode
from backend.services.activity_logger import log_activity

router = APIRouter()

# Request and Response Models
class TranslationRequest(BaseModel):
    """Base model for translation requests."""
    content: str = Field(..., description="The content to translate")
    source_language: str = Field("auto", description="Source language code (defaults to auto-detect)")
    target_language: str = Field(..., description="Target language code")
    mode: str = Field("document", description="Translation mode (document, conversation, cultural, technical, literary)")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for translation")
    model: Optional[str] = Field(None, description="AI model to use (defaults to sonar-large-online)")

class TranslationResponse(BaseModel):
    """Base model for translation responses."""
    translated_text: str
    mode: str
    source_language: str
    target_language: str
    metadata: Dict[str, Any]

class DocumentTranslationRequest(TranslationRequest):
    """Model for document translation requests."""
    preserve_formatting: bool = Field(True, description="Whether to preserve document formatting")
    include_cultural_context: bool = Field(True, description="Whether to include cultural context")

class ConversationTranslationRequest(TranslationRequest):
    """Model for conversation translation requests."""
    conversation_context: Optional[Dict[str, Any]] = Field(None, description="Context about the conversation")
    participants: Optional[List[str]] = Field(None, description="List of conversation participants")

class CulturalAdaptationRequest(TranslationRequest):
    """Model for cultural adaptation requests."""
    target_culture: str = Field(..., description="Target culture for adaptation")
    adaptation_level: str = Field("moderate", description="Level of cultural adaptation (minimal, moderate, extensive)")

class TechnicalTranslationRequest(TranslationRequest):
    """Model for technical translation requests."""
    domain: str = Field(..., description="Technical domain (e.g., IT, medical, legal)")
    terminology_style: str = Field("formal", description="Style of technical terminology")

class LiteraryTranslationRequest(TranslationRequest):
    """Model for literary translation requests."""
    style_context: Dict[str, Any] = Field(..., description="Context about the literary style")
    preserve_poetic_elements: bool = Field(True, description="Whether to preserve poetic elements")

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

@router.post("/translate", response_model=TranslationResponse)
async def process_translation(
    request: TranslationRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Process a translation request using the Translation Agent.
    
    This endpoint handles various types of translation tasks including:
    - Document translation with format preservation
    - Real-time conversation translation
    - Cultural adaptation
    - Technical content translation
    - Literary content translation
    """
    try:
        # Check permissions
        has_permission = await permission_repository.check_permission(
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            resource_type="AGENT",
            resource_id="translation",
            permission_type="EXECUTE"
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to use the translation agent"
            )

        # Create translation agent
        agent_config = {
            "model": request.model or "sonar-large-online",
            "model_config": {
                "model": request.model or "sonar-large-online",
                "temperature": 0.7,
                "max_tokens": 2048
            }
        }
        
        translation_agent = TranslationAgent(agent_config)
        
        # Process the request
        result = await translation_agent.process_async({
            "content": request.content,
            "source_language": request.source_language,
            "target_language": request.target_language,
            "mode": request.mode,
            "context": request.context or {}
        })
        
        if "error" in result:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=result["error"]
            )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=current_user.organization_id,
            activity_type="translation.translated",
            details={
                "mode": request.mode,
                "source_language": request.source_language,
                "target_language": request.target_language,
                "content_preview": request.content[:100],
            },
            related_resource_type="AGENT",
            agent_name="Translation Agent",
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing translation request: {str(e)}"
        )

@router.post("/translate/document", response_model=TranslationResponse)
async def translate_document(
    request: DocumentTranslationRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Translate a document with format preservation.
    
    This endpoint provides document translation with:
    - Format preservation
    - Cultural context inclusion
    - Technical accuracy
    - Style maintenance
    """
    try:
        # Set request type for document translation
        request.mode = TranslationMode.DOCUMENT.value
        
        # Add document-specific context
        context = request.context or {}
        context.update({
            "preserve_formatting": request.preserve_formatting,
            "include_cultural_context": request.include_cultural_context
        })
        
        # Process using the main endpoint
        return await process_translation(
            TranslationRequest(
                content=request.content,
                source_language=request.source_language,
                target_language=request.target_language,
                mode=request.mode,
                context=context,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error translating document: {str(e)}"
        )

@router.post("/translate/conversation", response_model=TranslationResponse)
async def translate_conversation(
    request: ConversationTranslationRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Translate conversation content for real-time communication.
    
    This endpoint provides:
    - Natural, conversational translations
    - Context-aware responses
    - Real-time processing
    - Multi-participant support
    """
    try:
        # Set request type for conversation translation
        request.mode = TranslationMode.CONVERSATION.value
        
        # Add conversation-specific context
        context = request.context or {}
        context.update({
            "conversation_context": request.conversation_context,
            "participants": request.participants
        })
        
        # Process using the main endpoint
        return await process_translation(
            TranslationRequest(
                content=request.content,
                source_language=request.source_language,
                target_language=request.target_language,
                mode=request.mode,
                context=context,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error translating conversation: {str(e)}"
        )

@router.post("/translate/cultural", response_model=TranslationResponse)
async def adapt_culturally(
    request: CulturalAdaptationRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Adapt content culturally while maintaining meaning.
    
    This endpoint provides:
    - Cultural adaptation
    - Context preservation
    - Idiom handling
    - Cultural notes
    """
    try:
        # Set request type for cultural adaptation
        request.mode = TranslationMode.CULTURAL.value
        
        # Add cultural adaptation context
        context = request.context or {}
        context.update({
            "target_culture": request.target_culture,
            "adaptation_level": request.adaptation_level
        })
        
        # Process using the main endpoint
        return await process_translation(
            TranslationRequest(
                content=request.content,
                source_language=request.source_language,
                target_language=request.target_language,
                mode=request.mode,
                context=context,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error adapting content culturally: {str(e)}"
        )

@router.post("/translate/technical", response_model=TranslationResponse)
async def translate_technical(
    request: TechnicalTranslationRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Translate technical content with domain-specific accuracy.
    
    This endpoint provides:
    - Technical terminology accuracy
    - Domain-specific translations
    - Format preservation
    - Technical term glossary
    """
    try:
        # Set request type for technical translation
        request.mode = TranslationMode.TECHNICAL.value
        
        # Add technical translation context
        context = request.context or {}
        context.update({
            "domain": request.domain,
            "terminology_style": request.terminology_style
        })
        
        # Process using the main endpoint
        return await process_translation(
            TranslationRequest(
                content=request.content,
                source_language=request.source_language,
                target_language=request.target_language,
                mode=request.mode,
                context=context,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error translating technical content: {str(e)}"
        )

@router.post("/translate/literary", response_model=TranslationResponse)
async def translate_literary(
    request: LiteraryTranslationRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Translate literary content preserving style and artistic elements.
    
    This endpoint provides:
    - Literary style preservation
    - Poetic elements handling
    - Cultural adaptation
    - Style analysis
    """
    try:
        # Set request type for literary translation
        request.mode = TranslationMode.LITERARY.value
        
        # Add literary translation context
        context = request.context or {}
        context.update({
            "style_context": request.style_context,
            "preserve_poetic_elements": request.preserve_poetic_elements
        })
        
        # Process using the main endpoint
        return await process_translation(
            TranslationRequest(
                content=request.content,
                source_language=request.source_language,
                target_language=request.target_language,
                mode=request.mode,
                context=context,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error translating literary content: {str(e)}"
        )

@router.get("/languages", response_model=List[Dict[str, Any]])
async def list_supported_languages(
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    List supported languages for translation.
    
    This endpoint returns a list of all supported languages
    with their codes and names.
    """
    # This would typically be loaded from a configuration or database
    languages = [
        {"code": "en", "name": "English", "native_name": "English"},
        {"code": "es", "name": "Spanish", "native_name": "Español"},
        {"code": "fr", "name": "French", "native_name": "Français"},
        {"code": "de", "name": "German", "native_name": "Deutsch"},
        {"code": "it", "name": "Italian", "native_name": "Italiano"},
        {"code": "pt", "name": "Portuguese", "native_name": "Português"},
        {"code": "ru", "name": "Russian", "native_name": "Русский"},
        {"code": "zh", "name": "Chinese", "native_name": "中文"},
        {"code": "ja", "name": "Japanese", "native_name": "日本語"},
        {"code": "ko", "name": "Korean", "native_name": "한국어"},
        {"code": "ar", "name": "Arabic", "native_name": "العربية"},
        {"code": "hi", "name": "Hindi", "native_name": "हिन्दी"}
    ]
    
    return languages

@router.get("/analytics", response_model=Dict[str, Any])
async def get_translation_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get translation analytics.
    
    This endpoint provides analytics about:
    - Translation volume by language pair
    - Translation modes usage
    - Processing times
    - Error rates
    - Quality metrics
    """
    # This would typically fetch from a database
    analytics = {
        "time_range": time_range,
        "total_translations": 5000,
        "average_processing_time": 0.8,  # seconds
        "language_pairs": [
            {"source": "en", "target": "es", "count": 1500},
            {"source": "en", "target": "fr", "count": 1000},
            {"source": "en", "target": "de", "count": 800},
            {"source": "es", "target": "en", "count": 700},
            {"source": "fr", "target": "en", "count": 500}
        ],
        "mode_usage": {
            "document": 2500,
            "conversation": 1500,
            "cultural": 500,
            "technical": 300,
            "literary": 200
        },
        "quality_metrics": {
            "average_confidence": 0.95,
            "error_rate": 0.02,
            "user_satisfaction": 0.92
        }
    }
    
    return analytics 