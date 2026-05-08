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
from backend.agents.customer_service_agent import CustomerServiceAgent
from backend.services.activity_logger import log_activity

router = APIRouter()

# Request and Response Models
class CustomerServiceRequest(BaseModel):
    """Base model for customer service requests."""
    content: str = Field(..., description="The content to process (inquiry, feedback, etc.)")
    request_type: str = Field(
        ...,
        description="Type of request (generate_response, analyze_feedback, generate_faq, create_template, satisfaction_strategy)"
    )
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for the request")
    model: Optional[str] = Field(None, description="AI model to use (defaults to sonar-large-online)")

class CustomerServiceResponse(BaseModel):
    """Base model for customer service responses."""
    response: Dict[str, Any]
    raw_response: str
    processed_at: datetime
    model_used: str
    request_type: str

class FeedbackAnalysisRequest(CustomerServiceRequest):
    """Model for feedback analysis requests."""
    categories: Optional[List[str]] = Field(None, description="Specific categories to analyze")
    include_sentiment: bool = Field(True, description="Whether to include sentiment analysis")

class FAQGenerationRequest(CustomerServiceRequest):
    """Model for FAQ generation requests."""
    topic: str = Field(..., description="Topic for FAQ generation")
    target_audience: Optional[str] = Field(None, description="Target audience for the FAQ")
    style: Optional[str] = Field("professional", description="Style of the FAQ content")

class TemplateRequest(CustomerServiceRequest):
    """Model for response template requests."""
    template_category: str = Field(..., description="Category of the template")
    variables: Optional[List[str]] = Field(None, description="Variables to include in the template")
    tone: Optional[str] = Field("professional_friendly", description="Tone of the template")

class SatisfactionStrategyRequest(CustomerServiceRequest):
    """Model for customer satisfaction strategy requests."""
    focus_areas: Optional[List[str]] = Field(None, description="Areas to focus on for improvement")
    timeframe: Optional[str] = Field(None, description="Timeframe for strategy implementation")
    priority_level: Optional[str] = Field("medium", description="Priority level of the strategy")

# Helper function to get agent service
def get_agent_service() -> AgentService:
    """Get or create an instance of AgentService."""
    # This would typically load from configuration
    config = {
        "model": "sonar-large-online",
        "model_config": {
            "model": "sonar-large-online",
            "temperature": 0.7,
            "max_tokens": 2048
        }
    }
    return AgentService(config)

@router.post("/process", response_model=CustomerServiceResponse)
async def process_customer_service_request(
    request: CustomerServiceRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Process a customer service request using the Customer Service Agent.
    
    This endpoint handles various types of customer service tasks including:
    - Generating responses to customer inquiries
    - Analyzing customer feedback
    - Generating FAQ content
    - Creating response templates
    - Suggesting customer satisfaction strategies
    """
    try:
        # `UserInDB` doesn't always carry organization_id — use getattr so
        # personal accounts don't blow up.
        org_id = getattr(current_user, "organization_id", None)

        # Check permissions (org_id=None passes through as permitted).
        has_permission = await permission_repository.check_permission(
            user_id=current_user.id,
            organization_id=org_id,
            resource_type="AGENT",
            resource_id="customer_service",
            permission_type="EXECUTE"
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to use the customer service agent"
            )

        # Build the agent config.  Default to whatever provider/model the
        # platform is configured with so we don't hardcode Perplexity here.
        from backend.core.config import settings
        provider = (settings.DEFAULT_LLM_PROVIDER or "gemini").lower()
        default_model = {
            "gemini": getattr(settings, "GEMINI_MODEL", None) or "gemini-2.5-flash",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-haiku-4-5-20251001",
            "mistral": "mistral-small-latest",
            "perplexity": "sonar",
        }.get(provider, "sonar")
        chosen_model = request.model or default_model

        agent_config = {
            "provider": provider,
            "model": chosen_model,
            # BaseAgent reads from `agent_model_config`, NOT `model_config`.
            "agent_model_config": {
                "model": chosen_model,
                "temperature": 0.7,
                "max_tokens": 2048,
            },
        }

        customer_service_agent = CustomerServiceAgent(agent_config)

        # Process the request
        result = await customer_service_agent.process_async({
            "content": request.content,
            "request_type": request.request_type,
            "context": request.context or {}
        })

        await log_activity(
            user_id=str(current_user.id),
            organization_id=org_id,
            activity_type="customer_service.ticket_handled",
            details={"request_type": request.request_type, "content_preview": request.content[:100]},
            related_resource_type="AGENT",
            agent_name="Customer Service Agent",
        )
        return result
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing customer service request: {str(e)}"
        )

@router.post("/analyze-feedback", response_model=CustomerServiceResponse)
async def analyze_customer_feedback(
    request: FeedbackAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Analyze customer feedback using the Customer Service Agent.
    
    This endpoint provides detailed analysis of customer feedback, including:
    - Sentiment analysis
    - Category classification
    - Actionable insights
    - Trend identification
    """
    try:
        # Set request type for feedback analysis
        request.request_type = "analyze_feedback"
        
        # Add feedback-specific context
        context = request.context or {}
        context.update({
            "categories": request.categories,
            "include_sentiment": request.include_sentiment
        })
        
        # Process using the main endpoint
        return await process_customer_service_request(
            CustomerServiceRequest(
                content=request.content,
                request_type=request.request_type,
                context=context,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing customer feedback: {str(e)}"
        )

@router.post("/generate-faq", response_model=CustomerServiceResponse)
async def generate_faq_content(
    request: FAQGenerationRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Generate FAQ content using the Customer Service Agent.
    
    This endpoint creates FAQ entries based on:
    - Common customer questions
    - Product/service information
    - Target audience needs
    - Specific topics or categories
    """
    try:
        # Set request type for FAQ generation
        request.request_type = "generate_faq"
        
        # Add FAQ-specific context
        context = request.context or {}
        context.update({
            "topic": request.topic,
            "target_audience": request.target_audience,
            "style": request.style
        })
        
        # Process using the main endpoint
        return await process_customer_service_request(
            CustomerServiceRequest(
                content=request.content,
                request_type=request.request_type,
                context=context,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating FAQ content: {str(e)}"
        )

@router.post("/create-template", response_model=CustomerServiceResponse)
async def create_response_template(
    request: TemplateRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Create response templates using the Customer Service Agent.
    
    This endpoint generates reusable response templates for:
    - Common customer inquiries
    - Technical support
    - Billing issues
    - Feature requests
    - Complaint handling
    """
    try:
        # Set request type for template creation
        request.request_type = "create_template"
        
        # Add template-specific context
        context = request.context or {}
        context.update({
            "template_category": request.template_category,
            "variables": request.variables,
            "tone": request.tone
        })
        
        # Process using the main endpoint
        return await process_customer_service_request(
            CustomerServiceRequest(
                content=request.content,
                request_type=request.request_type,
                context=context,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error creating response template: {str(e)}"
        )

@router.post("/satisfaction-strategy", response_model=CustomerServiceResponse)
async def generate_satisfaction_strategy(
    request: SatisfactionStrategyRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Generate customer satisfaction strategies using the Customer Service Agent.
    
    This endpoint provides strategies for:
    - Improving customer satisfaction
    - Addressing specific pain points
    - Implementing best practices
    - Measuring success metrics
    """
    try:
        # Set request type for strategy generation
        request.request_type = "satisfaction_strategy"
        
        # Add strategy-specific context
        context = request.context or {}
        context.update({
            "focus_areas": request.focus_areas,
            "timeframe": request.timeframe,
            "priority_level": request.priority_level
        })
        
        # Process using the main endpoint
        return await process_customer_service_request(
            CustomerServiceRequest(
                content=request.content,
                request_type=request.request_type,
                context=context,
                model=request.model
            ),
            current_user,
            agent_service
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating satisfaction strategy: {str(e)}"
        )

@router.get("/templates", response_model=List[Dict[str, Any]])
async def list_response_templates(
    category: Optional[str] = Query(None, description="Filter templates by category", max_length=64),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """List response templates for the caller's organization.

    Tenant-scoped, persisted in `response_templates`.  On the first call
    for an org the five canonical templates are seeded automatically
    (idempotent — subsequent reads do not re-seed).
    """
    from backend.services.customer_service import templates as templates_svc

    user_id = str(current_user.id)
    org_id = getattr(current_user, "organization_id", None) or user_id
    return await templates_svc.list_templates(org_id, category=category)


@router.get("/analytics", response_model=Dict[str, Any])
async def get_customer_service_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Real customer-service analytics aggregated over `support_tickets`,
    `ticket_replies`, and `response_templates`.  Same response shape as
    the legacy mock so existing frontends don't move; values are now
    real per-tenant numbers.
    """
    from backend.services.customer_service import analytics as analytics_svc

    user_id = str(current_user.id)
    org_id = getattr(current_user, "organization_id", None) or user_id
    return await analytics_svc.get_analytics(org_id, time_range=time_range)


# ─── FAQ → Knowledge Base ────────────────────────────────────────────────


class FaqToKnowledgeBaseRequest(BaseModel):
    """Persist a generated FAQ into the RAG ingest pipeline so future
    AI Drafts cite it.  Optionally also save it as a help-center
    article surfaced on the public portal."""
    topic: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=10, max_length=50_000)
    target_audience: Optional[str] = Field(None, max_length=200)
    tags: Optional[List[str]] = None
    # When true, ALSO create a published help-center article from the same content.
    publish_as_article: bool = False
    article_category: Optional[str] = Field(None, max_length=64)


@router.post("/faq/save-to-knowledge-base", status_code=201)
async def save_faq_to_knowledge_base(
    request: FaqToKnowledgeBaseRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Push a generated FAQ into the org's RAG knowledge base.

    The content lands as a `manual_entry` document that the existing
    AI Draft retrieval path picks up automatically.  Optionally
    creates a help-center article from the same content so end users
    can self-serve.
    """
    user_id = str(current_user.id)
    org_id = getattr(current_user, "organization_id", None) or user_id

    # 1. Push into RAG via the existing document_processor.
    try:
        from backend.services.document_processor import document_processor
        result = await document_processor.process_text(
            request.content,
            metadata={
                "user_id": user_id,
                "organization_id": org_id,
                "source": "manual_entry",
                "title": f"FAQ — {request.topic}",
                "tags": ["faq"] + list(request.tags or []),
                "topic": request.topic,
                "target_audience": request.target_audience,
                "mime_type": "text/markdown",
                "channel": "customer_service.faq",
            },
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to ingest FAQ into knowledge base: {e}",
        )

    rag_document_id = getattr(result, "document_id", None) or (
        result.get("document_id") if isinstance(result, dict) else None
    )

    article_dict: Optional[Dict[str, Any]] = None
    if request.publish_as_article:
        try:
            from backend.services.customer_service import articles as articles_svc
            article_dict = await articles_svc.create_article(
                organization_id=org_id,
                title=f"{request.topic}",
                body=request.content,
                summary=(request.target_audience and f"For: {request.target_audience}") or None,
                category=request.article_category or "faq",
                tags=["faq"] + list(request.tags or []),
                published=True,
                featured=False,
                created_by_user_id=user_id,
            )
            if rag_document_id and article_dict:
                await articles_svc.link_rag_document(
                    org_id, article_dict["id"], rag_document_id,
                )
                article_dict["rag_document_id"] = rag_document_id
        except Exception as e:
            # Article creation is best-effort — the RAG ingest already succeeded.
            await log_activity(
                user_id=user_id,
                organization_id=org_id,
                activity_type="customer_service.faq_article_create_failed",
                details={"topic": request.topic, "error": str(e)},
                agent_name="Customer Service Agent",
            )

    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="customer_service.faq_saved_to_kb",
        details={
            "topic": request.topic,
            "rag_document_id": rag_document_id,
            "article_id": (article_dict or {}).get("id"),
        },
        related_resource_type="DOCUMENT",
        related_resource_id=rag_document_id or "",
        agent_name="Customer Service Agent",
    )

    return {
        "rag_document_id": rag_document_id,
        "rag_status": getattr(result, "status", None) or (result.get("status") if isinstance(result, dict) else None),
        "article": article_dict,
        "topic": request.topic,
    }