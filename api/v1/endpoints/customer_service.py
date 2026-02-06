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
        # Check permissions
        has_permission = await permission_repository.check_permission(
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            resource_type="AGENT",
            resource_id="customer_service",
            permission_type="EXECUTE"
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to use the customer service agent"
            )

        # Create customer service agent
        agent_config = {
            "model": request.model or "sonar-large-online",
            "model_config": {
                "model": request.model or "sonar-large-online",
                "temperature": 0.7,
                "max_tokens": 2048
            }
        }
        
        customer_service_agent = CustomerServiceAgent(agent_config)
        
        # Process the request
        result = await customer_service_agent.process_async({
            "content": request.content,
            "request_type": request.request_type,
            "context": request.context or {}
        })
        
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
    category: Optional[str] = Query(None, description="Filter templates by category"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    List available response templates.
    
    This endpoint returns a list of pre-defined and custom templates
    that can be used for customer service responses.
    """
    # This would typically fetch from a database
    templates = [
        {
            "id": "template_1",
            "category": "general_inquiry",
            "name": "General Inquiry Response",
            "description": "Template for handling general customer inquiries",
            "variables": ["customer_name", "product_name", "issue_details"],
            "created_at": datetime.utcnow().isoformat()
        },
        # Add more templates...
    ]
    
    if category:
        templates = [t for t in templates if t["category"] == category]
    
    return templates

@router.get("/analytics", response_model=Dict[str, Any])
async def get_customer_service_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get customer service analytics for the specified time range.
    
    This endpoint provides analytics about:
    - Response times
    - Customer satisfaction
    - Common issues
    - Template usage
    - Feedback trends
    """
    # This would typically fetch from a database
    analytics = {
        "time_range": time_range,
        "total_requests": 1000,
        "average_response_time": 0.5,  # seconds
        "satisfaction_rate": 0.95,
        "common_issues": [
            {"issue": "Technical Support", "count": 300},
            {"issue": "Billing", "count": 200},
            {"issue": "Feature Requests", "count": 150}
        ],
        "template_usage": {
            "general_inquiry": 400,
            "technical_support": 300,
            "billing_issue": 200,
            "feature_request": 100
        },
        "feedback_trends": {
            "positive": 0.75,
            "neutral": 0.15,
            "negative": 0.10
        }
    }
    
    return analytics 