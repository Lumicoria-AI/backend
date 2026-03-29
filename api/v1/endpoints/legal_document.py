from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from pydantic import BaseModel, Field
from datetime import datetime

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.agent_universe_repository import agent_universe_repository
from backend.db.mongodb.repositories.permission_repository import permission_repository
from backend.models.user import User
from backend.agents.agent_service import AgentService
from backend.agents.legal_document_agent import LegalDocumentAgent, LegalDocumentMode
from backend.services.activity_logger import log_activity

router = APIRouter()

# Request and Response Models
class LegalDocumentRequest(BaseModel):
    """Base model for legal document analysis requests."""
    data: Dict[str, Any] = Field(..., description="Legal document data to analyze")
    mode: str = Field("clause_extraction", description="Analysis mode")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context for analysis")
    parameters: Optional[Dict[str, Any]] = Field(None, description="Analysis-specific parameters")
    model: Optional[str] = Field(None, description="AI model to use (defaults to sonar-large-online)")

class LegalDocumentResponse(BaseModel):
    """Base model for legal document analysis responses."""
    results: Dict[str, Any]
    metadata: Dict[str, Any]

class ClauseExtractionRequest(LegalDocumentRequest):
    """Model for clause extraction requests."""
    include_metadata: bool = Field(True, description="Whether to include clause metadata")
    highlight_obligations: bool = Field(True, description="Whether to highlight obligations")
    extract_dates: bool = Field(True, description="Whether to extract dates")

class RiskAnalysisRequest(LegalDocumentRequest):
    """Model for risk analysis requests."""
    risk_threshold: float = Field(0.7, description="Threshold for risk identification")
    include_recommendations: bool = Field(True, description="Whether to include risk recommendations")
    categorize_risks: bool = Field(True, description="Whether to categorize risks by type")

class VersionComparisonRequest(LegalDocumentRequest):
    """Model for version comparison requests."""
    old_version: str = Field(..., description="Previous version of the document")
    new_version: str = Field(..., description="New version of the document")
    track_changes: bool = Field(True, description="Whether to track detailed changes")
    summarize_changes: bool = Field(True, description="Whether to provide a change summary")

class PlainLanguageRequest(LegalDocumentRequest):
    """Model for plain language summary requests."""
    simplify_terms: bool = Field(True, description="Whether to simplify legal terms")
    include_examples: bool = Field(True, description="Whether to include practical examples")
    maintain_legal_accuracy: bool = Field(True, description="Whether to maintain legal accuracy")

class ComplianceCheckRequest(LegalDocumentRequest):
    """Model for compliance check requests."""
    jurisdiction: str = Field("global", description="Jurisdiction for compliance check")
    industry_specific: bool = Field(True, description="Whether to include industry-specific checks")
    include_citations: bool = Field(True, description="Whether to include regulatory citations")

# Helper function to get agent service
def get_agent_service() -> AgentService:
    """Get or create an instance of AgentService."""
    config = {
        "model": "sonar-large-online",
        "model_config": {
            "model": "sonar-large-online",
            "temperature": 0.3,
            "max_tokens": 4096
        }
    }
    return AgentService(config)

@router.post("/analyze", response_model=LegalDocumentResponse)
async def process_legal_document(
    request: LegalDocumentRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Process a legal document analysis request using the Legal Document Agent.
    
    This endpoint handles various types of analysis tasks including:
    - Clause extraction
    - Risk analysis
    - Version comparison
    - Plain language summaries
    - Compliance checks
    """
    try:
        # Check permissions
        has_permission = await permission_repository.check_permission(
            user_id=current_user.id,
            organization_id=current_user.organization_id,
            resource_type="AGENT",
            resource_id="legal_document",
            permission_type="EXECUTE"
        )
        if not has_permission:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Not enough permissions to use the legal document agent"
            )

        # Create legal document agent
        agent_config = {
            "model": request.model or "sonar-large-online",
            "model_config": {
                "model": request.model or "sonar-large-online",
                "temperature": 0.3,
                "max_tokens": 4096
            }
        }
        
        legal_document_agent = LegalDocumentAgent(agent_config)
        
        # Process the request
        result = await legal_document_agent.process_async({
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

        await log_activity(
            user_id=str(current_user.id),
            organization_id=current_user.organization_id,
            activity_type="legal.document_analyzed",
            details={"mode": request.mode},
            related_resource_type="AGENT",
            agent_name="Legal Document Agent",
        )
        return result

    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing legal document analysis: {str(e)}"
        )

@router.post("/analyze/clauses", response_model=LegalDocumentResponse)
async def extract_clauses(
    request: ClauseExtractionRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Extract key clauses, obligations, and deadlines from legal documents.
    
    This endpoint provides:
    - Clause identification and extraction
    - Obligation highlighting
    - Deadline extraction
    - Clause metadata
    """
    try:
        # Set request type for clause extraction
        request.mode = LegalDocumentMode.CLAUSE_EXTRACTION.value
        
        # Add clause extraction parameters
        parameters = request.parameters or {}
        parameters.update({
            "include_metadata": request.include_metadata,
            "highlight_obligations": request.highlight_obligations,
            "extract_dates": request.extract_dates
        })
        
        # Process using the main endpoint
        return await process_legal_document(
            LegalDocumentRequest(
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
            detail=f"Error extracting clauses: {str(e)}"
        )

@router.post("/analyze/risks", response_model=LegalDocumentResponse)
async def analyze_risks(
    request: RiskAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Analyze potential risks and unusual terms in legal documents.
    
    This endpoint provides:
    - Risk identification
    - Risk categorization
    - Risk recommendations
    - Risk severity assessment
    """
    try:
        # Set request type for risk analysis
        request.mode = LegalDocumentMode.RISK_ANALYSIS.value
        
        # Add risk analysis parameters
        parameters = request.parameters or {}
        parameters.update({
            "risk_threshold": request.risk_threshold,
            "include_recommendations": request.include_recommendations,
            "categorize_risks": request.categorize_risks
        })
        
        # Process using the main endpoint
        return await process_legal_document(
            LegalDocumentRequest(
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
            detail=f"Error analyzing risks: {str(e)}"
        )

@router.post("/compare/versions", response_model=LegalDocumentResponse)
async def compare_versions(
    request: VersionComparisonRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Compare different versions of legal documents and identify changes.
    
    This endpoint provides:
    - Change identification
    - Change tracking
    - Change summaries
    - Version comparison
    """
    try:
        # Set request type for version comparison
        request.mode = LegalDocumentMode.VERSION_COMPARISON.value
        
        # Add version comparison parameters
        parameters = request.parameters or {}
        parameters.update({
            "track_changes": request.track_changes,
            "summarize_changes": request.summarize_changes
        })
        
        # Add versions to data
        data = request.data.copy()
        data.update({
            "old_version": request.old_version,
            "new_version": request.new_version
        })
        
        # Process using the main endpoint
        return await process_legal_document(
            LegalDocumentRequest(
                data=data,
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
            detail=f"Error comparing versions: {str(e)}"
        )

@router.post("/summarize/plain", response_model=LegalDocumentResponse)
async def generate_plain_language(
    request: PlainLanguageRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Generate plain language summaries of legal documents.
    
    This endpoint provides:
    - Plain language summaries
    - Simplified legal terms
    - Practical examples
    - Key points extraction
    """
    try:
        # Set request type for plain language summary
        request.mode = LegalDocumentMode.PLAIN_LANGUAGE.value
        
        # Add plain language parameters
        parameters = request.parameters or {}
        parameters.update({
            "simplify_terms": request.simplify_terms,
            "include_examples": request.include_examples,
            "maintain_legal_accuracy": request.maintain_legal_accuracy
        })
        
        # Process using the main endpoint
        return await process_legal_document(
            LegalDocumentRequest(
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
            detail=f"Error generating plain language summary: {str(e)}"
        )

@router.post("/check/compliance", response_model=LegalDocumentResponse)
async def check_compliance(
    request: ComplianceCheckRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service: AgentService = Depends(get_agent_service)
) -> Any:
    """
    Check regulatory compliance based on document content.
    
    This endpoint provides:
    - Compliance analysis
    - Regulatory citations
    - Compliance recommendations
    - Severity assessment
    """
    try:
        # Set request type for compliance check
        request.mode = LegalDocumentMode.COMPLIANCE_CHECK.value
        
        # Add compliance check parameters
        parameters = request.parameters or {}
        parameters.update({
            "jurisdiction": request.jurisdiction,
            "industry_specific": request.industry_specific,
            "include_citations": request.include_citations
        })
        
        # Process using the main endpoint
        return await process_legal_document(
            LegalDocumentRequest(
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
            detail=f"Error checking compliance: {str(e)}"
        )

@router.get("/analytics", response_model=Dict[str, Any])
async def get_legal_document_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get legal document analysis analytics.
    
    This endpoint provides analytics about:
    - Document processing volume
    - Analysis types distribution
    - Risk levels distribution
    - Compliance issues
    """
    # This would typically fetch from a database
    analytics = {
        "time_range": time_range,
        "total_documents": 500,
        "analysis_types": {
            "clause_extraction": 200,
            "risk_analysis": 150,
            "version_comparison": 50,
            "plain_language": 75,
            "compliance_check": 25
        },
        "risk_levels": {
            "critical": 10,
            "high": 25,
            "medium": 45,
            "low": 70
        },
        "compliance_issues": {
            "regulatory": 30,
            "contractual": 45,
            "industry_specific": 15
        },
        "document_types": {
            "contracts": 300,
            "agreements": 100,
            "policies": 50,
            "regulations": 50
        },
        "average_processing_time": 15.5,  # seconds
        "success_rate": 0.98
    }
    
    return analytics 