from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from agents.meeting_fact_checker_agent import (
    MeetingFactCheckerAgent,
    ClaimType,
    VerificationStatus,
    ClaimSeverity
)
from api.dependencies import get_agent_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/meeting-fact-checker",
    tags=["meeting-fact-checker"],
    responses={404: {"description": "Not found"}},
)

# Request/Response Models
class ClaimVerificationRequest(BaseModel):
    """Request model for claim verification."""
    session_id: str = Field(..., description="Active session ID")
    claim: str = Field(..., description="Claim to verify")
    speaker: str = Field(..., description="Speaker making the claim")
    claim_type: ClaimType = Field(
        default=ClaimType.ASSERTION,
        description="Type of claim"
    )
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context for verification"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Verification parameters"
    )

class SessionStartRequest(BaseModel):
    """Request model for starting a meeting session."""
    title: str = Field(..., description="Meeting title")
    participants: List[str] = Field(..., description="Meeting participants")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Session context"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Session parameters"
    )

class SessionEndRequest(BaseModel):
    """Request model for ending a meeting session."""
    session_id: str = Field(..., description="Session ID to end")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="End session context"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="End session parameters"
    )

class SummaryRequest(BaseModel):
    """Request model for getting session summary."""
    session_id: str = Field(..., description="Session ID to summarize")
    context: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary context"
    )
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Summary parameters"
    )

# API Endpoints
@router.post("/verify-claim")
async def verify_claim(
    request: ClaimVerificationRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Verify a claim made during a meeting."""
    try:
        agent = agent_service.get_agent("meeting_fact_checker")
        if not isinstance(agent, MeetingFactCheckerAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "verify",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error verifying claim: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/start-session")
async def start_session(
    request: SessionStartRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Start a new meeting session."""
    try:
        agent = agent_service.get_agent("meeting_fact_checker")
        if not isinstance(agent, MeetingFactCheckerAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "start_session",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error starting session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/end-session")
async def end_session(
    request: SessionEndRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """End a meeting session and generate summary."""
    try:
        agent = agent_service.get_agent("meeting_fact_checker")
        if not isinstance(agent, MeetingFactCheckerAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "end_session",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error ending session: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/get-summary")
async def get_session_summary(
    request: SummaryRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Get summary of an active session."""
    try:
        agent = agent_service.get_agent("meeting_fact_checker")
        if not isinstance(agent, MeetingFactCheckerAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "get_summary",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error getting session summary: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/claim-types")
async def get_claim_types() -> List[str]:
    """Get available claim types."""
    return [claim_type.value for claim_type in ClaimType]

@router.get("/verification-statuses")
async def get_verification_statuses() -> List[str]:
    """Get available verification statuses."""
    return [status.value for status in VerificationStatus]

@router.get("/claim-severities")
async def get_claim_severities() -> List[str]:
    """Get available claim severity levels."""
    return [severity.value for severity in ClaimSeverity] 