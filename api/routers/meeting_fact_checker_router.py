from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from backend.agents.meeting_fact_checker_agent import (
    MeetingFactCheckerAgent,
    ClaimType,
    VerificationStatus,
    ClaimSeverity,
)
from backend.core.dependencies import get_agent_service
from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.db.postgres import get_async_sessionmaker
from backend.db.postgres_models import FactCheckSessionSQL, FactCheckClaimSQL
from sqlalchemy import select

logger = logging.getLogger(__name__)

router = APIRouter(
    responses={404: {"description": "Not found"}},
)


# ── Request Models ────────────────────────────────────────────────

class ClaimVerificationRequest(BaseModel):
    session_id: str = Field(..., description="Active session ID")
    claim: str = Field(..., description="Claim to verify")
    speaker: str = Field(..., description="Speaker making the claim")
    claim_type: ClaimType = Field(default=ClaimType.ASSERTION)
    context: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class SessionStartRequest(BaseModel):
    title: str = Field(..., description="Meeting title")
    participants: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class SessionEndRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to end")
    context: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class SummaryRequest(BaseModel):
    session_id: str = Field(..., description="Session ID to summarize")
    context: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)


# ── Response Models (for library endpoints) ───────────────────────

class FactCheckSessionItem(BaseModel):
    id: str
    title: str
    participants: List[str] = []
    summary: Optional[str] = None
    verification_stats: Dict[str, Any] = {}
    total_claims: int = 0
    started_at: str
    ended_at: Optional[str] = None
    created_at: str


class FactCheckClaimItem(BaseModel):
    id: str
    content: str
    speaker: str
    claim_type: str = "assertion"
    verification_status: str = "pending"
    confidence: int = 0
    severity: str = "medium"
    citations: List[str] = []
    corrections: List[str] = []
    summary: Optional[str] = None
    created_at: str


class FactCheckSessionDetail(FactCheckSessionItem):
    claims: List[FactCheckClaimItem] = []


class FactCheckSessionList(BaseModel):
    total: int
    page: int
    limit: int
    sessions: List[FactCheckSessionItem]


# ── Helpers ───────────────────────────────────────────────────────

async def _save_session_to_db(
    user_id: str,
    session_id: str,
    title: str,
    participants: List[str],
) -> None:
    """Save a new session to Postgres."""
    try:
        sf = get_async_sessionmaker()
        async with sf() as session:
            row = FactCheckSessionSQL(
                id=session_id,
                user_id=user_id,
                title=title,
                participants=participants,
            )
            session.add(row)
            await session.commit()
    except Exception as e:
        logger.error(f"Error saving session to DB: {e}")


async def _save_claim_to_db(
    user_id: str,
    session_id: str,
    claim_data: Dict[str, Any],
    result: Dict[str, Any],
) -> None:
    """Save a verified claim to Postgres."""
    try:
        results = result.get("results", result)
        confidence_raw = results.get("confidence", 0)
        if isinstance(confidence_raw, float) and confidence_raw <= 1:
            confidence_int = round(confidence_raw * 100)
        else:
            confidence_int = int(confidence_raw) if confidence_raw else 0

        sf = get_async_sessionmaker()
        async with sf() as session:
            row = FactCheckClaimSQL(
                id=results.get("claim_id", None),
                session_id=session_id,
                user_id=user_id,
                content=claim_data.get("claim", ""),
                speaker=claim_data.get("speaker", "Unknown"),
                claim_type=claim_data.get("claim_type", "assertion"),
                verification_status=results.get("verification_status", "pending"),
                confidence=confidence_int,
                severity=results.get("severity", "medium"),
                citations=results.get("citations", []),
                corrections=results.get("corrections", []),
                summary=results.get("summary", ""),
            )
            session.add(row)
            await session.commit()
    except Exception as e:
        logger.error(f"Error saving claim to DB: {e}")


async def _end_session_in_db(
    session_id: str,
    summary: str,
    verification_stats: Dict[str, Any],
) -> None:
    """Update session with end time, summary, and stats."""
    try:
        sf = get_async_sessionmaker()
        async with sf() as session:
            q = select(FactCheckSessionSQL).where(FactCheckSessionSQL.id == session_id)
            row = (await session.execute(q)).scalar_one_or_none()
            if row:
                row.ended_at = datetime.utcnow()
                row.summary = summary
                row.verification_stats = verification_stats
                await session.commit()
    except Exception as e:
        logger.error(f"Error ending session in DB: {e}")


# ── Core Endpoints (with auto-save) ──────────────────────────────

@router.post("/start-session")
async def start_session(
    request: SessionStartRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service=Depends(get_agent_service),
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
            "parameters": request.parameters,
        })

        sid = result.get("results", {}).get("session_id")
        if sid:
            await _save_session_to_db(
                user_id=str(current_user.id),
                session_id=sid,
                title=request.title,
                participants=request.participants,
            )

        return result
    except Exception as e:
        logger.error(f"Error starting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/verify-claim")
async def verify_claim(
    request: ClaimVerificationRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service=Depends(get_agent_service),
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
            "parameters": request.parameters,
        })

        # Auto-save claim to Postgres
        await _save_claim_to_db(
            user_id=str(current_user.id),
            session_id=request.session_id,
            claim_data=request.dict(),
            result=result,
        )

        return result
    except Exception as e:
        logger.error(f"Error verifying claim: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/end-session")
async def end_session(
    request: SessionEndRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service=Depends(get_agent_service),
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
            "parameters": request.parameters,
        })

        # Save end state to Postgres
        results = result.get("results", {})
        await _end_session_in_db(
            session_id=request.session_id,
            summary=results.get("summary", ""),
            verification_stats=results.get("verification_stats", {}),
        )

        return result
    except Exception as e:
        logger.error(f"Error ending session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/get-summary")
async def get_session_summary(
    request: SummaryRequest,
    current_user: User = Depends(get_current_active_user),
    agent_service=Depends(get_agent_service),
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
            "parameters": request.parameters,
        })

        return result
    except Exception as e:
        logger.error(f"Error getting session summary: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ── Enum Endpoints ────────────────────────────────────────────────

@router.get("/claim-types")
async def get_claim_types() -> List[str]:
    return [ct.value for ct in ClaimType]


@router.get("/verification-statuses")
async def get_verification_statuses() -> List[str]:
    return [s.value for s in VerificationStatus]


@router.get("/claim-severities")
async def get_claim_severities() -> List[str]:
    return [s.value for s in ClaimSeverity]


# ── Library Endpoints (Postgres persistence) ─────────────────────

@router.get("/sessions", response_model=FactCheckSessionList)
async def list_sessions(
    page: int = 1,
    limit: int = 20,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """List all fact-check sessions for the current user."""
    try:
        sf = get_async_sessionmaker()
        async with sf() as session:
            base = select(FactCheckSessionSQL).where(
                FactCheckSessionSQL.user_id == str(current_user.id),
                FactCheckSessionSQL.deleted_at.is_(None),
            )

            from sqlalchemy import func
            total = (await session.execute(
                select(func.count()).select_from(base.subquery())
            )).scalar() or 0

            offset = (page - 1) * limit
            rows = (await session.execute(
                base.order_by(FactCheckSessionSQL.created_at.desc())
                .offset(offset).limit(limit)
            )).scalars().all()

            # Count claims per session
            sessions_out = []
            for s in rows:
                claim_count_q = select(func.count()).where(
                    FactCheckClaimSQL.session_id == s.id
                )
                claim_count = (await session.execute(claim_count_q)).scalar() or 0

                sessions_out.append(FactCheckSessionItem(
                    id=s.id,
                    title=s.title,
                    participants=s.participants or [],
                    summary=s.summary,
                    verification_stats=s.verification_stats or {},
                    total_claims=claim_count,
                    started_at=s.started_at.isoformat() if s.started_at else s.created_at.isoformat(),
                    ended_at=s.ended_at.isoformat() if s.ended_at else None,
                    created_at=s.created_at.isoformat(),
                ))

            return FactCheckSessionList(total=total, page=page, limit=limit, sessions=sessions_out)

    except Exception as e:
        logger.error(f"Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/sessions/{session_id}", response_model=FactCheckSessionDetail)
async def get_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get a session with all its claims."""
    try:
        sf = get_async_sessionmaker()
        async with sf() as session:
            q = select(FactCheckSessionSQL).where(
                FactCheckSessionSQL.id == session_id,
                FactCheckSessionSQL.user_id == str(current_user.id),
                FactCheckSessionSQL.deleted_at.is_(None),
            )
            s = (await session.execute(q)).scalar_one_or_none()
            if not s:
                raise HTTPException(status_code=404, detail="Session not found")

            claims_q = (
                select(FactCheckClaimSQL)
                .where(FactCheckClaimSQL.session_id == session_id)
                .order_by(FactCheckClaimSQL.created_at.desc())
            )
            claims = (await session.execute(claims_q)).scalars().all()

            return FactCheckSessionDetail(
                id=s.id,
                title=s.title,
                participants=s.participants or [],
                summary=s.summary,
                verification_stats=s.verification_stats or {},
                total_claims=len(claims),
                started_at=s.started_at.isoformat() if s.started_at else s.created_at.isoformat(),
                ended_at=s.ended_at.isoformat() if s.ended_at else None,
                created_at=s.created_at.isoformat(),
                claims=[
                    FactCheckClaimItem(
                        id=c.id,
                        content=c.content,
                        speaker=c.speaker,
                        claim_type=c.claim_type,
                        verification_status=c.verification_status,
                        confidence=c.confidence,
                        severity=c.severity,
                        citations=c.citations or [],
                        corrections=c.corrections or [],
                        summary=c.summary,
                        created_at=c.created_at.isoformat(),
                    )
                    for c in claims
                ],
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/sessions/{session_id}")
async def delete_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, str]:
    """Soft-delete a session."""
    try:
        sf = get_async_sessionmaker()
        async with sf() as session:
            q = select(FactCheckSessionSQL).where(
                FactCheckSessionSQL.id == session_id,
                FactCheckSessionSQL.user_id == str(current_user.id),
                FactCheckSessionSQL.deleted_at.is_(None),
            )
            s = (await session.execute(q)).scalar_one_or_none()
            if not s:
                raise HTTPException(status_code=404, detail="Session not found")

            s.deleted_at = datetime.utcnow()
            await session.commit()
            return {"status": "deleted", "session_id": session_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))
