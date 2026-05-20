"""Legal Document Agent endpoints.

All routes are tenant-scoped on `organization_id` and gated by the
platform permission system.  Mutating routes are rate-limited and
audit-logged.  The agent runs through
`services.legal_document.orchestrator`, which:

  - resolves the right LLM (Gemini / Claude / Perplexity / etc.)
  - pulls text from an already-uploaded RAG document when requested
  - persists every analysis into Mongo so a user can revisit history
  - returns the analysis id so the UI can open or delete the run

Older callers that hit this module crashed with
`'UserInDB' object has no attribute 'organization_id'` because the User
model does not expose the field directly — we now resolve it defensively
via the permission system, matching the pattern in the Knowledge Graph
and Customer Service routers.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator

from backend.api.deps import get_current_active_user
from backend.core.security import rate_limit
from backend.db.mongodb.repositories.permission_repository import permission_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.legal_document import orchestrator
from backend.services.legal_document import repository as legal_repo
from backend.services.legal_document.sanitize import (
    MAX_DOCUMENT_CHARS,
    clean_label,
    clean_parameters,
    clean_text,
)

logger = structlog.get_logger(__name__)

router = APIRouter()


# ── Permission helper ───────────────────────────────────────────────


async def _require_legal_permission(current_user: User) -> str:
    """Resolve the tenant scope id, or raise 403 if the caller doesn't
    have AGENT/legal_document/EXECUTE.  Same pattern KG / Customer
    Service use; defensive on `organization_id` because the User model
    does not always expose it as an attribute."""
    user_id = str(current_user.id)
    permission_org = getattr(current_user, "organization_id", None)
    has = await permission_repository.check_permission(
        user_id=user_id,
        organization_id=permission_org,
        resource_type="AGENT",
        resource_id="legal_document",
        permission_type="EXECUTE",
    )
    if not has:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to use the legal document agent",
        )
    # Personal accounts: organization_id is None; fall back to user_id
    # for tenant scoping so each personal account gets its own history.
    return str(permission_org) if permission_org else user_id


# ── Pydantic models ─────────────────────────────────────────────────


PROVIDER_LITERALS = ("gemini", "anthropic", "claude", "perplexity", "openai", "mistral", "deepseek")


class BaseLegalRequest(BaseModel):
    """Common fields shared by all legal analysis modes.

    Either `content` OR `rag_document_id` must be set (except for
    /compare/versions, which uses its own old/new fields).  Inline
    content has a hard cap matching `MAX_DOCUMENT_CHARS`.
    """
    content: Optional[str] = Field(None, max_length=MAX_DOCUMENT_CHARS)
    rag_document_id: Optional[str] = Field(None, max_length=128)
    context: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provider: Optional[str] = Field(None, max_length=24)
    model: Optional[str] = Field(None, max_length=64)

    @field_validator("content")
    @classmethod
    def _validate_content(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return clean_text(v, max_len=MAX_DOCUMENT_CHARS)

    @field_validator("provider")
    @classmethod
    def _validate_provider(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        v = v.strip().lower()
        if v not in PROVIDER_LITERALS:
            raise ValueError(
                f"provider must be one of {', '.join(PROVIDER_LITERALS)}"
            )
        return v


class ClauseExtractionRequest(BaseLegalRequest):
    include_metadata: bool = True
    highlight_obligations: bool = True
    extract_dates: bool = True


class RiskAnalysisRequest(BaseLegalRequest):
    risk_threshold: float = Field(0.7, ge=0.0, le=1.0)
    include_recommendations: bool = True
    categorize_risks: bool = True


class PlainLanguageRequest(BaseLegalRequest):
    simplify_terms: bool = True
    include_examples: bool = True
    maintain_legal_accuracy: bool = True


class ComplianceCheckRequest(BaseLegalRequest):
    jurisdiction: str = Field("global", max_length=64)
    industry_specific: bool = True
    include_citations: bool = True

    @field_validator("jurisdiction")
    @classmethod
    def _clean_jurisdiction(cls, v: str) -> str:
        return clean_label(v, max_len=64) or "global"


class VersionComparisonRequest(BaseLegalRequest):
    old_version: str = Field(..., max_length=MAX_DOCUMENT_CHARS)
    new_version: str = Field(..., max_length=MAX_DOCUMENT_CHARS)
    track_changes: bool = True
    summarize_changes: bool = True

    @field_validator("old_version", "new_version")
    @classmethod
    def _clean_version(cls, v: str) -> str:
        return clean_text(v, max_len=MAX_DOCUMENT_CHARS)


class LegalDocumentResponse(BaseModel):
    results: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None


# ── Internal dispatcher ─────────────────────────────────────────────


async def _dispatch(
    *,
    mode: str,
    organization_id: str,
    user_id: str,
    content: Optional[str],
    rag_document_id: Optional[str],
    parameters: Dict[str, Any],
    context: Dict[str, Any],
    metadata: Dict[str, Any],
    provider: Optional[str],
    model: Optional[str],
    extra_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run the orchestrator and translate exceptions into HTTPException."""
    try:
        result = await orchestrator.run(
            organization_id=organization_id,
            user_id=user_id,
            mode=mode,
            content=content,
            rag_document_id=rag_document_id,
            extra_data=extra_data,
            context=context,
            parameters=parameters,
            provider=provider,
            model_name=model,
            request_metadata=metadata,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("legal_dispatch_failed", mode=mode, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error processing legal document analysis: {str(e)}",
        )
    return result


async def _log_activity_safely(
    *,
    user_id: str,
    organization_id: str,
    mode: str,
    analysis_id: Optional[str],
    provider: Optional[str],
    model: Optional[str],
) -> None:
    try:
        await log_activity(
            user_id=user_id,
            organization_id=organization_id,
            activity_type=f"legal.{mode}",
            details={"provider": provider, "model": model},
            related_resource_type="AGENT",
            related_resource_id=analysis_id,
            agent_name="Legal Document Agent",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("legal_activity_log_failed", error=str(e))


# ── Mode endpoints ─────────────────────────────────────────────────


@router.post("/analyze/clauses", response_model=LegalDocumentResponse)
@rate_limit(limit=20, window=900)
async def extract_clauses(
    request: Request,
    payload: ClauseExtractionRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Extract key clauses, obligations, and deadlines."""
    org_id = await _require_legal_permission(current_user)
    user_id = str(current_user.id)

    params = clean_parameters(payload.parameters)
    params.update({
        "include_metadata": payload.include_metadata,
        "highlight_obligations": payload.highlight_obligations,
        "extract_dates": payload.extract_dates,
    })

    result = await _dispatch(
        mode="clause_extraction",
        organization_id=org_id,
        user_id=user_id,
        content=payload.content,
        rag_document_id=payload.rag_document_id,
        parameters=params,
        context=payload.context,
        metadata=payload.metadata,
        provider=payload.provider,
        model=payload.model,
    )
    meta = result.get("metadata") or {}
    await _log_activity_safely(
        user_id=user_id,
        organization_id=org_id,
        mode="clause_extraction",
        analysis_id=meta.get("analysis_id"),
        provider=meta.get("model_provider"),
        model=meta.get("model_name"),
    )
    return result


@router.post("/analyze/risks", response_model=LegalDocumentResponse)
@rate_limit(limit=20, window=900)
async def analyze_risks(
    request: Request,
    payload: RiskAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Identify high-risk clauses and unusual terms."""
    org_id = await _require_legal_permission(current_user)
    user_id = str(current_user.id)

    params = clean_parameters(payload.parameters)
    params.update({
        "risk_threshold": payload.risk_threshold,
        "include_recommendations": payload.include_recommendations,
        "categorize_risks": payload.categorize_risks,
    })

    result = await _dispatch(
        mode="risk_analysis",
        organization_id=org_id,
        user_id=user_id,
        content=payload.content,
        rag_document_id=payload.rag_document_id,
        parameters=params,
        context=payload.context,
        metadata=payload.metadata,
        provider=payload.provider,
        model=payload.model,
    )
    meta = result.get("metadata") or {}
    await _log_activity_safely(
        user_id=user_id,
        organization_id=org_id,
        mode="risk_analysis",
        analysis_id=meta.get("analysis_id"),
        provider=meta.get("model_provider"),
        model=meta.get("model_name"),
    )
    return result


@router.post("/summarize/plain", response_model=LegalDocumentResponse)
@rate_limit(limit=20, window=900)
async def generate_plain_language(
    request: Request,
    payload: PlainLanguageRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Plain-English translation of a legal document."""
    org_id = await _require_legal_permission(current_user)
    user_id = str(current_user.id)

    params = clean_parameters(payload.parameters)
    params.update({
        "simplify_terms": payload.simplify_terms,
        "include_examples": payload.include_examples,
        "maintain_legal_accuracy": payload.maintain_legal_accuracy,
    })

    result = await _dispatch(
        mode="plain_language",
        organization_id=org_id,
        user_id=user_id,
        content=payload.content,
        rag_document_id=payload.rag_document_id,
        parameters=params,
        context=payload.context,
        metadata=payload.metadata,
        provider=payload.provider,
        model=payload.model,
    )
    meta = result.get("metadata") or {}
    await _log_activity_safely(
        user_id=user_id,
        organization_id=org_id,
        mode="plain_language",
        analysis_id=meta.get("analysis_id"),
        provider=meta.get("model_provider"),
        model=meta.get("model_name"),
    )
    return result


@router.post("/check/compliance", response_model=LegalDocumentResponse)
@rate_limit(limit=20, window=900)
async def check_compliance(
    request: Request,
    payload: ComplianceCheckRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Regulatory and contractual compliance check."""
    org_id = await _require_legal_permission(current_user)
    user_id = str(current_user.id)

    params = clean_parameters(payload.parameters)
    params.update({
        "jurisdiction": payload.jurisdiction,
        "industry_specific": payload.industry_specific,
        "include_citations": payload.include_citations,
    })

    result = await _dispatch(
        mode="compliance_check",
        organization_id=org_id,
        user_id=user_id,
        content=payload.content,
        rag_document_id=payload.rag_document_id,
        parameters=params,
        context=payload.context,
        metadata=payload.metadata,
        provider=payload.provider,
        model=payload.model,
    )
    meta = result.get("metadata") or {}
    await _log_activity_safely(
        user_id=user_id,
        organization_id=org_id,
        mode="compliance_check",
        analysis_id=meta.get("analysis_id"),
        provider=meta.get("model_provider"),
        model=meta.get("model_name"),
    )
    return result


@router.post("/compare/versions", response_model=LegalDocumentResponse)
@rate_limit(limit=10, window=900)
async def compare_versions(
    request: Request,
    payload: VersionComparisonRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Diff two versions of a legal document."""
    org_id = await _require_legal_permission(current_user)
    user_id = str(current_user.id)

    params = clean_parameters(payload.parameters)
    params.update({
        "track_changes": payload.track_changes,
        "summarize_changes": payload.summarize_changes,
    })

    result = await _dispatch(
        mode="version_comparison",
        organization_id=org_id,
        user_id=user_id,
        content=None,
        rag_document_id=None,
        parameters=params,
        context=payload.context,
        metadata=payload.metadata,
        provider=payload.provider,
        model=payload.model,
        extra_data={
            "old_version": payload.old_version,
            "new_version": payload.new_version,
        },
    )
    meta = result.get("metadata") or {}
    await _log_activity_safely(
        user_id=user_id,
        organization_id=org_id,
        mode="version_comparison",
        analysis_id=meta.get("analysis_id"),
        provider=meta.get("model_provider"),
        model=meta.get("model_name"),
    )
    return result


# ── Generic /analyze (mode-routed) — kept for backwards compat ─────


class GenericLegalRequest(BaseLegalRequest):
    mode: str = Field(..., max_length=32)

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in legal_repo.VALID_MODES:
            raise ValueError(
                f"mode must be one of {', '.join(legal_repo.VALID_MODES)}"
            )
        return v


@router.post("/analyze", response_model=LegalDocumentResponse)
@rate_limit(limit=20, window=900)
async def process_legal_document(
    request: Request,
    payload: GenericLegalRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Mode-routed legal document analysis entry point."""
    org_id = await _require_legal_permission(current_user)
    user_id = str(current_user.id)

    result = await _dispatch(
        mode=payload.mode,
        organization_id=org_id,
        user_id=user_id,
        content=payload.content,
        rag_document_id=payload.rag_document_id,
        parameters=clean_parameters(payload.parameters),
        context=payload.context,
        metadata=payload.metadata,
        provider=payload.provider,
        model=payload.model,
    )
    meta = result.get("metadata") or {}
    await _log_activity_safely(
        user_id=user_id,
        organization_id=org_id,
        mode=payload.mode,
        analysis_id=meta.get("analysis_id"),
        provider=meta.get("model_provider"),
        model=meta.get("model_name"),
    )
    return result


# ── History endpoints ─────────────────────────────────────────────


@router.get("/history")
async def list_history(
    mode: Optional[str] = Query(None, max_length=32),
    time_range: Optional[str] = Query(None, pattern="^(1d|7d|30d|90d|1y)$"),
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """List recent analyses for the caller's organization."""
    org_id = await _require_legal_permission(current_user)
    if mode:
        mode = mode.strip().lower()
        if mode not in legal_repo.VALID_MODES:
            mode = None
    return await legal_repo.list_analyses(
        org_id,
        mode=mode,
        time_range=time_range,
        limit=limit,
        offset=offset,
    )


@router.get("/history/{analysis_id}")
async def get_history_item(
    analysis_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Open a past analysis with its full result body."""
    org_id = await _require_legal_permission(current_user)
    row = await legal_repo.get_analysis(org_id, analysis_id)
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return row


@router.delete("/history/{analysis_id}")
async def delete_history_item(
    analysis_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Soft-delete a past analysis."""
    org_id = await _require_legal_permission(current_user)
    user_id = str(current_user.id)
    ok = await legal_repo.soft_delete_analysis(org_id, analysis_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Analysis not found")
    await _log_activity_safely(
        user_id=user_id,
        organization_id=org_id,
        mode="history_delete",
        analysis_id=analysis_id,
        provider=None,
        model=None,
    )
    return {"deleted": True, "id": analysis_id}


# ── Analytics ─────────────────────────────────────────────────────


@router.get("/analytics")
async def get_legal_document_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Real counts pulled from the Mongo history collection."""
    org_id = await _require_legal_permission(current_user)
    return await legal_repo.get_analytics(org_id, time_range=time_range)
