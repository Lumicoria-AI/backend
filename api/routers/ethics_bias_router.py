"""Ethics & Bias Agent endpoints.

All routes are tenant-scoped on `organization_id` and gated by the
platform permission system.  Mutating routes are rate-limited and
audit-logged.  The agent runs through
`services.ethics_bias.orchestrator`, which:

  - resolves the LLM provider (Gemini / Claude / Perplexity / etc.)
  - persists every analysis to Mongo so users can see history
  - derives a 0..100 ethics score and an issue count for the UI
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator

from backend.agents.ethics_bias_agent import BiasType, EthicsCategory, IssueSeverity
from backend.api.deps import get_current_active_user
from backend.core.security import rate_limit
from backend.db.mongodb.repositories.permission_repository import permission_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.ethics_bias import orchestrator
from backend.services.ethics_bias import repository as ethics_repo
from backend.services.ethics_bias.sanitize import (
    MAX_CONTENT_CHARS,
    clean_label,
    clean_parameters,
    clean_string_list,
    clean_text,
)

logger = structlog.get_logger(__name__)

router = APIRouter(
    responses={404: {"description": "Not found"}},
)


# ── Permission helper ───────────────────────────────────────────────


async def _require_ethics_bias_permission(current_user: User) -> str:
    """Resolve the tenant scope id, or raise 403 if the caller doesn't
    have AGENT/ethics_bias/EXECUTE.  Defensive on `organization_id`
    because the User model does not always expose it as an attribute
    (same pattern KG / Legal / Customer Service use)."""
    user_id = str(current_user.id)
    permission_org = getattr(current_user, "organization_id", None)
    has = await permission_repository.check_permission(
        user_id=user_id,
        organization_id=permission_org,
        resource_type="AGENT",
        resource_id="ethics_bias",
        permission_type="EXECUTE",
    )
    if not has:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to use the ethics & bias agent",
        )
    # Personal accounts: organization_id is None; fall back to user_id
    # for tenant scoping so each personal account gets its own history.
    return str(permission_org) if permission_org else user_id


# ── Pydantic request models ─────────────────────────────────────────


PROVIDER_LITERALS = (
    "gemini", "anthropic", "claude", "perplexity", "openai", "mistral", "deepseek"
)


class _BaseEthicsRequest(BaseModel):
    """Common fields shared by every ethics-bias action."""
    context: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    provider: Optional[str] = Field(None, max_length=24)
    model: Optional[str] = Field(None, max_length=64)

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


class ContentAnalysisRequest(_BaseEthicsRequest):
    """Analyze content for ethical issues and bias."""
    content: str = Field(..., max_length=MAX_CONTENT_CHARS)
    content_type: str = Field("text", max_length=32)

    @field_validator("content")
    @classmethod
    def _clean_content(cls, v: str) -> str:
        return clean_text(v, max_len=MAX_CONTENT_CHARS)

    @field_validator("content_type")
    @classmethod
    def _clean_ctype(cls, v: str) -> str:
        return clean_label(v, max_len=32) or "text"


class GuidelineCheckRequest(_BaseEthicsRequest):
    """Check content against ethical guidelines."""
    content: str = Field(..., max_length=MAX_CONTENT_CHARS)
    guidelines_focus: List[str] = Field(default_factory=list)

    @field_validator("content")
    @classmethod
    def _clean_content(cls, v: str) -> str:
        return clean_text(v, max_len=MAX_CONTENT_CHARS)

    @field_validator("guidelines_focus")
    @classmethod
    def _clean_focus(cls, v: List[str]) -> List[str]:
        return clean_string_list(v)


class SuggestionRequest(_BaseEthicsRequest):
    """Generate suggestions for addressing a set of issues."""
    issues: List[Dict[str, Any]] = Field(...)
    content: str = Field("", max_length=MAX_CONTENT_CHARS)

    @field_validator("content")
    @classmethod
    def _clean_content(cls, v: str) -> str:
        return clean_text(v, max_len=MAX_CONTENT_CHARS)


class CitationRequest(_BaseEthicsRequest):
    """Find citations for an ethics topic."""
    topic: str = Field(..., max_length=500)

    @field_validator("topic")
    @classmethod
    def _clean_topic(cls, v: str) -> str:
        return clean_label(v, max_len=500) or ""


# ── Internal dispatcher ─────────────────────────────────────────────


async def _dispatch(
    *,
    action: str,
    organization_id: str,
    user_id: str,
    content: Optional[str] = None,
    content_type: Optional[str] = None,
    issues: Optional[List[Dict[str, Any]]] = None,
    topic: Optional[str] = None,
    guidelines_focus: Optional[List[str]] = None,
    context: Dict[str, Any],
    parameters: Dict[str, Any],
    metadata: Dict[str, Any],
    provider: Optional[str],
    model: Optional[str],
) -> Dict[str, Any]:
    """Run the orchestrator and translate exceptions into HTTPException."""
    try:
        result = await orchestrator.run(
            organization_id=organization_id,
            user_id=user_id,
            action=action,
            content=content,
            content_type=content_type,
            issues=issues,
            topic=topic,
            guidelines_focus=guidelines_focus,
            context=context,
            parameters=parameters,
            provider=provider,
            model_name=model,
            request_metadata=metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("ethics_bias_dispatch_failed", action=action, error=str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Error processing ethics & bias request: {str(e)}",
        )
    return result


async def _log_activity_safely(
    *,
    user_id: str,
    organization_id: str,
    action: str,
    analysis_id: Optional[str],
    provider: Optional[str],
    model: Optional[str],
) -> None:
    try:
        await log_activity(
            user_id=user_id,
            organization_id=organization_id,
            activity_type=f"ethics_bias.{action}",
            details={"provider": provider, "model": model},
            related_resource_type="AGENT",
            related_resource_id=analysis_id,
            agent_name="Ethics & Bias Agent",
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("ethics_bias_activity_log_failed", error=str(e))


# ── Action endpoints ────────────────────────────────────────────────


@router.post("/analyze")
@rate_limit(limit=20, window=900)
async def analyze_content(
    request: Request,
    payload: ContentAnalysisRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Analyze content for ethical issues and bias."""
    org_id = await _require_ethics_bias_permission(current_user)
    user_id = str(current_user.id)

    result = await _dispatch(
        action="analyze",
        organization_id=org_id,
        user_id=user_id,
        content=payload.content,
        content_type=payload.content_type,
        context=payload.context,
        parameters=clean_parameters(payload.parameters),
        metadata=payload.metadata,
        provider=payload.provider,
        model=payload.model,
    )
    meta = result.get("metadata") or {}
    await _log_activity_safely(
        user_id=user_id,
        organization_id=org_id,
        action="analyze",
        analysis_id=meta.get("analysis_id"),
        provider=meta.get("model_provider"),
        model=meta.get("model_name"),
    )
    return result


@router.post("/check-guidelines")
@rate_limit(limit=20, window=900)
async def check_guidelines(
    request: Request,
    payload: GuidelineCheckRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Check content against ethical guidelines."""
    org_id = await _require_ethics_bias_permission(current_user)
    user_id = str(current_user.id)

    result = await _dispatch(
        action="check_guidelines",
        organization_id=org_id,
        user_id=user_id,
        content=payload.content,
        guidelines_focus=payload.guidelines_focus,
        context=payload.context,
        parameters=clean_parameters(payload.parameters),
        metadata=payload.metadata,
        provider=payload.provider,
        model=payload.model,
    )
    meta = result.get("metadata") or {}
    await _log_activity_safely(
        user_id=user_id,
        organization_id=org_id,
        action="check_guidelines",
        analysis_id=meta.get("analysis_id"),
        provider=meta.get("model_provider"),
        model=meta.get("model_name"),
    )
    return result


@router.post("/generate-suggestions")
@rate_limit(limit=20, window=900)
async def generate_suggestions(
    request: Request,
    payload: SuggestionRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Generate suggestions for addressing a set of issues."""
    org_id = await _require_ethics_bias_permission(current_user)
    user_id = str(current_user.id)

    result = await _dispatch(
        action="generate_suggestions",
        organization_id=org_id,
        user_id=user_id,
        content=payload.content,
        issues=payload.issues,
        context=payload.context,
        parameters=clean_parameters(payload.parameters),
        metadata=payload.metadata,
        provider=payload.provider,
        model=payload.model,
    )
    meta = result.get("metadata") or {}
    await _log_activity_safely(
        user_id=user_id,
        organization_id=org_id,
        action="generate_suggestions",
        analysis_id=meta.get("analysis_id"),
        provider=meta.get("model_provider"),
        model=meta.get("model_name"),
    )
    return result


@router.post("/get-citations")
@rate_limit(limit=20, window=900)
async def get_citations(
    request: Request,
    payload: CitationRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Find citations and references on an ethics topic."""
    org_id = await _require_ethics_bias_permission(current_user)
    user_id = str(current_user.id)

    result = await _dispatch(
        action="get_citations",
        organization_id=org_id,
        user_id=user_id,
        topic=payload.topic,
        context=payload.context,
        parameters=clean_parameters(payload.parameters),
        metadata=payload.metadata,
        provider=payload.provider,
        model=payload.model,
    )
    meta = result.get("metadata") or {}
    await _log_activity_safely(
        user_id=user_id,
        organization_id=org_id,
        action="get_citations",
        analysis_id=meta.get("analysis_id"),
        provider=meta.get("model_provider"),
        model=meta.get("model_name"),
    )
    return result


# ── History endpoints ──────────────────────────────────────────────


@router.get("/history")
async def list_history(
    action: Optional[str] = Query(None, max_length=32),
    time_range: Optional[str] = Query(None, pattern="^(1d|7d|30d|90d|1y)$"),
    limit: int = Query(30, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """List recent analyses for the caller's organization."""
    org_id = await _require_ethics_bias_permission(current_user)
    if action:
        action = action.strip().lower()
        if action not in ethics_repo.VALID_ACTIONS:
            action = None
    return await ethics_repo.list_analyses(
        org_id,
        action=action,
        time_range=time_range,
        limit=limit,
        offset=offset,
    )


@router.get("/history/{analysis_id}")
async def get_history_item(
    analysis_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Reopen a past analysis with its full result body."""
    org_id = await _require_ethics_bias_permission(current_user)
    row = await ethics_repo.get_analysis(org_id, analysis_id)
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return row


@router.delete("/history/{analysis_id}")
async def delete_history_item(
    analysis_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Soft-delete a past analysis."""
    org_id = await _require_ethics_bias_permission(current_user)
    user_id = str(current_user.id)
    ok = await ethics_repo.soft_delete_analysis(org_id, analysis_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Analysis not found")
    await _log_activity_safely(
        user_id=user_id,
        organization_id=org_id,
        action="history_delete",
        analysis_id=analysis_id,
        provider=None,
        model=None,
    )
    return {"deleted": True, "id": analysis_id}


@router.get("/analytics")
async def get_ethics_bias_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Workspace analytics derived from persisted history."""
    org_id = await _require_ethics_bias_permission(current_user)
    return await ethics_repo.get_analytics(org_id, time_range=time_range)


# ── Constants endpoints (unchanged, no auth required) ──────────────


@router.get("/ethics-categories")
async def get_ethics_categories() -> List[str]:
    """Get available ethics categories."""
    return [category.value for category in EthicsCategory]


@router.get("/bias-types")
async def get_bias_types() -> List[str]:
    """Get available bias types."""
    return [bias_type.value for bias_type in BiasType]


@router.get("/severity-levels")
async def get_severity_levels() -> List[str]:
    """Get available severity levels."""
    return [severity.value for severity in IssueSeverity]
