"""Service-layer orchestrator for the Legal Document Agent.

Responsibilities:
  1. Resolve content from either inline text or a RAG document id.
  2. Build the right agent config for the chosen model provider
     (Gemini, Claude/Anthropic, Perplexity, OpenAI, Mistral, DeepSeek).
  3. Run the agent.
  4. Persist the analysis row to Mongo (history + audit).
  5. Return the enriched result, augmented with the analysis id so the
     frontend can open / delete it later.

The agent itself stays single-purpose: it processes text into clauses,
risks, summaries, or compliance findings.  Anything tenant-shaped lives
here, so the agent cannot leak across organizations.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, Optional

import structlog

from . import repository as legal_repo
from .sanitize import (
    MAX_DOCUMENT_CHARS,
    clean_label,
    clean_parameters,
    clean_text,
    make_preview,
    make_title,
)

logger = structlog.get_logger(__name__)


# Per-organization async locks so two simultaneous calls do not collide
# on the in-process agent instance.  Single-process guarantee; we rely
# on Mongo's atomic update for cross-process safety.
_org_locks: Dict[str, asyncio.Lock] = {}


def _lock_for(org_id: str) -> asyncio.Lock:
    if org_id not in _org_locks:
        _org_locks[org_id] = asyncio.Lock()
    return _org_locks[org_id]


# ── Provider / model resolution ────────────────────────────────────


# Mapping of friendly provider names → default model id for each.  The
# UI sends `provider`; the orchestrator picks a sensible default model
# unless the caller overrides it via `model_name`.
PROVIDER_DEFAULTS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "claude": "claude-haiku-4-5-20251001",  # alias users tend to type
    # Gemini default = 2.5-flash.  2.5-pro is a better fit for legal
    # work but its free-tier quota is zero — callers should opt in
    # explicitly via `model="gemini-2.5-pro"` once their Google Cloud
    # project has billing enabled.
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "perplexity": "sonar",
    "mistral": "mistral-small-latest",
    "deepseek": "deepseek-chat",
}


def _resolve_provider(requested: Optional[str]) -> str:
    """Normalize a user-supplied provider string.  Fallback to the
    platform default if the value is unknown."""
    from ...core.config import settings

    if requested:
        provider = requested.strip().lower()
        if provider in PROVIDER_DEFAULTS:
            return "anthropic" if provider == "claude" else provider
    return (settings.DEFAULT_LLM_PROVIDER or "gemini").lower()


def _build_agent_config(
    provider: Optional[str], model_override: Optional[str]
) -> Dict[str, Any]:
    """Build the BaseAgent config dict for the legal analysis call."""
    resolved_provider = _resolve_provider(provider)
    model = (
        model_override
        or PROVIDER_DEFAULTS.get(resolved_provider)
        or "gemini-2.5-flash"
    )
    return {
        "provider": resolved_provider,
        "model": model,
        # BaseAgent reads from `agent_model_config`, not `model_config`.
        "agent_model_config": {
            "model": model,
            "temperature": 0.3,
            # Legal output is long and structured; Claude needs space
            # for the full clause list, Gemini 2.5 burns part of its
            # budget on internal reasoning — give both room to breathe.
            "max_tokens": 16384,
        },
    }, resolved_provider, model


def _new_agent(provider: Optional[str], model_override: Optional[str]):
    """Instantiate a fresh LegalDocumentAgent for this request.  We do
    not reuse a singleton because per-call config (model, provider)
    can differ between requests."""
    from ...agents.legal_document_agent import LegalDocumentAgent

    config, resolved_provider, model = _build_agent_config(provider, model_override)
    return LegalDocumentAgent(config), resolved_provider, model


# ── RAG document loader ────────────────────────────────────────────


async def _load_text_from_rag(
    rag_document_id: str, user_id: str, org_id: str
) -> str:
    """Resolve a RAG document id to its plain text.  Tenant-scoped via
    the org_id check so a leaked id cannot pull a foreign document."""
    try:
        from ..rag_document_registry import get as registry_get
        from ..storage_service import storage_service
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Document storage unavailable: {e}")

    doc = await registry_get(rag_document_id, user_id=user_id)
    if not doc:
        raise FileNotFoundError("Document not found")

    doc_org = doc.get("organization_id")
    if doc_org and doc_org != org_id and doc.get("user_id") != org_id:
        raise FileNotFoundError("Document not found")

    try:
        raw = await storage_service.download_file(doc["s3_key"])
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(f"Could not fetch document: {e}")

    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = str(raw)
    return clean_text(text, max_len=MAX_DOCUMENT_CHARS)


# ── Public entry point ─────────────────────────────────────────────


async def run(
    *,
    organization_id: str,
    user_id: str,
    mode: str,
    content: Optional[str] = None,
    rag_document_id: Optional[str] = None,
    extra_data: Optional[Dict[str, Any]] = None,
    context: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    request_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute one legal-document analysis end to end.

    Returns the agent result enriched with `analysis_id`, `model_provider`,
    `model_name`, and `processing_time_ms` in its metadata block so the
    UI can immediately open or delete the run.
    """
    if not organization_id:
        raise ValueError("organization_id is required")

    started = time.perf_counter()

    # Resolve content from the right source.
    source_kind = None
    source_ref: Optional[str] = None
    resolved_text = clean_text(content or "", max_len=MAX_DOCUMENT_CHARS)
    if rag_document_id:
        resolved_text = await _load_text_from_rag(rag_document_id, user_id, organization_id)
        source_kind = "rag_document"
        source_ref = rag_document_id
    elif resolved_text:
        source_kind = "inline"

    # For version_comparison we need two payloads, not a single content
    # body.  The endpoint passes them through `extra_data`.
    extra = extra_data or {}
    if mode == "version_comparison":
        if not (extra.get("old_version") and extra.get("new_version")):
            raise ValueError("version_comparison requires old_version and new_version")
    else:
        if not resolved_text:
            raise ValueError("content or rag_document_id is required")

    # Persist a `running` row so the history shows the in-flight work
    # and a slow LLM doesn't leave the UI with a blank canvas.
    preview = make_preview(
        resolved_text or extra.get("old_version") or extra.get("new_version") or ""
    )
    title = clean_label(
        (request_metadata or {}).get("title")
    ) or make_title(mode, preview)

    clean_params = clean_parameters(parameters)

    config_provider, config_model = (
        _resolve_provider(provider),
        model_name or PROVIDER_DEFAULTS.get(_resolve_provider(provider)) or "gemini-2.5-flash",
    )

    record = await legal_repo.create_analysis(
        organization_id=organization_id,
        user_id=user_id,
        mode=mode,
        title=title,
        content_preview=preview,
        source_kind=source_kind,
        source_ref=source_ref,
        model_provider=config_provider,
        model_name=config_model,
        parameters=clean_params,
        metadata=request_metadata or {},
    )
    analysis_id = record["id"]

    # Agent dispatch — single-flight per org within this process.
    async with _lock_for(organization_id):
        agent, _provider, _model = _new_agent(provider, model_name)

        # The agent's mode handlers read the document text from
        # `data["document"]`.  We populate `content` AND `document` so
        # callers using either key (legacy or new) work unchanged.
        agent_data: Dict[str, Any] = {
            "content": resolved_text,
            "document": resolved_text,
            **extra,
        }
        agent_request = {
            "mode": mode,
            "data": agent_data,
            "context": context or {},
            "parameters": clean_params,
        }

        try:
            result = await agent.process_async(agent_request)
        except Exception as e:  # noqa: BLE001
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.error("legal_agent_failed", error=str(e), mode=mode)
            await legal_repo.finalize_analysis(
                organization_id,
                analysis_id,
                status="error",
                error_message=str(e),
                processing_time_ms=elapsed_ms,
            )
            raise

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if not isinstance(result, dict):
        result = {"results": {}, "metadata": {}}

    if "error" in result:
        await legal_repo.finalize_analysis(
            organization_id,
            analysis_id,
            status="error",
            error_message=str(result.get("error"))[:2000],
            processing_time_ms=elapsed_ms,
        )
    else:
        await legal_repo.finalize_analysis(
            organization_id,
            analysis_id,
            status="ready",
            result_payload={
                "results": result.get("results") or result.get("data") or {},
                "metadata": result.get("metadata") or {},
            },
            processing_time_ms=elapsed_ms,
        )

    # Augment the return so the frontend gets identifiers immediately.
    meta = result.setdefault("metadata", {})
    meta["analysis_id"] = analysis_id
    meta["model_provider"] = config_provider
    meta["model_name"] = config_model
    meta["processing_time_ms"] = elapsed_ms
    meta["source_kind"] = source_kind
    meta["source_ref"] = source_ref
    return result
