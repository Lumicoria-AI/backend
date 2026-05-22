"""Service-layer orchestrator for the Ethics & Bias Agent.

Responsibilities:
  1. Build the right agent config for the chosen model provider
     (Gemini, Claude/Anthropic, Perplexity, OpenAI, etc.).
  2. Run the agent.
  3. Persist the analysis to Mongo (history + audit).
  4. Compute a friendly ethics score from the raw result so the
     frontend doesn't need to.
  5. Return the enriched result, augmented with the analysis id so
     the frontend can open / delete it later.

The agent stays single-purpose; anything tenant-shaped lives here so
the agent cannot leak across organizations.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import structlog

from . import repository as ethics_repo
from .sanitize import (
    MAX_CONTENT_CHARS,
    clean_label,
    clean_parameters,
    clean_string_list,
    clean_text,
    coerce_jsonable,
    make_preview,
    make_title,
)

logger = structlog.get_logger(__name__)


# Per-organization locks keep within-process correctness if two
# requests arrive at the same time on the same worker.
_org_locks: Dict[str, asyncio.Lock] = {}


def _lock_for(org_id: str) -> asyncio.Lock:
    if org_id not in _org_locks:
        _org_locks[org_id] = asyncio.Lock()
    return _org_locks[org_id]


# ── Provider / model resolution ────────────────────────────────────


PROVIDER_DEFAULTS = {
    "anthropic": "claude-haiku-4-5-20251001",
    "claude": "claude-haiku-4-5-20251001",
    "gemini": "gemini-2.5-flash",
    "openai": "gpt-4o-mini",
    "perplexity": "sonar",
    "mistral": "mistral-small-latest",
    "deepseek": "deepseek-chat",
}


def _resolve_provider(requested: Optional[str]) -> str:
    """Normalize a user-supplied provider string; fall back to the
    platform default for unknown values."""
    from ...core.config import settings

    if requested:
        provider = requested.strip().lower()
        if provider in PROVIDER_DEFAULTS:
            return "anthropic" if provider == "claude" else provider
    return (settings.DEFAULT_LLM_PROVIDER or "gemini").lower()


def _build_agent_config(
    provider: Optional[str], model_override: Optional[str]
) -> Dict[str, Any]:
    """Build the BaseAgent config dict for the analysis call."""
    resolved_provider = _resolve_provider(provider)
    model = (
        model_override
        or PROVIDER_DEFAULTS.get(resolved_provider)
        or "gemini-2.5-flash"
    )
    return {
        "provider": resolved_provider,
        "model": model,
        # BaseAgent reads `agent_model_config`, not `model_config`.
        "agent_model_config": {
            "model": model,
            "temperature": 0.3,
            # 16k token ceiling: Gemini 2.5 spends part of its budget
            # on internal reasoning before any text reaches the wire,
            # and Claude's structured ethics output is verbose.
            "max_tokens": 16384,
        },
    }, resolved_provider, model


def _new_agent(provider: Optional[str], model_override: Optional[str]):
    """Instantiate a fresh agent — never reuse a singleton across
    requests because we don't want one tenant's parameters to bleed
    into another's."""
    from ...agents.ethics_bias_agent import EthicsBiasAgent

    config, resolved_provider, model = _build_agent_config(provider, model_override)
    return EthicsBiasAgent(config), resolved_provider, model


# ── Score derivation ───────────────────────────────────────────────


_SEVERITY_PENALTY = {
    "critical": 25,
    "high": 12,
    "medium": 6,
    "low": 2,
    "info": 0,
}


def _derive_score_and_count(result: Dict[str, Any]) -> tuple[int, int]:
    """Turn a parsed result into a single 0..100 score plus an issue
    count.  Lower-severity issues take smaller bites; floors at 0."""
    if not isinstance(result, dict):
        return 100, 0
    issues: List[Dict[str, Any]] = []
    for key in ("ethics_issues", "bias_issues", "violations", "issues"):
        val = result.get(key)
        if isinstance(val, list):
            issues.extend([i for i in val if isinstance(i, dict)])

    if not issues:
        return 100, 0

    score = 100
    for issue in issues:
        sev = str(issue.get("severity") or "medium").lower()
        score -= _SEVERITY_PENALTY.get(sev, 6)
    return max(0, min(100, int(score))), len(issues)


# ── Public entry point ─────────────────────────────────────────────


async def run(
    *,
    organization_id: str,
    user_id: str,
    action: str,
    content: Optional[str] = None,
    content_type: Optional[str] = None,
    issues: Optional[List[Dict[str, Any]]] = None,
    topic: Optional[str] = None,
    guidelines_focus: Optional[List[str]] = None,
    context: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    provider: Optional[str] = None,
    model_name: Optional[str] = None,
    request_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute one ethics & bias action end to end.

    Returns the agent's result enriched with `analysis_id`, the model
    used, ethics_score, issue_count, and processing_time_ms so the UI
    can immediately open / delete the run.
    """
    if not organization_id:
        raise ValueError("organization_id is required")

    started = time.perf_counter()

    cleaned_content = clean_text(content or "", max_len=MAX_CONTENT_CHARS) if content else ""
    cleaned_topic = clean_label(topic, max_len=500) or ""
    cleaned_focus = clean_string_list(guidelines_focus)
    cleaned_params = clean_parameters(parameters)
    cleaned_metadata = clean_parameters(request_metadata)

    if action in ("analyze", "check_guidelines") and not cleaned_content:
        raise ValueError("content is required for this action")
    if action == "generate_suggestions" and not issues:
        raise ValueError("issues are required for generate_suggestions")
    if action == "get_citations" and not cleaned_topic:
        raise ValueError("topic is required for get_citations")

    # Persist a `running` row up front so a slow LLM doesn't leave the
    # history UI blank during the wait.
    preview_source = cleaned_content or cleaned_topic or (
        ", ".join(cleaned_focus) if cleaned_focus else ""
    )
    title = (
        clean_label((cleaned_metadata or {}).get("title"))
        or make_title(action, preview_source)
    )
    preview = make_preview(preview_source) if preview_source else None

    config_provider = _resolve_provider(provider)
    config_model = model_name or PROVIDER_DEFAULTS.get(config_provider) or "gemini-2.5-flash"

    record = await ethics_repo.create_analysis(
        organization_id=organization_id,
        user_id=user_id,
        action=action,
        title=title,
        content_preview=preview,
        content_type=content_type,
        model_provider=config_provider,
        model_name=config_model,
        parameters=cleaned_params,
        metadata=cleaned_metadata,
    )
    analysis_id = record["id"]

    # Agent dispatch — single-flight per org within this process.
    async with _lock_for(organization_id):
        agent, _provider, _model = _new_agent(provider, model_name)

        # Each agent action expects a slightly different `data` shape.
        # We assemble that here so the router stays narrow.
        if action == "analyze":
            agent_data: Dict[str, Any] = {
                "content": cleaned_content,
                "content_type": content_type or "text",
                "metadata": cleaned_metadata or {},
            }
        elif action == "check_guidelines":
            agent_data = {
                "content": cleaned_content,
                "guidelines_focus": cleaned_focus,
            }
        elif action == "generate_suggestions":
            agent_data = {
                "issues": issues or [],
                "content": cleaned_content,
            }
        elif action == "get_citations":
            agent_data = {
                "topic": cleaned_topic,
            }
        else:
            agent_data = {"content": cleaned_content}

        agent_request = {
            "action": action,
            "data": agent_data,
            "context": context or {},
            "parameters": cleaned_params,
        }

        try:
            result = await agent.process_async(agent_request)
        except Exception as e:  # noqa: BLE001
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            logger.error("ethics_bias_agent_failed", error=str(e), action=action)
            await ethics_repo.finalize_analysis(
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

    # The agent returns `{"results": {...}, "metadata": {...}}`.  We
    # persist that shape and derive a friendly score on the way out.
    inner = result.get("results") or {}
    ethics_score, issue_count = _derive_score_and_count(inner)

    if "error" in result:
        await ethics_repo.finalize_analysis(
            organization_id,
            analysis_id,
            status="error",
            error_message=str(result.get("error"))[:2000],
            processing_time_ms=elapsed_ms,
        )
    else:
        await ethics_repo.finalize_analysis(
            organization_id,
            analysis_id,
            status="ready",
            result_payload=coerce_jsonable({
                "results": inner,
                "metadata": result.get("metadata") or {},
            }),
            ethics_score=ethics_score,
            issue_count=issue_count,
            processing_time_ms=elapsed_ms,
        )

    # Augment the return so the frontend gets identifiers immediately.
    meta = result.setdefault("metadata", {})
    meta["analysis_id"] = analysis_id
    meta["model_provider"] = config_provider
    meta["model_name"] = config_model
    meta["processing_time_ms"] = elapsed_ms
    meta["ethics_score"] = ethics_score
    meta["issue_count"] = issue_count
    return result
