"""
Mistral AI Provider — LLMClient Implementation

Wraps the official Mistral Python SDK (mistral-common / mistralai) behind
the vendor-neutral LLMClient interface. Uses `Mistral` async client for
true async without thread-pool overhead.

Capabilities:
- Chat completions (Mistral Large, Medium, Small, Nemo, Codestral, Pixtral)
- True async streaming via server-sent events
- Embeddings (mistral-embed)
- Native system-message support (no rewriting needed)
- JSON mode, function/tool calling (via LLMConfig.extra)
- Health check via lightweight model listing

Mistral-Specific Design Decisions:
- Mistral uses the OpenAI-compatible message format (system / user / assistant),
  so role mapping is a straightforward 1-to-1 conversion — no rewriting needed.
- Token usage: Mistral returns `prompt_tokens` / `completion_tokens` / `total_tokens`
  in the same structure as OpenAI, so mapping is direct.
- Finish reasons: Mistral uses "stop", "length", "tool_calls", "model_length" —
  "model_length" is normalized to "length" for platform consistency.
- Embeddings: Mistral provides a dedicated embedding model (mistral-embed).

Error Mapping:
    mistralai.models.sdkerror.SDKError                   → LLMProviderError
    HTTP 401 / authentication errors                      → LLMAuthenticationError
    HTTP 429 / rate limit errors                          → LLMRateLimitError
    HTTP 404 / model not found                            → LLMModelNotFoundError
    Connection / timeout errors                           → LLMConnectionError
    Content moderation / policy errors                    → LLMContentFilterError
    *                                                     → LLMProviderError
"""

from __future__ import annotations

import logging
from typing import AsyncIterator, Dict, List, Optional, Set

from backend.ai_models.base import (
    LLMClient,
    LLMConfig,
    LLMConnectionError,
    LLMAuthenticationError,
    LLMContentFilterError,
    LLMMessage,
    LLMModelNotFoundError,
    LLMProviderError,
    LLMRateLimitError,
    LLMResponse,
    LLMStreamChunk,
    MessageRole,
    TokenUsage,
)
from backend.ai_models.registry import LLMRegistry

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_CHAT_MODELS: Set[str] = {
    # Mistral Large (flagship)
    "mistral-large-latest",
    "mistral-large-2411",
    # Mistral Medium
    "mistral-medium-latest",
    # Mistral Small (efficient)
    "mistral-small-latest",
    "mistral-small-2501",
    # Mistral Nemo (open-weight, 12B)
    "open-mistral-nemo",
    "open-mistral-nemo-2407",
    # Codestral (code-specialized)
    "codestral-latest",
    "codestral-2501",
    # Pixtral (multimodal)
    "pixtral-large-latest",
    "pixtral-12b-2409",
    # Mistral Saba (multilingual)
    "mistral-saba-latest",
    # Legacy
    "open-mixtral-8x22b",
    "open-mixtral-8x7b",
    "open-mistral-7b",
}

_EMBEDDING_MODELS: Set[str] = {
    "mistral-embed",
}

_DEFAULT_CHAT_MODEL = "mistral-large-latest"
_DEFAULT_EMBEDDING_MODEL = "mistral-embed"

# Mistral finish_reason → platform finish_reason mapping
_FINISH_REASON_MAP: Dict[str, str] = {
    "stop": "stop",
    "length": "length",
    "model_length": "length",  # Mistral-specific: context window exhausted
    "tool_calls": "tool_calls",
    "error": "error",
}


# ═══════════════════════════════════════════════════════════════════════════
# Provider Implementation
# ═══════════════════════════════════════════════════════════════════════════

class MistralProvider(LLMClient):
    """
    Mistral AI LLMClient implementation.

    Uses the official `mistralai` async client for fully async I/O.
    The SDK client is lazily initialized on first use and reused for
    the lifetime of this instance.

    Key design decisions:
    - Message format is OpenAI-compatible (system/user/assistant) — direct mapping
    - Token usage fields match OpenAI naming — direct mapping to TokenUsage
    - Embeddings supported via dedicated mistral-embed model
    - All SDK exceptions normalized to platform error hierarchy
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Args:
            api_key: Mistral API key. Falls back to settings.MISTRAL_API_KEY
                     then the MISTRAL_API_KEY env var (SDK auto-detects).
            **kwargs: Forwarded to Mistral client (e.g. server_url, timeout).
        """
        self._api_key = api_key
        self._kwargs = kwargs
        self._client = None  # lazy init

    # ── Abstract properties ────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "mistral"

    @property
    def default_model(self) -> str:
        try:
            from backend.core.config import get_settings
            model = getattr(get_settings(), "MISTRAL_MODEL", None)
            if model:
                return model
        except Exception:
            pass
        return _DEFAULT_CHAT_MODEL

    @property
    def supported_models(self) -> List[str]:
        return sorted(_CHAT_MODELS | _EMBEDDING_MODELS)

    # ── Lazy SDK init ──────────────────────────────────────────────────

    def _get_client(self):
        """Return (or create) the async Mistral client."""
        if self._client is not None:
            return self._client

        try:
            from mistralai import Mistral
        except ImportError as exc:
            raise LLMProviderError(
                "mistralai package is not installed. "
                "Install with: pip install mistralai>=1.0.0",
                provider="mistral",
                cause=exc,
            )

        api_key = self._api_key
        if not api_key:
            try:
                from backend.core.config import get_settings
                api_key = getattr(get_settings(), "MISTRAL_API_KEY", None)
            except Exception:
                pass

        if not api_key:
            # Unlike OpenAI, the Mistral SDK doesn't auto-detect from env
            import os
            api_key = os.environ.get("MISTRAL_API_KEY")

        # Resolve timeout from settings
        timeout = self._kwargs.pop("timeout", None)
        if timeout is None:
            try:
                from backend.core.config import get_settings
                timeout = getattr(get_settings(), "MISTRAL_TIMEOUT", 30)
            except Exception:
                timeout = 30

        self._client = Mistral(
            api_key=api_key or "",
            timeout_ms=int(timeout) * 1000,  # SDK uses milliseconds
            **self._kwargs,
        )
        return self._client

    # ── generate() ─────────────────────────────────────────────────────

    async def generate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Send a chat-completion request and return a normalized LLMResponse."""
        config = self._resolve_config(config)
        model = config.model or self.default_model
        client = self._get_client()

        mistral_messages = self._to_mistral_messages(
            self._normalize_messages(messages)
        )

        # Build request kwargs
        request_kwargs: Dict = {
            "model": model,
            "messages": mistral_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        if config.max_tokens:
            request_kwargs["max_tokens"] = config.max_tokens

        # Optional extras (JSON mode, response_format, tools, etc.)
        extra = config.extra or {}
        if "response_format" in extra:
            request_kwargs["response_format"] = extra["response_format"]
        if "tools" in extra:
            request_kwargs["tools"] = extra["tools"]
        if "tool_choice" in extra:
            request_kwargs["tool_choice"] = extra["tool_choice"]
        if "safe_prompt" in extra:
            request_kwargs["safe_prompt"] = extra["safe_prompt"]

        try:
            response = await client.chat.complete_async(**request_kwargs)
            return self._to_llm_response(response, model)

        except Exception as e:
            raise self._normalize_error(e) from e

    # ── stream() ───────────────────────────────────────────────────────

    async def stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream chat-completion tokens as they arrive."""
        config = self._resolve_config(config)
        model = config.model or self.default_model
        client = self._get_client()

        mistral_messages = self._to_mistral_messages(
            self._normalize_messages(messages)
        )

        request_kwargs: Dict = {
            "model": model,
            "messages": mistral_messages,
            "temperature": config.temperature,
            "top_p": config.top_p,
        }

        if config.max_tokens:
            request_kwargs["max_tokens"] = config.max_tokens

        # Optional extras
        extra = config.extra or {}
        if "response_format" in extra:
            request_kwargs["response_format"] = extra["response_format"]
        if "tools" in extra:
            request_kwargs["tools"] = extra["tools"]
        if "safe_prompt" in extra:
            request_kwargs["safe_prompt"] = extra["safe_prompt"]

        try:
            stream = await client.chat.stream_async(**request_kwargs)

            async for event in stream:
                chunk = event.data
                if not chunk or not chunk.choices:
                    continue

                choice = chunk.choices[0]
                delta = choice.delta
                content = delta.content if delta else None
                finish_reason_raw = choice.finish_reason

                # Normalize finish reason
                finish_reason = None
                if finish_reason_raw is not None:
                    reason_str = str(finish_reason_raw)
                    # Handle enum values (e.g. FinishReason.STOP → "stop")
                    if hasattr(finish_reason_raw, "value"):
                        reason_str = finish_reason_raw.value
                    finish_reason = _FINISH_REASON_MAP.get(
                        reason_str, reason_str
                    )

                # Extract usage from final chunk if available
                usage = None
                if hasattr(chunk, "usage") and chunk.usage:
                    usage = TokenUsage(
                        prompt_tokens=getattr(chunk.usage, "prompt_tokens", 0) or 0,
                        completion_tokens=getattr(chunk.usage, "completion_tokens", 0) or 0,
                        total_tokens=getattr(chunk.usage, "total_tokens", 0) or 0,
                    )

                yield LLMStreamChunk(
                    content=content or "",
                    model=getattr(chunk, "model", None) or model,
                    provider=self.provider_name,
                    finish_reason=finish_reason,
                    usage=usage,
                )

        except Exception as e:
            raise self._normalize_error(e) from e

    # ── generate_embeddings() ──────────────────────────────────────────

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate embeddings via Mistral's embedding API."""
        model = model or _DEFAULT_EMBEDDING_MODEL
        client = self._get_client()

        if model not in _EMBEDDING_MODELS:
            raise LLMModelNotFoundError(
                f"Unknown Mistral embedding model: {model}. "
                f"Supported: {sorted(_EMBEDDING_MODELS)}",
                provider="mistral",
            )

        try:
            response = await client.embeddings.create_async(
                model=model,
                inputs=texts,
            )

            # Response data is a list of EmbeddingObject with .embedding
            return [item.embedding for item in response.data]

        except Exception as e:
            raise self._normalize_error(e) from e

    # ── health_check() ─────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Verify the API key is valid by listing models."""
        try:
            client = self._get_client()
            response = await client.models.list_async()
            return bool(response.data)
        except Exception as e:
            logger.warning(
                "mistral_health_check_failed",
                error=str(e),
            )
            return False

    # ── close() ────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Release the underlying HTTP client."""
        if self._client is not None:
            try:
                # The Mistral SDK uses httpx internally
                if hasattr(self._client, "_client") and self._client._client:
                    await self._client._client.__aexit__(None, None, None)
            except Exception:
                pass
            self._client = None

    # ═══════════════════════════════════════════════════════════════════
    # Internal Helpers
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _to_mistral_messages(
        messages: List[LLMMessage],
    ) -> List[Dict[str, str]]:
        """
        Convert LLMMessage list to Mistral's message format.

        Mistral uses the OpenAI-compatible format with "system", "user",
        and "assistant" roles — straightforward 1-to-1 mapping.
        """
        result = []
        for msg in messages:
            role = msg.role
            if isinstance(role, MessageRole):
                role = role.value

            result.append({
                "role": role,
                "content": msg.content,
            })
        return result

    def _to_llm_response(self, response, model: str) -> LLMResponse:
        """Normalize a Mistral ChatCompletionResponse to our LLMResponse DTO."""
        choice = response.choices[0] if response.choices else None

        content = ""
        finish_reason = "stop"

        if choice:
            content = choice.message.content or ""
            # Handle finish_reason which may be an enum or string
            raw_reason = choice.finish_reason
            if raw_reason is not None:
                reason_str = str(raw_reason)
                if hasattr(raw_reason, "value"):
                    reason_str = raw_reason.value
                finish_reason = _FINISH_REASON_MAP.get(reason_str, reason_str)

        # Token usage — Mistral uses the same field names as OpenAI
        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=getattr(response.usage, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(response.usage, "completion_tokens", 0) or 0,
                total_tokens=getattr(response.usage, "total_tokens", 0) or 0,
            )

        return LLMResponse(
            content=content,
            model=getattr(response, "model", None) or model,
            provider=self.provider_name,
            response_id=getattr(response, "id", "") or "",
            usage=usage,
            citations=[],
            search_queries=[],
            finish_reason=finish_reason,
        )

    def _normalize_error(self, error: Exception) -> LLMProviderError:
        """Convert Mistral SDK exceptions to the normalized error hierarchy."""
        if isinstance(error, LLMProviderError):
            return error

        # Try to import Mistral exception types for precise matching
        try:
            from mistralai.models import sdkerror

            if isinstance(error, sdkerror.SDKError):
                status_code = getattr(error, "status_code", None)
                error_str = str(error).lower()

                if status_code == 401 or "unauthorized" in error_str or "authentication" in error_str:
                    return LLMAuthenticationError(
                        f"Mistral authentication failed: {error}",
                        provider="mistral",
                        cause=error,
                    )

                if status_code == 429 or "rate" in error_str or "quota" in error_str:
                    # Try to extract retry-after from the error
                    retry_after = None
                    if hasattr(error, "headers"):
                        retry_str = getattr(error, "headers", {}).get("retry-after")
                        if retry_str:
                            try:
                                retry_after = float(retry_str)
                            except (ValueError, TypeError):
                                pass
                    return LLMRateLimitError(
                        f"Mistral rate limit exceeded: {error}",
                        provider="mistral",
                        retry_after=retry_after,
                        cause=error,
                    )

                if status_code == 404 or "not found" in error_str:
                    return LLMModelNotFoundError(
                        f"Mistral model not found: {error}",
                        provider="mistral",
                        cause=error,
                    )

                if status_code == 400:
                    if any(kw in error_str for kw in (
                        "content_filter", "moderation", "unsafe", "blocked", "policy",
                    )):
                        return LLMContentFilterError(
                            f"Mistral content filter triggered: {error}",
                            provider="mistral",
                            cause=error,
                        )
                    return LLMProviderError(
                        f"Mistral bad request: {error}",
                        provider="mistral",
                        cause=error,
                    )

                if status_code and status_code >= 500:
                    return LLMConnectionError(
                        f"Mistral server error (status {status_code}): {error}",
                        provider="mistral",
                        cause=error,
                    )

                return LLMProviderError(
                    f"Mistral API error (status {status_code}): {error}",
                    provider="mistral",
                    cause=error,
                )

        except ImportError:
            pass

        # Check for httpx / connection exceptions
        try:
            import httpx
            if isinstance(error, httpx.TimeoutException):
                return LLMConnectionError(
                    f"Mistral timeout: {error}",
                    provider="mistral",
                    cause=error,
                )
            if isinstance(error, httpx.ConnectError):
                return LLMConnectionError(
                    f"Mistral connection error: {error}",
                    provider="mistral",
                    cause=error,
                )
        except ImportError:
            pass

        # Fallback: string-based matching for edge cases
        error_str = str(error).lower()

        if "auth" in error_str or "api_key" in error_str or (
            "invalid" in error_str and "key" in error_str
        ) or "unauthorized" in error_str:
            return LLMAuthenticationError(
                f"Mistral authentication error: {error}",
                provider="mistral",
                cause=error,
            )

        if "rate" in error_str or "429" in error_str or "quota" in error_str:
            return LLMRateLimitError(
                f"Mistral rate limit error: {error}",
                provider="mistral",
                cause=error,
            )

        if "timeout" in error_str or "connection" in error_str or "connect" in error_str:
            return LLMConnectionError(
                f"Mistral connection error: {error}",
                provider="mistral",
                cause=error,
            )

        if "not found" in error_str or "404" in error_str:
            return LLMModelNotFoundError(
                f"Mistral model not found: {error}",
                provider="mistral",
                cause=error,
            )

        return LLMProviderError(
            f"Mistral error: {error}",
            provider="mistral",
            cause=error,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Self-registration
# ═══════════════════════════════════════════════════════════════════════════

LLMRegistry.register("mistral", MistralProvider)
