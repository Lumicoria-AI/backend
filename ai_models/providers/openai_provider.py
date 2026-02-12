"""
OpenAI (ChatGPT) Provider — LLMClient Implementation

Wraps the official OpenAI Python SDK (v1+) behind the vendor-neutral
LLMClient interface. Uses `AsyncOpenAI` for true async without
thread-pool overhead.

Capabilities:
- Chat completions (GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo, o1/o3 series)
- True async streaming via server-sent events
- Embeddings (text-embedding-3-small/large, ada-002)
- Native system-message support (no rewriting needed)
- JSON mode, function calling (via LLMConfig.extra)
- Health check via model listing

Error Mapping:
    openai.AuthenticationError   → LLMAuthenticationError
    openai.RateLimitError        → LLMRateLimitError
    openai.NotFoundError         → LLMModelNotFoundError
    openai.APIConnectionError    → LLMConnectionError
    openai.BadRequestError (content_filter) → LLMContentFilterError
    *                            → LLMProviderError
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
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-4-turbo-preview",
    "gpt-4",
    "gpt-3.5-turbo",
    "o1",
    "o1-mini",
    "o1-preview",
    "o3-mini",
}

_EMBEDDING_MODELS: Set[str] = {
    "text-embedding-3-small",
    "text-embedding-3-large",
    "text-embedding-ada-002",
}

_DEFAULT_CHAT_MODEL = "gpt-4o-mini"
_DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"


# ═══════════════════════════════════════════════════════════════════════════
# Provider Implementation
# ═══════════════════════════════════════════════════════════════════════════

class OpenAIProvider(LLMClient):
    """
    OpenAI LLMClient implementation.

    Uses AsyncOpenAI for fully async I/O. The SDK client is lazily
    initialized on first use and reused for the lifetime of this instance.
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Args:
            api_key: OpenAI API key. Falls back to settings.OPENAI_API_KEY
                     then the OPENAI_API_KEY env var (SDK auto-detects).
            **kwargs: Forwarded to AsyncOpenAI (e.g. organization, base_url).
        """
        self._api_key = api_key
        self._kwargs = kwargs
        self._client = None  # lazy init

    # ── Abstract properties ────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        try:
            from backend.core.config import get_settings
            model = getattr(get_settings(), "OPENAI_MODEL", None)
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
        """Return (or create) the AsyncOpenAI client."""
        if self._client is not None:
            return self._client

        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise LLMProviderError(
                "openai package is not installed. "
                "Install with: pip install openai>=1.0.0",
                provider="openai",
                cause=exc,
            )

        api_key = self._api_key
        if not api_key:
            try:
                from backend.core.config import get_settings
                api_key = getattr(get_settings(), "OPENAI_API_KEY", None)
            except Exception:
                pass

        # AsyncOpenAI will also check the OPENAI_API_KEY env var itself,
        # so passing None here is acceptable — the SDK will resolve it.
        self._client = AsyncOpenAI(
            api_key=api_key or None,
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

        openai_messages = self._to_openai_messages(messages)

        # Build request kwargs
        request_kwargs: Dict = {
            "model": model,
            "messages": openai_messages,
        }

        # Temperature — skip for o1/o3 models (they don't support it)
        if not self._is_reasoning_model(model):
            request_kwargs["temperature"] = config.temperature
            if config.max_tokens:
                request_kwargs["max_tokens"] = config.max_tokens
        else:
            # o1/o3 models use max_completion_tokens instead
            if config.max_tokens:
                request_kwargs["max_completion_tokens"] = config.max_tokens

        # Optional extras (JSON mode, response_format, tools, etc.)
        extra = config.extra or {}
        if "response_format" in extra:
            request_kwargs["response_format"] = extra["response_format"]
        if "tools" in extra:
            request_kwargs["tools"] = extra["tools"]
        if "tool_choice" in extra:
            request_kwargs["tool_choice"] = extra["tool_choice"]

        try:
            response = await client.chat.completions.create(**request_kwargs)
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

        openai_messages = self._to_openai_messages(messages)

        request_kwargs: Dict = {
            "model": model,
            "messages": openai_messages,
            "stream": True,
        }

        if not self._is_reasoning_model(model):
            request_kwargs["temperature"] = config.temperature
            if config.max_tokens:
                request_kwargs["max_tokens"] = config.max_tokens
        else:
            if config.max_tokens:
                request_kwargs["max_completion_tokens"] = config.max_tokens

        # Optional extras
        extra = config.extra or {}
        if "response_format" in extra:
            request_kwargs["response_format"] = extra["response_format"]

        try:
            stream = await client.chat.completions.create(**request_kwargs)

            async for chunk in stream:
                choice = chunk.choices[0] if chunk.choices else None
                if choice is None:
                    continue

                delta = choice.delta
                content = delta.content if delta else None
                finish_reason = choice.finish_reason

                yield LLMStreamChunk(
                    content=content or "",
                    is_final=finish_reason is not None,
                    finish_reason=finish_reason,
                    model=chunk.model or model,
                    provider=self.provider_name,
                )

        except Exception as e:
            raise self._normalize_error(e) from e

    # ── generate_embeddings() ──────────────────────────────────────────

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate embeddings for a batch of texts."""
        model = model or _DEFAULT_EMBEDDING_MODEL
        client = self._get_client()

        if model not in _EMBEDDING_MODELS:
            raise LLMModelNotFoundError(
                f"Unknown OpenAI embedding model: {model}. "
                f"Supported: {sorted(_EMBEDDING_MODELS)}",
                provider="openai",
            )

        try:
            response = await client.embeddings.create(
                model=model,
                input=texts,
            )

            # Sort by index to guarantee order matches input
            sorted_data = sorted(response.data, key=lambda d: d.index)
            return [item.embedding for item in sorted_data]

        except Exception as e:
            raise self._normalize_error(e) from e

    # ── health_check() ─────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Verify the API key is valid by listing models."""
        try:
            client = self._get_client()
            # A lightweight call — list a single model
            await client.models.retrieve("gpt-3.5-turbo")
            return True
        except Exception as e:
            logger.warning(
                "openai_health_check_failed",
                error=str(e),
            )
            return False

    # ── close() ────────────────────────────────────────────────────────

    async def close(self) -> None:
        """Release the underlying httpx client."""
        if self._client is not None:
            try:
                await self._client.close()
            except Exception:
                pass
            self._client = None

    # ═══════════════════════════════════════════════════════════════════
    # Internal Helpers
    # ═══════════════════════════════════════════════════════════════════

    @staticmethod
    def _to_openai_messages(
        messages: List[LLMMessage],
    ) -> List[Dict[str, str]]:
        """
        Convert LLMMessage list to OpenAI's message format.

        OpenAI natively supports "system", "user", and "assistant" roles,
        so this is a straightforward 1-to-1 mapping — no rewriting needed.
        """
        result = []
        for msg in messages:
            role = msg.role
            if isinstance(role, MessageRole):
                role = role.value

            # OpenAI expects role strings: system, user, assistant
            result.append({
                "role": role,
                "content": msg.content,
            })
        return result

    @staticmethod
    def _is_reasoning_model(model: str) -> bool:
        """Check if the model is an o1/o3 reasoning model.

        Reasoning models have restrictions: no system messages as a
        separate role, no temperature, use max_completion_tokens instead
        of max_tokens.
        """
        return model.startswith(("o1", "o3"))

    def _to_llm_response(self, response, model: str) -> LLMResponse:
        """Normalize an OpenAI ChatCompletion to our LLMResponse DTO."""
        choice = response.choices[0] if response.choices else None

        content = ""
        finish_reason = "stop"

        if choice:
            content = choice.message.content or ""
            finish_reason = choice.finish_reason or "stop"

        # Token usage
        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                prompt_tokens=response.usage.prompt_tokens or 0,
                completion_tokens=response.usage.completion_tokens or 0,
                total_tokens=response.usage.total_tokens or 0,
            )

        return LLMResponse(
            content=content,
            model=response.model or model,
            provider=self.provider_name,
            response_id=response.id or "",
            usage=usage,
            citations=[],
            search_queries=[],
            finish_reason=finish_reason,
        )

    def _normalize_error(self, error: Exception) -> LLMProviderError:
        """Convert OpenAI SDK exceptions to the normalized error hierarchy."""
        if isinstance(error, LLMProviderError):
            return error

        # Try to import OpenAI exception types for precise matching
        try:
            import openai as openai_mod

            if isinstance(error, openai_mod.AuthenticationError):
                return LLMAuthenticationError(
                    f"OpenAI authentication failed: {error}",
                    provider="openai",
                    cause=error,
                )

            if isinstance(error, openai_mod.RateLimitError):
                return LLMRateLimitError(
                    f"OpenAI rate limit exceeded: {error}",
                    provider="openai",
                    cause=error,
                )

            if isinstance(error, openai_mod.NotFoundError):
                return LLMModelNotFoundError(
                    f"OpenAI model not found: {error}",
                    provider="openai",
                    cause=error,
                )

            if isinstance(error, openai_mod.APIConnectionError):
                return LLMConnectionError(
                    f"OpenAI connection error: {error}",
                    provider="openai",
                    cause=error,
                )

            if isinstance(error, openai_mod.BadRequestError):
                error_str = str(error).lower()
                if "content_filter" in error_str or "content_policy" in error_str:
                    return LLMContentFilterError(
                        f"OpenAI content filter triggered: {error}",
                        provider="openai",
                        cause=error,
                    )
                return LLMProviderError(
                    f"OpenAI bad request: {error}",
                    provider="openai",
                    cause=error,
                )

            if isinstance(error, openai_mod.APIStatusError):
                # Catch-all for other API errors with status codes
                return LLMProviderError(
                    f"OpenAI API error (status {error.status_code}): {error}",
                    provider="openai",
                    cause=error,
                )

        except ImportError:
            pass

        # Fallback: string-based matching for edge cases
        error_str = str(error).lower()

        if "auth" in error_str or "api_key" in error_str or "invalid" in error_str and "key" in error_str:
            return LLMAuthenticationError(
                f"OpenAI authentication error: {error}",
                provider="openai",
                cause=error,
            )

        if "rate" in error_str or "429" in error_str or "quota" in error_str:
            return LLMRateLimitError(
                f"OpenAI rate limit error: {error}",
                provider="openai",
                cause=error,
            )

        if "timeout" in error_str or "connection" in error_str:
            return LLMConnectionError(
                f"OpenAI connection error: {error}",
                provider="openai",
                cause=error,
            )

        return LLMProviderError(
            f"OpenAI error: {error}",
            provider="openai",
            cause=error,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Self-registration
# ═══════════════════════════════════════════════════════════════════════════

LLMRegistry.register("openai", OpenAIProvider)
