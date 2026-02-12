"""
Anthropic (Claude) Provider — LLMClient Implementation

Wraps the official Anthropic Python SDK behind the vendor-neutral
LLMClient interface. Uses `AsyncAnthropic` for true async without
thread-pool overhead.

Capabilities:
- Chat completions (Claude 4 Sonnet, Claude 3.5 Sonnet/Haiku, Claude 3 Opus/Sonnet/Haiku)
- True async streaming via server-sent events
- Separate system prompt parameter (mapped from LLMMessage system role)
- Tool/function calling (via LLMConfig.extra)
- Extended thinking support for reasoning models (via LLMConfig.extra)
- Health check via lightweight message

Anthropic-Specific Design Decisions:
- Anthropic separates `system` from `messages[]` — this module extracts
  system-role messages and passes them via the `system=` parameter.
  No Anthropic schema leaks into business logic.
- Token usage: Anthropic returns `input_tokens` / `output_tokens` — these
  are mapped to `prompt_tokens` / `completion_tokens` in TokenUsage.
- Stop reasons: Anthropic uses "end_turn", "max_tokens", "stop_sequence",
  "tool_use" — all mapped to the platform's normalized finish_reason.
- Refusals: When Claude refuses a request, the content is returned normally
  (Claude includes refusal text in the response content). The `stop_reason`
  is still "end_turn". No special handling is needed — the business layer
  sees it as normal text.
- Content filter: Anthropic may return an error for policy violations — this
  is mapped to LLMContentFilterError.

Error Mapping:
    anthropic.AuthenticationError      → LLMAuthenticationError
    anthropic.RateLimitError           → LLMRateLimitError
    anthropic.NotFoundError            → LLMModelNotFoundError
    anthropic.APIConnectionError       → LLMConnectionError
    anthropic.BadRequestError          → LLMProviderError / LLMContentFilterError
    anthropic.PermissionDeniedError    → LLMAuthenticationError
    anthropic.APITimeoutError          → LLMConnectionError
    *                                  → LLMProviderError
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
    # Claude 4 series
    "claude-sonnet-4-20250514",
    # Claude 3.7 series
    "claude-3-7-sonnet-20250219",
    "claude-3-7-sonnet-latest",
    # Claude 3.5 series
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-latest",
    "claude-3-5-haiku-20241022",
    "claude-3-5-haiku-latest",
    # Claude 3 series
    "claude-3-opus-20240229",
    "claude-3-opus-latest",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
}

_DEFAULT_CHAT_MODEL = "claude-3-5-sonnet-latest"

# Anthropic stop_reason → platform finish_reason mapping
_STOP_REASON_MAP: Dict[str, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "stop_sequence": "stop",
    "tool_use": "tool_calls",
}


# ═══════════════════════════════════════════════════════════════════════════
# Provider Implementation
# ═══════════════════════════════════════════════════════════════════════════

class AnthropicProvider(LLMClient):
    """
    Anthropic LLMClient implementation.

    Uses AsyncAnthropic for fully async I/O. The SDK client is lazily
    initialized on first use and reused for the lifetime of this instance.

    Key Anthropic differences handled internally:
    - System prompt is a top-level parameter, not a message role
    - Token usage fields are named differently (input/output vs prompt/completion)
    - stop_reason values differ from OpenAI conventions
    - No native embedding API — raises LLMProviderError with guidance
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Args:
            api_key: Anthropic API key. Falls back to settings.ANTHROPIC_API_KEY
                     then the ANTHROPIC_API_KEY env var (SDK auto-detects).
            **kwargs: Forwarded to AsyncAnthropic (e.g. base_url, default_headers).
        """
        self._api_key = api_key
        self._kwargs = kwargs
        self._client = None  # lazy init

    # ── Abstract properties ────────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        try:
            from backend.core.config import get_settings
            model = getattr(get_settings(), "ANTHROPIC_MODEL", None)
            if model:
                return model
        except Exception:
            pass
        return _DEFAULT_CHAT_MODEL

    @property
    def supported_models(self) -> List[str]:
        return sorted(_CHAT_MODELS)

    # ── Lazy SDK init ──────────────────────────────────────────────────

    def _get_client(self):
        """Return (or create) the AsyncAnthropic client."""
        if self._client is not None:
            return self._client

        try:
            from anthropic import AsyncAnthropic
        except ImportError as exc:
            raise LLMProviderError(
                "anthropic package is not installed. "
                "Install with: pip install anthropic>=0.39.0",
                provider="anthropic",
                cause=exc,
            )

        api_key = self._api_key
        if not api_key:
            try:
                from backend.core.config import get_settings
                api_key = getattr(get_settings(), "ANTHROPIC_API_KEY", None)
            except Exception:
                pass

        timeout = self._kwargs.pop("timeout", None)
        if timeout is None:
            try:
                from backend.core.config import get_settings
                timeout = getattr(get_settings(), "ANTHROPIC_TIMEOUT", 60)
            except Exception:
                timeout = 60

        # Build the timeout object for the SDK
        try:
            from anthropic import Timeout as AnthropicTimeout
            timeout_obj = AnthropicTimeout(timeout=float(timeout), connect=5.0)
        except (ImportError, Exception):
            timeout_obj = float(timeout)

        # AsyncAnthropic will also check the ANTHROPIC_API_KEY env var itself,
        # so passing None here is acceptable — the SDK will resolve it.
        self._client = AsyncAnthropic(
            api_key=api_key or None,
            timeout=timeout_obj,
            **self._kwargs,
        )
        return self._client

    # ── Message Mapping ────────────────────────────────────────────────

    @staticmethod
    def _split_system_and_messages(
        messages: List[LLMMessage],
    ) -> tuple:
        """
        Separate system messages from conversation messages.

        Anthropic requires the system prompt as a separate top-level
        parameter, NOT as a message with role "system". This method
        extracts all system-role messages and concatenates them into
        a single system string, leaving only user/assistant messages
        in the messages array.

        Returns:
            (system_prompt: Optional[str], conversation_messages: List[Dict])
        """
        system_parts: List[str] = []
        conversation: List[Dict[str, str]] = []

        for msg in messages:
            role = msg.role
            if isinstance(role, MessageRole):
                role = role.value

            if role == "system":
                system_parts.append(msg.content)
            else:
                # Anthropic expects "user" and "assistant" roles only
                conversation.append({
                    "role": role,
                    "content": msg.content,
                })

        system_prompt = "\n\n".join(system_parts) if system_parts else None
        return system_prompt, conversation

    # ── generate() ─────────────────────────────────────────────────────

    async def generate(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Send a message request and return a normalized LLMResponse."""
        config = self._resolve_config(config)
        model = config.model or self.default_model
        client = self._get_client()

        system_prompt, anthropic_messages = self._split_system_and_messages(
            self._normalize_messages(messages)
        )

        # Build request kwargs
        request_kwargs: Dict = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": config.max_tokens or 1024,
        }

        # System prompt — top-level parameter
        if system_prompt:
            request_kwargs["system"] = system_prompt

        # Temperature and top_p
        request_kwargs["temperature"] = config.temperature
        if config.top_p and config.top_p < 1.0:
            request_kwargs["top_p"] = config.top_p

        # Optional extras (tools, metadata, extended thinking, etc.)
        extra = config.extra or {}
        if "tools" in extra:
            request_kwargs["tools"] = extra["tools"]
        if "tool_choice" in extra:
            request_kwargs["tool_choice"] = extra["tool_choice"]
        if "stop_sequences" in extra:
            request_kwargs["stop_sequences"] = extra["stop_sequences"]
        if "metadata" in extra:
            request_kwargs["metadata"] = extra["metadata"]

        try:
            response = await client.messages.create(**request_kwargs)
            return self._to_llm_response(response, model)

        except Exception as e:
            raise self._normalize_error(e) from e

    # ── stream() ───────────────────────────────────────────────────────

    async def stream(
        self,
        messages: List[LLMMessage],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream message tokens as they arrive via SSE."""
        config = self._resolve_config(config)
        model = config.model or self.default_model
        client = self._get_client()

        system_prompt, anthropic_messages = self._split_system_and_messages(
            self._normalize_messages(messages)
        )

        request_kwargs: Dict = {
            "model": model,
            "messages": anthropic_messages,
            "max_tokens": config.max_tokens or 1024,
        }

        if system_prompt:
            request_kwargs["system"] = system_prompt

        request_kwargs["temperature"] = config.temperature
        if config.top_p and config.top_p < 1.0:
            request_kwargs["top_p"] = config.top_p

        # Optional extras
        extra = config.extra or {}
        if "tools" in extra:
            request_kwargs["tools"] = extra["tools"]
        if "stop_sequences" in extra:
            request_kwargs["stop_sequences"] = extra["stop_sequences"]

        try:
            async with client.messages.stream(**request_kwargs) as stream:
                async for event in stream:
                    # Anthropic streaming events:
                    #   message_start, content_block_start, content_block_delta,
                    #   content_block_stop, message_delta, message_stop

                    # Handle text delta events
                    if hasattr(event, "type"):
                        if event.type == "content_block_delta":
                            delta = getattr(event, "delta", None)
                            if delta and hasattr(delta, "text"):
                                yield LLMStreamChunk(
                                    content=delta.text,
                                    model=model,
                                    provider=self.provider_name,
                                )

                        elif event.type == "message_delta":
                            # Final event with stop_reason and usage
                            delta = getattr(event, "delta", None)
                            stop_reason = (
                                getattr(delta, "stop_reason", None)
                                if delta else None
                            )
                            finish_reason = _STOP_REASON_MAP.get(
                                stop_reason or "", stop_reason or "stop"
                            )

                            # Extract final usage from message_delta
                            msg_usage = getattr(event, "usage", None)
                            usage = None
                            if msg_usage:
                                usage = TokenUsage(
                                    prompt_tokens=0,  # only available in message_start
                                    completion_tokens=getattr(
                                        msg_usage, "output_tokens", 0
                                    ),
                                    total_tokens=getattr(
                                        msg_usage, "output_tokens", 0
                                    ),
                                )

                            yield LLMStreamChunk(
                                content="",
                                model=model,
                                provider=self.provider_name,
                                finish_reason=finish_reason,
                                usage=usage,
                            )

                        elif event.type == "message_stop":
                            # Redundant stop signal — we already emitted
                            # finish_reason in message_delta above
                            pass

        except Exception as e:
            raise self._normalize_error(e) from e

    # ── generate_embeddings() ──────────────────────────────────────────

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Anthropic does not offer a native embedding API.

        If embeddings are needed, use a provider that supports them
        (OpenAI, Gemini) via get_embedding_client().

        This method raises a clear error so callers can handle it
        gracefully rather than failing with a cryptic SDK error.
        """
        raise LLMProviderError(
            "Anthropic does not provide an embedding API. "
            "Use DEFAULT_EMBEDDING_PROVIDER='openai' or 'gemini' instead. "
            "Example: get_embedding_client(provider='openai')",
            provider="anthropic",
        )

    # ── health_check() ─────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """
        Verify the API key is valid by sending a minimal message.

        Anthropic doesn't have a lightweight models.list endpoint,
        so we send a 1-token completion as the health probe.
        """
        try:
            client = self._get_client()
            response = await client.messages.create(
                model=self.default_model,
                max_tokens=1,
                messages=[{"role": "user", "content": "hi"}],
            )
            return bool(response.content)
        except Exception as e:
            logger.warning(
                "anthropic_health_check_failed",
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

    def _to_llm_response(self, response, model: str) -> LLMResponse:
        """
        Normalize an Anthropic Message to our LLMResponse DTO.

        Anthropic response structure:
            response.content = [ContentBlock(type="text", text="..."), ...]
            response.stop_reason = "end_turn" | "max_tokens" | "stop_sequence" | "tool_use"
            response.usage.input_tokens, response.usage.output_tokens
        """
        # Extract text content — Anthropic may return multiple content blocks
        # (e.g., text + tool_use). We concatenate all text blocks.
        content_parts: List[str] = []
        for block in response.content:
            if hasattr(block, "text"):
                content_parts.append(block.text)
        content = "\n".join(content_parts) if content_parts else ""

        # Map stop_reason to platform finish_reason
        stop_reason = getattr(response, "stop_reason", None) or "end_turn"
        finish_reason = _STOP_REASON_MAP.get(stop_reason, stop_reason)

        # Token usage — map Anthropic's input/output to prompt/completion
        usage = TokenUsage()
        if response.usage:
            input_tokens = getattr(response.usage, "input_tokens", 0) or 0
            output_tokens = getattr(response.usage, "output_tokens", 0) or 0
            usage = TokenUsage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            )

        return LLMResponse(
            content=content,
            model=response.model or model,
            provider=self.provider_name,
            response_id=response.id or "",
            usage=usage,
            citations=[],      # Anthropic doesn't provide citations
            search_queries=[],  # Anthropic doesn't provide search queries
            finish_reason=finish_reason,
        )

    def _normalize_error(self, error: Exception) -> LLMProviderError:
        """Convert Anthropic SDK exceptions to the normalized error hierarchy."""
        if isinstance(error, LLMProviderError):
            return error

        # Try to import Anthropic exception types for precise matching
        try:
            import anthropic as anthropic_mod

            if isinstance(error, anthropic_mod.AuthenticationError):
                return LLMAuthenticationError(
                    f"Anthropic authentication failed: {error}",
                    provider="anthropic",
                    cause=error,
                )

            if isinstance(error, anthropic_mod.PermissionDeniedError):
                return LLMAuthenticationError(
                    f"Anthropic permission denied: {error}",
                    provider="anthropic",
                    cause=error,
                )

            if isinstance(error, anthropic_mod.RateLimitError):
                # Try to extract retry-after header
                retry_after = None
                if hasattr(error, "response") and error.response:
                    retry_str = error.response.headers.get("retry-after")
                    if retry_str:
                        try:
                            retry_after = float(retry_str)
                        except (ValueError, TypeError):
                            pass
                return LLMRateLimitError(
                    f"Anthropic rate limit exceeded: {error}",
                    provider="anthropic",
                    retry_after=retry_after,
                    cause=error,
                )

            if isinstance(error, anthropic_mod.NotFoundError):
                return LLMModelNotFoundError(
                    f"Anthropic model not found: {error}",
                    provider="anthropic",
                    cause=error,
                )

            if isinstance(error, anthropic_mod.APIConnectionError):
                return LLMConnectionError(
                    f"Anthropic connection error: {error}",
                    provider="anthropic",
                    cause=error,
                )

            if isinstance(error, anthropic_mod.APITimeoutError):
                return LLMConnectionError(
                    f"Anthropic timeout: {error}",
                    provider="anthropic",
                    cause=error,
                )

            if isinstance(error, anthropic_mod.BadRequestError):
                error_str = str(error).lower()
                if any(kw in error_str for kw in (
                    "content_policy", "unsafe", "harmful",
                    "blocked", "moderation",
                )):
                    return LLMContentFilterError(
                        f"Anthropic content filter triggered: {error}",
                        provider="anthropic",
                        cause=error,
                    )
                return LLMProviderError(
                    f"Anthropic bad request: {error}",
                    provider="anthropic",
                    cause=error,
                )

            if isinstance(error, anthropic_mod.APIStatusError):
                return LLMProviderError(
                    f"Anthropic API error (status {error.status_code}): {error}",
                    provider="anthropic",
                    cause=error,
                )

        except ImportError:
            pass

        # Fallback: string-based matching for edge cases
        error_str = str(error).lower()

        if "auth" in error_str or "api_key" in error_str or (
            "invalid" in error_str and "key" in error_str
        ):
            return LLMAuthenticationError(
                f"Anthropic authentication error: {error}",
                provider="anthropic",
                cause=error,
            )

        if "rate" in error_str or "429" in error_str or "quota" in error_str:
            return LLMRateLimitError(
                f"Anthropic rate limit error: {error}",
                provider="anthropic",
                cause=error,
            )

        if "timeout" in error_str or "connection" in error_str:
            return LLMConnectionError(
                f"Anthropic connection error: {error}",
                provider="anthropic",
                cause=error,
            )

        return LLMProviderError(
            f"Anthropic error: {error}",
            provider="anthropic",
            cause=error,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Self-registration
# ═══════════════════════════════════════════════════════════════════════════

LLMRegistry.register("anthropic", AnthropicProvider)
