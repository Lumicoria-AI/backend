"""
Google Gemini Provider — Production-Grade LLMClient Implementation

Migrated to the current stable `google-genai` SDK (google.genai) from the
deprecated `google-generativeai` package.

Key improvements over the legacy provider:
  - Native async via `client.aio` — no ThreadPoolExecutor blocking the event loop
  - Exponential back-off with jitter on rate-limit / transient errors (429 / 503)
  - Connection reuse via a shared async httpx client managed by the SDK
  - Full model catalogue for the v1 API (Gemini 2.0, 2.5, 1.5 series)
  - Correct multipart / vision content handling
  - Proper streaming with token-usage accumulation on the final chunk
  - Self-registration with LLMRegistry on import

Requires:
    pip install google-genai

Environment:
    GEMINI_API_KEY   — Google AI Studio API key (required)
    GEMINI_MODEL     — model override, default gemini-2.0-flash
    GEMINI_TIMEOUT   — HTTP timeout in seconds, default 60
"""

from __future__ import annotations

import asyncio
import base64
import random
import re
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import structlog

from backend.ai_models.base import (
    LLMAuthenticationError,
    LLMClient,
    LLMConfig,
    LLMConnectionError,
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

logger = structlog.get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Model catalogue — tracks what is available on the v1 API
# ═══════════════════════════════════════════════════════════════════════════

_CHAT_MODELS: set[str] = {
    # Gemini 3 series (latest preview)
    "gemini-3-pro-preview",
    "gemini-3-flash-preview",
    "gemini-3.1-pro-preview",
    # Gemini 2.5 series
    "gemini-2.5-pro",
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash-preview-09-2025",
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-flash-preview-05-20",
    "gemini-2.5-flash-preview",
    "gemini-2.5-pro-preview",
    # Gemini 2.0 series
    "gemini-2.0-flash",
    "gemini-2.0-flash-001",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-lite-001",
    "gemini-2.0-flash-exp",
    "gemini-2.0-pro-exp",
    "gemini-2.0-pro-exp-03-25",
    "gemini-2.0-flash-thinking-exp",
    # Gemini 1.5 series
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-8b",
    "gemini-1.5-flash-8b-latest",
    # Convenience aliases
    "gemini-flash-latest",
    "gemini-pro-latest",
}

_EMBEDDING_MODELS: set[str] = {
    "text-embedding-004",
    "text-multilingual-embedding-002",
}

_ALL_MODELS = _CHAT_MODELS | _EMBEDDING_MODELS

_DEFAULT_CHAT_MODEL = "gemini-2.5-flash"
_DEFAULT_EMBEDDING_MODEL = "text-embedding-004"

# Back-off: up to 4 retries, doubling each time + ±10 % jitter, cap 32 s
_MAX_RETRIES = 4
_BACKOFF_BASE = 1.0          # seconds
_BACKOFF_CAP = 32.0          # seconds
_RETRYABLE_STATUS = {429, 500, 503}

# Gemini role mapping (new SDK uses "user" / "model")
_ROLE_MAP = {
    MessageRole.USER:      "user",
    MessageRole.ASSISTANT: "model",
    MessageRole.SYSTEM:    "user",   # handled via system_instruction
}

# Safety settings — permissive so application layer owns moderation
_SAFETY_SETTINGS = [
    {"category": "HARM_CATEGORY_HARASSMENT",        "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_HATE_SPEECH",       "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
]


# ═══════════════════════════════════════════════════════════════════════════
# Helper — exponential back-off with jitter
# ═══════════════════════════════════════════════════════════════════════════

def _backoff(attempt: int) -> float:
    """Return sleep duration (seconds) for the given retry attempt (0-indexed)."""
    base = min(_BACKOFF_BASE * (2 ** attempt), _BACKOFF_CAP)
    return base * (0.9 + random.random() * 0.2)  # ± 10 % jitter


# ═══════════════════════════════════════════════════════════════════════════
# Provider
# ═══════════════════════════════════════════════════════════════════════════

class GeminiProvider(LLMClient):
    """
    Production-grade LLMClient for Google Gemini (google-genai SDK).

    The provider is designed to be instantiated once (singleton via the
    registry) and reused across the lifetime of the server process.

    Thread-safety: all public methods are async and safe for concurrent use.
    The underlying sdk client manages its own connection pool.
    """

    def __init__(self, api_key: Optional[str] = None, **_kwargs: Any) -> None:
        self._api_key = api_key
        self._client: Any = None            # google.genai.Client, lazy-init

    # ── Lazy client initialisation ────────────────────────────────────────

    def _get_client(self) -> Any:
        """
        Return (and lazily create) the `google.genai.Client` instance.

        Using lazy init keeps import errors from crashing startup when the
        package is not installed.
        """
        if self._client is not None:
            return self._client

        try:
            from google import genai
        except ImportError as exc:
            raise LLMProviderError(
                "google-genai package not installed. "
                "Run: pip install google-genai",
                provider="gemini",
            ) from exc

        api_key = self._resolve_api_key()
        self._client = genai.Client(api_key=api_key)
        logger.info("gemini_client_created", model=self.default_model)
        return self._client

    def _resolve_api_key(self) -> str:
        if self._api_key:
            return self._api_key
        try:
            from backend.core.config import settings
            key = getattr(settings, "GEMINI_API_KEY", None)
            if key:
                return key
        except Exception:
            pass
        raise LLMAuthenticationError(
            "GEMINI_API_KEY is not set.", provider="gemini"
        )

    def _sanitize_model_name(self, model: str) -> str:
        """Strip the 'models/' prefix that the ListModels API returns."""
        return model.removeprefix("models/")

    def _resolve_model(self, cfg_model: Optional[str]) -> str:
        """Precedence: call-time model > env GEMINI_MODEL > hardcoded default."""
        raw = cfg_model if cfg_model else self.default_model
        return self._sanitize_model_name(raw)

    # ── LLMClient interface ── properties ─────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        try:
            from backend.core.config import settings
            m = getattr(settings, "GEMINI_MODEL", None)
            if m:
                return m
        except Exception:
            pass
        return _DEFAULT_CHAT_MODEL

    @property
    def supported_models(self) -> List[str]:
        return sorted(_ALL_MODELS)

    # ── Content builders ──────────────────────────────────────────────────

    def _build_contents_and_system(
        self,
        messages: List[LLMMessage],
    ) -> tuple[Optional[str], List[Dict[str, Any]]]:
        """
        Split messages into (system_instruction, contents_list).

        The new SDK expects:
            contents = [{"role": "user"|"model", "parts": [{"text": "..."}]}]
        """
        system_parts: List[str] = []
        contents: List[Dict[str, Any]] = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_parts.append(msg.content)
                continue

            role = _ROLE_MAP[msg.role]
            parts: List[Dict[str, Any]] = [{"text": msg.content}]

            # Multimodal: inline images
            if msg.images:
                for img in msg.images:
                    if img.startswith("data:"):
                        # data:image/jpeg;base64,<payload>
                        match = re.match(r"data:([^;]+);base64,(.+)", img, re.S)
                        if match:
                            parts.append({
                                "inline_data": {
                                    "mime_type": match.group(1),
                                    "data": match.group(2),
                                }
                            })
                    elif img.startswith("http://") or img.startswith("https://"):
                        parts.append({"file_data": {"file_uri": img}})
                    else:
                        # Assume raw base64 JPEG
                        parts.append({
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": img,
                            }
                        })

            contents.append({"role": role, "parts": parts})

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return system_instruction, contents

    def _build_generate_config(self, cfg: LLMConfig) -> Dict[str, Any]:
        """Return the GenerateContentConfig kwargs for the new SDK."""
        kw: Dict[str, Any] = {
            "temperature": cfg.temperature,
            "top_p": cfg.top_p,
            "safety_settings": _SAFETY_SETTINGS,
        }
        if cfg.max_tokens is not None:
            kw["max_output_tokens"] = cfg.max_tokens
        return kw

    # ── Retry wrapper ─────────────────────────────────────────────────────

    async def _with_retry(self, coro_fn, *args, **kwargs) -> Any:
        """
        Execute an async coroutine with exponential back-off retry.

        Retries on rate-limit (429) and transient server errors (500, 503).
        Raises immediately on auth / model-not-found / content-filter errors.
        """
        last_exc: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return await coro_fn(*args, **kwargs)
            except LLMRateLimitError as exc:
                last_exc = exc
                if attempt == _MAX_RETRIES:
                    break
                wait = exc.retry_after if exc.retry_after else _backoff(attempt)
                logger.warning(
                    "gemini_rate_limit_retry",
                    attempt=attempt + 1,
                    wait_seconds=round(wait, 2),
                )
                await asyncio.sleep(wait)
            except LLMConnectionError as exc:
                last_exc = exc
                if attempt == _MAX_RETRIES:
                    break
                wait = _backoff(attempt)
                logger.warning(
                    "gemini_connection_retry",
                    attempt=attempt + 1,
                    wait_seconds=round(wait, 2),
                )
                await asyncio.sleep(wait)
            except (LLMAuthenticationError, LLMModelNotFoundError,
                    LLMContentFilterError, LLMProviderError) as exc:
                # Non-retryable
                raise exc

        raise last_exc  # type: ignore[misc]

    # ── Core generate ─────────────────────────────────────────────────────

    async def _do_generate(
        self,
        messages: List[LLMMessage],
        cfg: LLMConfig,
    ) -> LLMResponse:
        from google.genai import types as genai_types

        client = self._get_client()
        model_name = self._resolve_model(cfg.model)
        system_instruction, contents = self._build_contents_and_system(messages)
        gen_config = self._build_generate_config(cfg)

        if system_instruction:
            gen_config["system_instruction"] = system_instruction

        try:
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(**gen_config),
            )
        except Exception as exc:
            raise self._normalize_error(exc) from exc

        return self._parse_response(response, model_name)

    async def generate(
        self,
        messages: List[Union[LLMMessage, Dict[str, str]]],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Generate a completion — fully async, with retry."""
        cfg = self._resolve_config(config)
        normalized = self._normalize_messages(messages)
        return await self._with_retry(self._do_generate, normalized, cfg)

    # ── Streaming ─────────────────────────────────────────────────────────

    async def stream(
        self,
        messages: List[Union[LLMMessage, Dict[str, str]]],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream a completion. Yields LLMStreamChunk on each token batch."""
        from google.genai import types as genai_types

        cfg = self._resolve_config(config)
        normalized = self._normalize_messages(messages)
        client = self._get_client()
        model_name = self._resolve_model(cfg.model)
        system_instruction, contents = self._build_contents_and_system(normalized)
        gen_config = self._build_generate_config(cfg)
        if system_instruction:
            gen_config["system_instruction"] = system_instruction

        prompt_tokens = 0
        completion_tokens = 0

        try:
            async for chunk in await client.aio.models.generate_content_stream(
                model=model_name,
                contents=contents,
                config=genai_types.GenerateContentConfig(**gen_config),
            ):
                text = ""
                try:
                    text = chunk.text or ""
                except (ValueError, AttributeError):
                    pass

                # Accumulate token counts from usageMetadata when present
                if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
                    um = chunk.usage_metadata
                    prompt_tokens = getattr(um, "prompt_token_count", prompt_tokens) or prompt_tokens
                    completion_tokens = getattr(um, "candidates_token_count", completion_tokens) or completion_tokens

                if text:
                    yield LLMStreamChunk(
                        content=text,
                        model=model_name,
                        provider=self.provider_name,
                    )

        except Exception as exc:
            raise self._normalize_error(exc) from exc

        # Final sentinel chunk with token usage
        yield LLMStreamChunk(
            content="",
            model=model_name,
            provider=self.provider_name,
            finish_reason="stop",
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    # ── Embeddings ────────────────────────────────────────────────────────

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Batch-embed texts using the Gemini embedding API."""
        if not texts:
            return []

        from google.genai import types as genai_types

        client = self._get_client()
        embedding_model = model or _DEFAULT_EMBEDDING_MODEL

        try:
            results: List[List[float]] = []
            for text in texts:
                response = await client.aio.models.embed_content(
                    model=embedding_model,
                    contents=text,
                    config=genai_types.EmbedContentConfig(
                        task_type="RETRIEVAL_DOCUMENT",
                    ),
                )
                embedding = response.embeddings[0].values if response.embeddings else []
                results.append(list(embedding))
            return results
        except Exception as exc:
            logger.error("gemini_embedding_error", error=str(exc))
            raise self._normalize_error(exc) from exc

    # ── Health check ──────────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """Ping Gemini with a minimal request. Returns True if healthy."""
        try:
            from google.genai import types as genai_types
            client = self._get_client()
            response = await client.aio.models.generate_content(
                model=self.default_model,
                contents=[{"role": "user", "parts": [{"text": "ping"}]}],
                config=genai_types.GenerateContentConfig(
                    max_output_tokens=5,
                    safety_settings=_SAFETY_SETTINGS,
                ),
            )
            return bool(response.text)
        except Exception as exc:
            logger.warning("gemini_health_check_failed", error=str(exc))
            return False

    # ── Resource management ───────────────────────────────────────────────

    async def close(self) -> None:
        """Release the underlying HTTP client connections."""
        if self._client is not None:
            try:
                # The google-genai client exposes a close() method in newer versions
                close_fn = getattr(self._client, "close", None) or getattr(
                    getattr(self._client, "aio", None), "close", None
                )
                if close_fn and asyncio.iscoroutinefunction(close_fn):
                    await close_fn()
                elif close_fn:
                    close_fn()
            except Exception:
                pass
            self._client = None

    # ── Response parser ───────────────────────────────────────────────────

    def _parse_response(self, response: Any, model_name: str) -> LLMResponse:
        """Convert a Gemini GenerateContentResponse to a normalized LLMResponse."""
        content = ""
        finish_reason = "stop"

        try:
            content = response.text or ""
        except (ValueError, AttributeError):
            finish_reason = "content_filter"
            try:
                candidate = response.candidates[0]
                if candidate.content and candidate.content.parts:
                    content = candidate.content.parts[0].text or ""
                finish_reason = str(
                    getattr(getattr(candidate, "finish_reason", None), "name", "content_filter")
                )
            except Exception:
                pass
            if not content:
                raise LLMContentFilterError(
                    "Response blocked by Gemini safety filters",
                    provider="gemini",
                )

        usage = TokenUsage()
        try:
            um = response.usage_metadata
            if um:
                usage = TokenUsage(
                    prompt_tokens=getattr(um, "prompt_token_count", 0) or 0,
                    completion_tokens=getattr(um, "candidates_token_count", 0) or 0,
                    total_tokens=getattr(um, "total_token_count", 0) or 0,
                )
        except Exception:
            pass

        return LLMResponse(
            content=content,
            model=model_name,
            provider=self.provider_name,
            response_id="",
            usage=usage,
            citations=[],
            search_queries=[],
            finish_reason=finish_reason,
        )

    # ── Error normalisation ───────────────────────────────────────────────

    def _normalize_error(self, error: Exception) -> LLMProviderError:
        """
        Map google-genai exceptions → normalized LLMProviderError hierarchy.

        We inspect the exception type name *and* string message to cover both
        typed SDK exceptions and raw HTTP errors wrapped as generic exceptions.
        """
        if isinstance(error, LLMProviderError):
            return error

        err_type = type(error).__name__.lower()
        err_str  = str(error).lower()
        raw_str  = str(error)

        # --- Authentication -------------------------------------------------
        if (
            "api_key" in err_str
            or "api key" in err_str
            or "unauthenticated" in err_str
            or "permission" in err_str
            or "403" in err_str
        ):
            return LLMAuthenticationError(
                f"Gemini authentication error: {raw_str}",
                provider="gemini",
                cause=error,
            )

        # --- Model not found ------------------------------------------------
        if (
            "not found" in err_str
            or "404" in err_str
            or "does not exist" in err_str
            or "unsupported" in err_str
            or "not supported" in err_str
        ):
            return LLMModelNotFoundError(
                f"Gemini model not found: {raw_str}",
                provider="gemini",
                cause=error,
            )

        # --- Rate limit / quota --------------------------------------------
        retry_after: Optional[float] = None
        if "retry_delay" in err_str:
            try:
                match = re.search(r"seconds:\s*(\d+)", raw_str)
                if match:
                    retry_after = float(match.group(1))
            except Exception:
                pass

        if (
            "quota" in err_str
            or "rate" in err_str
            or "429" in err_str
            or "resource_exhausted" in err_str
            or "resourceexhausted" in err_type
            or "ratelimit" in err_type
        ):
            return LLMRateLimitError(
                f"Gemini rate limit exceeded: {raw_str}",
                provider="gemini",
                retry_after=retry_after,
                cause=error,
            )

        # --- Safety / content filter ----------------------------------------
        if "blocked" in err_str or "safety" in err_str or "harmful" in err_str:
            return LLMContentFilterError(
                f"Gemini content filter blocked response: {raw_str}",
                provider="gemini",
                cause=error,
            )

        # --- Connection / timeout -------------------------------------------
        if (
            "timeout" in err_str
            or "deadline" in err_str
            or "connection" in err_str
            or "503" in err_str
            or "unavailable" in err_str
        ):
            return LLMConnectionError(
                f"Gemini connection error: {raw_str}",
                provider="gemini",
                cause=error,
            )

        return LLMProviderError(
            f"Gemini error: {raw_str}",
            provider="gemini",
            cause=error,
        )


# ═══════════════════════════════════════════════════════════════════════════
# Self-registration — auto-runs on import
# ═══════════════════════════════════════════════════════════════════════════

LLMRegistry.register("gemini", GeminiProvider)
