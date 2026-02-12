"""
Perplexity Provider — LLMClient Implementation

Wraps the existing PerplexityClient (backend.ai_models.perplexity) behind
the provider-agnostic LLMClient interface. All Perplexity-specific logic
(API format, response parsing, retries) stays in this module.

The original perplexity.py is kept as-is for backward compatibility.
This adapter delegates to it.
"""

from typing import Any, AsyncIterator, Dict, List, Optional, Union
import structlog

from backend.ai_models.base import (
    LLMClient,
    LLMConfig,
    LLMMessage,
    LLMResponse,
    LLMStreamChunk,
    LLMProviderError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMModelNotFoundError,
    TokenUsage,
)
from backend.ai_models.registry import LLMRegistry

logger = structlog.get_logger(__name__)

# Perplexity-specific constants
PERPLEXITY_MODELS = [
    "sonar-small-chat",
    "sonar-medium-chat",
    "sonar-large-chat",
    "sonar-large-online",
    "sonar-medium-online",
    "mistral-7b-instruct",
    "mixtral-8x7b-instruct",
    "llama-3-70b-flash",
    "llama-3-8b-instruct",
]

DEFAULT_PERPLEXITY_MODEL = "sonar-large-online"
PERPLEXITY_EMBEDDING_MODEL = "sonar-embedding-001"


class PerplexityProvider(LLMClient):
    """
    LLMClient implementation for Perplexity Sonar API.

    Delegates to the existing PerplexityClient for HTTP calls, retries,
    and response parsing. This adapter handles:
    - Message normalization (LLMMessage → Perplexity format)
    - Response normalization (Perplexity response → LLMResponse)
    - Error normalization (httpx errors → LLMProviderError hierarchy)
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the Perplexity provider.

        Args:
            api_key: Perplexity API key. If None, reads from settings.
        """
        from backend.core.config import settings

        self._api_key = api_key or settings.PERPLEXITY_API_KEY
        if not self._api_key:
            raise LLMAuthenticationError(
                "PERPLEXITY_API_KEY is required", provider="perplexity"
            )

        # Lazy-init the underlying client
        self._client = None

    def _get_client(self):
        """Lazy-initialize the underlying Perplexity HTTP client."""
        if self._client is None:
            from backend.ai_models.perplexity import PerplexityClient

            self._client = PerplexityClient({"api_key": self._api_key})
        return self._client

    # ── Interface properties ──────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "perplexity"

    @property
    def default_model(self) -> str:
        return DEFAULT_PERPLEXITY_MODEL

    @property
    def supported_models(self) -> List[str]:
        return list(PERPLEXITY_MODELS)

    # ── Core methods ──────────────────────────────────────────────────

    async def generate(
        self,
        messages: List[Union[LLMMessage, Dict[str, str]]],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Generate a completion via Perplexity chat/completions API."""
        cfg = self._resolve_config(config)
        normalized = self._normalize_messages(messages)

        # Convert to Perplexity message format
        pplx_messages = [{"role": m.role.value, "content": m.content} for m in normalized]

        # Build kwargs for the underlying client
        kwargs: Dict[str, Any] = {}
        if cfg.model:
            kwargs["model"] = cfg.model
        if cfg.temperature is not None:
            kwargs["temperature"] = cfg.temperature
        if cfg.max_tokens is not None:
            kwargs["max_tokens"] = cfg.max_tokens
        if cfg.top_p is not None:
            kwargs["top_p"] = cfg.top_p
        kwargs.update(cfg.extra)

        try:
            client = self._get_client()
            response = await client.chat_completion(pplx_messages, stream=False, **kwargs)

            return self._to_llm_response(response)

        except Exception as e:
            raise self._normalize_error(e) from e

    async def stream(
        self,
        messages: List[Union[LLMMessage, Dict[str, str]]],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        Stream a completion from Perplexity.
        
        Note: The current Perplexity client doesn't implement true streaming
        response parsing. This provides a single-chunk "stream" as a 
        compatibility shim. When streaming is added to the raw client,
        this method will yield real chunks.
        """
        # For now, fall back to non-streaming and yield the full response
        response = await self.generate(messages, config)
        yield LLMStreamChunk(
            content=response.content,
            model=response.model,
            provider=self.provider_name,
            finish_reason=response.finish_reason,
            usage=response.usage,
        )

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate embeddings via Perplexity embedding API."""
        if not texts:
            return []

        try:
            client = self._get_client()
            embeddings = await client.generate_embeddings(
                texts=texts,
                model=model or PERPLEXITY_EMBEDDING_MODEL,
            )
            return embeddings

        except Exception as e:
            logger.error("perplexity_embedding_error", error=str(e))
            raise self._normalize_error(e) from e

    async def health_check(self) -> bool:
        """Verify Perplexity API is reachable."""
        try:
            client = self._get_client()
            response = await client.chat_completion(
                [{"role": "user", "content": "ping"}],
                max_tokens=5,
            )
            return bool(response.content)
        except Exception as e:
            logger.warning("perplexity_health_check_failed", error=str(e))
            return False

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    # ── Private helpers ───────────────────────────────────────────────

    def _to_llm_response(self, pplx_response) -> LLMResponse:
        """Convert a PerplexityResponse to a normalized LLMResponse."""
        # Extract usage if available
        usage_data = {}
        if hasattr(pplx_response, "usage") and pplx_response.usage:
            usage_data = pplx_response.usage if isinstance(pplx_response.usage, dict) else {}

        usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
        )

        # Extract citations
        citations = []
        try:
            raw_citations = pplx_response.citations
            for c in raw_citations:
                citations.append({
                    "text": getattr(c, "text", ""),
                    "url": getattr(c.metadata, "url", "") if hasattr(c, "metadata") else "",
                    "title": getattr(c.metadata, "title", "") if hasattr(c, "metadata") else "",
                })
        except Exception:
            pass

        # Extract search queries
        search_queries = []
        try:
            search_queries = pplx_response.search_queries_list
        except Exception:
            pass

        # Finish reason
        finish_reason = "stop"
        try:
            if pplx_response.choices:
                finish_reason = pplx_response.choices[0].get("finish_reason", "stop") or "stop"
        except Exception:
            pass

        return LLMResponse(
            content=pplx_response.content,
            model=pplx_response.model,
            provider=self.provider_name,
            response_id=pplx_response.id,
            usage=usage,
            citations=citations,
            search_queries=search_queries,
            finish_reason=finish_reason,
            raw=pplx_response.dict() if hasattr(pplx_response, "dict") else None,
        )

    def _normalize_error(self, error: Exception) -> LLMProviderError:
        """Convert provider-specific errors to the normalized hierarchy."""
        import httpx

        error_str = str(error)

        if isinstance(error, httpx.HTTPStatusError):
            status = error.response.status_code
            if status == 401:
                return LLMAuthenticationError(
                    "Invalid Perplexity API key", provider="perplexity", cause=error
                )
            elif status == 429:
                retry_after = error.response.headers.get("retry-after")
                return LLMRateLimitError(
                    "Perplexity rate limit exceeded",
                    provider="perplexity",
                    retry_after=float(retry_after) if retry_after else None,
                    cause=error,
                )
            elif status == 404:
                return LLMModelNotFoundError(
                    f"Model not found: {error_str}", provider="perplexity", cause=error
                )
            else:
                return LLMProviderError(
                    f"Perplexity HTTP {status}: {error_str}",
                    provider="perplexity",
                    cause=error,
                )

        if isinstance(error, (httpx.RequestError, httpx.TimeoutException, ConnectionError, TimeoutError)):
            return LLMConnectionError(
                f"Connection error: {error_str}", provider="perplexity", cause=error
            )

        if isinstance(error, LLMProviderError):
            return error

        return LLMProviderError(
            f"Unexpected error: {error_str}", provider="perplexity", cause=error
        )


# ═══════════════════════════════════════════════════════════════════════════
# Self-registration
# ═══════════════════════════════════════════════════════════════════════════

LLMRegistry.register("perplexity", PerplexityProvider)
