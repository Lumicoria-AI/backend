"""
Google Gemini Provider — LLMClient Implementation

Implements the provider-agnostic LLMClient interface for Google's Gemini API
(Google AI Studio / Vertex AI). Matches Perplexity's behavioral contract:
- Same message format semantics
- Same error handling guarantees
- Same response shape

Requires:
    pip install google-generativeai

Environment:
    GEMINI_API_KEY — Google AI Studio API key
"""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional, Union
from concurrent.futures import ThreadPoolExecutor

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
    LLMContentFilterError,
    TokenUsage,
    MessageRole,
)
from backend.ai_models.registry import LLMRegistry

logger = structlog.get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════

_CHAT_MODELS: set[str] = {
    # Gemini 2.5 series (latest)
    "gemini-2.5-pro-preview-06-05",
    "gemini-2.5-flash-preview-05-20",
    # Gemini 2.0 series
    "gemini-2.0-pro",
    "gemini-2.0-pro-exp",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-2.0-flash-thinking-exp",
    # Gemini 1.5 series
    "gemini-1.5-pro",
    "gemini-1.5-pro-latest",
    "gemini-1.5-flash",
    "gemini-1.5-flash-latest",
    "gemini-1.5-flash-8b",
    # Legacy
    "gemini-1.0-pro",
}

_EMBEDDING_MODELS: set[str] = {
    "text-embedding-004",
}

_DEFAULT_CHAT_MODEL = "gemini-2.0-flash"
_DEFAULT_EMBEDDING_MODEL = "text-embedding-004"

# Merged set for supported_models property
_ALL_MODELS = _CHAT_MODELS | _EMBEDDING_MODELS

# Gemini's role mapping (Gemini uses "model" instead of "assistant")
_ROLE_MAP = {
    MessageRole.SYSTEM: "user",      # Gemini handles system via system_instruction
    MessageRole.USER: "user",
    MessageRole.ASSISTANT: "model",
}


class GeminiProvider(LLMClient):
    """
    LLMClient implementation for Google Gemini API.

    Uses the google-generativeai SDK. Since the SDK is synchronous,
    all calls are wrapped in a thread pool executor for async compatibility.

    Design decisions:
    - System messages → Gemini's `system_instruction` parameter
    - Gemini "model" role → normalized to "assistant" in responses
    - Safety settings → configurable, defaults to BLOCK_NONE for flexibility
    - Embedding → uses text-embedding-004 model
    """

    def __init__(self, api_key: Optional[str] = None, **kwargs):
        """
        Initialize the Gemini provider.

        Args:
            api_key: Gemini API key. If None, reads from settings.
        """
        self._api_key = api_key
        self._sdk = None
        self._executor = ThreadPoolExecutor(max_workers=4)

    def _get_api_key(self) -> str:
        """Resolve API key from argument or settings."""
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
            "GEMINI_API_KEY is required", provider="gemini"
        )

    def _get_sdk(self):
        """Lazy-initialize the Gemini SDK."""
        if self._sdk is None:
            try:
                import google.generativeai as genai
            except ImportError:
                raise LLMProviderError(
                    "google-generativeai package is not installed. "
                    "Install with: pip install google-generativeai",
                    provider="gemini",
                )
            genai.configure(api_key=self._get_api_key())
            self._sdk = genai
        return self._sdk

    # ── Interface properties ──────────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "gemini"

    @property
    def default_model(self) -> str:
        try:
            from backend.core.config import get_settings
            model = getattr(get_settings(), "GEMINI_MODEL", None)
            if model:
                return model
        except Exception:
            pass
        return _DEFAULT_CHAT_MODEL

    @property
    def supported_models(self) -> List[str]:
        return sorted(_ALL_MODELS)

    # ── Core methods ──────────────────────────────────────────────────

    async def generate(
        self,
        messages: List[Union[LLMMessage, Dict[str, str]]],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """Generate a completion via Gemini API."""
        cfg = self._resolve_config(config)
        normalized = self._normalize_messages(messages)

        # Separate system instruction from conversation messages
        system_instruction, conversation = self._split_system(normalized)

        # Convert to Gemini message format
        gemini_messages = self._to_gemini_messages(conversation)

        model_name = cfg.model or self.default_model

        try:
            genai = self._get_sdk()

            # Build generation config
            generation_config = genai.GenerationConfig(
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
            )

            # Configure safety settings (permissive — let the app layer handle moderation)
            safety_settings = self._get_safety_settings(genai)

            # Create model instance
            model_kwargs = {
                "model_name": model_name,
                "generation_config": generation_config,
                "safety_settings": safety_settings,
            }
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction

            model = genai.GenerativeModel(**model_kwargs)

            # Run synchronous SDK call in executor
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor,
                lambda: model.generate_content(gemini_messages),
            )

            return self._to_llm_response(response, model_name)

        except Exception as e:
            raise self._normalize_error(e) from e

    async def stream(
        self,
        messages: List[Union[LLMMessage, Dict[str, str]]],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """Stream a completion from Gemini."""
        cfg = self._resolve_config(config)
        normalized = self._normalize_messages(messages)

        system_instruction, conversation = self._split_system(normalized)
        gemini_messages = self._to_gemini_messages(conversation)

        model_name = cfg.model or self.default_model

        try:
            genai = self._get_sdk()

            generation_config = genai.GenerationConfig(
                temperature=cfg.temperature,
                max_output_tokens=cfg.max_tokens,
                top_p=cfg.top_p,
            )
            safety_settings = self._get_safety_settings(genai)

            model_kwargs = {
                "model_name": model_name,
                "generation_config": generation_config,
                "safety_settings": safety_settings,
            }
            if system_instruction:
                model_kwargs["system_instruction"] = system_instruction

            model = genai.GenerativeModel(**model_kwargs)

            # Run streaming in executor (SDK is synchronous)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                self._executor,
                lambda: model.generate_content(gemini_messages, stream=True),
            )

            # Yield chunks
            for chunk in response:
                if chunk.text:
                    yield LLMStreamChunk(
                        content=chunk.text,
                        model=model_name,
                        provider=self.provider_name,
                    )

            # Final chunk with finish reason
            yield LLMStreamChunk(
                content="",
                model=model_name,
                provider=self.provider_name,
                finish_reason="stop",
            )

        except Exception as e:
            raise self._normalize_error(e) from e

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """Generate embeddings via Gemini embedding API."""
        if not texts:
            return []

        embedding_model = model or _DEFAULT_EMBEDDING_MODEL

        try:
            genai = self._get_sdk()

            loop = asyncio.get_event_loop()

            # Gemini supports batch embedding
            result = await loop.run_in_executor(
                self._executor,
                lambda: genai.embed_content(
                    model=embedding_model,
                    content=texts,
                    task_type="retrieval_document",
                ),
            )

            # The result contains a list of embeddings
            if isinstance(result, dict) and "embedding" in result:
                embeddings = result["embedding"]
                # If single text, it returns a single embedding
                if texts and len(texts) == 1 and isinstance(embeddings[0], float):
                    return [embeddings]
                return embeddings

            return [[0.0] * 768] * len(texts)  # Gemini default dimension

        except Exception as e:
            logger.error("gemini_embedding_error", error=str(e))
            raise self._normalize_error(e) from e

    async def health_check(self) -> bool:
        """Verify Gemini API is reachable."""
        try:
            genai = self._get_sdk()
            loop = asyncio.get_event_loop()

            model = genai.GenerativeModel(model_name=self.default_model)
            response = await loop.run_in_executor(
                self._executor,
                lambda: model.generate_content(
                    "Respond with OK",
                    generation_config=genai.GenerationConfig(max_output_tokens=5),
                ),
            )
            return bool(response.text)
        except Exception as e:
            logger.warning("gemini_health_check_failed", error=str(e))
            return False

    async def close(self) -> None:
        """Release resources."""
        self._executor.shutdown(wait=False)
        self._sdk = None

    # ── Private helpers ───────────────────────────────────────────────

    def _split_system(self, messages: List[LLMMessage]):
        """
        Separate system messages from conversation messages.

        Gemini handles system prompts via `system_instruction` parameter,
        not as a message in the conversation.
        """
        system_parts = []
        conversation = []

        for msg in messages:
            if msg.role == MessageRole.SYSTEM:
                system_parts.append(msg.content)
            else:
                conversation.append(msg)

        system_instruction = "\n\n".join(system_parts) if system_parts else None
        return system_instruction, conversation

    def _to_gemini_messages(self, messages: List[LLMMessage]) -> list:
        """
        Convert LLMMessage list to Gemini's content format.

        Gemini expects:
          [{"role": "user", "parts": ["text"]}, {"role": "model", "parts": ["text"]}]
        """
        gemini_msgs = []

        for msg in messages:
            role = _ROLE_MAP.get(msg.role, "user")
            parts = [msg.content]

            # Handle multimodal (images)
            if msg.images:
                import base64
                for img in msg.images:
                    if img.startswith("data:") or img.startswith("http"):
                        # URL-based images — Gemini SDK handles these differently
                        parts.append({"type": "image_url", "image_url": img})
                    else:
                        # Base64 encoded image
                        try:
                            img_bytes = base64.b64decode(img)
                            parts.append({
                                "mime_type": "image/jpeg",
                                "data": img_bytes,
                            })
                        except Exception:
                            parts.append(img)

            gemini_msgs.append({"role": role, "parts": parts})

        return gemini_msgs

    def _get_safety_settings(self, genai) -> list:
        """
        Get safety settings for Gemini.
        
        Default: BLOCK_NONE for all categories to let the application
        layer handle content moderation. Override via config if needed.
        """
        try:
            return [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
            ]
        except Exception:
            return []

    def _to_llm_response(self, response, model_name: str) -> LLMResponse:
        """Convert a Gemini response to a normalized LLMResponse."""
        content = ""
        finish_reason = "stop"

        try:
            content = response.text
        except ValueError:
            # Response was blocked by safety filters
            finish_reason = "content_filter"
            try:
                # Try to get partial content
                if response.candidates:
                    candidate = response.candidates[0]
                    if candidate.content and candidate.content.parts:
                        content = candidate.content.parts[0].text
                    finish_reason = str(getattr(candidate, "finish_reason", "content_filter"))
            except Exception:
                pass

            if not content:
                raise LLMContentFilterError(
                    "Response blocked by Gemini safety filters",
                    provider="gemini",
                )

        # Extract usage metadata
        usage = TokenUsage()
        try:
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                um = response.usage_metadata
                usage = TokenUsage(
                    prompt_tokens=getattr(um, "prompt_token_count", 0),
                    completion_tokens=getattr(um, "candidates_token_count", 0),
                    total_tokens=getattr(um, "total_token_count", 0),
                )
        except Exception:
            pass

        return LLMResponse(
            content=content,
            model=model_name,
            provider=self.provider_name,
            response_id="",  # Gemini doesn't return a response ID
            usage=usage,
            citations=[],  # Gemini doesn't provide citations like Perplexity
            search_queries=[],
            finish_reason=finish_reason,
        )

    def _normalize_error(self, error: Exception) -> LLMProviderError:
        """Convert Gemini-specific errors to the normalized hierarchy."""
        error_str = str(error).lower()

        if isinstance(error, LLMProviderError):
            return error

        # Check for common Gemini error patterns
        if "api key" in error_str or "api_key" in error_str or "invalid" in error_str and "key" in error_str:
            return LLMAuthenticationError(
                f"Invalid Gemini API key: {error}", provider="gemini", cause=error
            )

        if "quota" in error_str or "rate" in error_str or "429" in error_str or "resource_exhausted" in error_str:
            return LLMRateLimitError(
                f"Gemini rate limit exceeded: {error}", provider="gemini", cause=error
            )

        if "not found" in error_str or "404" in error_str or "does not exist" in error_str:
            return LLMModelNotFoundError(
                f"Gemini model not found: {error}", provider="gemini", cause=error
            )

        if "blocked" in error_str or "safety" in error_str:
            return LLMContentFilterError(
                f"Gemini content filter: {error}", provider="gemini", cause=error
            )

        if "timeout" in error_str or "deadline" in error_str or "connection" in error_str:
            return LLMConnectionError(
                f"Gemini connection error: {error}", provider="gemini", cause=error
            )

        return LLMProviderError(
            f"Gemini error: {error}", provider="gemini", cause=error
        )


# ═══════════════════════════════════════════════════════════════════════════
# Self-registration
# ═══════════════════════════════════════════════════════════════════════════

LLMRegistry.register("gemini", GeminiProvider)
