"""
LLM Base Interface — Provider-Agnostic Contract

This module defines the abstract interface that ALL LLM providers must implement.
Business logic depends ONLY on these types — never on provider-specific SDKs.

Design principles:
- Single interface for generate / stream / embed / health_check
- Normalized error hierarchy (no provider errors leak upward)
- Token usage reporting on every response
- Async-first (all methods are async)
- No provider-specific imports in this file
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    Any,
    AsyncIterator,
    Dict,
    List,
    Optional,
    Union,
)

import structlog

logger = structlog.get_logger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Error Hierarchy — normalized across all providers
# ═══════════════════════════════════════════════════════════════════════════


class LLMProviderError(Exception):
    """Base exception for all LLM provider errors."""

    def __init__(self, message: str, provider: str = "unknown", *, cause: Optional[Exception] = None):
        self.provider = provider
        self.cause = cause
        super().__init__(f"[{provider}] {message}")


class LLMRateLimitError(LLMProviderError):
    """Provider rate limit exceeded. Caller should back off."""

    def __init__(self, message: str, provider: str = "unknown", retry_after: Optional[float] = None, **kw):
        self.retry_after = retry_after
        super().__init__(message, provider, **kw)


class LLMAuthenticationError(LLMProviderError):
    """API key invalid or expired."""
    pass


class LLMModelNotFoundError(LLMProviderError):
    """Requested model does not exist on this provider."""
    pass


class LLMConnectionError(LLMProviderError):
    """Network / timeout error reaching the provider."""
    pass


class LLMContentFilterError(LLMProviderError):
    """Response blocked by provider safety filters."""
    pass


# ═══════════════════════════════════════════════════════════════════════════
# Data Transfer Objects
# ═══════════════════════════════════════════════════════════════════════════


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class LLMMessage:
    """A single message in a conversation."""
    role: MessageRole
    content: str
    # For multimodal (vision) — provider adapters translate this to their format
    images: Optional[List[str]] = None  # base64 or URLs

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.images:
            d["images"] = self.images
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "LLMMessage":
        return cls(
            role=MessageRole(data["role"]),
            content=data["content"],
            images=data.get("images"),
        )


@dataclass
class TokenUsage:
    """Token usage statistics from a single request."""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass
class LLMResponse:
    """Normalized response from any LLM provider."""
    content: str
    model: str
    provider: str
    response_id: str = ""
    usage: TokenUsage = field(default_factory=TokenUsage)
    citations: List[Dict[str, Any]] = field(default_factory=list)
    search_queries: List[str] = field(default_factory=list)
    finish_reason: str = "stop"
    raw: Optional[Dict[str, Any]] = None  # original provider response for debugging

    @property
    def has_citations(self) -> bool:
        return len(self.citations) > 0


@dataclass
class LLMStreamChunk:
    """A single chunk from a streaming response."""
    content: str
    model: str
    provider: str
    finish_reason: Optional[str] = None
    usage: Optional[TokenUsage] = None


@dataclass
class LLMConfig:
    """Provider-agnostic configuration for an LLM request."""
    model: Optional[str] = None  # None → use provider default
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    timeout: int = 60
    stream: bool = False
    # Provider-specific overrides (pass-through, not interpreted by the interface)
    extra: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
# Abstract LLM Client Interface
# ═══════════════════════════════════════════════════════════════════════════


class LLMClient(ABC):
    """
    Abstract base class for all LLM providers.

    Every provider module (Perplexity, Gemini, OpenAI, …) must subclass this
    and implement all abstract methods. Business logic ONLY depends on this
    interface — never on provider internals.

    Contract guarantees:
    - `generate()` always returns `LLMResponse`
    - `stream()` always yields `LLMStreamChunk`
    - `generate_embeddings()` always returns `List[List[float]]`
    - All errors are normalized to `LLMProviderError` subclasses
    - `health_check()` returns True/False, never raises
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the canonical name of this provider (e.g., 'perplexity', 'gemini')."""
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Return the default model name for this provider."""
        ...

    @property
    @abstractmethod
    def supported_models(self) -> List[str]:
        """Return list of models this provider supports."""
        ...

    @abstractmethod
    async def generate(
        self,
        messages: List[Union[LLMMessage, Dict[str, str]]],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        """
        Generate a completion from a list of messages.

        Args:
            messages: Conversation messages (system, user, assistant).
            config: Generation configuration (model, temperature, etc.).

        Returns:
            Normalized LLMResponse.

        Raises:
            LLMRateLimitError: Provider rate limit exceeded.
            LLMAuthenticationError: API key invalid.
            LLMModelNotFoundError: Requested model not available.
            LLMConnectionError: Network / timeout error.
            LLMContentFilterError: Blocked by safety filters.
            LLMProviderError: Any other provider error.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: List[Union[LLMMessage, Dict[str, str]]],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        Stream a completion from a list of messages.

        Yields LLMStreamChunk objects. The final chunk has finish_reason set.
        """
        ...

    @abstractmethod
    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Generate embedding vectors for a list of texts.

        Args:
            texts: List of strings to embed.
            model: Optional embedding model override.

        Returns:
            List of embedding vectors (List[float] per text).
        """
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """
        Verify the provider is reachable and the API key is valid.

        Returns True if healthy, False otherwise. Never raises.
        """
        ...

    @abstractmethod
    async def close(self) -> None:
        """Release any held resources (HTTP clients, connections)."""
        ...

    # ── Convenience helpers (not abstract — shared by all providers) ──

    def _normalize_messages(
        self, messages: List[Union[LLMMessage, Dict[str, str]]]
    ) -> List[LLMMessage]:
        """Convert mixed message formats into a list of LLMMessage."""
        normalized: List[LLMMessage] = []
        for msg in messages:
            if isinstance(msg, LLMMessage):
                normalized.append(msg)
            elif isinstance(msg, dict):
                normalized.append(LLMMessage.from_dict(msg))
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")
        return normalized

    def _resolve_config(self, config: Optional[LLMConfig]) -> LLMConfig:
        """Return the provided config or a sensible default."""
        return config or LLMConfig()

    # Context manager support
    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
