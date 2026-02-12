"""
AI Models — Multi-Provider LLM Abstraction Layer

This package provides a vendor-neutral interface for LLM providers.
All business logic interacts with the abstract `LLMClient` interface,
never with provider-specific SDKs directly.

Supported providers:
- Perplexity Sonar (default)
- Google Gemini
- OpenAI (GPT-4o, GPT-4 Turbo, GPT-3.5 Turbo, o1/o3 series)
- (extensible — add new providers in providers/)

Usage:
    from backend.ai_models import get_llm_client, LLMClient

    client: LLMClient = get_llm_client()  # uses DEFAULT_LLM_PROVIDER
    response = await client.generate(messages=[...])
"""

from .base import (
    LLMClient,
    LLMResponse,
    LLMMessage,
    LLMConfig,
    LLMProviderError,
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMModelNotFoundError,
    LLMConnectionError,
    LLMContentFilterError,
    TokenUsage,
    MessageRole,
)
from .registry import get_llm_client, get_embedding_client, LLMRegistry

__all__ = [
    "LLMClient",
    "LLMResponse",
    "LLMMessage",
    "LLMConfig",
    "LLMProviderError",
    "LLMRateLimitError",
    "LLMAuthenticationError",
    "LLMModelNotFoundError",
    "LLMConnectionError",
    "LLMContentFilterError",
    "TokenUsage",
    "MessageRole",
    "get_llm_client",
    "get_embedding_client",
    "LLMRegistry",
]
