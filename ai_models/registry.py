"""
LLM Provider Registry — Runtime Model Switching

This module provides the factory / registry for creating LLM clients.
It supports:
- Config-based default provider (DEFAULT_LLM_PROVIDER env var)
- Request-level provider override
- Lazy initialization and caching of provider clients
- Provider-specific fallback behavior

Usage:
    # Default provider (from env / config)
    client = get_llm_client()

    # Explicit provider override
    client = get_llm_client(provider="gemini")

    # Request-level override
    client = get_llm_client(provider="perplexity", model="sonar-large-online")
"""

from typing import Dict, Optional, Type
import structlog

from .base import LLMClient, LLMProviderError

logger = structlog.get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Provider Registry
# ═══════════════════════════════════════════════════════════════════════════

class LLMRegistry:
    """
    Central registry for LLM provider classes.

    Providers register themselves at import time via `register()`.
    The registry lazily instantiates and caches provider clients.
    """

    _providers: Dict[str, Type[LLMClient]] = {}
    _instances: Dict[str, LLMClient] = {}

    @classmethod
    def register(cls, name: str, provider_class: Type[LLMClient]) -> None:
        """
        Register an LLM provider class.

        Args:
            name: Canonical provider name (e.g., 'perplexity', 'gemini').
            provider_class: The LLMClient subclass to register.
        """
        cls._providers[name.lower()] = provider_class
        logger.info("llm_provider_registered", provider=name)

    @classmethod
    def get_provider_class(cls, name: str) -> Type[LLMClient]:
        """Get a registered provider class by name."""
        name_lower = name.lower()
        if name_lower not in cls._providers:
            available = list(cls._providers.keys())
            raise LLMProviderError(
                f"Unknown LLM provider '{name}'. Available: {available}",
                provider=name,
            )
        return cls._providers[name_lower]

    @classmethod
    def get_or_create(cls, name: str, **kwargs) -> LLMClient:
        """
        Get a cached provider instance, or create one.

        This ensures we don't create multiple HTTP clients for the same provider.
        Pass force_new=True in kwargs to skip the cache.
        """
        name_lower = name.lower()
        force_new = kwargs.pop("force_new", False)

        if not force_new and name_lower in cls._instances:
            return cls._instances[name_lower]

        provider_class = cls.get_provider_class(name_lower)
        instance = provider_class(**kwargs)
        cls._instances[name_lower] = instance
        logger.info("llm_provider_created", provider=name_lower)
        return instance

    @classmethod
    async def close_all(cls) -> None:
        """Close all cached provider instances. Call at app shutdown."""
        for name, instance in cls._instances.items():
            try:
                await instance.close()
                logger.info("llm_provider_closed", provider=name)
            except Exception as e:
                logger.error("llm_provider_close_error", provider=name, error=str(e))
        cls._instances.clear()

    @classmethod
    def available_providers(cls) -> list[str]:
        """Return list of registered provider names."""
        return list(cls._providers.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registrations and instances (for testing)."""
        cls._providers.clear()
        cls._instances.clear()


# ═══════════════════════════════════════════════════════════════════════════
# Public Factory Functions
# ═══════════════════════════════════════════════════════════════════════════

def _get_default_provider() -> str:
    """Read the default LLM provider from settings."""
    try:
        from backend.core.config import settings
        return getattr(settings, "DEFAULT_LLM_PROVIDER", "perplexity").lower()
    except Exception:
        return "perplexity"


def _get_default_embedding_provider() -> str:
    """Read the default embedding provider from settings."""
    try:
        from backend.core.config import settings
        return getattr(settings, "DEFAULT_EMBEDDING_PROVIDER", None) or _get_default_provider()
    except Exception:
        return "perplexity"


def get_llm_client(
    provider: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """
    Get an LLM client instance.

    Args:
        provider: Provider name override. If None, uses DEFAULT_LLM_PROVIDER.
        **kwargs: Provider-specific constructor arguments.

    Returns:
        An LLMClient instance (cached by default).
    """
    # Ensure providers are loaded
    _ensure_providers_loaded()

    provider_name = provider or _get_default_provider()
    return LLMRegistry.get_or_create(provider_name, **kwargs)


def get_embedding_client(
    provider: Optional[str] = None,
    **kwargs,
) -> LLMClient:
    """
    Get an LLM client for embedding generation.

    Some providers use different endpoints for embeddings.
    This function routes to the appropriate provider.

    Args:
        provider: Provider name override.
        **kwargs: Provider-specific constructor arguments.

    Returns:
        An LLMClient instance that supports generate_embeddings().
    """
    _ensure_providers_loaded()

    provider_name = provider or _get_default_embedding_provider()
    return LLMRegistry.get_or_create(provider_name, **kwargs)


# ═══════════════════════════════════════════════════════════════════════════
# Provider Auto-Discovery
# ═══════════════════════════════════════════════════════════════════════════

_providers_loaded = False


def _ensure_providers_loaded():
    """Import provider modules so they self-register."""
    global _providers_loaded
    if _providers_loaded:
        return

    # Import all provider modules — each calls LLMRegistry.register() on import
    try:
        from backend.ai_models.providers import perplexity_provider  # noqa: F401
    except ImportError as e:
        logger.warning("perplexity_provider_import_failed", error=str(e))

    try:
        from backend.ai_models.providers import gemini_provider  # noqa: F401
    except ImportError as e:
        logger.debug("gemini_provider_import_skipped", error=str(e))

    try:
        from backend.ai_models.providers import openai_provider  # noqa: F401
    except ImportError as e:
        logger.debug("openai_provider_import_skipped", error=str(e))

    _providers_loaded = True
