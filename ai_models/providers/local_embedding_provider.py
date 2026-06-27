"""
Local Embedding Provider — FastEmbed + BAAI/bge-base-en-v1.5 by default.

Why this exists
---------------
Gemini embeddings are rate-limited and metered.  For a RAG workload every
uploaded document, URL, note, and chat turn is round-tripped through the
Gemini embeddings API, which quickly exhausts the free tier.

This provider runs a quantized ONNX embedding model locally — zero API
calls, zero quota, CPU-only, and lightweight enough to co-locate with the
API server.  Output is 768-dimensional so it is a drop-in for the existing
`VECTOR_STORE_DIMENSION=768` schema (no Weaviate rebuild required).

Key production guarantees
-------------------------
* **Model loaded once** at process startup (or lazily on first call) and
  shared across every async request via a class-level cache.  Thread-safe
  via a module lock.
* **True batching** — FastEmbed's `embed()` does one ONNX inference pass
  per batch; we never loop per-text like the Gemini provider.
* **Non-blocking** — synchronous ONNX inference is off-loaded to a worker
  thread via `asyncio.to_thread`, so the event loop stays responsive under
  high concurrency.
* **Optional process-level parallelism** — for large reindex jobs, pass
  `LOCAL_EMBEDDING_PARALLEL=0` to use all CPU cores or a positive integer
  to cap FastEmbed's multiprocessing pool.
* **Warm-up hook** — `warmup()` preloads the ONNX session and runs one
  dummy inference so the first real request does not pay the cold-start
  penalty (~2-5 s on CPU).

Env variables
-------------
LOCAL_EMBEDDING_MODEL       — HF model id (default: BAAI/bge-base-en-v1.5)
LOCAL_EMBEDDING_CACHE_DIR   — ONNX model cache on disk (default: ./models/fastembed)
LOCAL_EMBEDDING_BATCH_SIZE  — texts per inference batch (default: 64)
LOCAL_EMBEDDING_PARALLEL    — none = single-process, 0 = all cores,
                              N = fan-out across N workers (default: 0)

Registered as provider name: "local"
"""

from __future__ import annotations

import asyncio
import os
import threading
from typing import Any, AsyncIterator, List, Optional, Union

import structlog

from backend.ai_models.base import (
    LLMClient,
    LLMConfig,
    LLMMessage,
    LLMProviderError,
    LLMResponse,
    LLMStreamChunk,
)
from backend.ai_models.registry import LLMRegistry

logger = structlog.get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════════════
# Model catalogue — name → embedding dimension
# ═══════════════════════════════════════════════════════════════════════════

_SUPPORTED_MODELS: dict[str, int] = {
    # 768-dim (matches the Lumicoria default schema)
    "BAAI/bge-base-en-v1.5": 768,
    "snowflake/snowflake-arctic-embed-m": 768,
    "intfloat/multilingual-e5-base": 768,
    "jinaai/jina-embeddings-v2-base-en": 768,
    "nomic-ai/nomic-embed-text-v1.5": 768,
    "thenlper/gte-base": 768,
    # 384-dim (would require VECTOR_STORE_DIMENSION=384 and Weaviate rebuild)
    "BAAI/bge-small-en-v1.5": 384,
    # 1024-dim (would require VECTOR_STORE_DIMENSION=1024 and Weaviate rebuild)
    "BAAI/bge-large-en-v1.5": 1024,
    "intfloat/multilingual-e5-large": 1024,
}

_DEFAULT_MODEL = "BAAI/bge-base-en-v1.5"
_DEFAULT_CACHE_DIR = "./models/fastembed"
_DEFAULT_BATCH_SIZE = 64
_DEFAULT_PARALLEL = 0  # 0 = FastEmbed chooses all available CPU cores


def _configured_value(name: str) -> Any:
    """Read a value from central settings without making import-time config fatal."""
    try:
        from backend.core.config import settings

        return getattr(settings, name, None)
    except Exception:
        return None


def _coerce_optional_int(value: Any, *, default: Optional[int]) -> Optional[int]:
    """Parse ints while preserving the provider's historical 'none' escape hatch."""
    if value is None:
        return default
    if isinstance(value, str) and value.lower() == "none":
        return None
    return int(value)


# ═══════════════════════════════════════════════════════════════════════════
# Process-wide model cache (thread-safe)
# ═══════════════════════════════════════════════════════════════════════════

_model_cache: dict[str, Any] = {}
_model_lock = threading.Lock()


def _load_model(model_name: str, cache_dir: str) -> Any:
    """Load a FastEmbed model, memoised per (model_name) across all callers."""
    with _model_lock:
        if model_name in _model_cache:
            return _model_cache[model_name]

        try:
            from fastembed import TextEmbedding
        except ImportError as exc:  # pragma: no cover
            raise LLMProviderError(
                "fastembed is not installed.  Install with `pip install fastembed`.",
                provider="local",
            ) from exc

        os.makedirs(cache_dir, exist_ok=True)
        logger.info(
            "local_embedding_model_loading",
            model=model_name,
            cache_dir=cache_dir,
        )
        # TextEmbedding downloads + opens an ONNX session.  This is the ~1-5 s
        # cold-start we amortise via warmup().
        instance = TextEmbedding(model_name=model_name, cache_dir=cache_dir)
        _model_cache[model_name] = instance
        logger.info("local_embedding_model_loaded", model=model_name)
        return instance


# ═══════════════════════════════════════════════════════════════════════════
# Provider
# ═══════════════════════════════════════════════════════════════════════════


class LocalEmbeddingProvider(LLMClient):
    """
    Embedding-only LLMClient backed by FastEmbed (ONNX runtime).

    Chat / completion methods raise NotImplementedError on purpose — this
    provider is designed to coexist with a separate chat provider (Gemini,
    OpenAI, etc.).  Configure `DEFAULT_LLM_PROVIDER=gemini` +
    `DEFAULT_EMBEDDING_PROVIDER=local` to get the hybrid setup.
    """

    def __init__(
        self,
        model: Optional[str] = None,
        cache_dir: Optional[str] = None,
        batch_size: Optional[int] = None,
        parallel: Optional[int] = None,
        **_kwargs: Any,
    ) -> None:
        configured_model = _configured_value("LOCAL_EMBEDDING_MODEL")
        configured_cache_dir = _configured_value("LOCAL_EMBEDDING_CACHE_DIR")
        configured_batch_size = _configured_value("LOCAL_EMBEDDING_BATCH_SIZE")
        configured_parallel = _configured_value("LOCAL_EMBEDDING_PARALLEL")

        self._model_name = (
            model
            or configured_model
            or os.getenv("LOCAL_EMBEDDING_MODEL")
            or _DEFAULT_MODEL
        )
        self._cache_dir = (
            cache_dir
            or configured_cache_dir
            or os.getenv("LOCAL_EMBEDDING_CACHE_DIR")
            or _DEFAULT_CACHE_DIR
        )
        batch_env = os.getenv("LOCAL_EMBEDDING_BATCH_SIZE")
        self._batch_size = (
            batch_size
            if batch_size is not None
            else int(
                configured_batch_size
                if configured_batch_size is not None
                else (batch_env if batch_env else _DEFAULT_BATCH_SIZE)
            )
        )
        parallel_env = os.getenv("LOCAL_EMBEDDING_PARALLEL")
        if parallel is not None:
            self._parallel: Optional[int] = parallel
        elif configured_parallel is not None:
            self._parallel = _coerce_optional_int(configured_parallel, default=_DEFAULT_PARALLEL)
        elif parallel_env is not None:
            self._parallel = _coerce_optional_int(parallel_env, default=_DEFAULT_PARALLEL)
        else:
            self._parallel = _DEFAULT_PARALLEL

        if self._model_name not in _SUPPORTED_MODELS:
            logger.warning(
                "local_embedding_model_unverified",
                model=self._model_name,
                hint="dimension unknown; ensure VECTOR_STORE_DIMENSION matches output",
            )
        self._expected_dim = _SUPPORTED_MODELS.get(self._model_name, 768)

    # ── Required LLMClient properties ──────────────────────────────────────

    @property
    def provider_name(self) -> str:
        return "local"

    @property
    def default_model(self) -> str:
        return self._model_name

    @property
    def supported_models(self) -> List[str]:
        return sorted(_SUPPORTED_MODELS.keys())

    @property
    def embedding_dimension(self) -> int:
        """Expected vector dimension for the configured model."""
        return self._expected_dim

    # ── Warm-up ────────────────────────────────────────────────────────────

    async def warmup(self) -> None:
        """
        Preload the ONNX model and run one dummy inference so the first
        real user-facing request pays no cold-start cost.  Safe to call
        from FastAPI's lifespan hook; no-op on subsequent calls.
        """
        def _warm() -> int:
            mdl = _load_model(self._model_name, self._cache_dir)
            # One tiny embed call forces session allocation + first kernel compile
            _ = list(mdl.embed(["warmup"], batch_size=1))
            return 1

        try:
            await asyncio.to_thread(_warm)
            logger.info(
                "local_embedding_warmup_complete",
                model=self._model_name,
                dim=self._expected_dim,
            )
        except Exception as exc:
            # Warm-up failures are non-fatal — the provider can still try
            # lazy-loading on first call.  Log and move on.
            logger.warning("local_embedding_warmup_failed", error=str(exc))

    # ── generate_embeddings ────────────────────────────────────────────────

    async def generate_embeddings(
        self,
        texts: List[str],
        model: Optional[str] = None,
    ) -> List[List[float]]:
        """
        Embed a batch of texts.  One ONNX forward pass per `batch_size`
        inputs; the entire call runs on a worker thread so the event loop
        is never blocked.
        """
        if not texts:
            return []

        model_name = model or self._model_name
        batch_size = self._batch_size
        parallel = self._parallel

        # FastEmbed spawns its own subprocess pool when `parallel` is set —
        # but daemon processes (Celery prefork workers) can't have children.
        # Detect that context and force single-process embedding.  ONNX
        # Runtime still uses multi-threaded intra-op parallelism internally,
        # so throughput stays reasonable.
        if parallel is not None and parallel != None:  # noqa: E711
            try:
                import multiprocessing
                if multiprocessing.current_process().daemon:
                    parallel = None
            except Exception:
                pass

        def _embed_sync() -> List[List[float]]:
            mdl = _load_model(model_name, self._cache_dir)
            # FastEmbed returns a generator of numpy arrays (float32).
            # `parallel=0` -> use all cores; `None` -> single process;
            # positive int -> N worker processes.
            kwargs: dict[str, Any] = {"batch_size": batch_size}
            if parallel is not None:
                kwargs["parallel"] = parallel
            vectors = mdl.embed(texts, **kwargs)
            return [vec.tolist() for vec in vectors]

        try:
            embeddings = await asyncio.to_thread(_embed_sync)
        except Exception as exc:
            logger.error(
                "local_embedding_failed",
                model=model_name,
                count=len(texts),
                error=str(exc),
            )
            raise LLMProviderError(
                f"Local embedding failed: {exc}",
                provider="local",
            ) from exc

        if embeddings and len(embeddings[0]) != self._expected_dim:
            logger.warning(
                "local_embedding_dim_mismatch",
                model=model_name,
                expected=self._expected_dim,
                actual=len(embeddings[0]),
            )

        logger.debug(
            "local_embedding_batch_complete",
            model=model_name,
            count=len(texts),
            dim=len(embeddings[0]) if embeddings else 0,
            batch_size=batch_size,
            parallel=parallel,
        )
        return embeddings

    # ── Chat / completion — intentionally unsupported ──────────────────────

    async def generate(
        self,
        messages: List[Union[LLMMessage, dict]],
        config: Optional[LLMConfig] = None,
    ) -> LLMResponse:
        raise NotImplementedError(
            "LocalEmbeddingProvider is embedding-only.  "
            "Configure DEFAULT_LLM_PROVIDER to a chat provider "
            "(gemini, openai, anthropic, perplexity, mistral)."
        )

    async def stream(
        self,
        messages: List[Union[LLMMessage, dict]],
        config: Optional[LLMConfig] = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        raise NotImplementedError(
            "LocalEmbeddingProvider is embedding-only."
        )
        # The following yield keeps the type-checker happy — unreachable.
        yield  # type: ignore[misc]  # pragma: no cover

    # ── Health / lifecycle ─────────────────────────────────────────────────

    async def health_check(self) -> bool:
        """True iff the ONNX session can produce at least one vector."""
        try:
            vecs = await self.generate_embeddings(["health"])
            return bool(vecs and len(vecs[0]) == self._expected_dim)
        except Exception as exc:
            logger.warning("local_embedding_health_check_failed", error=str(exc))
            return False

    async def close(self) -> None:
        """
        FastEmbed has no network connections to release; the ONNX session
        is owned by the module-level cache and lives for the process lifetime.
        """
        return None


# ═══════════════════════════════════════════════════════════════════════════
# Self-registration
# ═══════════════════════════════════════════════════════════════════════════

LLMRegistry.register("local", LocalEmbeddingProvider)
