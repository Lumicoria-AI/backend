"""
Vector store registry for Lumicoria.ai.

This module provides a thin factory that returns the correct vector store
implementation based on configuration. Imports are lazy to avoid hard
dependencies when services are not enabled.
"""

from __future__ import annotations

from typing import Optional
import structlog

from ...core.config import settings

logger = structlog.get_logger(__name__)

_vector_store = None


def _load_weaviate():
    from .weaviate import WeaviateDocumentStore  # type: ignore
    return WeaviateDocumentStore


def _load_qdrant():
    from .qdrant import QdrantDocumentStore  # type: ignore
    return QdrantDocumentStore


def _load_chroma():
    from .chroma import ChromaDocumentStore  # type: ignore
    return ChromaDocumentStore


def get_vector_store():
    """
    Return a singleton vector store implementation.

    Supported types: weaviate, qdrant, chroma
    """
    global _vector_store
    if _vector_store is not None:
        return _vector_store

    store_type = settings.db.VECTOR_STORE_TYPE.lower()
    if store_type == "weaviate":
        Store = _load_weaviate()
        _vector_store = Store(
            collection_name=settings.db.VECTOR_STORE_COLLECTION,
            embedding_dimension=settings.db.VECTOR_STORE_DIMENSION,
        )
    elif store_type == "qdrant":
        Store = _load_qdrant()
        _vector_store = Store(
            collection_name=settings.db.VECTOR_STORE_COLLECTION,
            embedding_dimension=settings.db.VECTOR_STORE_DIMENSION,
        )
    elif store_type == "chroma":
        Store = _load_chroma()
        _vector_store = Store(
            collection_name=settings.db.VECTOR_STORE_COLLECTION,
            embedding_dimension=settings.db.VECTOR_STORE_DIMENSION,
        )
    else:
        raise ValueError(f"Unsupported vector store type: {store_type}")

    logger.info("Vector store selected", type=store_type)
    return _vector_store

