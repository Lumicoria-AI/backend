"""
ChromaDB Vector Store Implementation for Lumicoria.ai

Provides a Chroma-backed document store compatible with the existing
RAG pipeline interfaces (add_documents, similarity_search, get_documents, etc.).
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
from urllib.parse import urlparse
import asyncio
import structlog
import uuid

from ...core.config import settings

logger = structlog.get_logger(__name__)

try:
    import chromadb
except Exception:  # pragma: no cover - optional dependency
    chromadb = None


class ChromaDocumentStore:
    def __init__(
        self,
        collection_name: str = "documents",
        url: Optional[str] = None,
        embedding_dimension: int = 1536,
    ):
        self.collection_name = collection_name
        self.url = url or settings.db.VECTOR_STORE_URL
        self.embedding_dimension = embedding_dimension
        self.client = None
        self.collection = None

    async def connect(self) -> None:
        if chromadb is None:
            raise ImportError("chromadb is not installed")

        def _connect_sync():
            parsed = urlparse(self.url) if self.url else None
            if parsed and parsed.scheme in ("http", "https"):
                host = parsed.hostname or "localhost"
                port = parsed.port or 8000
                client = chromadb.HttpClient(
                    host=host,
                    port=port,
                    ssl=(parsed.scheme == "https"),
                )
            else:
                # Local persistent path
                path = parsed.path if parsed else "./chroma_data"
                client = chromadb.PersistentClient(path=path)

            collection = client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            return client, collection

        self.client, self.collection = await asyncio.to_thread(_connect_sync)
        logger.info("Connected to Chroma", collection=self.collection_name)

    async def disconnect(self) -> None:
        self.client = None
        self.collection = None
        logger.info("Disconnected from Chroma")

    def _ensure_collection(self) -> None:
        if not self.client or not self.collection:
            raise RuntimeError("Chroma client not initialized")

    async def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if not self.client or not self.collection:
            await self.connect()
        self._ensure_collection()

        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]

        def _add_sync():
            self.collection.add(
                ids=ids,
                documents=texts,
                embeddings=embeddings,
                metadatas=metadatas,
            )

        await asyncio.to_thread(_add_sync)
        return ids

    async def similarity_search(
        self,
        query_vector: List[float],
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.client or not self.collection:
            await self.connect()
        self._ensure_collection()

        def _query_sync():
            return self.collection.query(
                query_embeddings=[query_vector],
                n_results=k,
                where=filters,
                include=["documents", "metadatas", "distances"],
            )

        result = await asyncio.to_thread(_query_sync)

        docs = result.get("documents", [[]])[0]
        metas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        formatted = []
        for doc, meta, dist in zip(docs, metas, distances):
            score = 1.0 - dist if dist is not None else 0.0
            formatted.append({
                "id": meta.get("id") if isinstance(meta, dict) else None,
                "content": doc or "",
                "metadata": meta or {},
                "score": score,
            })
        return formatted

    async def get_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: Optional[str] = None,
        sort_order: str = "desc",
    ) -> List[Dict[str, Any]]:
        if not self.client or not self.collection:
            await self.connect()
        self._ensure_collection()

        def _get_sync():
            return self.collection.get(
                where=filters,
                limit=limit,
                offset=offset,
                include=["documents", "metadatas"],
            )

        result = await asyncio.to_thread(_get_sync)
        documents = result.get("documents", [])
        metadatas = result.get("metadatas", [])
        ids = result.get("ids", [])

        formatted = []
        for doc_id, doc, meta in zip(ids, documents, metadatas):
            formatted.append({
                "id": doc_id,
                "content": doc or "",
                "metadata": meta or {},
            })
        return formatted

    async def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.client or not self.collection:
            await self.connect()
        self._ensure_collection()

        def _delete_sync():
            if ids:
                self.collection.delete(ids=ids)
                return True
            if filters:
                self.collection.delete(where=filters)
                return True
            return False

        return await asyncio.to_thread(_delete_sync)

    async def get_document_count(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        if not self.client or not self.collection:
            await self.connect()
        self._ensure_collection()

        def _count_sync():
            if not filters:
                return self.collection.count()
            result = self.collection.get(where=filters, include=["ids"])
            return len(result.get("ids", []))

        return await asyncio.to_thread(_count_sync)

    async def batch_add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        if not self.client or not self.collection:
            await self.connect()
        self._ensure_collection()

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [d["content"] for d in batch]
            metadatas = [d.get("metadata", {}) for d in batch]
            embeddings = [d.get("embedding") for d in batch]
            if not all(embeddings):
                raise ValueError("Embeddings required for Chroma batch add")
            await self.add_documents(texts=texts, embeddings=embeddings, metadatas=metadatas)
