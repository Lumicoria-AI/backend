"""
Qdrant Vector Store Implementation for Lumicoria.ai

Provides a Qdrant-backed document store compatible with the existing
RAG pipeline interfaces (add_documents, similarity_search, get_documents, etc.).
"""

from __future__ import annotations

from typing import Dict, Any, List, Optional
import asyncio
import structlog
import uuid

from ...core.config import settings

logger = structlog.get_logger(__name__)

try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models as qmodels
except Exception:  # pragma: no cover - optional dependency
    QdrantClient = None
    qmodels = None


class QdrantDocumentStore:
    def __init__(
        self,
        collection_name: str = "documents",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_dimension: int = 1536,
    ):
        self.collection_name = collection_name
        self.url = url or settings.db.VECTOR_STORE_URL
        self.api_key = api_key or settings.db.VECTOR_STORE_API_KEY
        self.embedding_dimension = embedding_dimension
        self.client: Optional[QdrantClient] = None

    async def connect(self) -> None:
        if QdrantClient is None or qmodels is None:
            raise ImportError("qdrant-client is not installed")

        def _connect_sync():
            client = QdrantClient(url=self.url, api_key=self.api_key, timeout=60)
            collections = client.get_collections().collections
            if not any(c.name == self.collection_name for c in collections):
                client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=qmodels.VectorParams(
                        size=self.embedding_dimension,
                        distance=qmodels.Distance.COSINE,
                    ),
                )
            return client

        self.client = await asyncio.to_thread(_connect_sync)
        logger.info("Connected to Qdrant", collection=self.collection_name)

    async def disconnect(self) -> None:
        if self.client:
            await asyncio.to_thread(self.client.close)
            self.client = None
            logger.info("Disconnected from Qdrant")

    def _ensure_client(self) -> None:
        if not self.client:
            raise RuntimeError("Qdrant client not initialized")

    def _build_payload(self, text: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        payload = {
            "content": text,
            "metadata": metadata or {},
        }
        # Promote common fields for filtering
        for key in [
            "source",
            "user_id",
            "organization_id",
            "document_id",
            "chunk_id",
            "created_at",
            "mime_type",
            "filename",
            "title",
            "tags",
            "url",
            "conversation_id",
        ]:
            if key in (metadata or {}):
                payload[key] = metadata[key]
        return payload

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[qmodels.Filter]:
        if not filters:
            return None
        must = []
        for key, value in filters.items():
            if isinstance(value, list):
                must.append(
                    qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchAny(any=value),
                    )
                )
            else:
                must.append(
                    qmodels.FieldCondition(
                        key=key,
                        match=qmodels.MatchValue(value=value),
                    )
                )
        return qmodels.Filter(must=must)

    async def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        if not self.client:
            await self.connect()
        self._ensure_client()

        metadatas = metadatas or [{} for _ in texts]
        ids = ids or [str(uuid.uuid4()) for _ in texts]

        if not all(len(emb) == self.embedding_dimension for emb in embeddings):
            raise ValueError("Embedding dimension mismatch")

        points = []
        for text, emb, metadata, doc_id in zip(texts, embeddings, metadatas, ids):
            payload = self._build_payload(text, metadata)
            points.append(qmodels.PointStruct(id=doc_id, vector=emb, payload=payload))

        await asyncio.to_thread(
            self.client.upsert,
            collection_name=self.collection_name,
            points=points,
        )
        return ids

    async def similarity_search(
        self,
        query_vector: List[float],
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if not self.client:
            await self.connect()
        self._ensure_client()

        qfilter = self._build_filter(filters)
        results = await asyncio.to_thread(
            self.client.search,
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=k,
            query_filter=qfilter,
            with_payload=True,
        )

        formatted = []
        for res in results:
            payload = res.payload or {}
            formatted.append({
                "id": str(res.id),
                "content": payload.get("content", ""),
                "metadata": payload.get("metadata", {}),
                "score": res.score,
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
        if not self.client:
            await self.connect()
        self._ensure_client()

        qfilter = self._build_filter(filters)
        points, _ = await asyncio.to_thread(
            self.client.scroll,
            collection_name=self.collection_name,
            scroll_filter=qfilter,
            limit=limit,
            offset=offset,
            with_payload=True,
        )

        docs = []
        for point in points:
            payload = point.payload or {}
            docs.append({
                "id": str(point.id),
                "content": payload.get("content", ""),
                "metadata": payload.get("metadata", {}),
            })
        return docs

    async def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.client:
            await self.connect()
        self._ensure_client()

        if ids:
            await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=qmodels.PointIdsList(points=ids),
            )
            return True
        if filters:
            qfilter = self._build_filter(filters)
            await asyncio.to_thread(
                self.client.delete,
                collection_name=self.collection_name,
                points_selector=qmodels.FilterSelector(filter=qfilter),
            )
            return True
        return False

    async def get_document_count(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        if not self.client:
            await self.connect()
        self._ensure_client()

        qfilter = self._build_filter(filters)
        result = await asyncio.to_thread(
            self.client.count,
            collection_name=self.collection_name,
            count_filter=qfilter,
        )
        return int(result.count)

    async def batch_add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        if not self.client:
            await self.connect()
        self._ensure_client()

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            texts = [d["content"] for d in batch]
            metadatas = [d.get("metadata", {}) for d in batch]
            embeddings = [d.get("embedding") for d in batch]
            if not all(embeddings):
                raise ValueError("Embeddings required for Qdrant batch add")
            await self.add_documents(texts=texts, embeddings=embeddings, metadatas=metadatas)
