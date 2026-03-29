"""
Weaviate Vector Store Implementation for Lumicoria.ai (v4 client)

This module provides a comprehensive integration with Weaviate as a vector store
for storing and retrieving document embeddings and context data.
"""

import uuid
import json
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Property, DataType, Configure
from weaviate.classes.query import MetadataQuery, Filter
from weaviate.classes.data import DataObject
from typing import Dict, Any, List, Optional
from datetime import datetime, timezone
import structlog

from ...core.config import settings

logger = structlog.get_logger(__name__)


class WeaviateDocumentStore:
    """
    Vector store implementation using Weaviate v4 for document storage and retrieval.
    """

    def __init__(
        self,
        collection_name: str = "Documents",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_dimension: int = 768,
    ):
        self.collection_name = collection_name
        self.url = url or settings.db.VECTOR_STORE_URL
        self.api_key = api_key or settings.db.VECTOR_STORE_API_KEY
        self.embedding_dimension = embedding_dimension
        self.client: Optional[weaviate.WeaviateClient] = None

    async def connect(self) -> None:
        """Connect to Weaviate server and set up the collection schema."""
        try:
            if self.api_key:
                self.client = weaviate.connect_to_custom(
                    http_host=self.url.replace("http://", "").replace("https://", "").split(":")[0],
                    http_port=int(self.url.split(":")[-1]) if ":" in self.url.split("//")[-1] else 8080,
                    http_secure=self.url.startswith("https"),
                    grpc_host=self.url.replace("http://", "").replace("https://", "").split(":")[0],
                    grpc_port=50051,
                    grpc_secure=False,
                    auth_credentials=Auth.api_key(self.api_key),
                )
            else:
                # Parse host and port from URL
                url_no_scheme = self.url.replace("http://", "").replace("https://", "")
                parts = url_no_scheme.split(":")
                host = parts[0]
                port = int(parts[1]) if len(parts) > 1 else 8080

                self.client = weaviate.connect_to_custom(
                    http_host=host,
                    http_port=port,
                    http_secure=self.url.startswith("https"),
                    grpc_host=host,
                    grpc_port=50051,
                    grpc_secure=False,
                )

            # Create collection if it doesn't exist
            if not self.client.collections.exists(self.collection_name):
                await self._create_collection()

            logger.info("Connected to Weaviate", url=self.url, collection=self.collection_name)

        except Exception as e:
            logger.error("Failed to connect to Weaviate", error=str(e))
            raise

    async def _create_collection(self) -> None:
        """Create collection schema in Weaviate v4."""
        try:
            self.client.collections.create(
                name=self.collection_name,
                description="Document collection for Lumicoria RAG system",
                vectorizer_config=Configure.Vectorizer.none(),
                properties=[
                    Property(name="content", data_type=DataType.TEXT, description="Text content of the chunk"),
                    Property(name="metadata", data_type=DataType.TEXT, description="JSON string of metadata"),
                    Property(name="document_id", data_type=DataType.TEXT, description="Original document ID",
                             index_filterable=True, index_searchable=True),
                    Property(name="chunk_id", data_type=DataType.INT, description="Chunk index",
                             index_filterable=True),
                    Property(name="source", data_type=DataType.TEXT, description="Source (upload, drive, chat)",
                             index_filterable=True, index_searchable=True),
                    Property(name="user_id", data_type=DataType.TEXT, description="Owner user ID",
                             index_filterable=True),
                    Property(name="organization_id", data_type=DataType.TEXT, description="Owner org ID",
                             index_filterable=True),
                    Property(name="created_at", data_type=DataType.DATE, description="Created timestamp",
                             index_filterable=True),
                    Property(name="mime_type", data_type=DataType.TEXT, description="MIME type",
                             index_filterable=True),
                    Property(name="filename", data_type=DataType.TEXT, description="Original filename",
                             index_searchable=True),
                    Property(name="title", data_type=DataType.TEXT, description="Document title",
                             index_searchable=True),
                    Property(name="tags", data_type=DataType.TEXT_ARRAY, description="Tags",
                             index_filterable=True, index_searchable=True),
                ],
            )
            logger.info("Created Weaviate collection", collection=self.collection_name)

        except Exception as e:
            logger.error("Failed to create collection schema", error=str(e))
            raise

    async def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[str]:
        """Add documents to the vector store with their embeddings."""
        if not self.client:
            await self.connect()

        if metadatas is None:
            metadatas = [{} for _ in texts]
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        collection = self.client.collections.get(self.collection_name)
        results_ids = []
        batch_size = 50

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_embeddings = embeddings[i:i + batch_size]
            batch_metadatas = metadatas[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]

            data_objects = []
            for text, embedding, metadata, doc_id in zip(batch_texts, batch_embeddings, batch_metadatas, batch_ids):
                metadata_str = json.dumps(metadata)
                created_at = metadata.get("created_at", datetime.now(timezone.utc).isoformat())
                # Ensure RFC3339 compliance — append Z if no timezone info
                if created_at and not created_at.endswith(("Z", "+00:00")) and "+" not in created_at[19:]:
                    created_at = created_at + "Z"

                properties = {
                    "content": text,
                    "metadata": metadata_str,
                    "document_id": metadata.get("document_id", ""),
                    "chunk_id": metadata.get("chunk_id", 0),
                    "source": metadata.get("source", "upload"),
                    "user_id": metadata.get("user_id", ""),
                    "organization_id": metadata.get("organization_id", ""),
                    "created_at": created_at,
                    "mime_type": metadata.get("mime_type", "text/plain"),
                    "filename": metadata.get("filename", ""),
                    "title": metadata.get("title", ""),
                    "tags": metadata.get("tags", []),
                }

                data_objects.append(DataObject(
                    properties=properties,
                    uuid=doc_id,
                    vector=embedding,
                ))
                results_ids.append(doc_id)

            # Batch insert
            collection.data.insert_many(data_objects)

            logger.info(
                "Added documents to Weaviate batch",
                count=len(batch_texts),
                collection=self.collection_name,
            )

        return results_ids

    async def similarity_search(
        self,
        query_vector: List[float],
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar documents based on vector similarity."""
        if not self.client:
            await self.connect()

        collection = self.client.collections.get(self.collection_name)

        wv_filter = self._build_filter(filters) if filters else None

        try:
            response = collection.query.near_vector(
                near_vector=query_vector,
                limit=k,
                filters=wv_filter,
                return_metadata=MetadataQuery(distance=True),
                return_properties=["content", "metadata", "document_id", "chunk_id", "source"],
            )

            search_results = []
            for obj in response.objects:
                distance = obj.metadata.distance if obj.metadata.distance is not None else 0.0
                score = 1.0 - distance

                metadata = json.loads(obj.properties.get("metadata", "{}")) if obj.properties.get("metadata") else {}
                metadata["document_id"] = obj.properties.get("document_id", "")
                metadata["chunk_id"] = obj.properties.get("chunk_id", 0)
                metadata["source"] = obj.properties.get("source", "")

                search_results.append({
                    "id": str(obj.uuid),
                    "content": obj.properties.get("content", ""),
                    "metadata": metadata,
                    "score": score,
                })

            return search_results

        except Exception as e:
            logger.error("Error in similarity search", error=str(e))
            return []

    def _build_filter(self, filters: Dict[str, Any]) -> Optional[Filter]:
        """Build a Weaviate v4 Filter from a dictionary."""
        filter_parts = []

        for field, value in filters.items():
            if isinstance(value, dict):
                for op, val in value.items():
                    if op == "eq":
                        filter_parts.append(Filter.by_property(field).equal(val))
                    elif op == "ne":
                        filter_parts.append(Filter.by_property(field).not_equal(val))
                    elif op == "gt":
                        filter_parts.append(Filter.by_property(field).greater_than(val))
                    elif op == "gte":
                        filter_parts.append(Filter.by_property(field).greater_or_equal(val))
                    elif op == "lt":
                        filter_parts.append(Filter.by_property(field).less_than(val))
                    elif op == "lte":
                        filter_parts.append(Filter.by_property(field).less_or_equal(val))
                    elif op == "in":
                        filter_parts.append(Filter.by_property(field).contains_any(val))
                    elif op == "like":
                        filter_parts.append(Filter.by_property(field).like(val))
            elif isinstance(value, list):
                filter_parts.append(Filter.by_property(field).contains_any(value))
            else:
                filter_parts.append(Filter.by_property(field).equal(value))

        if not filter_parts:
            return None
        if len(filter_parts) == 1:
            return filter_parts[0]

        # Chain with AND
        result = filter_parts[0]
        for f in filter_parts[1:]:
            result = result & f
        return result

    async def get_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: Optional[str] = None,
        sort_order: str = "desc",
    ) -> List[Dict[str, Any]]:
        """Get documents from the vector store based on filters."""
        if not self.client:
            await self.connect()

        collection = self.client.collections.get(self.collection_name)
        wv_filter = self._build_filter(filters) if filters else None

        try:
            response = collection.query.fetch_objects(
                limit=limit,
                offset=offset,
                filters=wv_filter,
                return_metadata=MetadataQuery(creation_time=True),
                return_properties=["content", "metadata", "document_id", "chunk_id",
                                   "source", "title", "created_at", "tags"],
            )

            documents = []
            for obj in response.objects:
                metadata = json.loads(obj.properties.get("metadata", "{}")) if obj.properties.get("metadata") else {}
                metadata["document_id"] = obj.properties.get("document_id", "")
                metadata["chunk_id"] = obj.properties.get("chunk_id", 0)
                metadata["source"] = obj.properties.get("source", "")
                metadata["title"] = obj.properties.get("title", "")
                metadata["created_at"] = obj.properties.get("created_at", "")
                if obj.properties.get("tags"):
                    metadata["tags"] = obj.properties["tags"]

                documents.append({
                    "id": str(obj.uuid),
                    "content": obj.properties.get("content", ""),
                    "metadata": metadata,
                })

            return documents

        except Exception as e:
            logger.error("Error getting documents", error=str(e))
            return []

    async def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Delete documents from the vector store."""
        if not self.client:
            await self.connect()

        collection = self.client.collections.get(self.collection_name)

        try:
            if ids:
                for doc_id in ids:
                    collection.data.delete_by_id(doc_id)
                logger.info("Deleted documents by IDs", count=len(ids))
                return True

            elif filters:
                wv_filter = self._build_filter(filters)
                if wv_filter:
                    collection.data.delete_many(where=wv_filter)
                    logger.info("Deleted documents by filter", filters=filters)
                    return True

            return False

        except Exception as e:
            logger.error("Error deleting documents", error=str(e))
            return False

    async def get_document_count(
        self,
        filters: Optional[Dict[str, Any]] = None,
    ) -> int:
        """Get the total count of documents matching the filters."""
        if not self.client:
            await self.connect()

        collection = self.client.collections.get(self.collection_name)

        try:
            wv_filter = self._build_filter(filters) if filters else None
            result = collection.aggregate.over_all(total_count=True, filters=wv_filter)
            return result.total_count or 0
        except Exception as e:
            logger.error("Error getting document count", error=str(e))
            return 0

    async def batch_add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """Add multiple documents in batches."""
        if not self.client:
            await self.connect()

        collection = self.client.collections.get(self.collection_name)

        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]

            data_objects = []
            for doc in batch:
                meta = doc.get("metadata", {})
                properties = {
                    "content": doc["content"],
                    "metadata": json.dumps(meta),
                    "document_id": meta.get("document_id", str(uuid.uuid4())),
                    "chunk_id": meta.get("chunk_id", 0),
                    "source": meta.get("source", "unknown"),
                    "title": meta.get("title", ""),
                    "created_at": meta.get("created_at", datetime.now(timezone.utc).isoformat()),
                    "tags": meta.get("tags", []),
                }
                data_objects.append(DataObject(properties=properties))

            collection.data.insert_many(data_objects)

    async def disconnect(self) -> None:
        """Disconnect from Weaviate server."""
        if self.client:
            self.client.close()
            self.client = None
        logger.info("Disconnected from Weaviate")


# Create a singleton instance
weaviate_store = WeaviateDocumentStore(
    collection_name=settings.db.VECTOR_STORE_COLLECTION,
    embedding_dimension=settings.db.VECTOR_STORE_DIMENSION,
)
