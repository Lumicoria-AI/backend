"""
Weaviate Vector Store Implementation for Lumicoria.ai

This module provides a comprehensive integration with Weaviate as a vector store
for storing and retrieving document embeddings and context data.
"""

import uuid
import json
import weaviate
from typing import Dict, Any, List, Optional, Union
from datetime import datetime
import structlog
import asyncio
from concurrent.futures import ThreadPoolExecutor

from ...core.config import settings

logger = structlog.get_logger(__name__)

class WeaviateDocumentStore:
    """
    Vector store implementation using Weaviate for document storage and retrieval.
    This class handles document indexing, searching, and metadata management.
    """
    
    def __init__(
        self,
        collection_name: str = "Documents",
        url: Optional[str] = None,
        api_key: Optional[str] = None,
        embedding_dimension: int = 1536,
    ):
        """
        Initialize the Weaviate document store.
        
        Args:
            collection_name: Name of the collection to store documents
            url: Weaviate server URL
            api_key: Weaviate API key for authentication
            embedding_dimension: Dimension of embedding vectors
        """
        self.collection_name = collection_name
        self.url = url or settings.db.VECTOR_STORE_URL
        self.api_key = api_key or settings.db.VECTOR_STORE_API_KEY
        self.embedding_dimension = embedding_dimension
        self.client = None
        self.executor = ThreadPoolExecutor(max_workers=10)
        
    async def connect(self) -> None:
        """
        Connect to Weaviate server and set up the collection schema.
        """
        try:
            # Connect to Weaviate
            auth_client = None
            if self.api_key:
                auth_client = weaviate.AuthApiKey(api_key=self.api_key)
            
            def _create_client():
                return weaviate.Client(
                    url=self.url,
                    auth_client_secret=auth_client,
                    timeout_config=(5, 60)  # (connect_timeout, read_timeout)
                )
            
            # Run client creation in executor to avoid blocking
            self.client = await asyncio.get_event_loop().run_in_executor(
                self.executor, _create_client
            )
            
            # Check if collection exists
            if not self._collection_exists():
                await self._create_collection()
                
            logger.info("Connected to Weaviate", url=self.url, collection=self.collection_name)
        
        except Exception as e:
            logger.error("Failed to connect to Weaviate", error=str(e))
            raise
            
    def _collection_exists(self) -> bool:
        """Check if collection exists in Weaviate."""
        try:
            schema = self.client.schema.get()
            classes = schema.get("classes", [])
            return any(cls["class"] == self.collection_name for cls in classes)
        except Exception as e:
            logger.error("Error checking if collection exists", error=str(e))
            return False
    
    async def _create_collection(self) -> None:
        """Create collection schema in Weaviate."""
        try:
            class_obj = {
                "class": self.collection_name,
                "description": "Document collection for Lumicoria RAG system",
                "vectorizer": "none",  # We'll provide our own vectors
                "properties": [
                    {
                        "name": "content",
                        "dataType": ["text"],
                        "description": "The text content of the document chunk"
                    },
                    {
                        "name": "metadata",
                        "dataType": ["text"],
                        "description": "JSON string of document metadata"
                    },
                    {
                        "name": "document_id",
                        "dataType": ["string"],
                        "description": "Original document identifier",
                        "indexFilterable": True,
                        "indexSearchable": True
                    },
                    {
                        "name": "chunk_id",
                        "dataType": ["int"],
                        "description": "Chunk index within the document",
                        "indexFilterable": True
                    },
                    {
                        "name": "source",
                        "dataType": ["string"],
                        "description": "Source of the document (e.g., upload, drive, chat)",
                        "indexFilterable": True,
                        "indexSearchable": True
                    },
                    {
                        "name": "user_id",
                        "dataType": ["string"],
                        "description": "ID of the user who owns this document",
                        "indexFilterable": True
                    },
                    {
                        "name": "organization_id",
                        "dataType": ["string"],
                        "description": "ID of the organization that owns this document",
                        "indexFilterable": True
                    },
                    {
                        "name": "created_at",
                        "dataType": ["date"],
                        "description": "Timestamp when this document was added",
                        "indexFilterable": True
                    },
                    {
                        "name": "mime_type",
                        "dataType": ["string"],
                        "description": "MIME type of the original document",
                        "indexFilterable": True
                    },
                    {
                        "name": "filename",
                        "dataType": ["string"],
                        "description": "Original filename",
                        "indexSearchable": True
                    },
                    {
                        "name": "title",
                        "dataType": ["string"],
                        "description": "Document title",
                        "indexSearchable": True
                    },
                    {
                        "name": "tags",
                        "dataType": ["string[]"],
                        "description": "List of tags associated with this document",
                        "indexFilterable": True,
                        "indexSearchable": True
                    }
                ]
            }
            
            def _create_class():
                return self.client.schema.create_class(class_obj)
            
            await asyncio.get_event_loop().run_in_executor(
                self.executor, _create_class
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
        """
        Add documents to the vector store with their embeddings.
        
        Args:
            texts: List of text chunks to add
            embeddings: List of embedding vectors for each text chunk
            metadatas: Optional list of metadata dictionaries
            ids: Optional list of document IDs
            
        Returns:
            List of document IDs
        """
        if not self.client:
            await self.connect()
            
        if not all(len(emb) == self.embedding_dimension for emb in embeddings):
            logger.error(
                "Embedding dimension mismatch", 
                expected=self.embedding_dimension,
                found=[len(emb) for emb in embeddings]
            )
            raise ValueError("Embedding dimension mismatch")
            
        # Ensure metadatas and ids are properly initialized
        if metadatas is None:
            metadatas = [{} for _ in texts]
            
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]
            
        # Process in batches to avoid overwhelming the server
        batch_size = 50
        results_ids = []
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            batch_embeddings = embeddings[i:i+batch_size]
            batch_metadatas = metadatas[i:i+batch_size]
            batch_ids = ids[i:i+batch_size]
            
            # Create batch objects
            with self.client.batch as batch:
                batch.batch_size = min(len(batch_texts), 50)
                
                for text, embedding, metadata, doc_id in zip(batch_texts, batch_embeddings, batch_metadatas, batch_ids):
                    # Ensure metadata is json-serializable
                    metadata_str = json.dumps(metadata)
                    
                    # Get standard metadata fields
                    source = metadata.get("source", "upload")
                    user_id = metadata.get("user_id", "")
                    org_id = metadata.get("organization_id", "")
                    doc_id_main = metadata.get("document_id", "")
                    chunk_id = metadata.get("chunk_id", 0)
                    created_at = metadata.get("created_at", datetime.utcnow().isoformat())
                    mime_type = metadata.get("mime_type", "text/plain")
                    filename = metadata.get("filename", "")
                    title = metadata.get("title", "")
                    tags = metadata.get("tags", [])
                    
                    properties = {
                        "content": text,
                        "metadata": metadata_str,
                        "document_id": doc_id_main,
                        "chunk_id": chunk_id,
                        "source": source,
                        "user_id": user_id,
                        "organization_id": org_id,
                        "created_at": created_at,
                        "mime_type": mime_type,
                        "filename": filename,
                        "title": title,
                        "tags": tags
                    }
                    
                    # Add object to batch
                    batch.add_data_object(
                        data_object=properties,
                        class_name=self.collection_name,
                        uuid=doc_id,
                        vector=embedding
                    )
                    
                    results_ids.append(doc_id)
                    
            logger.info(
                "Added documents to Weaviate batch",
                count=len(batch_texts),
                collection=self.collection_name
            )
            
        return results_ids
    
    async def similarity_search(
        self,
        query_vector: List[float],
        k: int = 4,
        filters: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents based on vector similarity.
        
        Args:
            query_vector: Embedding vector to search with
            k: Number of results to return
            filters: Optional filters for search
            
        Returns:
            List of document dictionaries with content, metadata, and score
        """
        if not self.client:
            await self.connect()
            
        # Construct filter if provided
        where_filter = None
        if filters:
            where_filter = self._build_where_filter(filters)
            
        def _vector_search():
            query_result = (
                self.client.query
                .get(self.collection_name, ["content", "metadata", "document_id", "chunk_id", "source"])
                .with_near_vector({
                    "vector": query_vector
                })
                .with_additional(["id", "distance"])
                .with_limit(k)
            )
            
            # Apply filter if provided
            if where_filter:
                query_result = query_result.with_where(where_filter)
                
            result = query_result.do()
            return result
            
        # Execute search in executor
        result = await asyncio.get_event_loop().run_in_executor(
            self.executor, _vector_search
        )
        
        # Process results
        try:
            objects = result["data"]["Get"][self.collection_name]
            
            search_results = []
            for obj in objects:
                # Calculate score (1 - distance) so higher is better
                distance = obj["_additional"]["distance"]
                score = 1.0 - distance
                
                # Parse metadata
                metadata = json.loads(obj["metadata"]) if obj["metadata"] else {}
                
                # Include original fields
                metadata["document_id"] = obj["document_id"]
                metadata["chunk_id"] = obj["chunk_id"]
                metadata["source"] = obj["source"]
                
                # Create result object
                search_results.append({
                    "id": obj["_additional"]["id"],
                    "content": obj["content"],
                    "metadata": metadata,
                    "score": score
                })
                
            return search_results
            
        except (KeyError, json.JSONDecodeError) as e:
            logger.error("Error processing search results", error=str(e), result=result)
            return []
    
    def _build_where_filter(self, filters: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a Weaviate where filter from a dictionary of filters.
        Supports basic operators like eq, ne, gt, gte, lt, lte, in, nin.
        
        Args:
            filters: Dictionary of filters {field: value} or {field: {operator: value}}
            
        Returns:
            Weaviate where filter object
        """
        operator_map = {
            "eq": "Equal",
            "ne": "NotEqual", 
            "gt": "GreaterThan",
            "gte": "GreaterThanEqual",
            "lt": "LessThan", 
            "lte": "LessThanEqual",
            "in": "ContainsAny",
            "nin": "NotContainsAny",
            "contains": "Like",
            "not_contains": "NotLike"
        }
        
        def build_operands(field: str, value: Any) -> Dict[str, Any]:
            # If value is a dict with an operator
            if isinstance(value, dict):
                operands = []
                for op, val in value.items():
                    if op not in operator_map:
                        logger.warning(f"Unsupported operator {op} for field {field}")
                        continue
                        
                    operands.append({
                        "operator": operator_map[op],
                        "path": [field],
                        "valueType": "string" if isinstance(val, str) else "number",
                        "value": val
                    })
                return {"operands": operands, "operator": "And"}
                
            # Simple equality
            return {
                "operator": "Equal",
                "path": [field],
                "valueType": "string" if isinstance(value, str) else "number",
                "value": value
            }
            
        where_filter = {"operator": "And", "operands": []}
        
        for field, value in filters.items():
            where_filter["operands"].append(build_operands(field, value))
            
        return where_filter
    
    async def get_documents(
        self,
        filters: Optional[Dict[str, Any]] = None,
        limit: int = 100,
        offset: int = 0,
        sort_by: Optional[str] = None,
        sort_order: str = "desc"
    ) -> List[Dict[str, Any]]:
        """
        Get documents from the vector store based on filters.
        
        Args:
            filters: Optional filters to select documents
            limit: Maximum number of documents to return
            offset: Offset for pagination
            sort_by: Field to sort by (e.g., created_at, title)
            sort_order: Sort order (asc or desc)
            
        Returns:
            List of document dictionaries with content and metadata
        """
        if not self.client:
            await self.connect()
            
        # Construct filter if provided
        where_filter = None
        if filters:
            where_filter = self._build_where_filter(filters)
            
        def _get_documents():
            query = (
                self.client.query
                .get(self.collection_name, ["content", "metadata", "document_id", "chunk_id", 
                                          "source", "title", "created_at", "tags"])
                .with_additional(["id"])
                .with_limit(limit)
                .with_offset(offset)
            )
            
            # Apply sort if specified
            if sort_by:
                query = query.with_sort([{
                    "path": [sort_by],
                    "order": sort_order.lower()
                }])
            
            # Apply filter if provided
            if where_filter:
                query = query.with_where(where_filter)
                
            result = query.do()
            return result
            
        # Execute query in executor
        result = await asyncio.get_event_loop().run_in_executor(
            self.executor, _get_documents
        )
        
        # Process results
        try:
            objects = result["data"]["Get"][self.collection_name]
            
            documents = []
            for obj in objects:
                # Parse metadata
                metadata = json.loads(obj["metadata"]) if obj["metadata"] else {}
                
                # Include original fields
                metadata["document_id"] = obj["document_id"]
                metadata["chunk_id"] = obj["chunk_id"]
                metadata["source"] = obj["source"]
                metadata["title"] = obj["title"]
                metadata["created_at"] = obj["created_at"]
                if "tags" in obj:
                    metadata["tags"] = obj["tags"]
                
                # Create result object
                documents.append({
                    "id": obj["_additional"]["id"],
                    "content": obj["content"],
                    "metadata": metadata
                })
                
            return documents
            
        except (KeyError, json.JSONDecodeError) as e:
            logger.error("Error processing document results", error=str(e), result=result)
            return []
    
    async def delete_documents(
        self,
        ids: Optional[List[str]] = None,
        filters: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Delete documents from the vector store.
        
        Args:
            ids: Optional list of document IDs to delete
            filters: Optional filters to select documents to delete
            
        Returns:
            True if deletion was successful
        """
        if not self.client:
            await self.connect()
            
        try:            # Delete by IDs if provided
            if ids:
                for doc_id in ids:
                    def _delete_by_id(id_to_delete):
                        self.client.data_object.delete(
                            uuid=id_to_delete,
                            class_name=self.collection_name
                        )
                        
                    await asyncio.get_event_loop().run_in_executor(
                        self.executor,
                        lambda: _delete_by_id(doc_id)
                    )
                logger.info("Deleted documents by IDs", count=len(ids))
                return True
                
            # Delete by filters if provided
            elif filters:
                where_filter = self._build_where_filter(filters)
                
                def _delete_by_filter():
                    self.client.batch.delete_objects(
                        class_name=self.collection_name,
                        where=where_filter
                    )
                
                await asyncio.get_event_loop().run_in_executor(
                    self.executor, _delete_by_filter
                )
                
                logger.info("Deleted documents by filter", filters=filters)
                return True
                
            return False
            
        except Exception as e:
            logger.error("Error deleting documents", error=str(e))
            return False
      async def get_document_count(
        self,
        filters: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Get the total count of documents matching the filters.
        
        Args:
            filters: Optional filters to count matching documents
            
        Returns:
            Total number of matching documents
        """
        if not self.client:
            await self.connect()
            
        where_filter = None
        if filters:
            where_filter = self._build_where_filter(filters)
            
        def _get_count():
            query = (
                self.client.query
                .aggregate(self.collection_name)
                .with_meta_count()
            )
            
            if where_filter:
                query = query.with_where(where_filter)
                
            result = query.do()
            return result
            
        result = await asyncio.get_event_loop().run_in_executor(
            self.executor, _get_count
        )
        
        try:
            return result["data"]["Aggregate"][self.collection_name][0]["meta"]["count"]
        except (KeyError, IndexError):
            logger.error("Error getting document count", result=result)
            return 0
            
    async def batch_add_documents(self, documents: List[Dict[str, Any]], batch_size: int = 100):
        """
        Add multiple documents in batches.
        
        Args:
            documents: List of document dictionaries with content and metadata
            batch_size: Number of documents to process in each batch
        """
        if not self.client:
            await self.connect()
            
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i + batch_size]
            
            def _batch_import():
                with self.client.batch as batch_context:
                    # Configure batch
                    batch_context.batch_size = batch_size
                    batch_context.dynamic = True
                    
                    # Add each document
                    for doc in batch:
                        properties = {
                            "content": doc["content"],
                            "metadata": json.dumps(doc.get("metadata", {})),
                            "document_id": doc["metadata"].get("document_id", str(uuid.uuid4())),
                            "chunk_id": doc["metadata"].get("chunk_id", 0),
                            "source": doc["metadata"].get("source", "unknown"),
                            "title": doc["metadata"].get("title", ""),
                            "created_at": doc["metadata"].get("created_at", datetime.utcnow().isoformat()),
                            "tags": doc["metadata"].get("tags", [])
                        }
                        
                        # Add document to batch
                        batch_context.add_data_object(
                            data_object=properties,
                            class_name=self.collection_name
                        )
            
            # Execute batch in executor
            await asyncio.get_event_loop().run_in_executor(
                self.executor, _batch_import
            )
    
    async def disconnect(self) -> None:
        """Disconnect from Weaviate server."""
        self.client = None
        self.executor.shutdown(wait=False)
        logger.info("Disconnected from Weaviate")

# Create a singleton instance
weaviate_store = WeaviateDocumentStore(
    collection_name=settings.db.VECTOR_STORE_COLLECTION,
    embedding_dimension=settings.db.VECTOR_STORE_DIMENSION
)

