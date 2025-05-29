"""
Context Service for Lumicoria.ai RAG System

This service manages the retrieval and formatting of context from various sources,
including the vector database, recent conversations, and user-specific data.
"""

import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
import structlog
from datetime import datetime, timedelta
import json

from ..db.vector_stores.weaviate import weaviate_store
from ..services.document_processor import document_processor
from ..ai_models.perplexity import PerplexityClient, create_perplexity_client, create_perplexity_client_async

logger = structlog.get_logger(__name__)

class ContextService:
    """
    Service for retrieving and managing context for RAG from multiple sources.
    """
    def __init__(self, perplexity_client: Optional[PerplexityClient] = None):
        """Initialize the context service."""
        self.perplexity_client = perplexity_client
        
    async def initialize(self):
        """Ensure client is initialized."""
        if not self.perplexity_client:
            self.perplexity_client = await create_perplexity_client_async()
            
    async def get_context_for_query(
        self,
        query: str,
        user_id: str,
        organization_id: Optional[str] = None,
        k: int = 8,
        filters: Optional[Dict[str, Any]] = None,
        include_sources: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Get relevant context for a query from the vector store.
        
        Args:
            query: The user's query
            user_id: User ID for filtering context
            organization_id: Optional organization ID for filtering context
            k: Number of context chunks to retrieve
            filters: Additional filters to apply
            include_sources: List of source types to include (e.g., ["upload", "drive", "chat_history"])
            
        Returns:
            Dictionary with relevant context and metadata
        """
        await self.initialize()
        
        # Generate embedding for the query
        query_embedding = await self.perplexity_client.generate_embeddings(texts=[query])
        if not query_embedding or len(query_embedding) == 0:
            logger.error("Failed to generate query embedding")
            return {"context": [], "error": "Failed to generate query embedding"}
            
        # Build filters
        search_filters = filters or {}
        
        # Add user and org filters
        search_filters["user_id"] = user_id
        if organization_id:
            search_filters["organization_id"] = organization_id
            
        # Add source filter if specified
        if include_sources:
            search_filters["source"] = include_sources
            
        # Search vector store
        try:
            results = await weaviate_store.similarity_search(
                query_vector=query_embedding[0],
                k=k,
                filters=search_filters
            )
            
            # Format results
            formatted_context = self._format_context(results)
            
            return {
                "context": formatted_context,
                "query": query,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error("Error retrieving context", error=str(e))
            return {"context": [], "error": str(e)}
    
    async def add_chat_context(
        self,
        messages: List[Dict[str, Any]],
        user_id: str,
        organization_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store chat history in the vector store for future context.
        
        Args:
            messages: List of chat messages
            user_id: User ID who owns the chat
            organization_id: Optional organization ID
            conversation_id: Optional conversation ID
            
        Returns:
            Processing result
        """
        # Build metadata
        metadata = {
            "user_id": user_id,
            "source": "chat_history",
            "created_at": datetime.utcnow().isoformat()
        }
        
        if organization_id:
            metadata["organization_id"] = organization_id
            
        if conversation_id:
            metadata["conversation_id"] = conversation_id
            metadata["document_id"] = f"chat_{conversation_id}"
            
        # Process chat history
        result = await document_processor.process_chat_history(
            messages=messages,
            metadata=metadata
        )
        
        return {
            "document_id": result.document_id,
            "status": result.status,
            "chunk_count": result.chunk_count,
            "error": result.error
        }
    
    async def add_document_from_url(
        self,
        url: str,
        user_id: str,
        organization_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process and add a document from a URL to the vector store.
        
        Args:
            url: URL to process
            user_id: User ID who owns the document
            organization_id: Optional organization ID
            tags: Optional tags for the document
            title: Optional document title
            
        Returns:
            Processing result
        """
        # Build metadata
        metadata = {
            "user_id": user_id,
            "source": "web",
            "url": url,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if organization_id:
            metadata["organization_id"] = organization_id
            
        if tags:
            metadata["tags"] = tags
            
        if title:
            metadata["title"] = title
            
        # Process URL
        result = await document_processor.process_url(
            url=url,
            metadata=metadata
        )
        
        return {
            "document_id": result.document_id,
            "status": result.status,
            "chunk_count": result.chunk_count,
            "error": result.error
        }
    
    async def add_document_from_text(
        self,
        text: str,
        user_id: str,
        title: Optional[str] = None,
        organization_id: Optional[str] = None,
        source: str = "manual_entry",
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process and add a text document to the vector store.
        
        Args:
            text: Text content to process
            user_id: User ID who owns the document
            title: Optional document title
            organization_id: Optional organization ID
            source: Source of the document
            tags: Optional tags for the document
            
        Returns:
            Processing result
        """
        # Build metadata
        metadata = {
            "user_id": user_id,
            "source": source,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if organization_id:
            metadata["organization_id"] = organization_id
            
        if title:
            metadata["title"] = title
            
        if tags:
            metadata["tags"] = tags
            
        # Process text
        result = await document_processor.process_text(
            text=text,
            metadata=metadata
        )
        
        return {
            "document_id": result.document_id,
            "status": result.status,
            "chunk_count": result.chunk_count,
            "error": result.error
        }
    
    async def delete_user_context(
        self,
        user_id: str,
        source: Optional[str] = None,
        document_id: Optional[str] = None,
        older_than_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Delete context for a specific user.
        
        Args:
            user_id: User ID whose context to delete
            source: Optional source filter
            document_id: Optional document ID filter
            older_than_days: Optional age filter
            
        Returns:
            Result of deletion operation
        """
        filters = {"user_id": user_id}
        
        if source:
            filters["source"] = source
            
        if document_id:
            filters["document_id"] = document_id
            
        if older_than_days:
            cutoff_date = (datetime.utcnow() - timedelta(days=older_than_days)).isoformat()
            filters["created_before"] = cutoff_date
            
        try:
            success = await weaviate_store.delete_documents(filters=filters)
            
            return {
                "success": success,
                "filters": filters            }
            
        except Exception as e:
            logger.error("Error deleting user context", error=str(e), user_id=user_id)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def add_document_from_file(
        self,
        file_path: str,
        user_id: str,
        organization_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        title: Optional[str] = None,
        source: str = "upload"
    ) -> Dict[str, Any]:
        """
        Process and add a document from a file path to the vector store.
        
        Args:
            file_path: Path to the file
            user_id: User ID who owns the document
            organization_id: Optional organization ID
            tags: Optional tags for the document
            title: Optional document title
            source: Source of the document
            
        Returns:
            Processing result
        """
        # Build metadata
        metadata = {
            "user_id": user_id,
            "source": source,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if organization_id:
            metadata["organization_id"] = organization_id
            
        if tags:
            metadata["tags"] = tags
            
        if title:
            metadata["title"] = title
            
        # Process file
        result = await document_processor.process_file(
            file_path=file_path,
            metadata=metadata
        )
        
        return {
            "document_id": result.document_id,
            "status": result.status,
            "chunk_count": result.chunk_count,
            "error": result.error
        }
    
    async def add_document_from_google_drive(
        self,
        drive_file_id: str,
        user_id: str,
        organization_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process and add a document from Google Drive to the vector store.
        
        Args:
            drive_file_id: Google Drive file ID
            user_id: User ID who owns the document
            organization_id: Optional organization ID
            tags: Optional tags for the document
            title: Optional document title
            
        Returns:
            Processing result
        """
        # Build metadata
        metadata = {
            "user_id": user_id,
            "source": "drive",
            "drive_file_id": drive_file_id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if organization_id:
            metadata["organization_id"] = organization_id
            
        if tags:
            metadata["tags"] = tags
            
        if title:
            metadata["title"] = title
        
        # Process Google Drive document
        result = await document_processor.process_google_drive(
            drive_file_id=drive_file_id,
            metadata=metadata
        )
        
        return {
            "document_id": result.document_id,
            "status": result.status,            "chunk_count": result.chunk_count,
            "error": result.error
        }
    
    async def get_user_documents(
        self,
        user_id: str,
        organization_id: Optional[str] = None,
        source_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get a list of documents for a user.
        
        Args:
            user_id: User ID to filter by
            organization_id: Optional organization ID to filter by
            source_types: Optional list of source types to include
            tags: Optional list of tags to filter by
            limit: Maximum number of documents to return
            offset: Offset for pagination
            
        Returns:
            List of documents and metadata
        """
        # Build filters
        filters = {"user_id": user_id}
        
        if organization_id:
            filters["organization_id"] = organization_id
            
        if source_types:
            filters["source"] = source_types
            
        if tags:
            filters["tags"] = tags
            
        try:
            # Get document count for pagination info
            total_count = await weaviate_store.get_document_count(filters=filters)
            
            # Get document metadata from vector store
            documents = await weaviate_store.get_documents(
                filters=filters,
                limit=limit,
                offset=offset
            )
            
            # Group by document_id and extract summary data
            document_map = {}
            for doc in documents:
                doc_id = doc["metadata"].get("document_id")
                if not doc_id:
                    continue
                    
                if doc_id not in document_map:
                    document_map[doc_id] = {
                        "document_id": doc_id,
                        "title": doc["metadata"].get("title", "Unnamed document"),
                        "source": doc["metadata"].get("source", "unknown"),
                        "created_at": doc["metadata"].get("created_at"),
                        "tags": doc["metadata"].get("tags", []),
                        "chunk_count": 1,
                        "url": doc["metadata"].get("url", ""),
                        "mime_type": doc["metadata"].get("mime_type", ""),
                        "summary": doc["content"][:150] + "..." if len(doc["content"]) > 150 else doc["content"]
                    }
                else:
                    document_map[doc_id]["chunk_count"] += 1
            
            # Get unique documents
            unique_documents = list(document_map.values())
            
            # Sort by created_at (newest first)
            unique_documents.sort(
                key=lambda x: x.get("created_at", ""), 
                reverse=True
            )
            
            return {
                "documents": unique_documents,
                "total": total_count,
                "unique_count": len(unique_documents),
                "limit": limit,
                "offset": offset
            }
            
        except Exception as e:
            logger.error("Error retrieving user documents", error=str(e), user_id=user_id)
            return {
                "documents": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "error": str(e)
            }
    
    async def get_all_context_sources(
        self,
        query: str,
        user_id: str, 
        organization_id: Optional[str] = None,
        max_results_per_source: int = 3
    ) -> Dict[str, Any]:
        """
        Get context from all available sources for a comprehensive answer.
        This is used for the main 'Ask Lumicoria AI' feature.
        
        Args:
            query: The user's question
            user_id: User ID for filtering context
            organization_id: Optional organization ID for filtering
            max_results_per_source: Maximum results per source type
            
        Returns:
            Dict with context from different sources
        """
        await self.initialize()
        
        # Generate embedding for the query
        query_embedding = await self.perplexity_client.generate_embeddings(texts=[query])
        if not query_embedding or len(query_embedding) == 0:
            logger.error("Failed to generate query embedding")
            return {"context": [], "error": "Failed to generate query embedding"}
        
        # Define source types to search from
        source_types = ["upload", "drive", "web", "chat_history"]
        
        all_context = []
        total_chunks = 0
        context_by_source = {}
        
        # Build base filters
        base_filters = {"user_id": user_id}
        if organization_id:
            base_filters["organization_id"] = organization_id
        
        # Query each source type
        for source in source_types:
            try:
                # Create source-specific filters
                source_filters = base_filters.copy()
                source_filters["source"] = source
                
                # Search vector store
                results = await weaviate_store.similarity_search(
                    query_vector=query_embedding[0],
                    k=max_results_per_source,
                    filters=source_filters
                )
                
                if results:
                    # Format results
                    formatted_results = self._format_context(results)
                    all_context.extend(formatted_results)
                    context_by_source[source] = formatted_results
                    total_chunks += len(formatted_results)
            
            except Exception as e:
                logger.error(f"Error retrieving context from {source}", error=str(e))
        
        return {
            "context": all_context,
            "context_by_source": context_by_source,
            "total_chunks": total_chunks,
            "query": query,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _format_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format vector store results for context inclusion."""
        formatted_results = []
        
        for result in results:
            # Extract key information
            formatted_result = {
                "text": result["content"],
                "score": result["score"],
                "source": result["metadata"].get("source", "unknown"),
                "metadata": {}
            }
            
            # Include relevant metadata but filter out internal fields
            for key, value in result["metadata"].items():
                if key not in ["user_id", "organization_id", "chunk_id"]:
                    formatted_result["metadata"][key] = value
                    
            formatted_results.append(formatted_result)
            
        return formatted_results

# Create a singleton instance
context_service = ContextService()

# Initialize the service asynchronously - this needs to be called at application startup
async def initialize_context_service():
    """Initialize the context service and its dependencies."""
    await context_service.initialize()
