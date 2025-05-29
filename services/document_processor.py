"""
Document Processing Pipeline for Lumicoria.ai RAG

This module handles the processing of documents for the RAG system, 
including chunking, embedding, and storage in the vector database.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import structlog
import asyncio
import uuid
from datetime import datetime
import httpx
from pydantic import BaseModel, Field

# Import text processing utilities
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredEmailLoader
)
from langchain_core.documents import Document as LangchainDocument

from ..db.vector_stores.weaviate import weaviate_store
from ..core.config import settings
from ..ai_models.perplexity import create_perplexity_client, PerplexityClient

logger = structlog.get_logger(__name__)

class DocumentChunk(BaseModel):
    """A single document chunk for processing."""
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class ProcessedDocument(BaseModel):
    """Result of document processing."""
    document_id: str
    chunk_count: int
    metadata: Dict[str, Any]
    vector_ids: List[str]
    status: str = "success"
    error: Optional[str] = None

class DocumentProcessor:
    """
    Process documents for RAG by chunking, embedding, and storing them.
    """
    
    def __init__(self, perplexity_client: Optional[PerplexityClient] = None):
        """Initialize the document processor."""
        self.perplexity_client = perplexity_client
        self.loaders = {
            "text/plain": TextLoader,
            "application/pdf": PyPDFLoader,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Docx2txtLoader,
            "text/csv": CSVLoader,
            "text/html": UnstructuredHTMLLoader,
            "text/markdown": UnstructuredMarkdownLoader,
            "message/rfc822": UnstructuredEmailLoader
        }
        
        # Default chunking parameters
        self.default_chunk_size = 1000
        self.default_chunk_overlap = 100
        
    async def initialize(self):
        """Ensure client is initialized."""
        if not self.perplexity_client:
            self.perplexity_client = create_perplexity_client()
            
    async def process_file(
        self, 
        file_path: str,
        metadata: Dict[str, Any],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> ProcessedDocument:
        """
        Process a file by loading, chunking, embedding, and storing it.
        
        Args:
            file_path: Path to the file
            metadata: Document metadata including user_id and organization_id
            chunk_size: Size of text chunks (default: 1000 chars)
            chunk_overlap: Overlap between chunks (default: 100 chars)
            
        Returns:
            ProcessedDocument object with processing results
        """
        try:
            # Get document ID or generate a new one
            document_id = metadata.get("document_id", str(uuid.uuid4()))
            metadata["document_id"] = document_id
            
            # Add source information if not provided
            if "source" not in metadata:
                metadata["source"] = "upload"
                
            # Determine MIME type if not provided
            if "mime_type" not in metadata:
                metadata["mime_type"] = self._detect_mime_type(file_path)
            
            # Add file info to metadata
            file_path_obj = Path(file_path)
            metadata["filename"] = metadata.get("filename", file_path_obj.name)
            metadata["title"] = metadata.get("title", file_path_obj.stem)
            metadata["created_at"] = metadata.get("created_at", datetime.utcnow().isoformat())
            
            # Load the document
            documents = await self._load_document(file_path, metadata["mime_type"])
            
            # Process and store chunks
            processed_doc = await self._process_documents(
                documents=documents,
                metadata=metadata,
                chunk_size=chunk_size or self.default_chunk_size,
                chunk_overlap=chunk_overlap or self.default_chunk_overlap
            )
            
            return processed_doc
            
        except Exception as e:
            logger.error("Error processing document", error=str(e), file_path=file_path)
            return ProcessedDocument(
                document_id=metadata.get("document_id", str(uuid.uuid4())),
                chunk_count=0,
                metadata=metadata,
                vector_ids=[],
                status="error",
                error=str(e)
            )
    
    async def process_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> ProcessedDocument:
        """
        Process text directly without a file.
        
        Args:
            text: Text content to process
            metadata: Document metadata
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            ProcessedDocument object with processing results
        """
        try:
            # Get document ID or generate a new one
            document_id = metadata.get("document_id", str(uuid.uuid4()))
            metadata["document_id"] = document_id
            
            # Add source information if not provided
            if "source" not in metadata:
                metadata["source"] = "direct_text"
                
            # Set default MIME type
            metadata["mime_type"] = metadata.get("mime_type", "text/plain")
            
            # Add metadata
            metadata["filename"] = metadata.get("filename", f"{document_id}.txt")
            metadata["title"] = metadata.get("title", document_id)
            metadata["created_at"] = metadata.get("created_at", datetime.utcnow().isoformat())
            
            # Create document object
            document = LangchainDocument(page_content=text, metadata=metadata)
            
            # Process and store chunks
            processed_doc = await self._process_documents(
                documents=[document],
                metadata=metadata,
                chunk_size=chunk_size or self.default_chunk_size,
                chunk_overlap=chunk_overlap or self.default_chunk_overlap
            )
            
            return processed_doc
            
        except Exception as e:
            logger.error("Error processing text", error=str(e))
            return ProcessedDocument(
                document_id=metadata.get("document_id", str(uuid.uuid4())),
                chunk_count=0,
                metadata=metadata,
                vector_ids=[],
                status="error",
                error=str(e)
            )
    
    async def process_url(
        self,
        url: str,
        metadata: Dict[str, Any],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> ProcessedDocument:
        """
        Process content from a URL.
        
        Args:
            url: URL to fetch and process
            metadata: Document metadata
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            ProcessedDocument object with processing results
        """
        try:
            # Fetch content from URL
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                content = response.text
                
            # Add URL-specific metadata
            metadata["url"] = url
            metadata["source"] = metadata.get("source", "web")
            metadata["title"] = metadata.get("title", self._extract_title_from_html(content) or url)
            
            # Get document ID or generate a new one
            document_id = metadata.get("document_id", str(uuid.uuid4()))
            metadata["document_id"] = document_id
            
            # Determine if it's HTML or plain text
            mime_type = "text/html" if "<html" in content.lower() else "text/plain"
            metadata["mime_type"] = mime_type
            
            # Create document
            document = LangchainDocument(page_content=content, metadata=metadata)
            
            # Process and store chunks
            processed_doc = await self._process_documents(
                documents=[document],
                metadata=metadata,
                chunk_size=chunk_size or self.default_chunk_size,
                chunk_overlap=chunk_overlap or self.default_chunk_overlap
            )
            
            return processed_doc
            
        except Exception as e:
            logger.error("Error processing URL", error=str(e), url=url)
            return ProcessedDocument(
                document_id=metadata.get("document_id", str(uuid.uuid4())),
                chunk_count=0,
                metadata=metadata,
                vector_ids=[],
                status="error",
                error=str(e)
            )
    
    async def process_chat_history(
        self,
        messages: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> ProcessedDocument:
        """
        Process chat history for context retention.
        
        Args:
            messages: List of chat messages in {role, content} format
            metadata: Document metadata
            
        Returns:
            ProcessedDocument object with processing results
        """
        try:
            # Format chat history as text
            text = self._format_chat_history(messages)
            
            # Add chat-specific metadata
            metadata["source"] = "chat_history"
            metadata["message_count"] = len(messages)
            
            # Process as text
            return await self.process_text(
                text=text,
                metadata=metadata,
                chunk_size=1500,  # Larger chunks for chat history
                chunk_overlap=150
            )
            
        except Exception as e:
            logger.error("Error processing chat history", error=str(e))
            return ProcessedDocument(
                document_id=metadata.get("document_id", str(uuid.uuid4())),
                chunk_count=0,
                metadata=metadata,
                vector_ids=[],
                status="error",
                error=str(e)
            )
    
    async def _load_document(self, file_path: str, mime_type: str) -> List[LangchainDocument]:
        """Load a document using appropriate loader based on MIME type."""
        try:
            # Get the appropriate loader
            loader_class = self.loaders.get(mime_type)
            if not loader_class:
                raise ValueError(f"Unsupported MIME type: {mime_type}")
                
            # Create and use the loader
            loader = loader_class(file_path)
            documents = loader.load()
            
            return documents
            
        except Exception as e:
            logger.error("Error loading document", error=str(e), file_path=file_path, mime_type=mime_type)
            raise
    
    def _detect_mime_type(self, file_path: str) -> str:
        """Detect MIME type of a file based on extension."""
        # Simple extension-based detection
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.txt']:
            return "text/plain"
        elif ext in ['.pdf']:
            return "application/pdf"
        elif ext in ['.docx']:
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif ext in ['.csv']:
            return "text/csv"
        elif ext in ['.html', '.htm']:
            return "text/html"
        elif ext in ['.md', '.markdown']:
            return "text/markdown"
        elif ext in ['.eml']:
            return "message/rfc822"
        else:
            # Default to plain text
            return "text/plain"
    
    async def _process_documents(
        self,
        documents: List[LangchainDocument],
        metadata: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int
    ) -> ProcessedDocument:
        """
        Process loaded documents by chunking, embedding, and storing them.
        
        Args:
            documents: List of Langchain documents
            metadata: Document metadata
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            ProcessedDocument object with processing results
        """
        try:
            # Step 1: Chunk the documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", " ", ""],
                length_function=len
            )
            
            chunked_docs = []
            for doc in documents:
                # Combine document metadata with overall metadata
                combined_metadata = {**metadata, **doc.metadata}
                
                # Split into chunks
                chunks = text_splitter.split_text(doc.page_content)
                
                # Add chunk-specific metadata
                for i, chunk_text in enumerate(chunks):
                    chunk_metadata = combined_metadata.copy()
                    chunk_metadata["chunk_id"] = i
                    
                    chunked_docs.append(DocumentChunk(
                        text=chunk_text,
                        metadata=chunk_metadata
                    ))
                    
            # Step 2: Get embeddings for all chunks
            await self.initialize()  # Ensure client is initialized
            
            # Prepare texts for embedding
            texts = [chunk.text for chunk in chunked_docs]
            all_metadata = [chunk.metadata for chunk in chunked_docs]
            
            # Generate embeddings in batches
            batch_size = 20  # Keep batches small for API limits
            all_embeddings = []
            
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                # Get embeddings from Perplexity
                batch_embeddings = await self.perplexity_client.generate_embeddings(texts=batch_texts)
                all_embeddings.extend(batch_embeddings)
                
                # Short delay to avoid rate limiting
                if i + batch_size < len(texts):
                    await asyncio.sleep(0.5)
            
            # Step 3: Store in vector database with metadata
            vector_ids = await weaviate_store.add_documents(
                texts=texts,
                embeddings=all_embeddings,
                metadatas=all_metadata
            )
            
            # Step 4: Return processing result
            return ProcessedDocument(
                document_id=metadata["document_id"],
                chunk_count=len(chunked_docs),
                metadata=metadata,
                vector_ids=vector_ids,
                status="success"
            )
            
        except Exception as e:
            logger.error("Error processing documents", error=str(e))
            raise
    
    def _format_chat_history(self, messages: List[Dict[str, Any]]) -> str:
        """Format chat history as text for embedding."""
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
            
        return "\n\n".join(formatted)
    
    def _extract_title_from_html(self, html_content: str) -> Optional[str]:
        """Extract title from HTML content."""
        title_match = re.search(r"<title>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            return title_match.group(1).strip()
        return None
        
    async def process_google_drive(
        self,
        drive_file_id: str,
        metadata: Dict[str, Any],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> ProcessedDocument:
        """
        Process a document from Google Drive.
        
        Args:
            drive_file_id: Google Drive file ID
            metadata: Document metadata
            chunk_size: Size of text chunks (default: 1000 chars)
            chunk_overlap: Overlap between chunks (default: 100 chars)
            
        Returns:
            ProcessedDocument object with processing results
        """
        try:
            # Get document ID or generate a new one
            document_id = metadata.get("document_id", str(uuid.uuid4()))
            metadata["document_id"] = document_id
            
            # Make sure source is set
            if "source" not in metadata:
                metadata["source"] = "drive"
                
            metadata["drive_file_id"] = drive_file_id
            metadata["created_at"] = metadata.get("created_at", datetime.utcnow().isoformat())
            
            # Download and process the file from Google Drive
            # NOTE: This is a placeholder. In a real implementation, you would:
            # 1. Use Google Drive API to download the file
            # 2. Save it to a temporary location
            # 3. Process it like a regular file
            # 4. Delete the temporary file
            
            # For now, we'll simulate with a fake document
            logger.info(f"Processing Google Drive document: {drive_file_id}")
            
            # Create a minimal document for demonstration
            text = f"This is a placeholder for Google Drive document {drive_file_id}. In a real implementation, the content would be downloaded from Google Drive API."
            
            # Process as text
            return await self.process_text(
                text=text,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        except Exception as e:
            logger.error("Error processing Google Drive document", error=str(e), drive_file_id=drive_file_id)
            return ProcessedDocument(
                document_id=metadata.get("document_id", str(uuid.uuid4())),
                chunk_count=0,
                metadata=metadata,
                vector_ids=[],
                status="error",
                error=str(e)
            )
    
    async def process_chat_history(
        self,
        messages: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> ProcessedDocument:
        """
        Process chat history for context storage.
        
        Args:
            messages: List of chat messages
            metadata: Document metadata
            chunk_size: Size of text chunks (default: 1000 chars)
            chunk_overlap: Overlap between chunks (default: 100 chars)
            
        Returns:
            ProcessedDocument object with processing results
        """
        try:
            # Format chat history as text
            text = self._format_chat_history(messages)
            
            # Set document ID if not provided
            if "document_id" not in metadata and "conversation_id" in metadata:
                metadata["document_id"] = f"chat_{metadata['conversation_id']}"
            
            # Make sure source is set
            if "source" not in metadata:
                metadata["source"] = "chat_history"
                
            # Process the text
            return await self.process_text(
                text=text,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        except Exception as e:
            logger.error("Error processing chat history", error=str(e))
            return ProcessedDocument(
                document_id=metadata.get("document_id", str(uuid.uuid4())),
                chunk_count=0,
                metadata=metadata,
                vector_ids=[],
                status="error",
                error=str(e)
            )
    
    async def process_url(
        self,
        url: str,
        metadata: Dict[str, Any],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> ProcessedDocument:
        """
        Process content from a URL.
        
        Args:
            url: URL to process
            metadata: Document metadata
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            ProcessedDocument object with processing results
        """
        try:
            # Get document ID or generate a new one
            document_id = metadata.get("document_id", str(uuid.uuid4()))
            metadata["document_id"] = document_id
            metadata["url"] = url
            
            # Make sure source is set
            if "source" not in metadata:
                metadata["source"] = "web"
                
            # Add timestamp if not present
            if "created_at" not in metadata:
                metadata["created_at"] = datetime.utcnow().isoformat()
            
            # Fetch URL content
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(url)
                response.raise_for_status()
                html_content = response.text
                
                # Try to extract title if not provided
                if "title" not in metadata:
                    title = self._extract_title_from_html(html_content)
                    if title:
                        metadata["title"] = title
                    else:
                        metadata["title"] = "Web content"
                
                # Process as text (in a real impl, you'd want to extract text from HTML properly)
                # Here's a very simplistic HTML stripping
                text = re.sub(r'<[^>]+>', ' ', html_content)
                text = re.sub(r'\s+', ' ', text).strip()
                
                return await self.process_text(
                    text=text,
                    metadata=metadata,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap
                )
                
        except Exception as e:
            logger.error("Error processing URL", error=str(e), url=url)
            return ProcessedDocument(
                document_id=metadata.get("document_id", str(uuid.uuid4())),
                chunk_count=0,
                metadata=metadata,
                vector_ids=[],
                status="error",
                error=str(e)
            )

# Create a singleton instance
document_processor = DocumentProcessor()
