"""
RAG Agent for Lumicoria.ai

This agent uses Retrieval Augmented Generation to provide informative responses
based on user-specific context and documents.
"""

from typing import Dict, Any, List, Optional, Union
import structlog
import asyncio
import json
from datetime import datetime
import uuid

from .base_agent import BaseAgent
from ..services.context_service import context_service

# Configure logger
logger = structlog.get_logger(__name__)

class RAGAgent(BaseAgent):
    """
    Agent that uses Retrieval Augmented Generation with Perplexity API
    to provide context-aware responses based on user documents and history.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the RAG agent with configuration.
        
        Args:
            config: Agent configuration dictionary
        """
        super().__init__(config)
        self.context_settings = config.get("context_settings", {})
        self.max_context_chunks = self.context_settings.get("max_chunks", 8)
        
        # Maximum tokens for context to avoid exceeding API limits
        self.max_context_tokens = self.context_settings.get("max_tokens", 8000)
        
        # Default system prompt template
        self.system_prompt_template = self.config.get(
            "system_prompt_template",
            "You are Lumicoria.ai, an AI assistant with access to the user's documents and knowledge. "
            "Use the following context to help answer the user's question. "
            "If the context doesn't contain relevant information, draw on your general knowledge "
            "but prioritize what's in the user's documents. "
            "Be clear when you're using information from their documents vs. your general knowledge.\n\n"
            "{context}"
        )

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a user request using RAG.
        
        Args:
            data: User request data including query and user information
            
        Returns:
            Response dictionary with generated text
        """
        # Use asyncio to run the async processing function
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(self.process_async(data))

    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Asynchronously process a user request using RAG.
        
        Args:
            data: User request data including query and user information
            
        Returns:
            Response dictionary with generated text and context
        """
        start_time = datetime.utcnow()
        query = data.get("query")
        user_id = data.get("user_id")
        organization_id = data.get("organization_id")
        conversation_id = data.get("conversation_id")
        
        if not query or not user_id:
            return {
                "error": "Missing required parameters: query and user_id are required",
                "success": False
            }
        
        try:
            # Retrieve relevant context from the context service
            context_result = await context_service.get_context_for_query(
                query=query,
                user_id=user_id,
                organization_id=organization_id,
                k=self.max_context_chunks,
            )
            
            context_chunks = context_result.get("context", [])
            
            # Format context for the prompt
            formatted_context = self._format_context_for_prompt(context_chunks)
            
            # Create system prompt with context
            system_prompt = self.system_prompt_template.format(context=formatted_context)
            
            # Create messages for Perplexity API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            # Call Perplexity API
            response = await self.perplexity_client.chat_completion(
                messages=messages,
                model=self.model_config.get("model", "sonar-medium-online"),
                temperature=self.model_config.get("temperature", 0.7),
                max_tokens=self.model_config.get("max_tokens", 1024),
            )
            
            # Extract response text
            ai_response = response.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            # Calculate processing time
            end_time = datetime.utcnow()
            processing_time = (end_time - start_time).total_seconds()
            
            # Store the conversation in context if requested
            if data.get("save_to_context", True) and conversation_id:
                chat_messages = [
                    {"role": "user", "content": query},
                    {"role": "assistant", "content": ai_response}
                ]
                
                await context_service.add_chat_context(
                    messages=chat_messages,
                    user_id=user_id,
                    organization_id=organization_id,
                    conversation_id=conversation_id
                )
            
            # Return result
            return {
                "response": ai_response,
                "context_used": len(context_chunks),
                "sources": self._extract_sources(context_chunks),
                "processing_time_seconds": processing_time,
                "success": True,
                "conversation_id": conversation_id or str(uuid.uuid4())
            }
            
        except Exception as e:
            logger.error(f"Error in RAG process: {str(e)}")
            return {
                "error": str(e),
                "success": False
            }
    
    def _format_context_for_prompt(self, context_chunks: List[Dict[str, Any]]) -> str:
        """Format context chunks for inclusion in the prompt."""
        if not context_chunks:
            return "No relevant context found in your documents."
            
        formatted_chunks = []
        total_chars = 0
        
        for i, chunk in enumerate(context_chunks):
            # Estimate tokens (rough approximation: 4 chars ~ 1 token)
            chunk_text = chunk.get("text", "")
            chunk_chars = len(chunk_text)
            
            # Track total character count as a proxy for tokens
            if total_chars + chunk_chars > (self.max_context_tokens * 4):
                break
                
            # Add source metadata
            source_info = self._get_source_info(chunk)
            formatted_chunk = f"[CONTEXT {i+1}]\n{chunk_text}\nSource: {source_info}\n"
            
            formatted_chunks.append(formatted_chunk)
            total_chars += chunk_chars
            
        return "\n".join(formatted_chunks)
    
    def _get_source_info(self, chunk: Dict[str, Any]) -> str:
        """Extract source information from a context chunk's metadata."""
        source_type = chunk.get("source", "unknown")
        metadata = chunk.get("metadata", {})
        
        if source_type == "upload":
            return f"Uploaded document: {metadata.get('title', 'Unnamed document')}"
        elif source_type == "drive":
            return f"Google Drive: {metadata.get('title', 'Unnamed document')}"
        elif source_type == "chat_history":
            return f"Previous conversation {metadata.get('conversation_id', '')}"
        elif source_type == "web":
            return f"Web content: {metadata.get('url', metadata.get('title', 'Unknown website'))}"
        else:
            return metadata.get("title", f"Document ({source_type})")
    
    def _extract_sources(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Extract source information from context chunks for citation."""
        sources = []
        seen_sources = set()
        
        for chunk in context_chunks:
            metadata = chunk.get("metadata", {})
            source_type = chunk.get("source", "unknown")
            
            # Create a key to deduplicate sources
            source_key = None
            if source_type == "upload" or source_type == "drive":
                source_key = metadata.get("document_id")
            elif source_type == "web":
                source_key = metadata.get("url")
            elif source_type == "chat_history":
                source_key = metadata.get("conversation_id")
                
            if not source_key or source_key in seen_sources:
                continue
                
            seen_sources.add(source_key)
            
            source_info = {
                "type": source_type,
                "title": metadata.get("title", "Unnamed document")
            }
            
            # Add source-specific fields
            if source_type == "web":
                source_info["url"] = metadata.get("url")
            elif source_type in ["upload", "drive"]:
                source_info["document_id"] = metadata.get("document_id")
                source_info["created_at"] = metadata.get("created_at")
                
            sources.append(source_info)
            
        return sources

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the RAG agent asynchronously.
        
        Args:
            query: The query string to search and retrieve information for
            context: Optional context dictionary containing user and organization IDs
            
        Returns:
            Dictionary containing retrieved information and response
        """
        try:
            # Ensure Perplexity client is initialized
            if not self.perplexity_client:
                self.initialize_models()
                
            if not self.perplexity_client:
                return {"error": "Perplexity client not initialized"}
            
            # Get user and organization IDs from context
            user_id = context.get("user_id") if context else None
            organization_id = context.get("organization_id") if context else None
            
            if not user_id:
                return {"error": "User ID is required in context"}
            
            # Retrieve relevant context from the context service
            context_result = await context_service.get_context_for_query(
                query=query,
                user_id=user_id,
                organization_id=organization_id,
                k=self.max_context_chunks,
            )
            
            context_chunks = context_result.get("context", [])
            
            # Format context for the prompt
            formatted_context = self._format_context_for_prompt(context_chunks)
            
            # Create system prompt with context
            system_prompt = self.system_prompt_template.format(context=formatted_context)
            
            # Create messages for Perplexity API
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ]
            
            # Call Perplexity API
            response = await self.perplexity_client.chat_completion(
                messages=messages,
                model=self.model_config.get("model", "sonar-medium-online"),
                temperature=self.model_config.get("temperature", 0.7),
                max_tokens=self.model_config.get("max_tokens", 1024),
            )
            
            # Extract response text
            ai_response = response.content
            
            # Create comprehensive response
            result = {
                "response": ai_response,
                "context_chunks": context_chunks,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
                "metadata": {
                    "user_id": user_id,
                    "organization_id": organization_id,
                    "query": query,
                    "context_sources": [chunk.get("source", "") for chunk in context_chunks]
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error querying RAG agent: {str(e)}")
            return {"error": f"Information retrieval failed: {str(e)}"}
