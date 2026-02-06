"""
Perplexity API Integration Module for Lumicoria.ai

This module provides a comprehensive interface to Perplexity's Sonar API,
implementing all necessary functions for document processing, querying, and integrating with Lumicoria's Agent Universe.

The implementation focuses on Perplexity's strengths in:
- Real-time search and information retrieval
- Deep research capabilities with citations
- Chain-of-thought reasoning
"""

import os
import json
import httpx
import asyncio
from typing import Dict, Any, List, Optional, Union
import structlog
from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

# Configure logging
logger = structlog.get_logger(__name__)

# Define Perplexity API Models
class PerplexityModel(str, Enum):
    SONAR_SMALL = "sonar-small-chat"
    SONAR_MEDIUM = "sonar-medium-chat"
    SONAR_LARGE = "sonar-large-chat"
    SONAR_LARGE_ONLINE = "sonar-large-online"
    MISTRAL_7B = "mistral-7b-instruct"
    MIXTRAL_8X7B = "mixtral-8x7b-instruct"
    LLAMA_3_70B = "llama-3-70b-flash"
    LLAMA_3_8B = "llama-3-8b-instruct"
    CLAUDE_3_OPUS = "claude-3-opus-20240229"
    CLAUDE_3_SONNET = "claude-3-sonnet-20240229"
    CLAUDE_3_HAIKU = "claude-3-haiku-20240307"

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"

class PerplexityMessage(BaseModel):
    role: MessageRole
    content: str

class CitationMetadata(BaseModel):
    url: str
    title: Optional[str] = None
    snippet: Optional[str] = None
    published_date: Optional[str] = None
    
class Citation(BaseModel):
    start: int
    end: int
    text: str
    metadata: CitationMetadata

class SearchQuerySource(BaseModel):
    type: str
    query: str

class PerplexityResponse(BaseModel):
    id: str
    model: str
    created: int
    choices: List[Dict[str, Any]]
    search_queries: Optional[List[Dict[str, str]]] = None
    
    @property
    def content(self) -> str:
        """Extract response content from the choices"""
        if not self.choices:
            return ""
        message = self.choices[0].get("message", {})
        return message.get("content", "")
    
    @property
    def citations(self) -> List[Citation]:
        """Extract citations from the response"""
        if not self.choices:
            return []
        
        message = self.choices[0].get("message", {})
        if not message:
            return []
            
        citations = []
        for citation in message.get("context", {}).get("citations", []):
            try:
                citations.append(Citation(**citation))
            except Exception as e:
                logger.warning(f"Failed to parse citation: {e}")
                
        return citations
        
    @property
    def search_queries_list(self) -> List[str]:
        """Extract search queries as plain list"""
        if not self.search_queries:
            return []
        return [sq.get("query", "") for sq in self.search_queries]

class PerplexityConfig(BaseModel):
    api_key: str
    model: str = PerplexityModel.SONAR_LARGE_ONLINE.value
    temperature: float = 0.7
    max_tokens: int = 1024
    top_p: float = 0.9
    timeout: int = 60
    base_url: str = "https://api.perplexity.ai"

class PerplexityClient:
    def __init__(self, config: Union[Dict[str, Any], PerplexityConfig]):
        """Initialize the Perplexity client with configuration.
        
        Args:
            config: Either a PerplexityConfig or a dictionary with config parameters.
        """
        # Convert dict to PerplexityConfig if needed
        if isinstance(config, dict):
            config = PerplexityConfig(**config)
        
        self.config = config
        self._validate_config()
        
        # Setup HTTP client
        self.client = httpx.AsyncClient(
            base_url=self.config.base_url,
            timeout=self.config.timeout,
            headers={"Authorization": f"Bearer {self.config.api_key}"}
        )
        
        # Setup request options
        self.default_options = {
            "model": self.config.model,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "top_p": self.config.top_p
        }
    
    def _validate_config(self):
        """Validate the configuration"""
        if not self.config.api_key:
            raise ValueError("Perplexity API key is required.")
        
        # If model is not a string value from enum, validate the string
        if self.config.model not in [model.value for model in PerplexityModel]:
            logger.warning(f"Using custom model: {self.config.model}")
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.RequestError, httpx.HTTPStatusError))
    )
    async def _make_request(self, endpoint: str, data: Dict[str, Any]) -> Dict[str, Any]:
        """Make a request to the Perplexity API.
        
        Args:
            endpoint: API endpoint path
            data: Request data
            
        Returns:
            API response
        """
        url = f"{endpoint}"
        
        try:
            response = await self.client.post(url, json=data)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error: {e.response.status_code} - {e.response.text}")
            raise
        except httpx.RequestError as e:
            logger.error(f"Request error: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            raise
    
    async def close(self):
        """Close the HTTP client"""
        await self.client.aclose()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def generate_embeddings(
        self,
        texts: List[str],
        model: str = "sonar-embedding-001",
        **kwargs
    ) -> List[List[float]]:
        """Generate embeddings for a list of texts.
        
        Args:
            texts: List of text strings to generate embeddings for
            model: Embedding model to use
            kwargs: Additional options for the API
            
        Returns:
            List of embeddings, each embedding is a list of floats
        """
        if not texts:
            return []
            
        # Prepare request data
        data = {
            "model": model,
            "input": texts,
            **kwargs
        }
        
        try:
            # Make request to embeddings endpoint
            response_data = await self._make_request("/embeddings", data)
            
            # Extract embeddings from response
            embeddings = []
            for item in response_data.get("data", []):
                embeddings.append(item.get("embedding", []))
                
            return embeddings
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            # Return empty vectors as fallback
            return [[0.0] * 1024] * len(texts)
    
    async def chat_completion(
        self, 
        messages: List[Union[Dict[str, str], PerplexityMessage]],
        stream: bool = False,
        **kwargs
    ) -> PerplexityResponse:
        """Generate a chat completion from Perplexity.
        
        Args:
            messages: List of messages in the conversation
            stream: Whether to stream the response
            kwargs: Additional options for the API
            
        Returns:
            Perplexity chat completion response
        """
        # Format messages if needed
        formatted_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                formatted_messages.append(msg)
            elif isinstance(msg, PerplexityMessage):
                formatted_messages.append(msg.dict())
            else:
                raise ValueError(f"Invalid message type: {type(msg)}")
        
        # Prepare request data
        data = {
            **self.default_options,
            "messages": formatted_messages,
            "stream": stream,
            **kwargs
        }
        
        # Make request
        response_data = await self._make_request("/chat/completions", data)
        return PerplexityResponse(**response_data)
    
    async def query_document(
        self, 
        document: str, 
        query: str,
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> PerplexityResponse:
        """Query a document with a specific question.
        
        Args:
            document: Document content
            query: Question about the document
            system_prompt: Optional system prompt
            kwargs: Additional options for the API
            
        Returns:
            Perplexity response
        """
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                "role": "system",
                "content": system_prompt
            })
        else:
            # Default system prompt for document analysis
            messages.append({
                "role": "system",
                "content": "You are a helpful assistant that analyzes documents accurately. " + 
                          "When you find important information like dates, names, amounts, " +
                          "and action items, please highlight them clearly."
            })
        
        # Add document context and query
        messages.append({
            "role": "user",
            "content": f"Document text: {document}\n\nQuery: {query}"
        })
        
        # Call API
        return await self.chat_completion(messages, **kwargs)
    
    async def extract_document_info(
        self,
        document: str,
        extraction_targets: Optional[List[str]] = None,
        **kwargs
    ) -> PerplexityResponse:
        """Extract key information from a document.
        
        Args:
            document: Document content
            extraction_targets: List of information types to extract
            kwargs: Additional options for the API
            
        Returns:
            Extracted information
        """
        if not extraction_targets:
            extraction_targets = ["dates", "names", "organizations", "amounts", "action items", "key points"]
        
        targets_str = ", ".join(extraction_targets)
        
        system_prompt = (
            f"You are an expert document analyzer. Extract the following information "
            f"from the document: {targets_str}. Return the information in a clear, "
            f"structured format. For each item, include the exact text from the document "
            f"and context around it."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Document text: {document}"}
        ]
        
        return await self.chat_completion(messages, **kwargs)
    
    async def analyze_document_with_citations(
        self,
        document: str,
        prompt: Optional[str] = None,
        **kwargs
    ) -> PerplexityResponse:
        """Analyze a document with citations for facts.
        
        Args:
            document: Document content
            prompt: Specific analysis prompt
            kwargs: Additional options for the API
            
        Returns:
            Analysis with citations
        """
        # Ensure we're using an online model
        model = kwargs.get("model", self.config.model)
        if "online" not in model and model != PerplexityModel.SONAR_LARGE_ONLINE.value:
            logger.warning(f"Using model {model} which may not support online search.")
            model = PerplexityModel.SONAR_LARGE_ONLINE.value
            kwargs["model"] = model
        
        analysis_prompt = prompt or (
            "Please analyze this document carefully. Identify key facts, verify them with "
            "online sources where possible, and provide citations for your analysis. Focus on "
            "extracting critical information such as dates, names, organizations, monetary values, "
            "and action items."
        )
        
        messages = [
            {"role": "user", "content": f"{analysis_prompt}\n\nDocument: {document}"}
        ]
        
        return await self.chat_completion(messages, **kwargs)
    
    async def generate_tasks_from_document(
        self,
        document: str,
        user_context: Optional[str] = None,
        **kwargs
    ) -> PerplexityResponse:
        """Generate tasks from document content.
        
        Args:
            document: Document content
            user_context: Optional context about the user
            kwargs: Additional options for the API
            
        Returns:
            Response with generated tasks
        """
        system_prompt = (
            "You are a task extraction specialist. Your goal is to identify potential tasks, "
            "deadlines, and action items from the document. Format the output as a clear list "
            "of tasks with priorities, deadlines (if available), and responsible parties (if mentioned). "
            "Include only actionable items that require someone to do something."
        )
        
        user_prompt = (
            f"Extract all tasks, deadlines, and action items from this document and format them as a list.\n\n"
            f"Document: {document}"
        )
        
        if user_context:
            user_prompt = f"User context: {user_context}\n\n{user_prompt}"
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return await self.chat_completion(messages, **kwargs)
    
    async def generate_wellbeing_advice(
        self,
        user_data: Dict[str, Any],
        **kwargs
    ) -> PerplexityResponse:
        """Generate wellbeing advice based on user data.
        
        Args:
            user_data: User activity and context data
            kwargs: Additional options for the API
            
        Returns:
            Wellbeing recommendations
        """
        # Format user data context
        context = []
        for key, value in user_data.items():
            if key == "activity_log":
                activity_str = "\n".join([f"- {a}" for a in value[-10:]])  # Last 10 activities
                context.append(f"Recent activity:\n{activity_str}")
            elif key == "screen_time":
                context.append(f"Daily screen time: {value} minutes")
            elif key == "breaks":
                context.append(f"Breaks taken today: {value}")
            elif key == "focus_sessions":
                context.append(f"Focus sessions completed: {value}")
            else:
                context.append(f"{key}: {value}")
        
        user_context = "\n".join(context)
        
        system_prompt = (
            "You are a compassionate wellbeing coach focused on helping knowledge workers "
            "maintain balance and health. Provide personalized suggestions based on the user's "
            "data. Include recommendations for breaks, physical activity, focus techniques, "
            "and overall wellbeing. Be encouraging but not pushy."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Based on my current work pattern, can you provide some wellbeing advice?\n\n{user_context}"}
        ]
        
        return await self.chat_completion(messages, **kwargs)
    
    async def academic_research(
        self,
        query: str,
        depth: str = "detailed",  # Options: "brief", "detailed", "comprehensive"
        focus_areas: Optional[List[str]] = None,
        **kwargs
    ) -> PerplexityResponse:
        """Perform academic research on a topic.
        
        Args:
            query: Research query
            depth: Desired depth of research
            focus_areas: Specific aspects to focus on
            kwargs: Additional options for the API
            
        Returns:
            Research results
        """
        # Use online model for academic research
        if "model" not in kwargs:
            kwargs["model"] = PerplexityModel.SONAR_LARGE_ONLINE.value
        
        focus_str = ""
        if focus_areas:
            focus_str = f"Focus particularly on these aspects: {', '.join(focus_areas)}."
        
        system_prompt = (
            f"You are a thorough academic researcher. Provide a {depth} analysis of the topic "
            f"with citations to reputable sources. {focus_str} Structure your response clearly with "
            f"sections covering key aspects of the topic. Include recent developments and various perspectives."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Research query: {query}"}
        ]
        
        return await self.chat_completion(messages, **kwargs)
    
    async def create_agent_prompt(
        self,
        agent_type: str,
        agent_name: str,
        agent_description: str,
        capabilities: List[str],
        **kwargs
    ) -> PerplexityResponse:
        """Create a custom prompt for a new agent type.
        
        Args:
            agent_type: Type of agent (e.g., "document", "meeting", "creative")
            agent_name: Name of the agent
            agent_description: Description of the agent's purpose
            capabilities: List of capabilities the agent should have
            kwargs: Additional options for the API
            
        Returns:
            Custom agent prompt template
        """
        capability_str = "\n".join([f"- {cap}" for cap in capabilities])
        
        system_prompt = (
            "You are an AI system architect specializing in creating effective prompts "
            "for specialized AI agents. Create a detailed system prompt template that will "
            "guide an AI to fulfill the described role and capabilities effectively."
        )
        
        user_prompt = (
            f"Please create a detailed system prompt for a new AI agent with these specifications:\n\n"
            f"Agent Type: {agent_type}\n"
            f"Agent Name: {agent_name}\n"
            f"Description: {agent_description}\n"
            f"Capabilities:\n{capability_str}\n\n"
            f"The prompt should be comprehensive, clear, and effective at guiding the AI to perform "
            f"this specific role. Include appropriate tone, constraints, and response formats."
        )
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return await self.chat_completion(messages, **kwargs)

# Factory function to create a Perplexity client
def create_perplexity_client(
    api_key: Optional[str] = None, 
    config: Optional[Dict[str, Any]] = None
) -> PerplexityClient:
    """Create a Perplexity client instance with configuration.
    
    Args:
        api_key: Perplexity API key (if not provided in config)
        config: Configuration options
        
    Returns:
        PerplexityClient instance
    """
    # Get API key from centralized settings if not provided directly
    from backend.core.config import settings as app_settings
    final_api_key = api_key or (config or {}).get("api_key") or app_settings.PERPLEXITY_API_KEY
    
    # Create config
    if config is None:
        config = {}
    
    if final_api_key:
        config["api_key"] = final_api_key
        
    return PerplexityClient(config)

# Async factory function to create a Perplexity client
async def create_perplexity_client_async(
    api_key: Optional[str] = None, 
    config: Optional[Dict[str, Any]] = None
) -> PerplexityClient:
    """Asynchronously create a Perplexity client instance with configuration.
    
    This is an async wrapper for create_perplexity_client for consistent async interface.
    
    Args:
        api_key: Perplexity API key (if not provided in config)
        config: Configuration options
        
    Returns:
        PerplexityClient instance
    """
    return create_perplexity_client(api_key, config)

