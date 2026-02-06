from typing import Any, Dict, Optional, Union, List
import json
import pickle
from datetime import datetime, timedelta
import asyncio
import structlog
from functools import wraps
import hashlib
from enum import Enum
from dataclasses import dataclass
from ..utils.cache import CacheManager, CacheKey, cached

# Configure logging
logger = structlog.get_logger(__name__)

class AgentCacheType(Enum):
    """Types of caching strategies for different agents."""
    DOCUMENT = "document"  # Cache document analysis results
    WELLBEING = "wellbeing"  # Cache wellbeing metrics and recommendations
    VISION = "vision"  # Cache vision analysis results
    MEETING = "meeting"  # Cache meeting summaries and action items
    CREATIVE = "creative"  # Cache creative content generation
    STUDENT = "student"  # Cache study materials and learning progress

@dataclass
class AgentCacheMetadata:
    """Metadata for cached agent responses."""
    agent_type: AgentCacheType
    created_at: datetime
    expires_at: Optional[datetime]
    model_used: str  # e.g., "perplexity-sonar"
    confidence_score: float
    citations: List[Dict[str, str]]  # List of citations from Perplexity
    processing_time: float  # Time taken to generate response

class AgentCache:
    """Specialized cache implementation for agents."""
    
    def __init__(self, cache_manager: CacheManager):
        self.cache_manager = cache_manager
        self._lock = asyncio.Lock()
    
    def _generate_agent_key(self, agent_type: AgentCacheType, *args, **kwargs) -> str:
        """Generate a cache key specific to an agent type."""
        return CacheKey.generate_key(
            f"agent:{agent_type.value}",
            *args,
            **kwargs
        )
    
    async def get_agent_response(
        self,
        agent_type: AgentCacheType,
        input_data: Dict[str, Any],
        include_metadata: bool = False
    ) -> Optional[Union[Dict[str, Any], tuple]]:
        """
        Get a cached agent response.
        
        Args:
            agent_type: Type of agent
            input_data: Input data for the agent
            include_metadata: Whether to include cache metadata in response
            
        Returns:
            Cached response and optionally metadata
        """
        cache_key = self._generate_agent_key(agent_type, **input_data)
        
        async with self._lock:
            cached_data = await self.cache_manager.get(cache_key)
            
            if cached_data is None:
                return None
                
            if include_metadata:
                return cached_data["response"], cached_data["metadata"]
            return cached_data["response"]
    
    async def set_agent_response(
        self,
        agent_type: AgentCacheType,
        input_data: Dict[str, Any],
        response: Dict[str, Any],
        metadata: AgentCacheMetadata,
        ttl: Optional[int] = None
    ) -> bool:
        """
        Cache an agent response with metadata.
        
        Args:
            agent_type: Type of agent
            input_data: Input data for the agent
            response: Agent's response
            metadata: Metadata about the response
            ttl: Time to live in seconds
        """
        cache_key = self._generate_agent_key(agent_type, **input_data)
        
        async with self._lock:
            return await self.cache_manager.set(
                cache_key,
                {
                    "response": response,
                    "metadata": metadata
                },
                ttl
            )
    
    async def invalidate_agent_cache(
        self,
        agent_type: AgentCacheType,
        pattern: Optional[str] = None
    ) -> bool:
        """
        Invalidate cache entries for an agent type.
        
        Args:
            agent_type: Type of agent
            pattern: Optional pattern to match specific entries
        """
        if pattern:
            # For Redis, we can use pattern matching
            if isinstance(self.cache_manager.cache, RedisCache):
                pattern = f"agent:{agent_type.value}:{pattern}"
                keys = await self.cache_manager.cache.redis.keys(pattern)
                if keys:
                    await self.cache_manager.cache.redis.delete(*keys)
                return True
            else:
                # For memory cache, we need to iterate
                async with self._lock:
                    keys_to_delete = []
                    for key in self.cache_manager.cache._cache:
                        if key.startswith(f"agent:{agent_type.value}:{pattern}"):
                            keys_to_delete.append(key)
                    for key in keys_to_delete:
                        await self.cache_manager.delete(key)
                    return True
        else:
            # Delete all entries for this agent type
            return await self.invalidate_agent_cache(agent_type, "*")

def agent_cached(
    agent_type: AgentCacheType,
    ttl: Optional[int] = None,
    include_metadata: bool = False
):
    """
    Decorator for caching agent responses.
    
    Args:
        agent_type: Type of agent
        ttl: Time to live in seconds
        include_metadata: Whether to include cache metadata in response
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache instance from kwargs or use default
            cache = kwargs.pop("cache", None)
            if cache is None:
                cache = AgentCache(CacheManager("memory"))
            
            # Extract input data from kwargs
            input_data = kwargs.get("input_data", {})
            
            # Try to get from cache
            cached_result = await cache.get_agent_response(
                agent_type,
                input_data,
                include_metadata
            )
            
            if cached_result is not None:
                return cached_result
            
            # If not in cache, call function
            start_time = datetime.utcnow()
            response = await func(*args, **kwargs)
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create metadata
            metadata = AgentCacheMetadata(
                agent_type=agent_type,
                created_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=ttl) if ttl else None,
                model_used=kwargs.get("model", "perplexity-sonar"),
                confidence_score=kwargs.get("confidence_score", 1.0),
                citations=kwargs.get("citations", []),
                processing_time=processing_time
            )
            
            # Store in cache
            await cache.set_agent_response(
                agent_type,
                input_data,
                response,
                metadata,
                ttl
            )
            
            if include_metadata:
                return response, metadata
            return response
            
        return wrapper
    return decorator

# Example usage:
"""
# Initialize agent cache
agent_cache = AgentCache(CacheManager("redis", redis_url="redis://localhost:6379"))

# Use as decorator
@agent_cached(agent_type=AgentCacheType.DOCUMENT, ttl=3600)
async def process_document(agent_id: str, input_data: Dict[str, Any], cache: Optional[AgentCache] = None):
    # Document processing implementation
    pass

# Use directly
async def process_with_cache():
    input_data = {"document_id": "123", "content": "test"}
    metadata = AgentCacheMetadata(
        agent_type=AgentCacheType.DOCUMENT,
        created_at=datetime.utcnow(),
        expires_at=None,
        model_used="perplexity-sonar",
        confidence_score=0.95,
        citations=[{"title": "Example", "url": "https://example.com"}],
        processing_time=0.5
    )
    await agent_cache.set_agent_response(
        AgentCacheType.DOCUMENT,
        input_data,
        {"result": "processed"},
        metadata,
        ttl=3600
    )
    result = await agent_cache.get_agent_response(
        AgentCacheType.DOCUMENT,
        input_data
    )
"""
