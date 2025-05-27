from typing import Any, Dict, Optional, Union
import json
import pickle
from datetime import datetime, timedelta
import asyncio
import structlog
from functools import wraps
import hashlib
import redis.asyncio as redis
from redis.exceptions import RedisError

# Configure logging
logger = structlog.get_logger(__name__)

class CacheKey:
    """Helper class for generating consistent cache keys."""
    
    @staticmethod
    def generate_key(prefix: str, *args, **kwargs) -> str:
        """Generate a consistent cache key from prefix, args and kwargs."""
        # Convert args and kwargs to a sorted string
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}:{v}" for k, v in sorted(kwargs.items()))
        
        # Create a hash of the combined parts
        key_string = f"{prefix}:{':'.join(key_parts)}"
        return hashlib.md5(key_string.encode()).hexdigest()

class BaseCache:
    """Base cache class defining the interface for all cache implementations."""
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        raise NotImplementedError
        
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache with optional TTL in seconds."""
        raise NotImplementedError
        
    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        raise NotImplementedError
        
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        raise NotImplementedError
        
    async def clear(self) -> bool:
        """Clear all values from cache."""
        raise NotImplementedError

class MemoryCache(BaseCache):
    """In-memory cache implementation using a dictionary."""
    
    def __init__(self):
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from memory cache."""
        async with self._lock:
            if key not in self._cache:
                return None
                
            item = self._cache[key]
            if item["expires_at"] and datetime.utcnow() > item["expires_at"]:
                del self._cache[key]
                return None
                
            return item["value"]
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in memory cache with optional TTL."""
        async with self._lock:
            expires_at = None
            if ttl is not None:
                expires_at = datetime.utcnow() + timedelta(seconds=ttl)
                
            self._cache[key] = {
                "value": value,
                "expires_at": expires_at
            }
            return True
    
    async def delete(self, key: str) -> bool:
        """Delete a value from memory cache."""
        async with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in memory cache."""
        async with self._lock:
            if key not in self._cache:
                return False
                
            item = self._cache[key]
            if item["expires_at"] and datetime.utcnow() > item["expires_at"]:
                del self._cache[key]
                return False
                
            return True
    
    async def clear(self) -> bool:
        """Clear all values from memory cache."""
        async with self._lock:
            self._cache.clear()
            return True

class RedisCache(BaseCache):
    """Redis cache implementation using redis-py."""
    
    def __init__(self, redis_url: str, prefix: str = "lumicoria"):
        self.redis = redis.from_url(redis_url)
        self.prefix = prefix
        self._lock = asyncio.Lock()
    
    def _get_key(self, key: str) -> str:
        """Get the full Redis key with prefix."""
        return f"{self.prefix}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from Redis cache."""
        try:
            full_key = self._get_key(key)
            data = await self.redis.get(full_key)
            if data is None:
                return None
            return pickle.loads(data)
        except (RedisError, pickle.PickleError) as e:
            logger.error(f"Error getting value from Redis: {str(e)}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in Redis cache with optional TTL."""
        try:
            full_key = self._get_key(key)
            data = pickle.dumps(value)
            if ttl is not None:
                return await self.redis.setex(full_key, ttl, data)
            return await self.redis.set(full_key, data)
        except (RedisError, pickle.PickleError) as e:
            logger.error(f"Error setting value in Redis: {str(e)}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete a value from Redis cache."""
        try:
            full_key = self._get_key(key)
            return bool(await self.redis.delete(full_key))
        except RedisError as e:
            logger.error(f"Error deleting value from Redis: {str(e)}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in Redis cache."""
        try:
            full_key = self._get_key(key)
            return bool(await self.redis.exists(full_key))
        except RedisError as e:
            logger.error(f"Error checking key existence in Redis: {str(e)}")
            return False
    
    async def clear(self) -> bool:
        """Clear all values from Redis cache with the prefix."""
        try:
            pattern = f"{self.prefix}:*"
            keys = await self.redis.keys(pattern)
            if keys:
                await self.redis.delete(*keys)
            return True
        except RedisError as e:
            logger.error(f"Error clearing Redis cache: {str(e)}")
            return False

class CacheManager:
    """Manager class for handling different cache implementations."""
    
    def __init__(self, cache_type: str = "memory", **kwargs):
        """
        Initialize cache manager with specified cache type.
        
        Args:
            cache_type: Type of cache to use ("memory" or "redis")
            **kwargs: Additional arguments for cache initialization
        """
        if cache_type == "redis":
            if "redis_url" not in kwargs:
                raise ValueError("redis_url is required for Redis cache")
            self.cache = RedisCache(kwargs["redis_url"], kwargs.get("prefix", "lumicoria"))
        else:
            self.cache = MemoryCache()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get a value from cache."""
        return await self.cache.get(key)
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set a value in cache."""
        return await self.cache.set(key, value, ttl)
    
    async def delete(self, key: str) -> bool:
        """Delete a value from cache."""
        return await self.cache.delete(key)
    
    async def exists(self, key: str) -> bool:
        """Check if a key exists in cache."""
        return await self.cache.exists(key)
    
    async def clear(self) -> bool:
        """Clear all values from cache."""
        return await self.cache.clear()

def cached(ttl: Optional[int] = None, key_prefix: str = "cache"):
    """
    Decorator for caching function results.
    
    Args:
        ttl: Time to live in seconds
        key_prefix: Prefix for cache key
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Get cache instance from kwargs or use default
            cache = kwargs.pop("cache", None)
            if cache is None:
                # Use default memory cache if none provided
                cache = MemoryCache()
            
            # Generate cache key
            cache_key = CacheKey.generate_key(
                f"{key_prefix}:{func.__name__}",
                *args,
                **kwargs
            )
            
            # Try to get from cache
            cached_value = await cache.get(cache_key)
            if cached_value is not None:
                return cached_value
            
            # If not in cache, call function
            result = await func(*args, **kwargs)
            
            # Store in cache
            await cache.set(cache_key, result, ttl)
            
            return result
        return wrapper
    return decorator

# Example usage:
"""
# Initialize cache manager
cache_manager = CacheManager("redis", redis_url="redis://localhost:6379")

# Use as decorator
@cached(ttl=3600, key_prefix="agent")
async def get_agent_response(agent_id: str, input_data: Dict[str, Any]):
    # Function implementation
    pass

# Use directly
async def process_with_cache():
    key = CacheKey.generate_key("agent:response", agent_id="123", input="test")
    await cache_manager.set(key, {"result": "cached"}, ttl=3600)
    result = await cache_manager.get(key)
""" 