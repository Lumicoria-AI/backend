# MongoDB, Redis, Vector Store imports (always available)
from .mongodb.mongodb import get_mongodb, MongoDB
from .redis.redis import get_redis, RedisClient
from .vector_stores import get_vector_store

# Cassandra — lazy import only (driver needs asyncore, removed in Python 3.12+)
# Use: from backend.db.cassandra.cassandra import CassandraClient
# The import is deferred to when actually needed (e.g., main.py lifespan)

__all__ = [
    'get_mongodb',
    'MongoDB',
    'get_redis',
    'RedisClient',
    'get_vector_store',
]
