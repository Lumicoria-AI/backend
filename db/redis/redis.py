from redis.asyncio import Redis, ConnectionPool
from typing import Optional
from ...core.config import settings
import structlog

logger = structlog.get_logger()

class RedisClient:
    _pool: Optional[ConnectionPool] = None
    _client: Optional[Redis] = None

    @classmethod
    async def connect(cls) -> None:
        try:
            cls._pool = ConnectionPool(
                host=settings.db.REDIS_HOST,
                port=settings.db.REDIS_PORT,
                password=settings.db.REDIS_PASSWORD,
                db=settings.db.REDIS_DB,
                max_connections=settings.db.REDIS_POOL_SIZE,
                decode_responses=True
            )
            cls._client = Redis(connection_pool=cls._pool)
            # Verify connection
            await cls._client.ping()
            logger.info("Connected to Redis")
        except Exception as e:
            logger.error("Failed to connect to Redis", error=str(e))
            raise

    @classmethod
    async def disconnect(cls) -> None:
        if cls._client:
            await cls._client.close()
        if cls._pool:
            await cls._pool.disconnect()
        logger.info("Disconnected from Redis")

    @classmethod
    async def get_client(cls) -> Redis:
        if not cls._client:
            await cls.connect()
        return cls._client

    @classmethod
    async def set(cls, key: str, value: str, expire: Optional[int] = None) -> bool:
        client = await cls.get_client()
        return await client.set(key, value, ex=expire)

    @classmethod
    async def get(cls, key: str) -> Optional[str]:
        client = await cls.get_client()
        return await client.get(key)

    @classmethod
    async def delete(cls, key: str) -> int:
        client = await cls.get_client()
        return await client.delete(key)

    @classmethod
    async def exists(cls, key: str) -> bool:
        client = await cls.get_client()
        return bool(await client.exists(key))

    @classmethod
    async def increment(cls, key: str, amount: int = 1) -> int:
        client = await cls.get_client()
        return await client.incrby(key, amount)

    @classmethod
    async def set_hash(cls, name: str, mapping: dict, expire: Optional[int] = None) -> bool:
        client = await cls.get_client()
        result = await client.hset(name, mapping=mapping)
        if expire:
            await client.expire(name, expire)
        return result

    @classmethod
    async def get_hash(cls, name: str) -> dict:
        client = await cls.get_client()
        return await client.hgetall(name)


async def get_redis() -> RedisClient:
    return RedisClient() 