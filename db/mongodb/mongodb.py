from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional
from ...core.config import settings
import structlog

logger = structlog.get_logger()

class MongoDB:
    client: Optional[AsyncIOMotorClient] = None
    db: Optional[AsyncIOMotorDatabase] = None

    @classmethod
    async def connect(cls) -> None:
        try:
            cls.client = AsyncIOMotorClient(
                settings.db.MONGODB_URI,
                maxPoolSize=settings.db.MONGODB_MAX_POOL_SIZE,
                minPoolSize=settings.db.MONGODB_MIN_POOL_SIZE
            )
            cls.db = cls.client[settings.db.MONGODB_DB]
            # Verify connection
            await cls.client.admin.command('ping')
            logger.info("Connected to MongoDB")
        except Exception as e:
            logger.error("Failed to connect to MongoDB", error=str(e))
            raise

    @classmethod
    async def disconnect(cls) -> None:
        if cls.client:
            cls.client.close()
            logger.info("Disconnected from MongoDB")

    @classmethod
    async def get_database(cls) -> AsyncIOMotorDatabase:
        if cls.db is None:
            await cls.connect()
        # Ensure cls.db is not None after connect call
        if cls.db is None:
             raise ConnectionError("Failed to connect to MongoDB database")
        return cls.db

    @classmethod
    async def get_collection(cls, collection_name: str):
        db = await cls.get_database()
        return db[collection_name]

    @classmethod
    def reset_for_new_loop(cls) -> None:
        """Drop the cached Motor client so the next call rebuilds it on
        the current event loop. Used by Celery tasks that run on a fresh
        asyncio.run() loop each invocation — Motor binds Futures to the
        loop that ran the first I/O, so reuse across loops fails with
        'Future attached to a different loop'.
        """
        if cls.client is not None:
            try:
                cls.client.close()
            except Exception:
                pass
        cls.client = None
        cls.db = None


async def get_mongodb() -> AsyncIOMotorDatabase:
    return await MongoDB.get_database()

# Add wrapper functions for FastAPI startup/shutdown events
async def init_mongodb() -> None:
    """Initialize MongoDB connection."""
    await MongoDB.connect()

async def close_mongodb() -> None:
    """Close MongoDB connection."""
    await MongoDB.disconnect() 