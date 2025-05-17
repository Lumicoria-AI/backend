from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase
from typing import Optional
from core.config import settings
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
        if not cls.db:
            await cls.connect()
        return cls.db

    @classmethod
    async def get_collection(cls, collection_name: str):
        db = await cls.get_database()
        return db[collection_name]


async def get_mongodb() -> MongoDB:
    return MongoDB() 