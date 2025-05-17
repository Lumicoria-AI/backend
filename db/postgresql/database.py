from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from backend.core.config import settings

# Define the asynchronous engine
# Use the database URL from the settings
engine = create_async_engine(settings.DATABASE_URL, echo=True)

# Define the asynchronous session factory
AsyncSessionLocal = sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

async def get_db():
    """Dependency function to get an asynchronous database session."""
    async with AsyncSessionLocal() as session:
        yield session

# You might also want functions here for database initialization/migration using Alembic
# For example, functions to create tables if they don't exist, or run migrations. 