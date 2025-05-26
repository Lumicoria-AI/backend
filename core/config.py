from typing import List, Optional, Dict, Any, Union
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator, Field, HttpUrl, ConfigDict, PostgresDsn
import secrets
from functools import lru_cache
from pathlib import Path


class DatabaseSettings(BaseSettings):
    model_config = ConfigDict(extra='allow', case_sensitive=True)

    # PostgreSQL Settings
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "postgres"
    POSTGRES_DB: str = "lumicoria"
    POSTGRES_PORT: int = 5432
    POSTGRES_POOL_SIZE: int = 20
    POSTGRES_MAX_OVERFLOW: int = 10

    # MongoDB Settings (Optional for now)
    MONGODB_URI: Optional[str] = "mongodb://localhost:27017"
    MONGODB_DB: Optional[str] = "lumicoria"
    MONGODB_MAX_POOL_SIZE: int = 100
    MONGODB_MIN_POOL_SIZE: int = 10

    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_POOL_SIZE: int = 10

    # Cassandra Settings (Optional for now)
    CASSANDRA_HOSTS: Optional[List[str]] = ["localhost"]
    CASSANDRA_PORT: int = 9042
    CASSANDRA_USERNAME: Optional[str] = None
    CASSANDRA_PASSWORD: Optional[str] = None
    CASSANDRA_KEYSPACE: Optional[str] = "lumicoria"
    CASSANDRA_REPLICATION_FACTOR: int = 1

    # Vector Store Settings (Optional for now)
    VECTOR_STORE_TYPE: str = "weaviate"  # weaviate, qdrant, or chroma
    VECTOR_STORE_URL: Optional[str] = "http://localhost:8080"
    VECTOR_STORE_API_KEY: Optional[str] = None
    VECTOR_STORE_COLLECTION: str = "documents"
    VECTOR_STORE_DIMENSION: int = 1536  # Default for OpenAI embeddings

    @property
    def DATABASE_URL(self) -> str:
        """Construct the database URL from individual components."""
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @validator("CASSANDRA_HOSTS", pre=True)
    def assemble_cassandra_hosts(cls, v: Union[str, List[str]]) -> List[str]:
        if isinstance(v, str):
            if v.startswith("["):
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return [i.strip() for i in v.split(",")]
            return [i.strip() for i in v.split(",")]
        return v or ["localhost"]


class Settings(BaseSettings):
    model_config = ConfigDict(extra='allow', case_sensitive=True, env_file='.env', env_nested_delimiter='__')

    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:8080",  # Frontend development server
        "http://localhost:3000",  # Alternative frontend port
        "http://127.0.0.1:8080",
        "http://127.0.0.1:3000",
    ]

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str):
            if v.startswith("["):
                import json
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    return [i.strip() for i in v.split(",")]
            return [i.strip() for i in v.split(",")]
        return v or [
            "http://localhost:8080",
            "http://localhost:3000",
            "http://127.0.0.1:8080",
            "http://127.0.0.1:3000",
        ]

    # Firebase Configuration
    FIREBASE_CREDENTIALS_PATH: str = "firebase-credentials.json"
    
    # Database Configuration
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)

    # Expose database URL at top level for Alembic
    @property
    def DATABASE_URL(self) -> str:
        """Get the database URL from the database settings."""
        return self.db.DATABASE_URL

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    # Sentry Configuration
    SENTRY_DSN: Optional[str] = None

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = Field(default_factory=lambda: True if ENVIRONMENT == "development" else False)

    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # Database
    SQLALCHEMY_DATABASE_URI: Optional[PostgresDsn] = None

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        if isinstance(v, str):
            return v
        
        # Use the existing DATABASE_URL property
        db = values.get("db")
        if db:
            return db.DATABASE_URL
        
        return None

    # File Upload
    UPLOAD_DIR: Path = Path("uploads")
    MAX_UPLOAD_SIZE: int = 2 * 1024 * 1024  # 2MB
    ALLOWED_EXTENSIONS: set[str] = {"jpg", "jpeg", "png", "gif"}


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Create upload directory if it doesn't exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
