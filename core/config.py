from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator, Field
import secrets
from functools import lru_cache


class DatabaseSettings(BaseSettings):
    # PostgreSQL Settings
    POSTGRES_SERVER: str
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    POSTGRES_PORT: int = 5432
    POSTGRES_POOL_SIZE: int = 20
    POSTGRES_MAX_OVERFLOW: int = 10
    SQLALCHEMY_DATABASE_URI: Optional[str] = None

    # MongoDB Settings
    MONGODB_URI: str
    MONGODB_DB: str
    MONGODB_MAX_POOL_SIZE: int = 100
    MONGODB_MIN_POOL_SIZE: int = 10

    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_POOL_SIZE: int = 10

    # Cassandra Settings
    CASSANDRA_HOSTS: List[str]
    CASSANDRA_PORT: int = 9042
    CASSANDRA_USERNAME: Optional[str] = None
    CASSANDRA_PASSWORD: Optional[str] = None
    CASSANDRA_KEYSPACE: str
    CASSANDRA_REPLICATION_FACTOR: int = 1

    # Vector Store Settings
    VECTOR_STORE_TYPE: str = "weaviate"  # weaviate, qdrant, or chroma
    VECTOR_STORE_URL: str
    VECTOR_STORE_API_KEY: Optional[str] = None
    VECTOR_STORE_COLLECTION: str = "documents"
    VECTOR_STORE_DIMENSION: int = 1536  # Default for OpenAI embeddings

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> str:
        if isinstance(v, str):
            return v
        return f"postgresql+asyncpg://{values.get('POSTGRES_USER')}:{values.get('POSTGRES_PASSWORD')}@{values.get('POSTGRES_SERVER')}:{values.get('POSTGRES_PORT')}/{values.get('POSTGRES_DB')}"

    class Config:
        case_sensitive = True


class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = secrets.token_urlsafe(32)
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # CORS Configuration
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: str | List[str]) -> List[str] | str:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # Firebase Configuration
    FIREBASE_CREDENTIALS_PATH: str = "firebase-credentials.json"
    
    # Database Configuration
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    # Sentry Configuration
    SENTRY_DSN: Optional[str] = None

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = Field(default_factory=lambda: True if ENVIRONMENT == "development" else False)

    class Config:
        case_sensitive = True
        env_file = ".env"
        env_nested_delimiter = "__"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings() 
