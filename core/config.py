from typing import List, Optional, Dict, Any, Union
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator, Field, HttpUrl, ConfigDict
import secrets
from functools import lru_cache
from pathlib import Path


class DatabaseSettings(BaseSettings):
    model_config = ConfigDict(extra='allow', case_sensitive=True)

    # MongoDB Settings
    MONGODB_URI: str = "mongodb://localhost:27017"
    MONGODB_DB: str = "lumicoria"
    MONGODB_MAX_POOL_SIZE: int = 100
    MONGODB_MIN_POOL_SIZE: int = 10

    # Redis Settings
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    REDIS_DB: int = 0
    REDIS_POOL_SIZE: int = 10

    # Vector Store Settings
    VECTOR_STORE_TYPE: str = "weaviate"  # weaviate, qdrant, or chroma
    VECTOR_STORE_URL: Optional[str] = "http://localhost:8080"
    VECTOR_STORE_API_KEY: Optional[str] = None
    VECTOR_STORE_COLLECTION: str = "documents"
    VECTOR_STORE_DIMENSION: int = 1536  # Default for OpenAI embeddings


class Settings(BaseSettings):
    """Application settings."""
    
    # API Settings
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Lumicoria.ai"
    
    # Security
    SECRET_KEY: str = "your-secret-key-here"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 7  # 7 days
    
    # Database
    MONGODB_URL: str = "mongodb://localhost:27017/lumicoria"
    DATABASE_NAME: str = "lumicoria"
    
    # OpenAI
    OPENAI_API_KEY: str = "your-openai-api-key-here"
    OPENAI_MODEL: str = "gpt-4-turbo-preview"
    
    # Notion
    NOTION_API_KEY: Optional[str] = None
    
    # Google Workspace
    GOOGLE_CREDENTIALS_FILE: Optional[str] = None
    GOOGLE_TOKEN_FILE: Optional[str] = None
    
    # Slack
    SLACK_BOT_TOKEN: Optional[str] = None
    SLACK_APP_TOKEN: Optional[str] = None
    SLACK_SIGNING_SECRET: Optional[str] = None
      # Logging
    LOG_LEVEL: str = "INFO"
    
    model_config = ConfigDict(extra='allow', env_file=".env", case_sensitive=True)

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

    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60

    # Sentry Configuration
    SENTRY_DSN: Optional[str] = None

    # Environment
    ENVIRONMENT: str = "development"
    DEBUG: bool = True  # Will be overridden by environment variable if set

    @validator("DEBUG", pre=True)
    def set_debug(cls, v: Optional[bool], values: Dict[str, Any]) -> bool:
        if v is not None:
            return v
        return values.get("ENVIRONMENT", "development") == "development"

    # Logging
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # File Upload
    UPLOAD_DIR: Path = Path("uploads")
    MAX_UPLOAD_SIZE: int = 2 * 1024 * 1024  # 2MB
    ALLOWED_EXTENSIONS: set[str] = {"jpg", "jpeg", "png", "gif"}

    # Email Settings
    SMTP_SERVER: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USERNAME: str = "jacobasuquo199@gmail.com"
    SMTP_PASSWORD: str = "yesrqkhohhivjdkp"
    SMTP_FROM_EMAIL: str = "noreply@lumicoria.ai"
    SMTP_FROM_NAME: str = "Lumicoria.ai"


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()

# Create upload directory if it doesn't exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
