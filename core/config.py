"""
Lumicoria AI — Centralized Configuration Module

ALL configuration flows through this module. No other module should read
os.environ directly. Settings are validated at import time — if a required
variable is missing, the application refuses to start.

Environment variables are loaded from .env (via pydantic-settings) and can
be overridden by actual environment variables in production.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic_settings import BaseSettings
from pydantic import AnyHttpUrl, validator, Field, HttpUrl, ConfigDict
import secrets
from functools import lru_cache
from pathlib import Path
import os
import sys


# ---------------------------------------------------------------------------
# Database Settings (nested)
# ---------------------------------------------------------------------------

class DatabaseSettings(BaseSettings):
    model_config = ConfigDict(extra="allow", case_sensitive=True)

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
    VECTOR_STORE_URL: Optional[str] = "http://localhost:8081"
    VECTOR_STORE_API_KEY: Optional[str] = None
    VECTOR_STORE_COLLECTION: str = "documents"
    VECTOR_STORE_DIMENSION: int = 768  # Default for Gemini text-embedding-004
    VECTOR_STORE_ENABLED: bool = True

    # Cassandra Settings (optional)
    CASSANDRA_ENABLED: bool = False
    CASSANDRA_HOSTS: List[str] = Field(default_factory=lambda: ["localhost"])
    CASSANDRA_PORT: int = 9042
    CASSANDRA_KEYSPACE: str = "lumicoria"
    CASSANDRA_REPLICATION_FACTOR: int = 1
    CASSANDRA_USERNAME: Optional[str] = None
    CASSANDRA_PASSWORD: Optional[str] = None
    CASSANDRA_CONNECT_TIMEOUT: int = 5
    CASSANDRA_DUAL_WRITE: bool = False


# ---------------------------------------------------------------------------
# S3 / Object Storage Settings (nested)
# ---------------------------------------------------------------------------

class S3Settings(BaseSettings):
    """S3-compatible object storage with dual-write to MinIO + Cloudflare R2."""
    model_config = ConfigDict(extra="allow", case_sensitive=True)

    # MinIO (primary)
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "lumicoria-documents"
    MINIO_USE_SSL: bool = False

    # Cloudflare R2 (backup — simultaneous dual-write)
    R2_ENDPOINT: Optional[str] = None
    R2_ACCESS_KEY: Optional[str] = None
    R2_SECRET_KEY: Optional[str] = None
    R2_BUCKET: str = "lumicoria-documents"

    # Dual-write behaviour
    DUAL_WRITE_ENABLED: bool = True
    PRESIGNED_URL_EXPIRY: int = 3600  # seconds


# ---------------------------------------------------------------------------
# Rate Limiting Settings
# ---------------------------------------------------------------------------

class RateLimitSettings(BaseSettings):
    """Per-tier rate-limiting thresholds (requests per window)."""
    model_config = ConfigDict(extra="allow", case_sensitive=True)

    # Global defaults
    ENABLED: bool = True
    WINDOW_SECONDS: int = 60

    # Endpoint-specific limits (per IP per window)
    DEFAULT_LIMIT: int = 60
    AUTH_LIMIT: int = 10         # login / signup / refresh
    AI_AGENT_LIMIT: int = 20    # agent chat / execution
    UPLOAD_LIMIT: int = 10       # file uploads
    BURST_LIMIT: int = 5         # max burst within 5 seconds


# ---------------------------------------------------------------------------
# Main Application Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    """Application settings — validated at startup."""

    model_config = ConfigDict(
        extra="allow",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )

    # ── API ─────────────────────────────────────────────────────────────
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Lumicoria AI"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "AI-powered platform for personalized learning and productivity"

    # ── Security (REQUIRED — no safe defaults) ──────────────────────────
    SECRET_KEY: str = Field(
        ...,
        description=(
            "JWT signing key. MUST be set via env var. "
            "Generate: python -c \"import secrets; print(secrets.token_hex(32))\""
        ),
    )
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30       # 30 min (was 7 days)
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    ALGORITHM: str = "HS256"

    # ── AI / LLM Provider Switching ─────────────────────────────────────
    DEFAULT_LLM_PROVIDER: str = Field(
        default="perplexity",
        description=(
            "Default LLM provider. Options: 'perplexity', 'gemini', 'openai', 'anthropic', 'mistral'. "
            "Can be overridden per-request via the API."
        ),
    )
    DEFAULT_EMBEDDING_PROVIDER: Optional[str] = Field(
        default=None,
        description=(
            "Default embedding provider. If None, falls back to DEFAULT_LLM_PROVIDER. "
            "Set to a specific provider if you want embeddings from a different source. "
            "Use 'local' for self-hosted FastEmbed (no API quota)."
        ),
    )

    # ── Local embedding provider (FastEmbed, BGE by default) ─────────────
    LOCAL_EMBEDDING_MODEL: str = Field(
        default="BAAI/bge-base-en-v1.5",
        description=(
            "HuggingFace model id used by the 'local' embedding provider. "
            "Default (bge-base-en-v1.5) is 768-dim and matches VECTOR_STORE_DIMENSION."
        ),
    )
    LOCAL_EMBEDDING_CACHE_DIR: str = Field(
        default="./models/fastembed",
        description="On-disk cache for downloaded ONNX models.",
    )
    LOCAL_EMBEDDING_BATCH_SIZE: int = Field(
        default=64,
        description="Texts per ONNX inference batch (tune for your CPU/RAM).",
    )
    LOCAL_EMBEDDING_PARALLEL: int = Field(
        default=0,
        description=(
            "Multiprocessing parallelism for FastEmbed.  "
            "0 = use all CPU cores (good for big reindex jobs); "
            "set to 1 for single-process inside web workers to avoid fork overhead."
        ),
    )
    LOCAL_EMBEDDING_WARMUP_ON_STARTUP: bool = Field(
        default=True,
        description=(
            "If True and DEFAULT_EMBEDDING_PROVIDER='local', preload the ONNX "
            "model during app startup so the first embed call is warm."
        ),
    )
    INGEST_PROCESS_POOL_WORKERS: int = Field(
        default=4,
        description=(
            "Worker count for the PDF extraction ProcessPoolExecutor. Each "
            "worker parses one page range in parallel. Set to 1 to disable."
        ),
    )
    INGEST_PDF_PAGES_PER_WORKER: int = Field(
        default=25,
        description=(
            "Pages per ProcessPoolExecutor task when extracting PDFs. "
            "Smaller = more parallelism, more pickling overhead."
        ),
    )
    INGEST_PARSER_DEFAULT: str = Field(
        default="fast",
        description=(
            "Default parser preference for rich documents: 'fast' uses "
            "PyMuPDF for digital PDFs and falls back to Docling only for "
            "scanned pages; 'docling' forces Docling for every rich format."
        ),
    )
    INGEST_CHUNK_TOKENS_PROSE: int = Field(
        default=512,
        description="Target tokens per chunk for prose (PDF/DOCX/HTML body).",
    )
    INGEST_CHUNK_TOKENS_CHAT: int = Field(
        default=1500,
        description="Target tokens per chunk for chat history transcripts.",
    )
    INGEST_CHUNK_TOKENS_CODE: int = Field(
        default=1000,
        description="Max tokens per code chunk before splitting on structure.",
    )
    INGEST_CHUNK_OVERLAP_TOKENS: int = Field(
        default=50,
        description="Token overlap between adjacent prose chunks.",
    )
    INGEST_OCR_MIN_CHARS_PER_PAGE: int = Field(
        default=50,
        description=(
            "PyMuPDF pages returning fewer than this many characters are "
            "treated as scanned and routed to the OCR-capable fallback."
        ),
    )
    INGEST_CHUNK_DEDUP_ENABLED: bool = Field(
        default=True,
        description="Drop chunks whose content_sha256 has already been seen in this run.",
    )
    INGEST_DOC_DEDUP_ENABLED: bool = Field(
        default=True,
        description=(
            "If True, a new upload whose content sha256 matches an existing "
            "ready document for this user aliases to that document instead "
            "of re-running the pipeline."
        ),
    )
    # ── Celery ─────────────────────────────────────────────────────────
    CELERY_ENABLED: bool = Field(
        default=False,
        description=(
            "When True, ingest workers run on a Celery worker process pool "
            "(requires `celery[redis]` + a running worker). Falls back to "
            "FastAPI BackgroundTasks when False — useful for dev."
        ),
    )
    CELERY_BROKER_URL: Optional[str] = Field(
        default=None,
        description="Celery broker URL. Defaults to redis://{host}:{port}/{db} when unset.",
    )
    CELERY_RESULT_BACKEND: Optional[str] = Field(
        default=None,
        description="Celery result backend URL. Defaults to the broker URL when unset.",
    )
    CELERY_TASK_ALWAYS_EAGER: bool = Field(
        default=False,
        description="Run Celery tasks inline (dev/test). Production: False.",
    )
    CELERY_WORKER_CONCURRENCY: int = Field(
        default=4,
        description="Prefork workers per Celery process.",
    )
    LLM_FALLBACK_PROVIDER: Optional[str] = Field(
        default=None,
        description=(
            "Fallback LLM provider if the primary is unavailable. "
            "Set to 'gemini' for Perplexity → Gemini fallback, or vice versa."
        ),
    )

    # ── AI / LLM Keys ──────────────────────────────────────────────────
    PERPLEXITY_API_KEY: Optional[str] = Field(
        default=None,
        description=(
            "Perplexity Sonar API key. Required if DEFAULT_LLM_PROVIDER='perplexity'. "
            "Get from https://www.perplexity.ai/settings/api"
        ),
    )
    PERPLEXITY_MODEL: str = Field(
        default="sonar-pro",
        description=(
            "Default Perplexity chat model. Options: sonar, sonar-pro, "
            "sonar-reasoning-pro, sonar-deep-research."
        ),
    )
    GEMINI_API_KEY: Optional[str] = Field(
        default=None,
        description=(
            "Google Gemini API key. Required if DEFAULT_LLM_PROVIDER='gemini'. "
            "Get from https://aistudio.google.com/apikey"
        ),
    )
    GEMINI_MODEL: str = Field(
        default="gemini-2.5-flash",
        description=(
            "Default Gemini chat model. Options: gemini-2.5-pro-preview-06-05, "
            "gemini-2.5-flash-preview-05-20, gemini-2.0-pro, gemini-2.0-flash, "
            "gemini-2.0-flash-lite, gemini-1.5-pro, gemini-1.5-flash."
        ),
    )
    GEMINI_TIMEOUT: int = Field(
        default=60,
        description="HTTP timeout in seconds for Gemini API calls.",
    )
    OPENAI_API_KEY: Optional[str] = Field(
        default=None,
        description=(
            "OpenAI API key. Required if DEFAULT_LLM_PROVIDER='openai'. "
            "Get from https://platform.openai.com/api-keys"
        ),
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4o-mini",
        description=(
            "Default OpenAI chat model. Options: gpt-4o, gpt-4o-mini, "
            "gpt-4-turbo, gpt-3.5-turbo, o1, o1-mini, o3-mini."
        ),
    )
    OPENAI_TIMEOUT: int = Field(
        default=30,
        description="HTTP timeout in seconds for OpenAI API calls.",
    )
    ANTHROPIC_API_KEY: Optional[str] = Field(
        default=None,
        description=(
            "Anthropic API key. Required if DEFAULT_LLM_PROVIDER='anthropic'. "
            "Get from https://console.anthropic.com/settings/keys"
        ),
    )
    ANTHROPIC_MODEL: str = Field(
        default="claude-3-5-sonnet-latest",
        description=(
            "Default Anthropic chat model. Options: claude-3-5-sonnet-latest, "
            "claude-3-5-haiku-latest, claude-3-opus-latest, claude-sonnet-4-20250514."
        ),
    )
    ANTHROPIC_TIMEOUT: int = Field(
        default=60,
        description="HTTP timeout in seconds for Anthropic API calls.",
    )
    MISTRAL_API_KEY: Optional[str] = Field(
        default=None,
        description=(
            "Mistral AI API key. Required if DEFAULT_LLM_PROVIDER='mistral'. "
            "Get from https://console.mistral.ai/api-keys"
        ),
    )
    MISTRAL_MODEL: str = Field(
        default="mistral-large-latest",
        description=(
            "Default Mistral chat model. Options: mistral-large-latest, "
            "mistral-small-latest, open-mistral-nemo, codestral-latest, "
            "pixtral-large-latest, mistral-saba-latest."
        ),
    )
    MISTRAL_TIMEOUT: int = Field(
        default=30,
        description="HTTP timeout in seconds for Mistral API calls.",
    )

    # ── Stripe Billing ──────────────────────────────────────────────────
    STRIPE_SECRET_KEY: str = Field(
        default="",
        description=(
            "Stripe secret key (sk_live_... or sk_test_...). "
            "Get from https://dashboard.stripe.com/apikeys"
        ),
    )
    STRIPE_WEBHOOK_SECRET: str = Field(
        default="",
        description=(
            "Stripe webhook endpoint signing secret (whsec_...). "
            "Get from https://dashboard.stripe.com/webhooks"
        ),
    )
    STRIPE_PUBLISHABLE_KEY: str = Field(
        default="",
        description="Stripe publishable key (pk_live_... or pk_test_...).",
    )
    STRIPE_PRICE_STARTER_MONTHLY: str = Field(
        default="",
        description="Stripe Price ID for Starter plan (monthly).",
    )
    STRIPE_PRICE_STARTER_YEARLY: str = Field(
        default="",
        description="Stripe Price ID for Starter plan (yearly).",
    )
    STRIPE_PRICE_PRO_MONTHLY: str = Field(
        default="",
        description="Stripe Price ID for Professional plan (monthly).",
    )
    STRIPE_PRICE_PRO_YEARLY: str = Field(
        default="",
        description="Stripe Price ID for Professional plan (yearly).",
    )
    STRIPE_PRICE_ENTERPRISE: str = Field(
        default="",
        description="Stripe Price ID for Enterprise plan.",
    )
    STRIPE_SUCCESS_URL: str = Field(
        default="http://localhost:3000/dashboard?billing=success",
        description="Redirect URL after successful Stripe Checkout.",
    )
    STRIPE_CANCEL_URL: str = Field(
        default="http://localhost:3000/pricing?billing=canceled",
        description="Redirect URL after canceled Stripe Checkout.",
    )

    # ── Google OAuth ────────────────────────────────────────────────────
    GOOGLE_OAUTH_CLIENT_ID: str = Field(
        default="",
        description="Google OAuth 2.0 Client ID (web). Required for Google Sign-In.",
    )

    # ── Firebase ────────────────────────────────────────────────────────
    FIREBASE_CREDENTIALS_PATH: str = Field(
        default_factory=lambda: str(Path(__file__).parent.parent / "firebase-credentials.json"),
        description="Path to Firebase Admin SDK service account JSON.",
    )

    # ── Database (backward-compat aliases — prefer db.* for new code) ──
    MONGODB_URL: str = "mongodb://localhost:27017"
    MONGODB_DB_NAME: str = "lumicoria"

    # ── Nested database config ─────────────────────────────────────────
    db: DatabaseSettings = Field(default_factory=DatabaseSettings)
    s3: S3Settings = Field(default_factory=S3Settings)

    # ── PostgreSQL / SQLAlchemy (optional) ─────────────────────────────
    POSTGRES_ENABLED: bool = True
    POSTGRES_DUAL_WRITE: bool = True
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = "postgres"
    POSTGRES_PASSWORD: str = "lumicoria"
    POSTGRES_DB: str = "lumicoria"
    SQLALCHEMY_DATABASE_URI: Optional[str] = None
    SQLALCHEMY_ECHO: bool = False
    SQLALCHEMY_POOL_SIZE: int = 5
    SQLALCHEMY_MAX_OVERFLOW: int = 10

    # ── Rate Limiting ──────────────────────────────────────────────────
    RATE_LIMIT_PER_MINUTE: int = 60
    rate_limit: RateLimitSettings = Field(default_factory=RateLimitSettings)

    # ── CORS ───────────────────────────────────────────────────────────
    BACKEND_CORS_ORIGINS: List[str] = Field(
        default=[
            "http://localhost:8080",
            "http://localhost:3000",
            "http://127.0.0.1:8080",
            "http://127.0.0.1:3000",
        ],
        description="Allowed CORS origins. In production set to your actual domains.",
    )

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
        return v or []

    @validator("SQLALCHEMY_DATABASE_URI", pre=True)
    def assemble_sqlalchemy_uri(cls, v: Optional[str], values: Dict[str, Any]) -> Optional[str]:
        if v:
            return v
        host = values.get("POSTGRES_HOST")
        port = values.get("POSTGRES_PORT")
        user = values.get("POSTGRES_USER")
        password = values.get("POSTGRES_PASSWORD")
        db = values.get("POSTGRES_DB")
        if not host or not user or not db:
            return None
        if password:
            return f"postgresql://{user}:{password}@{host}:{port}/{db}"
        return f"postgresql://{user}@{host}:{port}/{db}"

    # ── Environment ────────────────────────────────────────────────────
    ENVIRONMENT: str = "development"
    DEBUG: bool = False  # Default OFF — opt-in only

    @validator("DEBUG", pre=True)
    def set_debug(cls, v: Optional[bool], values: Dict[str, Any]) -> bool:
        if v is not None:
            return bool(v)
        return values.get("ENVIRONMENT", "production") == "development"

    # ── Observability ──────────────────────────────────────────────────
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    SENTRY_DSN: Optional[str] = None
    SENTRY_TRACES_SAMPLE_RATE: float = Field(
        default=0.1,
        description="Sentry performance traces sample rate (0.0-1.0). Use 1.0 only in dev.",
    )

    # ── Speech-to-Text (Faster-Whisper) ─────────────────────────────────
    STT_MODEL_SIZE: str = Field(
        default="base",
        description="Faster-Whisper model size: tiny, base, small, medium, large-v3.",
    )
    STT_DEVICE: str = Field(
        default="cpu",
        description="Device for STT inference: cpu or cuda.",
    )
    STT_COMPUTE_TYPE: str = Field(
        default="int8",
        description="Compute type: int8 (CPU), float16 (GPU), float32.",
    )
    STT_LANGUAGE: Optional[str] = Field(
        default="en",
        description="Default STT language code (ISO 639-1). None = auto-detect.",
    )
    STT_CHUNK_DURATION: float = Field(
        default=4.0,
        description="Seconds of audio to buffer before transcribing a chunk.",
    )

    # ── File Upload ────────────────────────────────────────────────────
    UPLOAD_DIR: str = str(Path(__file__).parent.parent / "uploads")
    MAX_UPLOAD_SIZE: int = 10 * 1024 * 1024  # 10 MB
    ALLOWED_EXTENSIONS: set[str] = {"jpg", "jpeg", "png", "gif", "pdf", "doc", "docx"}

    # ── Email Providers ───────────────────────────────────────────────
    # SendGrid (primary) - Get key from https://sendgrid.com
    SENDGRID_API_KEY: Optional[str] = None  # Set via SENDGRID_API_KEY env var
    
    # Resend (fallback) - Get key from https://resend.com
    RESEND_API_KEY: Optional[str] = None  # Set via RESEND_API_KEY env var
    
    # Email configuration
    EMAIL_FROM_ADDRESS: str = "noreply@lumicoria.ai"
    EMAIL_FROM_NAME: str = "Lumicoria.ai"
    EMAIL_SANDBOX_MODE: bool = False  # Set True to disable actual sending (testing)

    # ── External Integrations (all optional) ───────────────────────────
    NOTION_API_KEY: Optional[str] = None
    GOOGLE_CREDENTIALS_FILE: Optional[str] = None
    GOOGLE_TOKEN_FILE: Optional[str] = None
    SLACK_BOT_TOKEN: Optional[str] = None
    SLACK_APP_TOKEN: Optional[str] = None
    SLACK_SIGNING_SECRET: Optional[str] = None

    # ── OAuth 2.0 Integration Credentials ─────────────────────────────
    # Google Workspace (GOOGLE_OAUTH_CLIENT_ID is above in the Google OAuth section)
    GOOGLE_OAUTH_CLIENT_SECRET: Optional[str] = None

    # Slack OAuth
    SLACK_CLIENT_ID: Optional[str] = None
    SLACK_CLIENT_SECRET: Optional[str] = None

    # Notion OAuth
    NOTION_OAUTH_CLIENT_ID: Optional[str] = None
    NOTION_OAUTH_CLIENT_SECRET: Optional[str] = None

    # Salesforce OAuth
    SALESFORCE_CLIENT_ID: Optional[str] = None
    SALESFORCE_CLIENT_SECRET: Optional[str] = None

    # Frontend URL for OAuth redirect URI construction
    FRONTEND_URL: str = "http://localhost:3000"

    # Public-facing base URL used in customer-facing emails / share links
    # (support portal, ticket status URLs, magic links, etc.).
    # Set this to your production domain — falls back to FRONTEND_URL
    # when unset so dev environments don't need to configure both.
    PUBLIC_BASE_URL: Optional[str] = None

    # ── Cloud Providers (optional) ─────────────────────────────────────
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_API_VERSION: str = "2023-05-15"
    GOOGLE_CLOUD_PROJECT: Optional[str] = None
    GOOGLE_APPLICATION_CREDENTIALS: Optional[str] = None
    AWS_ACCESS_KEY_ID: Optional[str] = None
    AWS_SECRET_ACCESS_KEY: Optional[str] = None
    AWS_REGION: str = "us-east-1"

    # ── Encryption ─────────────────────────────────────────────────────
    INTEGRATION_ENCRYPTION_KEY: Optional[str] = Field(
        default=None,
        description=(
            "Fernet key for encrypting integration credentials. "
            "Generate: python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
        ),
    )

    # ── Convenience helpers ────────────────────────────────────────────

    @property
    def is_production(self) -> bool:
        return self.ENVIRONMENT == "production"

    @property
    def is_development(self) -> bool:
        return self.ENVIRONMENT == "development"

    @property
    def docs_enabled(self) -> bool:
        """Disable Swagger/ReDoc in production unless DEBUG is on."""
        return not self.is_production or self.DEBUG


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

@lru_cache()
def get_settings() -> Settings:
    """
    Create and cache the Settings singleton.

    Pydantic will raise a ValidationError at this point if any required
    field (SECRET_KEY, PERPLEXITY_API_KEY) is missing from the environment.
    This is intentional — fail fast, fail loud.
    """
    try:
        return Settings()
    except Exception as e:
        print(f"\n{'='*60}", file=sys.stderr)
        print("FATAL: Application configuration is invalid.", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        print(f"\n{e}\n", file=sys.stderr)
        print("Required environment variables:", file=sys.stderr)
        print("  SECRET_KEY         — JWT signing key (64+ hex chars)", file=sys.stderr)
        print("  PERPLEXITY_API_KEY — Perplexity Sonar API key (if using Perplexity)", file=sys.stderr)
        print("  GEMINI_API_KEY     — Google Gemini API key (if using Gemini)", file=sys.stderr)
        print("  DEFAULT_LLM_PROVIDER — 'perplexity' or 'gemini' (default: perplexity)", file=sys.stderr)
        print("\nSee backend/.env.example for the full list.\n", file=sys.stderr)
        sys.exit(1)


settings = get_settings()

# Create upload directory if it doesn't exist
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
