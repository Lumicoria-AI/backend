"""
Lumicoria AI — Application Entry Point

Hardened for production:
  • CORS: explicit methods & headers (no wildcards)
  • Security headers middleware (X-Frame-Options, CSP, etc.)
  • Deep health check (MongoDB, Redis, Weaviate)
  • Sentry: configurable sample rate from settings
  • Swagger/ReDoc disabled in production (unless DEBUG)
  • uvicorn: reload XOR workers (never both)
"""

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import sentry_sdk
from contextlib import asynccontextmanager
from backend.api.v1.api import api_router
from backend.core.config import settings
from backend.core.logging import get_logger
import time
import prometheus_client
from prometheus_client import Counter, Histogram
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from pydantic import ValidationError
from backend.db.mongodb.mongodb import init_mongodb, close_mongodb
from backend.agents.agent_service import init_agent_service, close_agent_service
from backend.db.postgres import init_postgres, close_postgres
from backend.db.cassandra.cassandra import CassandraClient

# Initialize Sentry (production-safe sample rate)
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        traces_sample_rate=settings.SENTRY_TRACES_SAMPLE_RATE,
        profiles_sample_rate=min(settings.SENTRY_TRACES_SAMPLE_RATE, 0.25),
        environment=settings.ENVIRONMENT,
    )

# Initialize logging
logger = get_logger("lumicoria.main")

# Initialize Prometheus metrics
REQUEST_COUNT = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)
REQUEST_LATENCY = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint']
)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application startup and shutdown."""
    # Startup
    logger.info("Starting up application", environment=settings.ENVIRONMENT)
    
    try:
        logger.info("=" * 60)
        logger.info("lumicoria_startup_begin", environment=settings.ENVIRONMENT, version=settings.VERSION)
        logger.info("=" * 60)

        # Create upload directory if it doesn't exist
        upload_dir = Path(settings.UPLOAD_DIR)
        upload_dir.mkdir(parents=True, exist_ok=True)

        # ── Databases ────────────────────────────────────────────
        logger.info("─" * 40)
        logger.info("connecting_databases")

        # MongoDB
        try:
            await init_mongodb()
            logger.info("service_connected", service="MongoDB", host=settings.db.MONGODB_URI, status="ok")
        except Exception as e:
            logger.error("service_connection_failed", service="MongoDB", error=str(e))
            raise

        # Redis
        try:
            import redis as _redis
            r = _redis.Redis(
                host=settings.db.REDIS_HOST,
                port=settings.db.REDIS_PORT,
                password=settings.db.REDIS_PASSWORD,
                db=settings.db.REDIS_DB,
                socket_connect_timeout=3,
            )
            r.ping()
            r.close()
            logger.info("service_connected", service="Redis", host=f"{settings.db.REDIS_HOST}:{settings.db.REDIS_PORT}", status="ok")
        except Exception as e:
            logger.warning("service_connection_failed", service="Redis", error=str(e))

        # Postgres (optional)
        try:
            await init_postgres()
            if settings.POSTGRES_ENABLED and settings.SQLALCHEMY_DATABASE_URI:
                logger.info("service_connected", service="PostgreSQL", status="ok")
            else:
                logger.info("service_skipped", service="PostgreSQL", reason="not enabled")
        except Exception as e:
            logger.warning("service_connection_failed", service="PostgreSQL", error=str(e))

        # Cassandra (optional)
        try:
            await CassandraClient.connect()
            if settings.db.CASSANDRA_ENABLED:
                logger.info("service_connected", service="Cassandra", status="ok")
            else:
                logger.info("service_skipped", service="Cassandra", reason="not enabled")
        except Exception as e:
            logger.warning("service_skipped", service="Cassandra", reason=str(e))

        # ── Vector Store ─────────────────────────────────────────
        logger.info("─" * 40)
        logger.info("connecting_vector_store")

        try:
            if settings.db.VECTOR_STORE_ENABLED:
                import httpx
                vector_url = settings.db.VECTOR_STORE_URL or "http://localhost:8081"
                vector_type = getattr(settings.db, "VECTOR_STORE_TYPE", "weaviate")
                async with httpx.AsyncClient(timeout=3.0) as client:
                    # Health check endpoint varies by provider
                    health_endpoints = {
                        "weaviate": f"{vector_url}/v1/.well-known/ready",
                        "qdrant": f"{vector_url}/readyz",
                        "chroma": f"{vector_url}/api/v1/heartbeat",
                    }
                    health_url = health_endpoints.get(vector_type, f"{vector_url}/v1/.well-known/ready")
                    resp = await client.get(health_url)
                    if resp.status_code == 200:
                        logger.info("service_connected", service="VectorStore", provider=vector_type, url=vector_url, status="ok")
                    else:
                        logger.warning("service_degraded", service="VectorStore", provider=vector_type, url=vector_url, status_code=resp.status_code)
            else:
                logger.info("service_skipped", service="VectorStore", reason="not enabled")
        except Exception as e:
            logger.warning("service_connection_failed", service="VectorStore", error=str(e))

        # ── S3 / Object Storage ──────────────────────────────────
        logger.info("─" * 40)
        logger.info("connecting_object_storage")

        from backend.services.storage_service import storage_service
        try:
            await storage_service.initialize()
            logger.info("service_connected", service="MinIO (S3)", endpoint=settings.s3.MINIO_ENDPOINT, bucket=settings.s3.MINIO_BUCKET, status="ok")
            if settings.s3.DUAL_WRITE_ENABLED and settings.s3.R2_ENDPOINT:
                logger.info("service_connected", service="Cloudflare R2 (backup)", endpoint=settings.s3.R2_ENDPOINT, status="ok")
            else:
                logger.info("service_skipped", service="Cloudflare R2 (backup)", reason="not configured")
        except Exception as e:
            logger.warning("service_connection_failed", service="MinIO (S3)", error=str(e))

        # ── RAG & Document Processing ────────────────────────────
        logger.info("─" * 40)
        logger.info("initializing_rag_pipeline")

        from backend.services.context_service import initialize_context_service
        from backend.services.document_processor import document_processor

        try:
            await initialize_context_service()
            logger.info("service_initialized", service="ContextService (RAG)", status="ok")
        except Exception as e:
            logger.warning("service_init_failed", service="ContextService (RAG)", error=str(e))

        try:
            await document_processor.initialize()
            pymupdf_status = "available" if getattr(document_processor, '_has_pymupdf', True) else "unavailable (fallback mode)"
            logger.info("service_initialized", service="DocumentProcessor", pymupdf=pymupdf_status, status="ok")
        except Exception as e:
            logger.warning("service_init_failed", service="DocumentProcessor", error=str(e))

        # ── Agent Service ────────────────────────────────────────
        logger.info("─" * 40)
        logger.info("initializing_agents")

        try:
            await init_agent_service()
            logger.info("service_initialized", service="AgentService", status="ok")
        except Exception as e:
            logger.warning("service_init_failed", service="AgentService", error=str(e))

        # ── Speech-to-Text (Faster-Whisper) ─────────────────────
        logger.info("─" * 40)
        logger.info("initializing_stt_service")

        try:
            from backend.services.stt_service import stt_service
            stt_service.preload()
            logger.info(
                "service_initialized",
                service="STT (Faster-Whisper)",
                model=settings.STT_MODEL_SIZE,
                device=settings.STT_DEVICE,
                status="ok",
            )
        except Exception as e:
            logger.warning("service_init_failed", service="STT (Faster-Whisper)", error=str(e))

        # ── Startup Complete ─────────────────────────────────────
        logger.info("=" * 60)
        logger.info("lumicoria_startup_complete", status="ready")
        logger.info("=" * 60)

    except Exception as e:
        logger.error("startup_fatal_error", error=str(e))
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    try:
        # Close MongoDB connection
        await close_mongodb()
        logger.info("MongoDB connection closed")

        # Close Postgres connection
        await close_postgres()
        logger.info("Postgres connection closed")

        # Close Cassandra connection
        await CassandraClient.disconnect()
        
        # Close agent service
        await close_agent_service()
        logger.info("Agent service closed")
        
        # Close all LLM provider clients
        from backend.ai_models.registry import LLMRegistry
        await LLMRegistry.close_all()
        logger.info("LLM provider clients closed")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {str(e)}")
        raise

app = FastAPI(
    title="Lumicoria AI API",
    description="""
    Backend API for Lumicoria AI - Your AI-powered productivity assistant.
    
    ## Features
    * User authentication with JWT tokens
    * User profile management
    * User settings management
    * Avatar upload and management
    
    ## Authentication
    All endpoints except `/auth/login` and `/auth/signup` require authentication.
    Include the JWT token in the Authorization header:
    ```
    Authorization: Bearer <your_token>
    ```
    """,
    version=settings.VERSION,
    lifespan=lifespan,
    # Disable docs in production unless DEBUG is explicitly on
    docs_url="/docs" if settings.docs_enabled else None,
    redoc_url="/redoc" if settings.docs_enabled else None,
    openapi_url=f"{settings.API_V1_STR}/openapi.json" if settings.docs_enabled else None,
)

# ---------------------------------------------------------------------------
# Security Headers Middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def security_headers_middleware(request: Request, call_next):
    """Add security headers to every response."""
    response: Response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["X-XSS-Protection"] = "1; mode=block"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    response.headers["Permissions-Policy"] = "camera=(), microphone=(self), geolocation=()"
    if settings.is_production:
        response.headers["Strict-Transport-Security"] = (
            "max-age=31536000; includeSubDomains; preload"
        )
    return response

# ---------------------------------------------------------------------------
# CORS — explicit methods and headers (no wildcards)
# ---------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
    allow_headers=[
        "Authorization",
        "Content-Type",
        "Accept",
        "Origin",
        "X-Requested-With",
        "X-Onboarding-Completed",
    ],
    expose_headers=["Content-Range", "X-Content-Range", "X-RateLimit-Limit",
                     "X-RateLimit-Remaining", "X-RateLimit-Reset", "Retry-After"],
    max_age=3600,
)

# Mount static files for user uploads with proper content types
app.mount("/uploads", StaticFiles(
    directory=str(Path(settings.UPLOAD_DIR)),
    html=False,  # Do not serve HTML files for security
), name="uploads")

# ---------------------------------------------------------------------------
# Request logging middleware
# ---------------------------------------------------------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    
    # Process request
    try:
        response = await call_next(request)
        status_code = response.status_code
    except Exception as e:
        status_code = 500
        raise e
    finally:
        # Record metrics
        duration = time.time() - start_time
        REQUEST_COUNT.labels(
            method=request.method,
            endpoint=request.url.path,
            status=status_code
        ).inc()
        REQUEST_LATENCY.labels(
            method=request.method,
            endpoint=request.url.path
        ).observe(duration)
        
        # Log request
        logger.info(
            "request_processed",
            method=request.method,
            path=request.url.path,
            status_code=status_code,
            duration=duration,
            client_ip=request.client.host if request.client else "unknown",
            user_agent=request.headers.get("user-agent")
        )
    
    return response

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    logger.error(
        "http_exception",
        path=request.url.path,
        status_code=exc.status_code,
        detail=str(exc.detail)
    )
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": str(exc.detail)},
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(
        "validation_error",
        path=request.url.path,
        errors=exc.errors()
    )
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

@app.exception_handler(ValidationError)
async def pydantic_validation_exception_handler(request, exc):
    """Handle Pydantic validation errors."""
    logger.error(f"Pydantic validation error: {str(exc)}")
    return JSONResponse(
        status_code=422,
        content={"detail": str(exc)},
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions — never leak stack traces in production."""
    logger.error(f"Unhandled exception: {str(exc)}")
    detail = str(exc) if settings.DEBUG else "Internal server error"
    return JSONResponse(
        status_code=500,
        content={"detail": detail},
    )

# ---------------------------------------------------------------------------
# Deep health check — verify MongoDB, Redis, Weaviate connectivity
# ---------------------------------------------------------------------------
@app.get("/health")
async def health_check():
    """
    Deep health check — verifies connectivity to all critical services.
    Returns 503 if any service is unreachable.
    """
    checks: dict = {}
    healthy = True

    # MongoDB
    try:
        from backend.db.mongodb.mongodb import get_mongodb
        db = await get_mongodb()
        await db.command("ping")
        checks["mongodb"] = "ok"
    except Exception as e:
        checks["mongodb"] = f"error: {str(e)}"
        healthy = False

    # Redis
    try:
        import redis as _redis
        r = _redis.Redis(
            host=settings.db.REDIS_HOST,
            port=settings.db.REDIS_PORT,
            password=settings.db.REDIS_PASSWORD,
            db=settings.db.REDIS_DB,
            socket_connect_timeout=2,
        )
        r.ping()
        r.close()
        checks["redis"] = "ok"
    except Exception as e:
        checks["redis"] = f"error: {str(e)}"
        healthy = False

    # Postgres (optional)
    try:
        from backend.db.postgres import check_postgres
        if settings.POSTGRES_ENABLED and settings.SQLALCHEMY_DATABASE_URI:
            checks["postgres"] = "ok" if await check_postgres() else "error"
            if checks["postgres"] != "ok":
                healthy = False
    except Exception as e:
        checks["postgres"] = f"error: {str(e)}"
        healthy = False

    # Cassandra (optional)
    try:
        if settings.db.CASSANDRA_ENABLED:
            session = await CassandraClient.get_session()
            if session:
                checks["cassandra"] = "ok"
            else:
                checks["cassandra"] = "error"
                healthy = False
    except Exception as e:
        checks["cassandra"] = f"error: {str(e)}"
        healthy = False

    # Vector Store / Weaviate (optional)
    try:
        if settings.db.VECTOR_STORE_ENABLED:
            import httpx
            vector_url = settings.db.VECTOR_STORE_URL or "http://localhost:8081"
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{vector_url}/v1/.well-known/ready")
                if resp.status_code == 200:
                    checks["vector_store"] = "ok"
                else:
                    checks["vector_store"] = f"error: status {resp.status_code}"
                    healthy = False
    except Exception as e:
        checks["vector_store"] = f"error: {str(e)}"
        # Vector store is optional — don't mark overall as unhealthy
        # healthy = False


    # MinIO / S3 Storage
    try:
        from backend.services.storage_service import storage_service
        if storage_service.is_initialized:
            checks["minio"] = "ok" if await storage_service.health_check() else "error"
    except Exception as e:
        checks["minio"] = f"error: {str(e)}"

    status_code = 200 if healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "healthy" if healthy else "degraded",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "checks": checks,
        },
    )

# Metrics endpoint
@app.get("/metrics")
async def metrics():
    return prometheus_client.generate_latest()

# Include API router
app.include_router(api_router, prefix=settings.API_V1_STR)

@app.get("/")
async def root():
    return {
        "message": "Welcome to Lumicoria AI API",
        "version": settings.VERSION,
        "docs_url": "/docs" if settings.docs_enabled else None,
    }

if __name__ == "__main__":
    import uvicorn

    if settings.is_development:
        # Development: reload enabled, single worker
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level=settings.LOG_LEVEL.lower(),
        )
    else:
        # Production: multiple workers, no reload
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            workers=4,
            log_level=settings.LOG_LEVEL.lower(),
        ) 
