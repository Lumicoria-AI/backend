from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException
import sentry_sdk
import structlog
from contextlib import asynccontextmanager
from api.v1.api import api_router
from core.config import settings
import time
import prometheus_client
from prometheus_client import Counter, Histogram
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Initialize Sentry
if settings.SENTRY_DSN:
    sentry_sdk.init(
        dsn=settings.SENTRY_DSN,
        traces_sample_rate=1.0,
        profiles_sample_rate=1.0,
    )

# Initialize logging
structlog.configure(
    processors=[
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ]
)
logger = structlog.get_logger()

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
    # Startup
    logger.info("Starting up application")
    # Create upload directory if it doesn't exist
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Initialize services
    from services.context_service import initialize_context_service
    from services.document_processor import document_processor
    
    logger.info("Initializing context service and document processor")
    await initialize_context_service()
    await document_processor.initialize()
    logger.info("Services initialized successfully")
    
    yield
    # Shutdown
    logger.info("Shutting down application")

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
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
)

# Set up CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.BACKEND_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Range", "X-Content-Range"],
    max_age=3600,
)

# Mount static files for user uploads
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")

# Request logging middleware
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
            client_ip=request.client.host,
            user_agent=request.headers.get("user-agent")
        )
    
    return response

# Exception handlers
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

# Health check endpoint
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "1.0.0",
        "environment": settings.ENVIRONMENT
    }

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
        "version": "1.0.0",
        "docs_url": "/docs",
        "redoc_url": "/redoc",
        "openapi_url": f"{settings.API_V1_STR}/openapi.json",
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=4,
        log_level=settings.LOG_LEVEL.lower()
    ) 
