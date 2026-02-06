# ============================================================================
# Lumicoria AI — Backend Dockerfile (Production)
# ============================================================================
# Multi-stage build: install deps in builder, copy to slim runtime.
# Non-root user, env-driven config, fast startup.
# ============================================================================

# --- Stage 1: Build dependencies ---
FROM python:3.11-slim AS builder

WORKDIR /build

# Install system deps needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# --- Stage 2: Runtime ---
FROM python:3.11-slim AS runtime

# Security: run as non-root
RUN groupadd -r lumicoria && useradd -r -g lumicoria -d /app -s /sbin/nologin lumicoria

WORKDIR /app

# Install only runtime system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY . /app/backend/

# Create upload directory
RUN mkdir -p /app/backend/uploads && chown -R lumicoria:lumicoria /app

# Switch to non-root user
USER lumicoria

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Entrypoint — production mode by default
# Use ENVIRONMENT=development to enable reload (single worker)
CMD ["python", "-m", "uvicorn", "backend.main:app", \
     "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "4", "--log-level", "info"]
