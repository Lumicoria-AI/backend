"""
Lumicoria AI — Security Module

Handles JWT creation/verification, Firebase auth fallback, password hashing,
and production-grade Redis-backed rate limiting.

SECURITY FIX: Removed jwt.decode(token, options={"verify_signature": False})
which allowed complete authentication bypass.
"""

from datetime import datetime, timedelta
from typing import Any, Dict, Optional, Union
from jose import jwt, JWTError, ExpiredSignatureError
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import auth, credentials
from backend.core.config import settings
import structlog
from functools import wraps
import time

logger = structlog.get_logger()
security = HTTPBearer()

# ---------------------------------------------------------------------------
# Firebase Admin — initialise once per process (idempotent)
# ---------------------------------------------------------------------------
# Celery workers import this module after the push_notification_service has
# already called `initialize_app()`, which used to raise
# "The default Firebase app already exists" and crash the task.  Guard by
# checking for an existing default app first — same pattern used in
# push_notification_service._initialize_firebase.
try:
    firebase_admin.get_app()  # raises ValueError when no default app exists
except ValueError:
    try:
        cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
    except Exception as e:
        logger.error("Failed to initialize Firebase", error=str(e))
        raise
except Exception as e:  # noqa: BLE001
    # Any other unexpected error during the existence check — log and move on,
    # so an auth-side issue can't take down task workers.
    logger.warning("firebase_get_app_unexpected_error", error=str(e))

pwd_context = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")
ALGORITHM = settings.ALGORITHM


# ---------------------------------------------------------------------------
# JWT helpers
# ---------------------------------------------------------------------------

def create_access_token(
    subject: Union[str, Any],
    expires_delta: Optional[timedelta] = None,
    user_data: Optional[Dict[str, Any]] = None,
) -> str:
    """Create a signed JWT access token."""
    expire = datetime.utcnow() + (
        expires_delta
        if expires_delta
        else timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    to_encode: Dict[str, Any] = {
        "exp": expire,
        "sub": str(subject),
        "iat": datetime.utcnow(),
    }
    if user_data:
        to_encode["user"] = user_data

    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)


async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    """
    Verify an incoming Bearer token.

    Flow:
      1. Try to decode as our own HS256 JWT (signature ALWAYS verified).
      2. If that fails, try Firebase ``verify_id_token``.
      3. If both fail → 401.

    SECURITY: We NEVER decode a token without signature verification.
    """
    
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="No credentials provided",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Handle both HTTPAuthorizationCredentials (HTTP) and raw string (WebSocket)
    if hasattr(credentials, "credentials"):
        token = credentials.credentials
    elif isinstance(credentials, str):
        token = credentials
    else:
        # Fallback for unexpected types
        token = str(credentials)

    # Lazy import to avoid circular dependency at module level
    from backend.db.mongodb.repositories.user_repository import user_repository

    # ── 1. Try our own JWT (with full signature verification) ──────────
    try:
        payload = jwt.decode(
            token,
            settings.SECRET_KEY,
            algorithms=[ALGORITHM],
        )
        user_id = payload.get("sub")
        logger.info("JWT token verified successfully")

        # Fast path: user data embedded in token
        if "user" in payload:
            user_data = payload["user"]
            return {
                "uid": user_data.get("firebase_uid") or user_id,
                "user_id": user_id,
                "email": user_data.get("email"),
                "provider": "jwt",
                "is_active": user_data.get("is_active", True),
            }

        # Slow path: DB lookup
        user = await user_repository.get_user_by_id(user_id)
        if user:
            return {
                "uid": user.firebase_uid or user_id,
                "user_id": user_id,
                "email": user.email,
                "provider": "jwt",
                "is_active": user.is_active,
            }

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found",
        )

    except ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except JWTError:
        pass  # Fall through to Firebase verification

    # ── 2. Try Firebase ID token verification ─────────────────────────
    try:
        decoded_token = auth.verify_id_token(token, check_revoked=True)
        logger.info(
            "Firebase ID token verified",
            uid=decoded_token.get("uid"),
        )

        # Try to find the user in our DB
        try:
            user = await user_repository.get_user_by_firebase_uid(
                decoded_token["uid"]
            )
            if user:
                return {
                    "uid": decoded_token["uid"],
                    "user_id": str(user.id),
                    "email": decoded_token.get("email") or user.email,
                    "provider": "firebase",
                    "is_active": user.is_active,
                }
        except Exception as lookup_err:
            logger.error(
                "Firebase user DB lookup failed",
                error=str(lookup_err),
                uid=decoded_token.get("uid"),
            )

        # Fallback: return Firebase claims even if DB lookup fails
        return {
            "uid": decoded_token["uid"],
            "email": decoded_token.get("email"),
            "provider": "firebase",
        }
    except Exception as firebase_err:
        logger.warning(
            "Firebase token verification also failed",
            error=str(firebase_err),
        )

    # ── 3. Both paths failed ──────────────────────────────────────────
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )


# ---------------------------------------------------------------------------
# Password hashing
# ---------------------------------------------------------------------------

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


# ---------------------------------------------------------------------------
# Rate limiting — async-safe, Redis-backed, per-endpoint tiers
# ---------------------------------------------------------------------------

# Lazy Redis connection (initialised on first use, not at import time)
_redis_client = None


def _get_redis():
    """Get or create the Redis client (lazy init)."""
    global _redis_client
    if _redis_client is None:
        import redis as _redis

        try:
            _redis_client = _redis.Redis(
                host=settings.db.REDIS_HOST,
                port=settings.db.REDIS_PORT,
                password=settings.db.REDIS_PASSWORD,
                db=settings.db.REDIS_DB,
                decode_responses=True,
                max_connections=settings.db.REDIS_POOL_SIZE,
                socket_connect_timeout=2,
                socket_timeout=2,
            )
            _redis_client.ping()
        except Exception as e:
            logger.warning("Redis unavailable — rate limiting disabled", error=str(e))
            _redis_client = None
    return _redis_client


def rate_limit(
    limit: Optional[int] = None,
    window: Optional[int] = None,
    tier: str = "default",
):
    """
    Production rate-limit decorator.

    Args:
        limit:  Max requests per window.  Falls back to the tier default from settings.
        window: Window size in seconds.  Falls back to ``rate_limit.WINDOW_SECONDS``.
        tier:   One of ``default``, ``auth``, ``ai_agent``, ``upload``.
    """
    # Resolve the actual limit from tier settings
    tier_limits = {
        "default": settings.rate_limit.DEFAULT_LIMIT,
        "auth": settings.rate_limit.AUTH_LIMIT,
        "ai_agent": settings.rate_limit.AI_AGENT_LIMIT,
        "upload": settings.rate_limit.UPLOAD_LIMIT,
    }
    resolved_limit = limit or tier_limits.get(tier, settings.rate_limit.DEFAULT_LIMIT)
    resolved_window = window or settings.rate_limit.WINDOW_SECONDS

    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            if not settings.rate_limit.ENABLED:
                return await func(request, *args, **kwargs)

            redis_client = _get_redis()
            if redis_client is None:
                # If Redis is down, allow the request through (fail-open)
                return await func(request, *args, **kwargs)

            client_ip = request.client.host if request.client else "unknown"
            endpoint = request.url.path
            key = f"rl:{tier}:{client_ip}:{endpoint}"

            try:
                current = redis_client.get(key)
                if current is None:
                    pipe = redis_client.pipeline()
                    pipe.setex(key, resolved_window, 1)
                    pipe.execute()
                elif int(current) >= resolved_limit:
                    ttl = redis_client.ttl(key)
                    raise HTTPException(
                        status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                        detail="Too many requests",
                        headers={
                            "Retry-After": str(max(ttl, 1)),
                            "X-RateLimit-Limit": str(resolved_limit),
                            "X-RateLimit-Remaining": "0",
                            "X-RateLimit-Reset": str(max(ttl, 1)),
                        },
                    )
                else:
                    redis_client.incr(key)
            except HTTPException:
                raise
            except Exception as e:
                logger.warning("Rate limiting error — allowing request", error=str(e))

            return await func(request, *args, **kwargs)

        return wrapper

    return decorator
