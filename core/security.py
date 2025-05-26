from datetime import datetime, timedelta
from typing import Any, Optional, Union
from jose import jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import firebase_admin
from firebase_admin import auth, credentials
from core.config import settings
import structlog
# Rate limiting decorator
from functools import wraps
from fastapi import Request
import redis
import time

logger = structlog.get_logger()
security = HTTPBearer()

# Initialize Firebase Admin
try:
    cred = credentials.Certificate(settings.FIREBASE_CREDENTIALS_PATH)
    firebase_admin.initialize_app(cred)
except Exception as e:
    logger.error("Failed to initialize Firebase", error=str(e))
    raise

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

ALGORITHM = "HS256"

def create_access_token(
    subject: Union[str, Any], expires_delta: Optional[timedelta] = None
) -> str:
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(
            minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES
        )
    to_encode = {"exp": expire, "sub": str(subject)}
    encoded_jwt = jwt.encode(to_encode, settings.SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> dict:
    try:
        # First try to verify as Firebase token
        try:
            token = credentials.credentials
            decoded_token = auth.verify_id_token(token)
            return {
                "uid": decoded_token["uid"],
                "email": decoded_token.get("email"),
                "provider": "firebase"
            }
        except Exception as firebase_error:
            logger.warning("Firebase token verification failed", error=str(firebase_error))
            
            # Fallback to JWT verification
            try:
                payload = jwt.decode(credentials.credentials, settings.SECRET_KEY, algorithms=["HS256"])
                return {
                    "uid": payload["sub"],
                    "provider": "jwt"
                }
            except jwt.JWTError as jwt_error:
                logger.error("JWT verification failed", error=str(jwt_error))
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Could not validate credentials",
                    headers={"WWW-Authenticate": "Bearer"},
                )
    except Exception as e:
        logger.error("Token verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


redis_client = redis.Redis(
    host=settings.REDIS_HOST,
    port=settings.REDIS_PORT,
    password=settings.REDIS_PASSWORD,
    decode_responses=True
)

def rate_limit(limit: int = settings.RATE_LIMIT_PER_MINUTE):
    def decorator(func):
        @wraps(func)
        async def wrapper(request: Request, *args, **kwargs):
            client_ip = request.client.host
            key = f"rate_limit:{client_ip}"
            
            current = redis_client.get(key)
            if current is None:
                redis_client.setex(key, 60, 1)
            elif int(current) >= limit:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Too many requests"
                )
            else:
                redis_client.incr(key)
            
            return await func(request, *args, **kwargs)
        return wrapper
    return decorator 
