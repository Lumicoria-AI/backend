from typing import Generator, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from backend.core.security import verify_token
from backend.db.mongodb.repositories.user_repository import user_repository, get_user_repository
from backend.models.user import User
from backend.db.mongodb.mongodb import get_mongodb
from motor.motor_asyncio import AsyncIOMotorDatabase
import structlog

logger = structlog.get_logger()
security = HTTPBearer()

async def get_db() -> AsyncIOMotorDatabase:
    """Get MongoDB database instance."""
    try:
        db = await get_mongodb()
        return db
    except Exception as e:
        logger.error(f"Error getting database connection: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not connect to database"
        )

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> User:
    """Get current user from token."""
    try:
        token_data = await verify_token(credentials)
        
        # Handle both Firebase and JWT tokens
        user = None
        if token_data.get("provider") == "firebase" and token_data.get("uid"):
            user = await user_repository.get_user_by_firebase_uid(token_data["uid"])
        elif token_data.get("provider") == "jwt" and token_data.get("uid"):
            user = await user_repository.get_user_by_id(token_data["uid"])
            
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

async def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Inactive user"
        )
    return current_user

async def get_current_superuser(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Get current superuser."""
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user

async def get_current_user_id(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """Get current user ID from token."""
    try:
        # Log the token type for debugging
        token = credentials.credentials if credentials else None
        token_type = "JWT" if token and token.count('.') == 2 else "Unknown"
        logger.info("Processing authentication token", token_type=token_type)
        
        token_data = await verify_token(credentials)
        logger.info("Token verified successfully", token_keys=list(token_data.keys()))
        
        # Make sure we have the user repository
        repo = await get_user_repository()
        if not repo:
            logger.error("User repository not initialized")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal server error"
            )
        
        # Enhanced token data handling for better debugging
        user = None
        
        # Case 1: We have a direct user_id from our JWT token (preferred)
        if "user_id" in token_data:
            logger.info(f"Looking up user by user_id: {token_data['user_id']}")
            user = await repo.get_user_by_id(token_data["user_id"])
            
        # Case 2: We have a uid from Firebase or custom token
        elif "uid" in token_data:
            uid = token_data["uid"]
            provider = token_data.get("provider", "unknown")
            logger.info(f"Looking up user by uid: {uid} from provider: {provider}")
            
            # Try Firebase UID first if it's from Firebase
            if provider == "firebase":
                logger.info(f"Looking up user by firebase_uid: {uid}")
                user = await repo.get_user_by_firebase_uid(uid)
                
            # If not found or not from Firebase, try as MongoDB ID
            if not user:
                logger.info(f"Looking up user by id: {uid}")
                user = await repo.get_user_by_id(uid)
        
        if not user:
            logger.error("User lookup failed", token_data=token_data)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.info("User found successfully", user_id=str(user.id))
        return str(user.id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user ID: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )