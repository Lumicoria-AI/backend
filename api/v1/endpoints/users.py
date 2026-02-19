from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Request
from fastapi.responses import JSONResponse
from typing import Any, List
from backend.api.deps import get_db, get_current_user, get_current_active_user
from backend.db.mongodb.repositories.user_repository import user_repository
from backend.models.user import User, UserUpdate, UserResponse
from backend.core.security import rate_limit
import structlog

logger = structlog.get_logger()
router = APIRouter()

@router.get("/me", response_model=UserResponse)
@rate_limit()
async def get_current_user_profile(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get current user profile."""
    return current_user

@router.put("/me", response_model=UserResponse)
@rate_limit()
async def update_current_user(
    request: Request,
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Update current user profile."""
    try:
        updated_user = await user_repository.update_user(str(current_user.id), user_update)
        return updated_user
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating user profile"
        )

@router.get("/me/profile", response_model=UserResponse)
@rate_limit()
async def get_user_profile(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get detailed user profile."""
    return current_user

@router.put("/me/profile", response_model=UserResponse)
@rate_limit()
async def update_user_profile(
    request: Request,
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Update user profile."""
    try:
        updated_user = await user_repository.update_user(str(current_user.id), user_update)
        return updated_user
    except Exception as e:
        logger.error(f"Error updating user profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating user profile"
        )

@router.get("/me/settings", response_model=dict)
@rate_limit()
async def get_user_settings(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get user settings."""
    user_settings = await user_repository.get_user_settings(str(current_user.id))
    if user_settings:
        return user_settings.model_dump()
    return {}

@router.put("/me/settings", response_model=dict)
@rate_limit()
async def update_user_settings(
    request: Request,
    settings: dict,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Update user settings."""
    try:
        updated_user = await user_repository.update_user(
            str(current_user.id),
            UserUpdate(preferences=settings)
        )
        return updated_user.preferences
    except Exception as e:
        logger.error(f"Error updating user settings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating user settings"
        )

@router.post("/me/avatar", response_model=UserResponse)
@rate_limit()
async def upload_user_avatar(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Upload user avatar."""
    try:
        # Validate file type
        if not file.content_type.startswith("image/"):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )

        # Upload file and get URL
        avatar_url = await user_repository.upload_avatar(str(current_user.id), file)

        # Update user with new avatar URL
        updated_user = await user_repository.update_user(
            str(current_user.id),
            UserUpdate(avatar_url=avatar_url)
        )
        return updated_user
    except Exception as e:
        logger.error(f"Error uploading avatar: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error uploading avatar"
        ) 