from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Request
from fastapi.responses import JSONResponse
from typing import Any, List
from backend.api.deps import get_db, get_current_user, get_current_active_user
from backend.db.mongodb.repositories.user_repository import UserRepository, get_user_repository
from backend.models.user import User, UserUpdate, UserResponse
from backend.core.security import rate_limit
import structlog

logger = structlog.get_logger()
router = APIRouter()


def _serialize_user(user) -> dict:
    """Convert a User/UserInDB object to a safe dict with string id."""
    data = {}
    for field in ("email", "full_name", "is_active", "avatar_url", "created_at",
                  "updated_at", "onboarding_completed", "job_title", "company"):
        data[field] = getattr(user, field, None)
    data["id"] = str(user.id)
    if hasattr(user, "onboarding_info"):
        data["onboarding_info"] = getattr(user, "onboarding_info", None)
    if hasattr(user, "preferences"):
        data["preferences"] = getattr(user, "preferences", None)
    return data


@router.get("/me", response_model=None)
@rate_limit()
async def get_current_user_profile(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get current user profile."""
    return _serialize_user(current_user)

@router.put("/me", response_model=None)
@rate_limit()
async def update_current_user(
    request: Request,
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    repo: UserRepository = Depends(get_user_repository)
) -> Any:
    """Update current user profile."""
    try:
        updated_user = await repo.update_user(str(current_user.id), user_update)
        return _serialize_user(updated_user)
    except Exception as e:
        logger.error(f"Error updating user: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating user profile"
        )

@router.get("/me/profile", response_model=None)
@rate_limit()
async def get_user_profile(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get detailed user profile."""
    return _serialize_user(current_user)

@router.put("/me/profile", response_model=None)
@rate_limit()
async def update_user_profile(
    request: Request,
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user),
    repo: UserRepository = Depends(get_user_repository)
) -> Any:
    """Update user profile."""
    try:
        updated_user = await repo.update_user(str(current_user.id), user_update)
        return _serialize_user(updated_user)
    except Exception as e:
        logger.error(f"Error updating user profile: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating user profile"
        )

@router.get("/me/settings", response_model=None)
@rate_limit()
async def get_user_settings(
    request: Request,
    current_user: User = Depends(get_current_active_user),
    repo: UserRepository = Depends(get_user_repository)
) -> Any:
    """Get user settings."""
    user_settings = await repo.get_user_settings(str(current_user.id))
    if user_settings:
        data = user_settings.model_dump()
        # Convert ObjectId fields to strings for JSON serialization
        if "user_id" in data:
            data["user_id"] = str(data["user_id"])
        if "_id" in data:
            data["id"] = str(data.pop("_id"))
        return data
    # Return defaults if no settings document exists yet
    return {
        "email_notifications": True,
        "push_notifications": True,
        "task_reminders": True,
        "break_reminders": True,
        "work_hours_start": "09:00",
        "work_hours_end": "17:00",
        "break_interval_minutes": 60,
        "break_duration_minutes": 5,
        "preferred_ai_model": "gemini",
    }

@router.put("/me/settings", response_model=None)
@rate_limit()
async def update_user_settings(
    request: Request,
    settings: dict,
    current_user: User = Depends(get_current_active_user),
    repo: UserRepository = Depends(get_user_repository)
) -> Any:
    """Update user settings."""
    try:
        updated_settings = await repo.update_user_settings(str(current_user.id), settings)
        if updated_settings:
            data = updated_settings.model_dump()
            if "user_id" in data:
                data["user_id"] = str(data["user_id"])
            if "_id" in data:
                data["id"] = str(data.pop("_id"))
            return data
        return settings
    except Exception as e:
        logger.error(f"Error updating user settings: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error updating user settings"
        )

@router.post("/me/avatar", response_model=None)
@rate_limit()
async def upload_user_avatar(
    request: Request,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
    repo: UserRepository = Depends(get_user_repository)
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
        avatar_url = await repo.upload_avatar(str(current_user.id), file)

        # Update user with new avatar URL
        updated_user = await repo.update_user(
            str(current_user.id),
            UserUpdate(avatar_url=avatar_url)
        )
        return _serialize_user(updated_user)
    except Exception as e:
        logger.error(f"Error uploading avatar: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error uploading avatar"
        )
