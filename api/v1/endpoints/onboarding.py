from fastapi import APIRouter, Depends, HTTPException, status, Request, UploadFile, File
from typing import Any, Optional
from backend.core.security import rate_limit
from backend.db.mongodb.repositories.user_repository import UserRepository, get_user_repository
from backend.api.deps import get_current_user, get_current_user_id
from backend.models.user import UserInDB, UserUpdate, UserProfile, UserSettings
from pydantic import BaseModel
import structlog
from datetime import datetime
import uuid
import os

logger = structlog.get_logger()
router = APIRouter()

class OnboardingProfileRequest(BaseModel):
    """Model for onboarding profile request."""
    full_name: Optional[str] = None
    job_title: Optional[str] = None
    company: Optional[str] = None
    avatar_url: Optional[str] = None

class OnboardingPreferencesRequest(BaseModel):
    """Model for onboarding preferences request."""
    timezone: Optional[str] = None
    preferred_language: Optional[str] = None

class OnboardingCompleteRequest(BaseModel):
    """Complete onboarding request combining profile and preferences."""
    full_name: Optional[str] = None
    job_title: Optional[str] = None
    company: Optional[str] = None
    avatar_url: Optional[str] = None
    timezone: Optional[str] = None
    preferred_language: Optional[str] = None

@router.post("/profile")
@rate_limit()
async def update_onboarding_profile(
    profile_data: OnboardingProfileRequest,
    user_repository: UserRepository = Depends(get_user_repository),
    current_user_id: str = Depends(get_current_user_id)
) -> Any:
    """Update user profile during onboarding."""
    try:
        # Create update data
        update_data = profile_data.model_dump(exclude_unset=True)
        if not update_data:
            return {"message": "No data provided for update"}
        
        # Update user
        updated_user = await user_repository.update_user(
            current_user_id, 
            update_data
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "message": "Profile updated successfully",
            "user": {
                "id": str(updated_user.id),
                "email": updated_user.email,
                "full_name": updated_user.full_name,
                "job_title": updated_user.job_title,
                "company": updated_user.company,
                "avatar_url": updated_user.avatar_url
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update profile", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update profile"
        )

@router.post("/preferences")
@rate_limit()
async def update_onboarding_preferences(
    preferences_data: OnboardingPreferencesRequest,
    user_repository: UserRepository = Depends(get_user_repository),
    current_user_id: str = Depends(get_current_user_id)
) -> Any:
    """Update user preferences during onboarding."""
    try:
        # Create update data
        update_data = preferences_data.model_dump(exclude_unset=True)
        if not update_data:
            return {"message": "No data provided for update"}
        
        # Update user
        updated_user = await user_repository.update_user(
            current_user_id, 
            update_data
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "message": "Preferences updated successfully",
            "user": {
                "id": str(updated_user.id),
                "timezone": updated_user.timezone,
                "preferred_language": updated_user.preferred_language
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to update preferences", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update preferences"
        )

@router.post("/complete")
@rate_limit()
async def complete_onboarding(
    onboarding_data: OnboardingCompleteRequest,
    user_repository: UserRepository = Depends(get_user_repository),
    current_user_id: str = Depends(get_current_user_id)
) -> Any:
    """Complete the onboarding process with all user data at once."""
    try:
        # Create update data
        update_data = onboarding_data.model_dump(exclude_unset=True)
        if not update_data:
            return {"message": "No data provided for update"}
        
        # Add onboarding completion flag
        update_data["onboarding_completed"] = True
        update_data["onboarding_completed_at"] = datetime.utcnow()
        
        # Update user
        updated_user = await user_repository.update_user(
            current_user_id, 
            update_data
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "message": "Onboarding completed successfully",
            "user": {
                "id": str(updated_user.id),
                "email": updated_user.email,
                "full_name": updated_user.full_name,
                "job_title": updated_user.job_title,
                "company": updated_user.company,
                "avatar_url": updated_user.avatar_url,
                "timezone": updated_user.timezone,
                "preferred_language": updated_user.preferred_language,
                "onboarding_completed": updated_user.onboarding_completed,
                "onboarding_completed_at": updated_user.onboarding_completed_at
            }
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to complete onboarding", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete onboarding"
        )

@router.post("/avatar")
@rate_limit()
async def upload_avatar(
    file: UploadFile = File(...),
    user_repository: UserRepository = Depends(get_user_repository),
    current_user_id: str = Depends(get_current_user_id)
) -> Any:
    """Upload user avatar image."""
    try:
        # Check file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
            
        # Create uploads directory if it doesn't exist
        upload_dir = os.path.join("uploads", "avatars")
        os.makedirs(upload_dir, exist_ok=True)
        
        # Generate unique filename
        file_ext = os.path.splitext(file.filename)[1]
        unique_filename = f"{current_user_id}_{uuid.uuid4()}{file_ext}"
        file_path = os.path.join(upload_dir, unique_filename)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(await file.read())
        
        # Generate URL for the avatar
        avatar_url = f"/uploads/avatars/{unique_filename}"
        
        # Update user with avatar URL
        updated_user = await user_repository.update_user(
            current_user_id, 
            {"avatar_url": avatar_url}
        )
        
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return {
            "message": "Avatar uploaded successfully",
            "avatar_url": avatar_url
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to upload avatar", error=str(e), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload avatar"
        )
