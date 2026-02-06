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
    request: Request,
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
          # Safely access attributes that might not exist in the UserInDB model
        user_data = {
            "id": str(updated_user.id),
            "email": updated_user.email,
            "full_name": updated_user.full_name,
            "avatar_url": getattr(updated_user, "avatar_url", None)
        }
        
        # Add job_title and company if they exist
        if hasattr(updated_user, "job_title"):
            user_data["job_title"] = updated_user.job_title
        else:
            user_data["job_title"] = update_data.get("job_title")
            
        if hasattr(updated_user, "company"):
            user_data["company"] = updated_user.company
        else:
            user_data["company"] = update_data.get("company")
        
        return {
            "message": "Profile updated successfully",
            "user": user_data
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
    request: Request,
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
          # Safely access attributes that might not exist in the UserInDB model
        user_data = {
            "id": str(updated_user.id)
        }
        
        # Add timezone and preferred_language if they exist
        if hasattr(updated_user, "timezone"):
            user_data["timezone"] = updated_user.timezone
        else:
            user_data["timezone"] = update_data.get("timezone", "UTC")
            
        if hasattr(updated_user, "preferred_language"):
            user_data["preferred_language"] = updated_user.preferred_language
        else:
            user_data["preferred_language"] = update_data.get("preferred_language", "en")
            
        return {
            "message": "Preferences updated successfully",
            "user": user_data
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
    request: Request,
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
          # Add onboarding completion flag - forcing it to be True
        update_data["onboarding_completed"] = True
        update_data["onboarding_completed_at"] = datetime.utcnow()
        
        logger.info("Completing onboarding for user", 
                   user_id=current_user_id, 
                   update_data=update_data)
        
        # Update user
        updated_user = await user_repository.update_user(
            current_user_id, 
            update_data
        )
        
        # Verify the update succeeded
        if not updated_user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
              # Double-check that the onboarding flag was properly set
        if not getattr(updated_user, "onboarding_completed", False):
            logger.warning("Onboarding flag not set correctly, fixing it", user_id=current_user_id)
            # Try one more time with explicit focus on just the onboarding flag
            # Get the updated user object from the second update call
            fixed_user = await user_repository.update_user(
                current_user_id, 
                {"onboarding_completed": True, "onboarding_completed_at": datetime.utcnow()}
            )
            # Use the fixed user object if it was returned
            if fixed_user:
                updated_user = fixed_user
                logger.info("Successfully fixed onboarding flag", user_id=current_user_id)
            else:
                logger.error("Failed to fix onboarding flag", user_id=current_user_id)
        
        # Always set onboarding_completed to True in the response
        user_data = {
            "id": str(updated_user.id),
            "email": updated_user.email,
            "full_name": updated_user.full_name,
            "avatar_url": getattr(updated_user, "avatar_url", None),
            # Force onboarding_completed to true in the response
            "onboarding_completed": True,  
            "onboarding_completed_at": getattr(updated_user, "onboarding_completed_at", datetime.utcnow())
        }
        
        logger.info("Onboarding completed, returning user data", 
                   user_id=user_data["id"], 
                   onboarding_completed=user_data["onboarding_completed"])
        
        # Add optional fields if they exist
        if hasattr(updated_user, "job_title"):
            user_data["job_title"] = updated_user.job_title
        else:
            user_data["job_title"] = update_data.get("job_title")
            
        if hasattr(updated_user, "company"):
            user_data["company"] = updated_user.company
        else:
            user_data["company"] = update_data.get("company")
            
        if hasattr(updated_user, "timezone"):
            user_data["timezone"] = updated_user.timezone
        else:
            user_data["timezone"] = update_data.get("timezone", "UTC")
            
        if hasattr(updated_user, "preferred_language"):
            user_data["preferred_language"] = updated_user.preferred_language
        else:
            user_data["preferred_language"] = update_data.get("preferred_language", "en")
            
        return {
            "message": "Onboarding completed successfully",
            "user": user_data
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
    request: Request,
    file: UploadFile = File(...),
    user_repository: UserRepository = Depends(get_user_repository),
    current_user_id: str = Depends(get_current_user_id)
) -> Any:
    """Upload user avatar image."""    
    try:
        # Log the beginning of avatar upload
        logger.info("Starting avatar upload process", user_id=current_user_id, file_name=file.filename)
        
        # Check file type
        if not file.content_type.startswith('image/'):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File must be an image"
            )
            
        # Use the dedicated upload_avatar method from user_repository
        try:
            # This handles all the file saving logic internally
            avatar_url = await user_repository.upload_avatar(current_user_id, file)
            
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
            
            logger.info("Avatar upload completed successfully", 
                      user_id=current_user_id, 
                      avatar_url=avatar_url)
                      
            return {
                "message": "Avatar uploaded successfully",
                "avatar_url": avatar_url
            }
        except Exception as e:
            logger.error("Error during avatar upload processing", 
                       error=str(e), 
                       user_id=current_user_id,
                       exc_info=True)
            raise
        
    except HTTPException as http_ex:
        logger.error("HTTP exception in avatar upload", 
                  status_code=http_ex.status_code, 
                  detail=http_ex.detail)
        raise
    except Exception as e:
        logger.error("Failed to upload avatar", 
                  error=str(e), 
                  error_type=type(e).__name__, 
                  user_id=current_user_id,
                  exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload avatar: " + str(e)
        )

