from typing import Any, List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File
from sqlalchemy.orm import Session
import aiofiles
import os
from pathlib import Path

from api.deps import get_db, get_current_user, get_current_active_user
from core.config import settings
from models.user import (
    User,
    UserUpdate,
    UserInDB,
    UserProfile,
    UserProfileUpdate,
    UserProfileInDB,
    UserSettings,
    UserSettingsUpdate,
    UserSettingsInDB,
)

router = APIRouter()

@router.get("/me", response_model=UserInDB)
async def read_user_me(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get current user.
    """
    return current_user

@router.put("/me", response_model=UserInDB)
async def update_user_me(
    *,
    db: Session = Depends(get_db),
    user_in: UserUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Update own user.
    """
    if user_in.full_name is not None:
        current_user.full_name = user_in.full_name
    if user_in.job_title is not None:
        if not current_user.profile:
            current_user.profile = UserProfile(user_id=current_user.id)
        current_user.profile.job_title = user_in.job_title
    if user_in.company is not None:
        if not current_user.profile:
            current_user.profile = UserProfile(user_id=current_user.id)
        current_user.profile.company = user_in.company
    if user_in.timezone is not None:
        if not current_user.profile:
            current_user.profile = UserProfile(user_id=current_user.id)
        current_user.profile.timezone = user_in.timezone
    if user_in.preferred_language is not None:
        if not current_user.profile:
            current_user.profile = UserProfile(user_id=current_user.id)
        current_user.profile.preferred_language = user_in.preferred_language

    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    return current_user

@router.get("/me/profile", response_model=UserProfileInDB)
async def read_user_profile(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get current user's profile.
    """
    if not current_user.profile:
        raise HTTPException(
            status_code=404,
            detail="User profile not found",
        )
    return current_user.profile

@router.put("/me/profile", response_model=UserProfileInDB)
async def update_user_profile(
    *,
    db: Session = Depends(get_db),
    profile_in: UserProfileUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Update current user's profile.
    """
    if not current_user.profile:
        current_user.profile = UserProfile(user_id=current_user.id)
    
    update_data = profile_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(current_user.profile, field, value)

    db.add(current_user.profile)
    db.commit()
    db.refresh(current_user.profile)
    return current_user.profile

@router.get("/me/settings", response_model=UserSettingsInDB)
async def read_user_settings(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get current user's settings.
    """
    if not current_user.settings:
        raise HTTPException(
            status_code=404,
            detail="User settings not found",
        )
    return current_user.settings

@router.put("/me/settings", response_model=UserSettingsInDB)
async def update_user_settings(
    *,
    db: Session = Depends(get_db),
    settings_in: UserSettingsUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Update current user's settings.
    """
    if not current_user.settings:
        current_user.settings = UserSettings(user_id=current_user.id)
    
    update_data = settings_in.dict(exclude_unset=True)
    for field, value in update_data.items():
        setattr(current_user.settings, field, value)

    db.add(current_user.settings)
    db.commit()
    db.refresh(current_user.settings)
    return current_user.settings

@router.post("/me/avatar", response_model=UserInDB)
async def upload_avatar(
    *,
    db: Session = Depends(get_db),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Upload user avatar.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image",
        )
    
    if file.size > settings.MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File size must not exceed {settings.MAX_UPLOAD_SIZE} bytes",
        )
    
    file_ext = file.filename.split(".")[-1].lower()
    if file_ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File extension must be one of {settings.ALLOWED_EXTENSIONS}",
        )
    
    # Create user's upload directory if it doesn't exist
    user_upload_dir = settings.UPLOAD_DIR / str(current_user.id)
    user_upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Save file
    file_path = user_upload_dir / f"avatar.{file_ext}"
    async with aiofiles.open(file_path, "wb") as out_file:
        content = await file.read()
        await out_file.write(content)
    
    # Update user's avatar URL
    current_user.avatar_url = str(file_path.relative_to(settings.UPLOAD_DIR))
    db.add(current_user)
    db.commit()
    db.refresh(current_user)
    
    return current_user 