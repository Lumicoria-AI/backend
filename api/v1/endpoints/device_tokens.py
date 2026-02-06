"""
API endpoints for device token management.

Provides endpoints for registering and managing push notification device tokens.
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
import structlog

from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.db.mongodb.models.device_token import (
    DeviceToken,
    DevicePlatform,
    DeviceTokenResponse
)
from backend.db.mongodb.repositories.device_token_repository import (
    get_device_token_repository
)

logger = structlog.get_logger()
router = APIRouter()


class RegisterTokenRequest(BaseModel):
    """Request body for registering a device token."""
    token: str
    platform: DevicePlatform = DevicePlatform.UNKNOWN
    device_name: Optional[str] = None
    app_version: Optional[str] = None


class DeregisterTokenRequest(BaseModel):
    """Request body for deregistering a device token."""
    token: str


@router.post("/register", response_model=DeviceTokenResponse)
async def register_device_token(
    request: RegisterTokenRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Register a device token for push notifications.
    
    This should be called when:
    - User logs in on a new device
    - FCM token is refreshed
    - User enables push notifications
    """
    repository = await get_device_token_repository()
    
    device_token = await repository.register_token(
        user_id=current_user.id,
        token=request.token,
        platform=request.platform,
        device_name=request.device_name,
        app_version=request.app_version
    )
    
    return DeviceTokenResponse(
        id=str(device_token.id),
        user_id=device_token.user_id,
        platform=device_token.platform,
        device_name=device_token.device_name,
        created_at=device_token.created_at,
        is_active=device_token.is_active
    )


@router.delete("/deregister")
async def deregister_device_token(
    request: DeregisterTokenRequest,
    current_user: User = Depends(get_current_active_user)
):
    """
    Deregister a device token.
    
    This should be called when:
    - User logs out
    - User disables push notifications
    """
    repository = await get_device_token_repository()
    
    success = await repository.delete_token(
        user_id=current_user.id,
        token=request.token
    )
    
    if not success:
        raise HTTPException(status_code=404, detail="Token not found")
    
    return {"status": "success", "message": "Token deregistered"}


@router.get("/")
async def get_my_device_tokens(
    current_user: User = Depends(get_current_active_user)
):
    """Get all device tokens for the current user."""
    repository = await get_device_token_repository()
    tokens = await repository.get_user_tokens(current_user.id)
    
    return [
        DeviceTokenResponse(
            id=str(token.id),
            user_id=token.user_id,
            platform=token.platform,
            device_name=token.device_name,
            created_at=token.created_at,
            is_active=token.is_active
        )
        for token in tokens
    ]


@router.delete("/all")
async def delete_all_my_tokens(
    current_user: User = Depends(get_current_active_user)
):
    """Delete all device tokens for the current user."""
    repository = await get_device_token_repository()
    deleted_count = await repository.delete_all_user_tokens(current_user.id)
    
    return {
        "status": "success",
        "deleted_count": deleted_count
    }


@router.get("/stats")
async def get_token_stats(
    current_user: User = Depends(get_current_active_user)
):
    """Get device token statistics for the current user."""
    repository = await get_device_token_repository()
    stats = await repository.get_platform_stats(current_user.id)
    
    return stats
