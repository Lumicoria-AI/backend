from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import EmailStr
from backend.api.deps import get_current_active_user
from backend.services.notification_service import notification_service
from backend.models.mongodb_models import Notification, NotificationType, NotificationPriority
from backend.models.user import User

router = APIRouter()

@router.get("/", response_model=List[Notification])
async def get_notifications(
    current_user: User = Depends(get_current_active_user),
    unread_only: bool = False,
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    notification_type: Optional[NotificationType] = None
):
    """Get notifications for the current user."""
    if notification_type:
        return await notification_service.get_notifications_by_type(
            user_id=current_user.id,
            notification_type=notification_type,
            limit=limit,
            skip=skip
        )
    return await notification_service.get_user_notifications(
        user_id=current_user.id,
        unread_only=unread_only,
        limit=limit,
        skip=skip
    )

@router.get("/unread/count")
async def get_unread_count(
    current_user: User = Depends(get_current_active_user)
):
    """Get count of unread notifications."""
    return await notification_service.get_unread_count(current_user.id)

@router.post("/{notification_id}/read")
async def mark_as_read(
    notification_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Mark a notification as read."""
    success = await notification_service.mark_notification_as_read(
        notification_id=notification_id,
        user_id=current_user.id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "success"}

@router.post("/read/all")
async def mark_all_as_read(
    current_user: User = Depends(get_current_active_user)
):
    """Mark all notifications as read."""
    success = await notification_service.mark_all_notifications_as_read(
        user_id=current_user.id
    )
    return {"status": "success" if success else "error"}

@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: str,
    current_user: User = Depends(get_current_active_user)
):
    """Delete a notification."""
    success = await notification_service.delete_notification(
        notification_id=notification_id,
        user_id=current_user.id
    )
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "success"}

@router.post("/test/email")
async def send_test_email(
    email: EmailStr,
    current_user: User = Depends(get_current_active_user)
):
    """Send a test email notification (for development only)."""
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    success = await notification_service.send_email_notification(
        to_email=email,
        template_name="test_notification",
        template_data={
            "subject": "Test Notification",
            "user_name": current_user.full_name,
            "message": "This is a test notification from Lumicoria.ai"
        },
        priority=NotificationPriority.NORMAL
    )
    
    if not success:
        raise HTTPException(status_code=500, detail="Failed to send email")
    return {"status": "success"}

@router.post("/test/in-app")
async def create_test_in_app_notification(
    current_user: User = Depends(get_current_active_user)
):
    """Create a test in-app notification (for development only)."""
    if not current_user.is_superuser:
        raise HTTPException(status_code=403, detail="Not authorized")
    
    notification = await notification_service.create_in_app_notification(
        user_id=current_user.id,
        title="Test Notification",
        content="This is a test in-app notification",
        notification_type=NotificationType.SYSTEM,
        priority=NotificationPriority.NORMAL,
        metadata={"test": True}
    )
    return notification 