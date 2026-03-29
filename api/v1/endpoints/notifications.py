from typing import List, Optional, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import EmailStr
from backend.core.auth import get_current_user
from backend.services.notification_service import notification_service
from backend.db.mongodb.models.notification import (
    Notification,
    NotificationType,
    NotificationPriority,
)
import structlog

logger = structlog.get_logger()

router = APIRouter()


def _serialize_notification(n: Notification) -> dict:
    """Convert a Notification model to a JSON-safe dict."""
    return {
        "id": str(n.id) if n.id else None,
        "user_id": n.user_id,
        "user_email": n.user_email,
        "notification_type": n.notification_type.value if n.notification_type else None,
        "title": n.title,
        "content": n.content,
        "priority": n.priority.value if n.priority else None,
        "metadata": n.metadata or {},
        "created_at": n.created_at.isoformat() if n.created_at else None,
        "read": n.read,
        "read_at": n.read_at.isoformat() if n.read_at else None,
    }


@router.get("/")
async def get_notifications(
    current_user: Dict[str, Any] = Depends(get_current_user),
    unread_only: bool = False,
    limit: int = Query(50, ge=1, le=100),
    skip: int = Query(0, ge=0),
    notification_type: Optional[NotificationType] = None
):
    """Get notifications for the current user."""
    user_id = current_user["id"]
    logger.info("fetching_notifications", user_id=user_id, unread_only=unread_only)
    if notification_type:
        notifications = await notification_service.get_notifications_by_type(
            user_id=user_id,
            notification_type=notification_type,
            limit=limit,
            skip=skip
        )
    else:
        notifications = await notification_service.get_user_notifications(
            user_id=user_id,
            unread_only=unread_only,
            limit=limit,
            skip=skip
        )
    logger.info("notifications_fetched", count=len(notifications), user_id=user_id)
    return [_serialize_notification(n) for n in notifications]

@router.get("/unread/count")
async def get_unread_count(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Get count of unread notifications."""
    return await notification_service.get_unread_count(current_user["id"])

@router.post("/{notification_id}/read")
async def mark_as_read(
    notification_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Mark a notification as read."""
    success = await notification_service.mark_notification_as_read(
        notification_id=notification_id,
        user_id=current_user["id"]
    )
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "success"}

@router.post("/read/all")
async def mark_all_as_read(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Mark all notifications as read."""
    success = await notification_service.mark_all_notifications_as_read(
        user_id=current_user["id"]
    )
    return {"status": "success" if success else "error"}

@router.delete("/{notification_id}")
async def delete_notification(
    notification_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Delete a notification."""
    success = await notification_service.delete_notification(
        notification_id=notification_id,
        user_id=current_user["id"]
    )
    if not success:
        raise HTTPException(status_code=404, detail="Notification not found")
    return {"status": "success"}

@router.post("/test/email")
async def send_test_email(
    email: EmailStr,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Send a test email notification (for development only)."""
    success = await notification_service.send_email_notification(
        to_email=email,
        template_name="test_notification",
        template_data={
            "subject": "Test Notification",
            "user_name": current_user.get("full_name", "User"),
            "message": "This is a test notification from Lumicoria.ai"
        },
        priority=NotificationPriority.NORMAL
    )

    if not success:
        raise HTTPException(status_code=500, detail="Failed to send email")
    return {"status": "success"}

@router.post("/test/in-app")
async def create_test_in_app_notification(
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Create a test in-app notification (for development only)."""
    notification = await notification_service.create_in_app_notification(
        user_id=current_user["id"],
        title="Test Notification",
        content="This is a test in-app notification",
        notification_type=NotificationType.SYSTEM,
        priority=NotificationPriority.NORMAL,
        metadata={"test": True}
    )
    return _serialize_notification(notification) 