"""
Celery tasks for asynchronous notification processing.

This module provides background task processing for:
- Email notifications (async SMTP)
- Push notifications (FCM)
- Bulk notification delivery
- Scheduled cleanup of old notifications
"""

from celery import Celery, shared_task
from kombu import Queue
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
import structlog

from backend.tasks.async_utils import run_worker_coro

logger = structlog.get_logger()

# Initialize Celery app
# Redis URL will be loaded from environment
import os
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

celery_app = Celery(
    "lumicoria_notifications",
    broker=REDIS_URL,
    backend=REDIS_URL
)

# Celery configuration
celery_app.conf.update(
    task_default_queue="platform",
    task_queues=(Queue("platform"),),
    task_routes={"backend.tasks.notification_tasks.*": {"queue": "platform"}},
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=300,  # 5 minute hard limit
    task_soft_time_limit=240,  # 4 minute soft limit
    worker_prefetch_multiplier=1,  # One task at a time per worker
    task_acks_late=True,  # Acknowledge after task completes
    task_reject_on_worker_lost=True,
    # Retry settings
    task_default_retry_delay=60,  # 1 minute default retry delay
    task_max_retries=3,
    # Result expiration
    result_expires=3600,  # 1 hour
)

# Scheduled tasks (Celery Beat)
celery_app.conf.beat_schedule = {
    "cleanup-old-notifications": {
        "task": "backend.tasks.notification_tasks.cleanup_old_notifications_task",
        "schedule": timedelta(hours=24),  # Daily at midnight
        "args": (30, True),  # older_than_days=30, read_only=True
    },
}


def run_async(coro):
    """Helper to run async functions on the worker's persistent loop."""
    return run_worker_coro(coro)


@celery_app.task(
    bind=True,
    name="backend.tasks.notification_tasks.send_email_task",
    max_retries=3,
    default_retry_delay=60
)
def send_email_task(
    self,
    to_email: str,
    template_name: str,
    template_data: Dict[str, Any],
    priority: str = "normal"
) -> Dict[str, Any]:
    """
    Send an email notification asynchronously.
    
    Args:
        to_email: Recipient email address
        template_name: Name of the email template
        template_data: Data to render in the template
        priority: Notification priority (low, normal, high, urgent)
    
    Returns:
        Dict with status and details
    """
    try:
        logger.info(
            "email_task_started",
            to_email=to_email,
            template=template_name,
            task_id=self.request.id
        )
        
        async def _send():
            from backend.services.notification_service import notification_service
            from backend.db.mongodb.models.notification import NotificationPriority
            
            priority_enum = NotificationPriority(priority)
            success = await notification_service.send_email_notification(
                to_email=to_email,
                template_name=template_name,
                template_data=template_data,
                priority=priority_enum
            )
            return success
        
        success = run_async(_send())
        
        logger.info(
            "email_task_completed",
            to_email=to_email,
            success=success,
            task_id=self.request.id
        )
        
        return {
            "status": "success" if success else "failed",
            "to_email": to_email,
            "template": template_name,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(
            "email_task_error",
            to_email=to_email,
            error=str(e),
            task_id=self.request.id
        )
        # Retry with exponential backoff
        raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))


@celery_app.task(
    bind=True,
    name="backend.tasks.notification_tasks.send_push_notification_task",
    max_retries=3,
    default_retry_delay=30
)
def send_push_notification_task(
    self,
    user_id: str,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Send a push notification asynchronously via FCM.
    
    Args:
        user_id: Target user ID
        title: Notification title
        body: Notification body text
        data: Optional additional data payload
    
    Returns:
        Dict with status and details
    """
    try:
        logger.info(
            "push_task_started",
            user_id=user_id,
            title=title,
            task_id=self.request.id
        )
        
        async def _send():
            from backend.services.push_notification_service import push_notification_service
            success = await push_notification_service.send_to_user(
                user_id=user_id,
                title=title,
                body=body,
                data=data
            )
            return success
        
        success = run_async(_send())
        
        logger.info(
            "push_task_completed",
            user_id=user_id,
            success=success,
            task_id=self.request.id
        )
        
        return {
            "status": "success" if success else "failed",
            "user_id": user_id,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error(
            "push_task_error",
            user_id=user_id,
            error=str(e),
            task_id=self.request.id
        )
        raise self.retry(exc=e, countdown=30 * (2 ** self.request.retries))


@celery_app.task(
    bind=True,
    name="backend.tasks.notification_tasks.send_bulk_notifications_task",
    max_retries=2,
    default_retry_delay=120
)
def send_bulk_notifications_task(
    self,
    notifications: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Send multiple notifications in bulk.
    
    Args:
        notifications: List of notification dicts with keys:
            - user_id: Target user
            - title: Notification title
            - content: Notification content
            - notification_type: Type of notification
            - priority: Priority level
            - channels: List of delivery channels (email, push, in_app)
    
    Returns:
        Dict with counts of successful/failed deliveries
    """
    try:
        logger.info(
            "bulk_notifications_started",
            count=len(notifications),
            task_id=self.request.id
        )
        
        async def _send_bulk():
            from backend.services.notification_service import notification_service
            from backend.db.mongodb.models.notification import NotificationType, NotificationPriority
            
            success_count = 0
            failure_count = 0
            
            for notif in notifications:
                try:
                    await notification_service.create_in_app_notification(
                        user_id=notif["user_id"],
                        title=notif["title"],
                        content=notif["content"],
                        notification_type=NotificationType(notif.get("notification_type", "system")),
                        priority=NotificationPriority(notif.get("priority", "normal")),
                        metadata=notif.get("metadata"),
                        send_realtime=True,
                        send_push="push" in notif.get("channels", [])
                    )
                    success_count += 1
                except Exception as e:
                    logger.error("bulk_notification_item_error", error=str(e), user_id=notif.get("user_id"))
                    failure_count += 1
            
            return success_count, failure_count
        
        success_count, failure_count = run_async(_send_bulk())
        
        logger.info(
            "bulk_notifications_completed",
            success_count=success_count,
            failure_count=failure_count,
            task_id=self.request.id
        )
        
        return {
            "status": "completed",
            "success_count": success_count,
            "failure_count": failure_count,
            "total": len(notifications),
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error("bulk_notifications_error", error=str(e), task_id=self.request.id)
        raise self.retry(exc=e)


@celery_app.task(
    bind=True,
    name="backend.tasks.notification_tasks.cleanup_old_notifications_task"
)
def cleanup_old_notifications_task(
    self,
    older_than_days: int = 30,
    read_only: bool = True
) -> Dict[str, Any]:
    """
    Clean up old notifications from the database.
    
    Args:
        older_than_days: Delete notifications older than this many days
        read_only: If True, only delete read notifications
    
    Returns:
        Dict with deletion count
    """
    try:
        logger.info(
            "cleanup_task_started",
            older_than_days=older_than_days,
            read_only=read_only,
            task_id=self.request.id
        )
        
        async def _cleanup():
            from backend.db.mongodb.repositories.notification_repository import get_notification_repository
            repository = await get_notification_repository()
            deleted_count = await repository.cleanup_old_notifications(
                older_than_days=older_than_days,
                read_only=read_only
            )
            return deleted_count
        
        deleted_count = run_async(_cleanup())
        
        logger.info(
            "cleanup_task_completed",
            deleted_count=deleted_count,
            task_id=self.request.id
        )
        
        return {
            "status": "completed",
            "deleted_count": deleted_count,
            "older_than_days": older_than_days,
            "read_only": read_only,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error("cleanup_task_error", error=str(e), task_id=self.request.id)
        return {
            "status": "error",
            "error": str(e),
            "task_id": self.request.id
        }


@celery_app.task(
    bind=True,
    name="backend.tasks.notification_tasks.send_scheduled_reminder_task"
)
def send_scheduled_reminder_task(
    self,
    user_id: str,
    reminder_type: str,
    message: str
) -> Dict[str, Any]:
    """
    Send a scheduled reminder (e.g., wellbeing, task due).
    
    Args:
        user_id: Target user ID
        reminder_type: Type of reminder (hydration, break, task_due, etc.)
        message: Reminder message content
    
    Returns:
        Dict with status
    """
    try:
        logger.info(
            "reminder_task_started",
            user_id=user_id,
            reminder_type=reminder_type,
            task_id=self.request.id
        )
        
        async def _send():
            from backend.services.notification_service import notification_service
            notification = await notification_service.send_wellbeing_reminder(
                user_id=user_id,
                reminder_type=reminder_type,
                message=message
            )
            return str(notification.id)
        
        notification_id = run_async(_send())
        
        logger.info(
            "reminder_task_completed",
            user_id=user_id,
            notification_id=notification_id,
            task_id=self.request.id
        )
        
        return {
            "status": "success",
            "notification_id": notification_id,
            "user_id": user_id,
            "reminder_type": reminder_type,
            "task_id": self.request.id
        }
        
    except Exception as e:
        logger.error("reminder_task_error", error=str(e), task_id=self.request.id)
        return {
            "status": "error",
            "error": str(e),
            "task_id": self.request.id
        }


# Helper functions for queuing tasks

def queue_email_notification(
    to_email: str,
    template_name: str,
    template_data: Dict[str, Any],
    priority: str = "normal"
) -> str:
    """Queue an email notification for async delivery. Returns task ID."""
    result = send_email_task.delay(to_email, template_name, template_data, priority)
    return result.id


def queue_push_notification(
    user_id: str,
    title: str,
    body: str,
    data: Optional[Dict[str, Any]] = None
) -> str:
    """Queue a push notification for async delivery. Returns task ID."""
    result = send_push_notification_task.delay(user_id, title, body, data)
    return result.id


def queue_bulk_notifications(notifications: List[Dict[str, Any]]) -> str:
    """Queue bulk notifications for async delivery. Returns task ID."""
    result = send_bulk_notifications_task.delay(notifications)
    return result.id


def schedule_reminder(
    user_id: str,
    reminder_type: str,
    message: str,
    eta: datetime
) -> str:
    """Schedule a reminder for future delivery. Returns task ID."""
    result = send_scheduled_reminder_task.apply_async(
        args=[user_id, reminder_type, message],
        eta=eta
    )
    return result.id
