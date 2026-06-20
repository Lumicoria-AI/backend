from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path
import json
from pydantic import BaseModel, EmailStr
import structlog

from ..core.config import settings
from ..db.mongodb.repositories.notification_repository import (
    get_notification_repository,
    NotificationRepository
)
from ..db.mongodb.models.notification import (
    Notification, 
    NotificationType, 
    NotificationPriority
)

logger = structlog.get_logger()


class NotificationTemplate(BaseModel):
    subject: str
    body: str
    template_name: str


# WebSocket connection manager for real-time notifications
class ConnectionManager:
    """Manages WebSocket connections for real-time notification delivery."""
    
    def __init__(self):
        # Map of user_id -> list of WebSocket connections
        self.active_connections: Dict[str, List[Any]] = {}
    
    async def connect(self, websocket, user_id: str):
        """Accept and store a WebSocket connection for a user."""
        await websocket.accept()
        self._register(websocket, user_id)
        logger.info("websocket_connected", user_id=user_id)

    def register(self, websocket, user_id: str) -> None:
        """Store an already-accepted WebSocket without re-accepting.

        Use this when the caller has called websocket.accept() itself —
        e.g. the huddle WS endpoint accepts after auth and then delegates
        to _serve_room. Calling connect() in that case re-accepts and
        Starlette raises:
        "Expected ASGI message 'websocket.send' or 'websocket.close',
         but got 'websocket.accept'".
        """
        self._register(websocket, user_id)
        logger.info("websocket_registered", user_id=user_id)

    def _register(self, websocket, user_id: str) -> None:
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
    
    def disconnect(self, websocket, user_id: str):
        """Remove a WebSocket connection for a user."""
        if user_id in self.active_connections:
            if websocket in self.active_connections[user_id]:
                self.active_connections[user_id].remove(websocket)
            if not self.active_connections[user_id]:
                del self.active_connections[user_id]
        logger.info("websocket_disconnected", user_id=user_id)
    
    async def send_to_user(self, user_id: str, message: dict):
        """Send a message to all connections for a specific user."""
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error("websocket_send_error", user_id=user_id, error=str(e))
    
    async def broadcast(self, message: dict):
        """Broadcast a message to all connected users."""
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_json(message)
                except Exception as e:
                    logger.error("websocket_broadcast_error", user_id=user_id, error=str(e))
    
    def is_user_connected(self, user_id: str) -> bool:
        """Check if a user has any active connections."""
        return user_id in self.active_connections and len(self.active_connections[user_id]) > 0


# Global connection manager instance
connection_manager = ConnectionManager()


class NotificationService:
    """Service for managing notifications across all channels."""
    
    def __init__(self):
        self.email_templates_dir = Path(__file__).parent / "templates" / "email"
        
        # Email service will be lazily initialized
        self._email_service = None
        
        # Repository will be lazily initialized
        self._repository: Optional[NotificationRepository] = None
    
    async def _get_email_service(self):
        """Get or initialize the email service."""
        if self._email_service is None:
            from .email_service import get_email_service
            self._email_service = await get_email_service()
        return self._email_service

    async def _get_repository(self) -> NotificationRepository:
        """Get or initialize the notification repository."""
        if self._repository is None:
            self._repository = await get_notification_repository()
        return self._repository

    async def send_email_notification(
        self,
        to_email: EmailStr,
        template_name: str,
        template_data: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> bool:
        """Send an email notification using a template via SendGrid/Resend."""
        try:
            # Get the email service
            email_service = await self._get_email_service()
            
            # Send email using the production email service
            result = await email_service.send(
                to=str(to_email),
                subject=template_data.get('subject', 'Notification from Lumicoria.ai'),
                template_name=template_name,
                template_data=template_data,
            )
            
            if result.success:
                # Store notification in database
                await self._store_notification(
                    notification_type=NotificationType.EMAIL,
                    title=template_data.get('subject', 'Email Notification'),
                    content=f"Email sent via {result.provider}",
                    user_email=to_email,
                    priority=priority,
                    metadata={
                        "template": template_name,
                        "template_data": template_data,
                        "provider": result.provider,
                        "message_id": result.message_id,
                    }
                )
                
                logger.info(
                    "email_notification_sent",
                    to_email=to_email,
                    template=template_name,
                    provider=result.provider,
                    message_id=result.message_id
                )
                return True
            else:
                logger.error(
                    "email_notification_failed",
                    to_email=to_email,
                    error=result.error_message,
                    error_code=result.error_code
                )
                return False
                
        except Exception as e:
            logger.error("email_notification_error", to_email=to_email, error=str(e))
            return False

    async def create_in_app_notification(
        self,
        user_id: str,
        title: str,
        content: str,
        notification_type: NotificationType,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None,
        send_realtime: bool = True,
        send_push: bool = True
    ) -> Notification:
        """Create and store an in-app notification with optional real-time and push delivery."""
        notification = await self._store_notification(
            notification_type=notification_type,
            title=title,
            content=content,
            user_id=user_id,
            priority=priority,
            metadata=metadata
        )
        
        # Send real-time WebSocket notification
        if send_realtime and connection_manager.is_user_connected(user_id):
            await connection_manager.send_to_user(user_id, {
                "type": "notification",
                "data": {
                    "id": str(notification.id),
                    "title": notification.title,
                    "content": notification.content,
                    "notification_type": notification.notification_type.value,
                    "priority": notification.priority.value,
                    "created_at": notification.created_at.isoformat(),
                    "metadata": notification.metadata
                }
            })
        
        # Send push notification if enabled
        if send_push:
            await self._send_push_notification(user_id, title, content, metadata)
        
        return notification

    async def _send_push_notification(
        self,
        user_id: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Send a push notification via Firebase Cloud Messaging."""
        try:
            # Import here to avoid circular imports and only if FCM is needed
            from .push_notification_service import push_notification_service
            return await push_notification_service.send_to_user(user_id, title, body, data)
        except ImportError:
            logger.warning("push_notification_service_not_available")
            return False
        except Exception as e:
            logger.error("push_notification_error", user_id=user_id, error=str(e))
            return False

    async def get_user_notifications(
        self,
        user_id: str,
        unread_only: bool = False,
        limit: int = 50,
        skip: int = 0
    ) -> List[Notification]:
        """Get notifications for a user."""
        repository = await self._get_repository()
        return await repository.get_user_notifications(
            user_id=user_id,
            unread_only=unread_only,
            limit=limit,
            skip=skip
        )

    async def get_notifications_by_type(
        self,
        user_id: str,
        notification_type: NotificationType,
        limit: int = 50,
        skip: int = 0
    ) -> List[Notification]:
        """Get notifications of a specific type for a user."""
        repository = await self._get_repository()
        return await repository.get_notifications_by_type(
            user_id=user_id,
            notification_type=notification_type,
            limit=limit,
            skip=skip
        )

    async def get_unread_count(self, user_id: str) -> Dict[str, int]:
        """Get count of unread notifications for a user."""
        repository = await self._get_repository()
        count = await repository.get_unread_count(user_id)
        return {"unread_count": count}

    async def mark_notification_as_read(
        self,
        notification_id: str,
        user_id: str
    ) -> bool:
        """Mark a notification as read."""
        repository = await self._get_repository()
        success = await repository.mark_as_read(
            notification_id=notification_id,
            user_id=user_id
        )
        
        # Notify connected clients about the read status change
        if success and connection_manager.is_user_connected(user_id):
            await connection_manager.send_to_user(user_id, {
                "type": "notification_read",
                "data": {"notification_id": notification_id}
            })
        
        return success

    async def mark_all_notifications_as_read(
        self,
        user_id: str
    ) -> Dict[str, int]:
        """Mark all notifications as read for a user."""
        repository = await self._get_repository()
        count = await repository.mark_all_as_read(user_id=user_id)
        
        # Notify connected clients
        if connection_manager.is_user_connected(user_id):
            await connection_manager.send_to_user(user_id, {
                "type": "all_notifications_read",
                "data": {"marked_count": count}
            })
        
        return {"marked_count": count}

    async def delete_notification(
        self,
        notification_id: str,
        user_id: str
    ) -> bool:
        """Delete a notification."""
        repository = await self._get_repository()
        success = await repository.delete_notification(
            notification_id=notification_id,
            user_id=user_id
        )
        
        # Notify connected clients
        if success and connection_manager.is_user_connected(user_id):
            await connection_manager.send_to_user(user_id, {
                "type": "notification_deleted",
                "data": {"notification_id": notification_id}
            })
        
        return success

    async def delete_all_notifications(self, user_id: str) -> Dict[str, int]:
        """Delete all notifications for a user."""
        repository = await self._get_repository()
        count = await repository.delete_all_notifications(user_id)
        return {"deleted_count": count}

    async def _store_notification(
        self,
        notification_type: NotificationType,
        title: str,
        content: str,
        user_id: Optional[str] = None,
        user_email: Optional[EmailStr] = None,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """Store a notification in the database."""
        repository = await self._get_repository()
        notification = Notification(
            user_id=user_id,
            user_email=user_email,
            notification_type=notification_type,
            title=title,
            content=content,
            priority=priority,
            metadata=metadata or {},
            created_at=datetime.utcnow(),
            read=False
        )
        return await repository.create_notification(notification)

    async def send_wellbeing_reminder(
        self,
        user_id: str,
        reminder_type: str,
        message: str
    ) -> Notification:
        """Send a wellbeing reminder notification."""
        return await self.create_in_app_notification(
            user_id=user_id,
            title=f"Wellbeing Reminder: {reminder_type}",
            content=message,
            notification_type=NotificationType.WELLBEING,
            priority=NotificationPriority.NORMAL,
            metadata={"reminder_type": reminder_type}
        )

    async def send_task_notification(
        self,
        user_id: str,
        task_id: str,
        action: str,
        task_title: str
    ) -> Notification:
        """Send a task-related notification."""
        action_titles = {
            "created": "New Task Created",
            "completed": "Task Completed",
            "due_soon": "Task Due Soon",
            "overdue": "Task Overdue"
        }
        return await self.create_in_app_notification(
            user_id=user_id,
            title=action_titles.get(action, "Task Update"),
            content=f"Task '{task_title}' has been {action}",
            notification_type=NotificationType.TASK,
            priority=NotificationPriority.HIGH if action in ["due_soon", "overdue"] else NotificationPriority.NORMAL,
            metadata={"task_id": task_id, "action": action}
        )

    async def send_document_notification(
        self,
        user_id: str,
        document_id: str,
        action: str,
        document_name: str
    ) -> Notification:
        """Send a document-related notification."""
        return await self.create_in_app_notification(
            user_id=user_id,
            title=f"Document {action.title()}",
            content=f"Document '{document_name}' has been {action}",
            notification_type=NotificationType.DOCUMENT,
            priority=NotificationPriority.NORMAL,
            metadata={"document_id": document_id, "action": action}
        )

    async def send_system_notification(
        self,
        user_id: str,
        title: str,
        content: str,
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> Notification:
        """Send a system notification."""
        return await self.create_in_app_notification(
            user_id=user_id,
            title=title,
            content=content,
            notification_type=NotificationType.SYSTEM,
            priority=priority
        )

    async def send_welcome_notification(
        self,
        user_id: str,
        email: str,
        name: str
    ) -> None:
        """Send welcome email + in-app notification on signup."""
        import asyncio

        title = "Welcome to Lumicoria.ai! 🎉"
        content = f"Hi {name}, your account is ready. Start exploring our AI agents to boost your productivity."
        metadata = {"action": "signup"}

        try:
            # Send welcome email
            await self.send_email_notification(
                to_email=email,
                template_name="welcome",
                template_data={
                    "user_name": name,
                    "dashboard_url": "https://lumicoria.ai/dashboard",
                    "getting_started_url": "https://lumicoria.ai/getting-started",
                },
                priority=NotificationPriority.NORMAL,
            )
        except Exception as e:
            logger.error("Failed to send welcome email", user_id=user_id, error=str(e))

        try:
            # Create in-app welcome notification (no push yet)
            await self.create_in_app_notification(
                user_id=user_id,
                title=title,
                content=content,
                notification_type=NotificationType.AUTH,
                priority=NotificationPriority.NORMAL,
                metadata=metadata,
                send_push=False,
            )
        except Exception as e:
            logger.error("Failed to create welcome in-app notification", user_id=user_id, error=str(e))

        # Delay push so frontend has time to register FCM token
        try:
            await asyncio.sleep(5)
            await self._send_push_notification(user_id, title, content, metadata)
        except Exception as e:
            logger.error("Failed to send welcome push notification", user_id=user_id, error=str(e))

    async def send_login_alert(
        self,
        user_id: str,
        email: str,
        name: str,
        ip_address: str = None,
        device: str = None,
        activity_time: str = None,
    ) -> None:
        """Send security alert email + in-app notification on login.

        The push notification is sent after a short delay to give the
        frontend time to register the FCM device token (which happens
        after the login response is received).
        """
        import asyncio
        from datetime import datetime as dt

        activity_time = activity_time or dt.utcnow().strftime("%Y-%m-%d %H:%M UTC")

        title = "New Sign-In Detected"
        content = f"A new sign-in was detected from {ip_address or 'an unknown location'} at {activity_time}."
        metadata = {
            "action": "login",
            "ip_address": ip_address,
            "device": device,
        }

        # 1. Send email alert (no delay needed)
        try:
            await self.send_email_notification(
                to_email=email,
                template_name="security_alert",
                template_data={
                    "user_name": name,
                    "alert_type": "a new sign-in",
                    "activity_description": "New login to your account",
                    "activity_time": activity_time,
                    "ip_address": ip_address or "Unknown",
                    "device": device or "Unknown device",
                    "secure_account_url": f"{settings.BACKEND_CORS_ORIGINS[0] if settings.BACKEND_CORS_ORIGINS else 'https://lumicoria.ai'}/security",
                    "review_activity_url": f"{settings.BACKEND_CORS_ORIGINS[0] if settings.BACKEND_CORS_ORIGINS else 'https://lumicoria.ai'}/security/activity",
                },
                priority=NotificationPriority.HIGH,
            )
        except Exception as e:
            logger.error("Failed to send login alert email", user_id=user_id, error=str(e))

        # 2. Store in-app notification immediately (send_push=False, we handle push separately)
        try:
            await self.create_in_app_notification(
                user_id=user_id,
                title=title,
                content=content,
                notification_type=NotificationType.AUTH,
                priority=NotificationPriority.HIGH,
                metadata=metadata,
                send_push=False,  # Don't push yet — device token may not be registered
            )
        except Exception as e:
            logger.error("Failed to create login in-app notification", user_id=user_id, error=str(e))

        # 3. Delay push notification so frontend has time to register FCM token
        try:
            await asyncio.sleep(5)
            await self._send_push_notification(user_id, title, content, metadata)
        except Exception as e:
            logger.error("Failed to send login push notification", user_id=user_id, error=str(e))

    async def send_billing_notification(
        self,
        user_id: str,
        email: str,
        event: str,
        details: dict = None,
    ) -> None:
        """Send billing-related in-app notification (+ email for critical events)."""
        details = details or {}

        event_config = {
            "checkout_completed": {
                "title": "Subscription Confirmed! 🎉",
                "content": f"Your {details.get('plan', 'subscription')} plan is now active. Enjoy your upgraded features!",
                "priority": NotificationPriority.HIGH,
                "send_email": True,
            },
            "subscription_updated": {
                "title": "Plan Updated",
                "content": f"Your subscription has been updated to {details.get('plan', 'a new plan')}.",
                "priority": NotificationPriority.NORMAL,
                "send_email": False,
            },
            "subscription_deleted": {
                "title": "Subscription Cancelled",
                "content": "Your subscription has been cancelled. You'll retain access until the end of your billing period.",
                "priority": NotificationPriority.HIGH,
                "send_email": True,
            },
            "payment_succeeded": {
                "title": "Payment Received ✓",
                "content": f"Payment of {details.get('amount', 'your invoice')} was successfully processed.",
                "priority": NotificationPriority.NORMAL,
                "send_email": False,
            },
            "payment_failed": {
                "title": "⚠️ Payment Failed",
                "content": "We couldn't process your payment. Please update your payment method to avoid service interruption.",
                "priority": NotificationPriority.URGENT,
                "send_email": True,
            },
        }

        config = event_config.get(event, {
            "title": "Billing Update",
            "content": f"There's an update regarding your billing: {event}",
            "priority": NotificationPriority.NORMAL,
            "send_email": False,
        })

        # In-app notification
        try:
            await self.create_in_app_notification(
                user_id=user_id,
                title=config["title"],
                content=config["content"],
                notification_type=NotificationType.BILLING,
                priority=config["priority"],
                metadata={"event": event, **details},
            )
        except Exception as e:
            logger.error("Failed to create billing in-app notification", user_id=user_id, error=str(e))

        # Email for critical events
        if config.get("send_email") and email:
            try:
                await self.send_email_notification(
                    to_email=email,
                    template_name="notification",
                    template_data={
                        "user_name": details.get("name", "there"),
                        "subject": config["title"],
                        "message": config["content"],
                    },
                    priority=config["priority"],
                )
            except Exception as e:
                logger.error("Failed to send billing email", user_id=user_id, error=str(e))


# Create singleton instance
notification_service = NotificationService()