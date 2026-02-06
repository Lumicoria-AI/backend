from typing import List, Dict, Any, Optional
from datetime import datetime
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from jinja2 import Environment, FileSystemLoader
from pathlib import Path
import json
from pydantic import BaseModel, EmailStr
from ..core.config import settings
from ..db.mongodb.repositories.notification_repository import notification_repository
from ..db.mongodb.models.notification import Notification, NotificationType, NotificationPriority

class NotificationTemplate(BaseModel):
    subject: str
    body: str
    template_name: str

class NotificationService:
    def __init__(self):
        self.email_templates_dir = Path(__file__).parent / "templates" / "email"
        self.jinja_env = Environment(loader=FileSystemLoader(str(self.email_templates_dir)))
        
        # Email settings from config
        self.smtp_server = settings.SMTP_SERVER
        self.smtp_port = settings.SMTP_PORT
        self.smtp_username = settings.SMTP_USERNAME
        self.smtp_password = settings.SMTP_PASSWORD
        self.smtp_from_email = settings.SMTP_FROM_EMAIL

    async def send_email_notification(
        self,
        to_email: EmailStr,
        template_name: str,
        template_data: Dict[str, Any],
        priority: NotificationPriority = NotificationPriority.NORMAL
    ) -> bool:
        """Send an email notification using a template."""
        try:
            # Load and render email template
            template = self.jinja_env.get_template(f"{template_name}.html")
            html_content = template.render(**template_data)
            
            # Create email message
            msg = MIMEMultipart()
            msg['From'] = self.smtp_from_email
            msg['To'] = to_email
            msg['Subject'] = template_data.get('subject', 'Notification from Lumicoria.ai')
            
            msg.attach(MIMEText(html_content, 'html'))
            
            # Send email
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            # Store notification in database
            await self._store_notification(
                notification_type=NotificationType.EMAIL,
                title=msg['Subject'],
                content=html_content,
                user_email=to_email,
                priority=priority,
                metadata={"template": template_name, "template_data": template_data}
            )
            
            return True
        except Exception as e:
            # Log error and return False
            print(f"Error sending email notification: {str(e)}")
            return False

    async def create_in_app_notification(
        self,
        user_id: str,
        title: str,
        content: str,
        notification_type: NotificationType,
        priority: NotificationPriority = NotificationPriority.NORMAL,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Notification:
        """Create and store an in-app notification."""
        notification = await self._store_notification(
            notification_type=notification_type,
            title=title,
            content=content,
            user_id=user_id,
            priority=priority,
            metadata=metadata
        )
        return notification

    async def get_user_notifications(
        self,
        user_id: str,
        unread_only: bool = False,
        limit: int = 50,
        skip: int = 0
    ) -> List[Notification]:
        """Get notifications for a user."""
        return await notification_repository.get_user_notifications(
            user_id=user_id,
            unread_only=unread_only,
            limit=limit,
            skip=skip
        )

    async def mark_notification_as_read(
        self,
        notification_id: str,
        user_id: str
    ) -> bool:
        """Mark a notification as read."""
        return await notification_repository.mark_as_read(
            notification_id=notification_id,
            user_id=user_id
        )

    async def mark_all_notifications_as_read(
        self,
        user_id: str
    ) -> bool:
        """Mark all notifications as read for a user."""
        return await notification_repository.mark_all_as_read(user_id=user_id)

    async def delete_notification(
        self,
        notification_id: str,
        user_id: str
    ) -> bool:
        """Delete a notification."""
        return await notification_repository.delete_notification(
            notification_id=notification_id,
            user_id=user_id
        )

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
        return await notification_repository.create_notification(notification)

# Create singleton instance
notification_service = NotificationService() 