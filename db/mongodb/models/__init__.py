"""
MongoDB Models Package
This package contains all MongoDB model classes.
"""

from .user import User, UserCreate, UserUpdate, UserProfile, UserSettings
from .notification import Notification, NotificationCreate, NotificationUpdate, NotificationType, NotificationPriority
from .document import Document, DocumentCreate, DocumentUpdate, DocumentType, DocumentStatus

__all__ = [
    # User models
    'User',
    'UserCreate',
    'UserUpdate',
    'UserProfile',
    'UserSettings',
    # Notification models
    'Notification',
    'NotificationCreate',
    'NotificationUpdate',
    'NotificationType',
    'NotificationPriority',
    # Document models
    'Document',
    'DocumentCreate',
    'DocumentUpdate',
    'DocumentType',
    'DocumentStatus'
] 