"""
Lumicoria.ai Backend Package
This is the main package initialization file that exposes our core modules.
"""

from .core.config import settings
from .db.mongodb import get_mongodb, MongoDB
from .services.notification_service import notification_service

__all__ = [
    'settings',
    'get_mongodb',
    'MongoDB',
    'notification_service'
]
