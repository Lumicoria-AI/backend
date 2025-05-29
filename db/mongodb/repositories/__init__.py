"""
MongoDB Repositories Package
This package contains all MongoDB repository classes.
"""

from .user_repository import get_user_repository
from .notification_repository import get_notification_repository
from .document_repository import get_document_repository

__all__ = [
    'get_user_repository',
    'get_notification_repository',
    'get_document_repository'
] 