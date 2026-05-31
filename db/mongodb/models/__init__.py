"""
MongoDB Models Package
This package contains all MongoDB model classes.
"""

from .user import User, UserCreate, UserUpdate, UserProfile, UserSettings, TaskReminderSettings
from .notification import Notification, NotificationCreate, NotificationUpdate, NotificationType, NotificationPriority
from .document import Document, DocumentCreate, DocumentUpdate, DocumentType, DocumentStatus

# ── Phase 1 additions ─────────────────────────────────────────────────────
from .calendar_event import (
    CalendarEvent,
    CalendarEventCreate,
    CalendarEventUpdate,
    CalendarEventSource,
    CalendarEventStatus,
    CalendarEventReminder,
)
from .invite import (
    Invite,
    InviteCreate,
    InviteUpdate,
    InviteStatus,
    InviteRole,
    InviteScope,
)
from .agent_run import (
    AgentRun,
    AgentRunCreate,
    AgentRunStatus,
    AgentRunTrigger,
)

__all__ = [
    # User
    'User', 'UserCreate', 'UserUpdate', 'UserProfile', 'UserSettings', 'TaskReminderSettings',
    # Notification
    'Notification', 'NotificationCreate', 'NotificationUpdate', 'NotificationType', 'NotificationPriority',
    # Document
    'Document', 'DocumentCreate', 'DocumentUpdate', 'DocumentType', 'DocumentStatus',
    # Phase 1: calendar
    'CalendarEvent', 'CalendarEventCreate', 'CalendarEventUpdate',
    'CalendarEventSource', 'CalendarEventStatus', 'CalendarEventReminder',
    # Phase 1: invites
    'Invite', 'InviteCreate', 'InviteUpdate', 'InviteStatus', 'InviteRole', 'InviteScope',
    # Phase 1: agent runs
    'AgentRun', 'AgentRunCreate', 'AgentRunStatus', 'AgentRunTrigger',
]
