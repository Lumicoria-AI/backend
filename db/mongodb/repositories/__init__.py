"""
MongoDB Repositories Package

This package contains all MongoDB repository classes.  Each repository is
exposed both as a singleton instance (for direct import) and via an async
factory (`get_*_repository`) for places that need lazy initialisation.
"""

from .user_repository import get_user_repository
from .notification_repository import get_notification_repository
from .document_repository import get_document_repository

# ── Phase 1 additions: calendar, invite, agent run ─────────────────────────
from .calendar_repository import calendar_repository, CalendarRepository
from .invite_repository import invite_repository, InviteRepository
from .agent_run_repository import agent_run_repository, AgentRunRepository

__all__ = [
    'get_user_repository',
    'get_notification_repository',
    'get_document_repository',
    # Phase 1
    'calendar_repository',
    'CalendarRepository',
    'invite_repository',
    'InviteRepository',
    'agent_run_repository',
    'AgentRunRepository',
]
