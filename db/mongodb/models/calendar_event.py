"""
Lumicoria-native calendar event model.

The Lumicoria calendar is the source of truth for scheduled work in the
app — tasks with due dates create events here automatically.  When a user
has connected Google Workspace, events are *mirrored* to Google Calendar
(see Phase 3) but Lumicoria's calendar always works without Google.

Collection: `lumicoria_calendar_events`
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field

from backend.models.mongodb_models import PyObjectId


class CalendarEventSource(str, Enum):
    """Where this calendar event originated."""
    TASK = "task"                    # auto-created from a Task with a due_date
    MANUAL = "manual"                # user created via Calendar UI
    GCAL_IMPORTED = "gcal_imported"  # imported from Google Calendar
    AGENT = "agent"                  # created by an agent (meeting agent etc.)


class CalendarEventStatus(str, Enum):
    SCHEDULED = "scheduled"
    COMPLETED = "completed"          # mirrors task.status == completed
    CANCELLED = "cancelled"


class CalendarEventReminder(BaseModel):
    """Optional inline reminders attached to a calendar event."""
    minutes_before: int = 30
    channel: str = "in_app"          # "in_app" | "email" | "push"


class CalendarEvent(BaseModel):
    """Mongo model for `lumicoria_calendar_events`.

    `task_id` links back to the originating task when source == "task".
    `gcal_event_id` is set when the event has been mirrored to Google.
    """
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    owner_user_id: PyObjectId                              # owner of the calendar
    organization_id: Optional[PyObjectId] = None
    task_id: Optional[PyObjectId] = None                   # tasks._id
    project_id: Optional[PyObjectId] = None                # lumicoria_projects._id

    title: str
    description: Optional[str] = None
    location: Optional[str] = None
    start: datetime
    end: datetime
    all_day: bool = False
    color: str = "#6C4AB0"                                  # default Lumicoria purple
    timezone: str = "UTC"

    source: CalendarEventSource = CalendarEventSource.MANUAL
    status: CalendarEventStatus = CalendarEventStatus.SCHEDULED
    gcal_event_id: Optional[str] = None                    # set if mirrored to Google
    gcal_calendar_id: Optional[str] = None                 # which Google calendar
    last_synced_at: Optional[datetime] = None

    attendees: List[Dict[str, Any]] = Field(default_factory=list)
    # e.g. [{"email": "x@y.com", "user_id": "...", "response": "accepted|declined|tentative"}]
    reminders: List[CalendarEventReminder] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    deleted_at: Optional[datetime] = None                  # soft delete

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str, datetime: lambda v: v.isoformat()},
    )


class CalendarEventCreate(BaseModel):
    """Payload for creating a calendar event via the API."""
    title: str
    description: Optional[str] = None
    location: Optional[str] = None
    start: datetime
    end: datetime
    all_day: bool = False
    color: str = "#6C4AB0"
    timezone: str = "UTC"
    task_id: Optional[PyObjectId] = None
    project_id: Optional[PyObjectId] = None
    source: CalendarEventSource = CalendarEventSource.MANUAL
    attendees: List[Dict[str, Any]] = Field(default_factory=list)
    reminders: List[CalendarEventReminder] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    sync_to_google: bool = False                            # opt-in mirror

    model_config = ConfigDict(arbitrary_types_allowed=True)


class CalendarEventUpdate(BaseModel):
    """Patch payload — every field optional."""
    title: Optional[str] = None
    description: Optional[str] = None
    location: Optional[str] = None
    start: Optional[datetime] = None
    end: Optional[datetime] = None
    all_day: Optional[bool] = None
    color: Optional[str] = None
    timezone: Optional[str] = None
    status: Optional[CalendarEventStatus] = None
    attendees: Optional[List[Dict[str, Any]]] = None
    reminders: Optional[List[CalendarEventReminder]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None

    model_config = ConfigDict(arbitrary_types_allowed=True)
