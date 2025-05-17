from datetime import datetime
from typing import Optional
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Enum, Text, JSON
from sqlalchemy.orm import relationship
import enum

from ..base import Base

class TaskPriority(str, enum.Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    URGENT = "urgent"

class TaskStatus(str, enum.Enum):
    TODO = "todo"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"

class Task(Base):
    __tablename__ = "tasks"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    assignee_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"))
    source_document_id = Column(Integer, ForeignKey("documents.id", ondelete="SET NULL"))
    
    # Task details
    priority = Column(Enum(TaskPriority), default=TaskPriority.MEDIUM)
    status = Column(Enum(TaskStatus), default=TaskStatus.TODO)
    due_date = Column(DateTime, nullable=True)
    estimated_hours = Column(Integer, nullable=True)
    actual_hours = Column(Integer, nullable=True)
    
    # Task metadata
    tags = Column(JSON, nullable=True)  # List of tags
    custom_fields = Column(JSON, nullable=True)  # For any additional fields
    ai_generated = Column(Boolean, default=False)  # Whether task was AI-generated
    
    # Task tracking
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    cancelled_at = Column(DateTime, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    assignee = relationship("User", back_populates="tasks")
    source_document = relationship("Document", back_populates="tasks")
    calendar_event = relationship("CalendarEvent", back_populates="task", uselist=False)

class CalendarEvent(Base):
    __tablename__ = "calendar_events"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    task_id = Column(Integer, ForeignKey("tasks.id", ondelete="SET NULL"), unique=True)
    source_document_id = Column(Integer, ForeignKey("documents.id", ondelete="SET NULL"))
    
    # Event details
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    all_day = Column(Boolean, default=False)
    location = Column(String, nullable=True)
    meeting_link = Column(String, nullable=True)
    
    # Event metadata
    event_type = Column(String, nullable=True)  # meeting, reminder, deadline, etc.
    recurrence_rule = Column(String, nullable=True)  # iCal RRULE format
    attendees = Column(JSON, nullable=True)  # List of attendee emails
    custom_fields = Column(JSON, nullable=True)  # For any additional fields
    ai_generated = Column(Boolean, default=False)  # Whether event was AI-generated
    
    # Calendar integration
    external_calendar_id = Column(String, nullable=True)  # ID in external calendar
    external_calendar_type = Column(String, nullable=True)  # google, outlook, etc.
    sync_status = Column(String, nullable=True)  # synced, pending, failed
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    task = relationship("Task", back_populates="calendar_event")
    source_document = relationship("Document", back_populates="calendar_events") 