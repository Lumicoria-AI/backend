from datetime import datetime
from typing import Optional
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Enum, JSON, Float
from sqlalchemy.orm import relationship
import enum

from ..base import Base

class ActivityType(str, enum.Enum):
    TYPING = "typing"
    MOUSE_MOVEMENT = "mouse_movement"
    SCROLLING = "scrolling"
    MEETING = "meeting"
    BREAK = "break"
    FOCUS = "focus"
    OTHER = "other"

class BreakType(str, enum.Enum):
    MICRO_BREAK = "micro_break"  # Short breaks (5-10 minutes)
    LUNCH_BREAK = "lunch_break"  # Lunch break
    EXERCISE = "exercise"  # Exercise break
    MEDITATION = "meditation"  # Meditation break
    OTHER = "other"

class WellbeingMetrics(Base):
    __tablename__ = "wellbeing_metrics"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    date = Column(DateTime, nullable=False)
    
    # Activity metrics
    total_typing_time = Column(Integer, default=0)  # In seconds
    total_mouse_movement = Column(Integer, default=0)  # In pixels
    total_scroll_distance = Column(Integer, default=0)  # In pixels
    total_meeting_time = Column(Integer, default=0)  # In seconds
    total_focus_time = Column(Integer, default=0)  # In seconds
    
    # Break metrics
    total_break_time = Column(Integer, default=0)  # In seconds
    break_count = Column(Integer, default=0)
    average_break_duration = Column(Float, default=0.0)  # In seconds
    
    # Health metrics
    stress_level = Column(Integer, nullable=True)  # 1-10 scale
    energy_level = Column(Integer, nullable=True)  # 1-10 scale
    focus_score = Column(Integer, nullable=True)  # 1-100 scale
    productivity_score = Column(Integer, nullable=True)  # 1-100 scale
    
    # Custom metrics
    custom_metrics = Column(JSON, nullable=True)  # For any additional metrics
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="wellbeing_metrics")
    activity_logs = relationship("ActivityLog", back_populates="metrics")
    break_reminders = relationship("BreakReminder", back_populates="metrics")

class ActivityLog(Base):
    __tablename__ = "activity_logs"

    id = Column(Integer, primary_key=True, index=True)
    metrics_id = Column(Integer, ForeignKey("wellbeing_metrics.id", ondelete="CASCADE"))
    activity_type = Column(Enum(ActivityType), nullable=False)
    
    # Activity details
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    duration = Column(Integer, nullable=False)  # In seconds
    
    # Activity-specific data
    typing_speed = Column(Integer, nullable=True)  # Words per minute
    mouse_distance = Column(Integer, nullable=True)  # In pixels
    scroll_distance = Column(Integer, nullable=True)  # In pixels
    meeting_type = Column(String, nullable=True)  # For meeting activities
    focus_score = Column(Integer, nullable=True)  # 1-100 scale
    
    # Custom data
    custom_data = Column(JSON, nullable=True)  # For any additional activity data
    
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    metrics = relationship("WellbeingMetrics", back_populates="activity_logs")

class BreakReminder(Base):
    __tablename__ = "break_reminders"

    id = Column(Integer, primary_key=True, index=True)
    metrics_id = Column(Integer, ForeignKey("wellbeing_metrics.id", ondelete="CASCADE"))
    break_type = Column(Enum(BreakType), nullable=False)
    
    # Break details
    scheduled_time = Column(DateTime, nullable=False)
    actual_start_time = Column(DateTime, nullable=True)
    actual_end_time = Column(DateTime, nullable=True)
    duration = Column(Integer, nullable=True)  # In seconds
    
    # Break status
    status = Column(String, nullable=False)  # scheduled, started, completed, skipped
    reminder_count = Column(Integer, default=0)  # Number of times reminded
    
    # Break activities
    suggested_activities = Column(JSON, nullable=True)  # List of suggested activities
    completed_activities = Column(JSON, nullable=True)  # List of completed activities
    
    # User feedback
    user_feedback = Column(String, nullable=True)  # User's feedback about the break
    mood_before = Column(Integer, nullable=True)  # 1-10 scale
    mood_after = Column(Integer, nullable=True)  # 1-10 scale
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    metrics = relationship("WellbeingMetrics", back_populates="break_reminders") 