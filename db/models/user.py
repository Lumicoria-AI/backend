from datetime import datetime
from typing import Optional, List
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Enum, JSON
from sqlalchemy.orm import relationship
import enum

from ..base import Base
from .conversation import Conversation

class UserRole(str, enum.Enum):
    USER = "user"
    ADMIN = "admin"
    ENTERPRISE = "enterprise"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    firebase_uid = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=True)  # For non-Firebase auth
    is_active = Column(Boolean, default=True)
    role = Column(Enum(UserRole), default=UserRole.USER)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    settings = relationship("UserSettings", back_populates="user", uselist=False)
    documents = relationship("Document", back_populates="owner")
    tasks = relationship("Task", back_populates="assignee")
    wellbeing_metrics = relationship("WellbeingMetrics", back_populates="user")
    agents = relationship("Agent", back_populates="owner") # This relationship is for agents *owned* by the user (custom agents)
    user_agents = relationship("UserAgent", back_populates="user") # This relationship links to the instances of agents the user is using

    # Organization and Team Relationships
    organizations = relationship("UserOrganization", back_populates="user")
    teams = relationship("UserTeam", back_populates="user")
    owned_organizations = relationship("Organization", back_populates="owner")

    # Integration Relationships
    google_workspace_integrations = relationship("GoogleWorkspaceIntegration", back_populates="user")
    slack_integrations = relationship("SlackIntegration", back_populates="user")
    notion_integrations = relationship("NotionIntegration", back_populates="user")
    salesforce_integrations = relationship("SalesforceIntegration", back_populates="user")

    # Conversation Relationship
    conversations = relationship("Conversation", back_populates="user")

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    full_name = Column(String)
    avatar_url = Column(String, nullable=True)
    job_title = Column(String, nullable=True)
    company = Column(String, nullable=True)
    timezone = Column(String, default="UTC")
    preferred_language = Column(String, default="en")
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="profile")

class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    
    # Notification settings
    email_notifications = Column(Boolean, default=True)
    push_notifications = Column(Boolean, default=True)
    task_reminders = Column(Boolean, default=True)
    break_reminders = Column(Boolean, default=True)
    
    # Well-being settings
    work_hours_start = Column(String, default="09:00")  # 24-hour format
    work_hours_end = Column(String, default="17:00")
    break_interval_minutes = Column(Integer, default=60)
    break_duration_minutes = Column(Integer, default=5)
    
    # AI preferences
    preferred_ai_model = Column(String, default="gemini")
    auto_task_creation = Column(Boolean, default=True)
    auto_calendar_sync = Column(Boolean, default=True)
    
    # Integration settings
    google_calendar_sync = Column(Boolean, default=False) # Deprecated: use GoogleWorkspaceIntegration model
    slack_integration = Column(Boolean, default=False) # Deprecated: use SlackIntegration model
    notion_integration = Column(Boolean, default=False) # Deprecated: use NotionIntegration model
    salesforce_integration = Column(Boolean, default=False) # Deprecated: use SalesforceIntegration model
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    user = relationship("User", back_populates="settings")

class UserAgent(Base):
    __tablename__ = "user_agents"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    agent_id = Column(Integer, ForeignKey("agents.id", ondelete="CASCADE")) # Link to the base agent definition

    custom_name = Column(String, nullable=True) # User-defined name for this instance
    custom_configuration = Column(JSON, nullable=True) # User-specific overrides or additions to agent config
    status = Column(String, default="active") # e.g., active, paused, archived

    usage_count = Column(Integer, default=0)
    added_at = Column(DateTime, default=datetime.utcnow)
    last_used_at = Column(DateTime, nullable=True)

    # Relationships
    user = relationship("User", back_populates="user_agents")
    agent = relationship("Agent") # Relationship to the base agent 