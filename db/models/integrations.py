from datetime import datetime
from typing import Optional
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, JSON
from sqlalchemy.orm import relationship

from ..base import Base

class GoogleWorkspaceIntegration(Base):
    __tablename__ = "google_workspace_integrations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    
    # Google Workspace specific fields
    google_user_id = Column(String, unique=True, nullable=False)
    access_token = Column(String, nullable=False) # Encrypted in production
    refresh_token = Column(String, nullable=False) # Encrypted in production
    expires_at = Column(DateTime, nullable=False)
    
    # Permissions granted
    scopes = Column(JSON, nullable=True) # List of granted scopes
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="google_workspace_integrations")

class SlackIntegration(Base):
    __tablename__ = "slack_integrations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    
    # Slack specific fields
    slack_user_id = Column(String, unique=True, nullable=False)
    slack_team_id = Column(String, nullable=False)
    access_token = Column(String, nullable=False) # Encrypted in production
    
    # Permissions granted
    scopes = Column(JSON, nullable=True) # List of granted scopes
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="slack_integrations")

class NotionIntegration(Base):
    __tablename__ = "notion_integrations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    
    # Notion specific fields
    notion_user_id = Column(String, unique=True, nullable=False)
    access_token = Column(String, nullable=False) # Encrypted in production
    workspace_id = Column(String, nullable=False)
    workspace_name = Column(String, nullable=True)
    workspace_icon = Column(String, nullable=True)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="notion_integrations")

class SalesforceIntegration(Base):
    __tablename__ = "salesforce_integrations"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"))
    
    # Salesforce specific fields
    salesforce_user_id = Column(String, unique=True, nullable=False)
    organization_id = Column(String, nullable=False)
    instance_url = Column(String, nullable=False)
    access_token = Column(String, nullable=False) # Encrypted in production
    refresh_token = Column(String, nullable=True) # Encrypted in production
    expires_at = Column(DateTime, nullable=True)
    
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="salesforce_integrations") 