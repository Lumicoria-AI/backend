from datetime import datetime
from typing import Optional
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Enum, JSON, Text
from sqlalchemy.orm import relationship
import enum

from ..base import Base
from .agent_studio import AgentWorkflow # Import the new workflow model
from .context import ContextStrategy # Import the new ContextStrategy model

class AgentType(str, enum.Enum):
    DOCUMENT = "document"  # Document processing agent
    WELLBEING = "wellbeing"  # Well-being coach agent
    MEETING = "meeting"  # Meeting assistant agent
    CREATIVE = "creative"  # Creative tasks agent
    VISION = "vision"  # Computer vision agent
    CUSTOM = "custom"  # User-defined custom agent

class AgentStatus(str, enum.Enum):
    ACTIVE = "active"
    INACTIVE = "inactive"
    ERROR = "error"
    UPDATING = "updating"

class Agent(Base):
    __tablename__ = "agents"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL")) # Personal agent owner
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True) # Agent owned by an organization
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="SET NULL"), nullable=True) # Agent owned by a team

    name = Column(String, nullable=False)
    description = Column(Text, nullable=True)
    agent_type = Column(Enum(AgentType), nullable=False)
    status = Column(Enum(AgentStatus), default=AgentStatus.ACTIVE)
    
    # Agent capabilities
    capabilities = Column(JSON, nullable=True)  # List of agent capabilities
    supported_file_types = Column(JSON, nullable=True)  # For document agents
    supported_languages = Column(JSON, nullable=True)  # List of supported languages
    
    # Agent configuration
    configuration = relationship("AgentConfiguration", back_populates="agent", uselist=False)
    workflow = relationship("AgentWorkflow", back_populates="agent", uselist=False) # Link for CUSTOM agents
    context_strategy_id = Column(Integer, ForeignKey("context_strategies.id", ondelete="SET NULL"), nullable=True) # Link to the context strategy
    context_strategy = relationship("ContextStrategy")
    
    # Agent metadata
    version = Column(String, nullable=False, default="1.0.0")
    is_public = Column(Boolean, default=False)  # Whether agent is available to all users
    usage_count = Column(Integer, default=0)  # Number of times agent has been used
    last_used_at = Column(DateTime, nullable=True)
    
    # Agent performance
    success_rate = Column(Integer, nullable=True)  # 0-100 scale
    average_response_time = Column(Integer, nullable=True)  # In milliseconds
    error_rate = Column(Integer, nullable=True)  # 0-100 scale
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    owner = relationship("User", back_populates="agents")
    organization = relationship("Organization", back_populates="agents")
    team = relationship("Team", back_populates="agents")

class AgentConfiguration(Base):
    __tablename__ = "agent_configurations"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id", ondelete="CASCADE"), unique=True)
    
    # AI model settings
    model_provider = Column(String, nullable=False)  # openai, anthropic, google, etc.
    model_name = Column(String, nullable=False)  # gpt-4, claude-2, gemini-pro, etc.
    model_version = Column(String, nullable=True)
    temperature = Column(Integer, nullable=True)  # 0-100 scale
    max_tokens = Column(Integer, nullable=True)
    
    # Agent behavior settings
    system_prompt = Column(Text, nullable=True)
    user_prompt_template = Column(Text, nullable=True)
    response_format = Column(JSON, nullable=True)  # Expected response format
    retry_policy = Column(JSON, nullable=True)  # Retry configuration
    rate_limits = Column(JSON, nullable=True)  # Rate limiting configuration
    
    # Integration settings
    api_keys = Column(JSON, nullable=True)  # Encrypted API keys
    webhooks = Column(JSON, nullable=True)  # Webhook configurations
    integrations = Column(JSON, nullable=True)  # Third-party integrations
    
    # Custom settings
    custom_settings = Column(JSON, nullable=True)  # Any additional settings
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    agent = relationship("Agent", back_populates="configuration") 