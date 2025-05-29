"""
Agent models for Lumicoria.ai
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field

class AgentType(str, Enum):
    """Types of agents available in the system."""
    DOCUMENT = "document"
    MEETING = "meeting"
    RESEARCH = "research"
    CREATIVE = "creative"
    STUDENT = "student"
    WELLBEING = "wellbeing"
    VISION = "vision"
    TRANSLATION = "translation"
    CUSTOM = "custom"

class AgentStatus(str, Enum):
    """Status of an agent."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    MAINTENANCE = "maintenance"
    DEPRECATED = "deprecated"

class AgentCapability(str, Enum):
    """Capabilities that an agent can have."""
    TEXT_PROCESSING = "text_processing"
    IMAGE_PROCESSING = "image_processing"
    AUDIO_PROCESSING = "audio_processing"
    DOCUMENT_ANALYSIS = "document_analysis"
    MEETING_SUMMARY = "meeting_summary"
    RESEARCH_ANALYSIS = "research_analysis"
    CREATIVE_WRITING = "creative_writing"
    STUDENT_ASSISTANCE = "student_assistance"
    WELLBEING_COACHING = "wellbeing_coaching"
    VISION_ANALYSIS = "vision_analysis"
    TRANSLATION = "translation"
    CUSTOM_TASK = "custom_task"

class AgentBase(BaseModel):
    """Base model for agent data."""
    name: str = Field(..., description="Name of the agent")
    description: str = Field(..., description="Description of the agent's purpose and capabilities")
    agent_type: AgentType = Field(..., description="Type of agent")
    capabilities: List[AgentCapability] = Field(..., description="List of agent capabilities")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Additional metadata for the agent")

class AgentCreate(AgentBase):
    """Model for creating a new agent."""
    organization_id: str = Field(..., description="ID of the organization this agent belongs to")
    created_by: str = Field(..., description="ID of the user creating the agent")
    preferred_model: Optional[str] = Field(None, description="Preferred AI model to use")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")

class AgentUpdate(BaseModel):
    """Model for updating an existing agent."""
    name: Optional[str] = Field(None, description="Updated name of the agent")
    description: Optional[str] = Field(None, description="Updated description")
    status: Optional[AgentStatus] = Field(None, description="Updated status")
    capabilities: Optional[List[AgentCapability]] = Field(None, description="Updated capabilities")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Updated metadata")
    preferred_model: Optional[str] = Field(None, description="Updated preferred AI model")
    system_prompt: Optional[str] = Field(None, description="Updated system prompt")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Updated configuration")

class Agent(AgentBase):
    """Complete agent model including database fields."""
    id: str = Field(..., description="Unique identifier for the agent")
    organization_id: str = Field(..., description="ID of the organization this agent belongs to")
    created_by: str = Field(..., description="ID of the user who created the agent")
    status: AgentStatus = Field(default=AgentStatus.ACTIVE, description="Current status of the agent")
    created_at: datetime = Field(..., description="Timestamp when the agent was created")
    updated_at: Optional[datetime] = Field(None, description="Timestamp when the agent was last updated")
    preferred_model: Optional[str] = Field(None, description="Preferred AI model to use")
    system_prompt: Optional[str] = Field(None, description="System prompt for the agent")
    configuration: Optional[Dict[str, Any]] = Field(None, description="Agent configuration")
    success_rate: Optional[float] = Field(None, description="Agent's success rate")
    error_rate: Optional[float] = Field(None, description="Agent's error rate")
    avg_response_time: Optional[float] = Field(None, description="Average response time in seconds")

    class Config:
        """Pydantic model configuration."""
        from_attributes = True  # For ORM mode
        json_encoders = {
            datetime: lambda v: v.isoformat()
        } 