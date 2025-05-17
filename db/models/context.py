from datetime import datetime
from typing import Optional
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, JSON, Enum
from sqlalchemy.orm import relationship
import enum

from ..base import Base

class ContextStrategyType(str, enum.Enum):
    DEFAULT = "default" # Standard context handling
    RAG_DOCUMENT = "rag_document" # Retrieve context from documents via RAG
    RAG_CONVERSATION = "rag_conversation" # Retrieve context from past conversations via RAG
    USER_PROFILE = "user_profile" # Include specific user profile details
    USER_SETTINGS = "user_settings" # Include specific user settings
    AGENT_CONFIG = "agent_config" # Include specific agent configuration
    WORKFLOW_STATE = "workflow_state" # Include current workflow execution state
    COMBINED = "combined" # Combination of multiple strategies
    OTHER = "other"

class ContextStrategy(Base):
    __tablename__ = "context_strategies"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, nullable=False) # e.g., "Document Agent Default Strategy", "Wellbeing Coach Short History"
    description = Column(Text, nullable=True)
    strategy_type = Column(Enum(ContextStrategyType), nullable=False)
    
    # Configuration for the strategy (JSON to allow flexibility)
    configuration = Column(JSON, nullable=True) 
    # Examples of configuration:
    # For RAG_DOCUMENT: {"document_types": ["contract", "invoice"], "max_results": 5, "similarity_threshold": 0.7}
    # For RAG_CONVERSATION: {"max_messages": 10, "filter_by_agent": True, "agent_id": 123}
    # For USER_PROFILE/SETTINGS: {"include_fields": ["job_title", "timezone", "preferred_ai_model"]}
    # For COMBINED: [{"strategy_type": "RAG_DOCUMENT", "config": {...}}, {"strategy_type": "USER_SETTINGS", "config": {...}}]

    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships (backref from Agent/AgentWorkflow) 