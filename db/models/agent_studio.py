from datetime import datetime
from typing import Optional
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, JSON, Enum
from sqlalchemy.orm import relationship
import enum

from ..base import Base
from .context import ContextStrategy # Import the new ContextStrategy model

class AgentComponentType(str, enum.Enum):
    INPUT = "input"
    OUTPUT = "output"
    PROCESSOR = "processor" # e.g., NLP, Vision, Data Manipulation
    INTEGRATION = "integration" # e.g., Google Search, Notion, Slack
    LOGIC = "logic" # e.g., Conditional, Loop
    OTHER = "other"

class AgentComponent(Base):
    __tablename__ = "agent_components"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, nullable=False) # e.g., "Text Input", "Google Search", "Summarizer"
    description = Column(Text, nullable=True)
    component_type = Column(Enum(AgentComponentType), nullable=False)
    configuration_schema = Column(JSON, nullable=True) # JSON schema for component-specific settings
    input_schema = Column(JSON, nullable=True) # JSON schema for input data structure
    output_schema = Column(JSON, nullable=True) # JSON schema for output data structure
    is_public = Column(Boolean, default=True) # Whether this component is available in the studio
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    workflows = relationship("AgentWorkflowNode", back_populates="component")

class AgentWorkflow(Base):
    __tablename__ = "agent_workflows"

    id = Column(Integer, primary_key=True, index=True)
    agent_id = Column(Integer, ForeignKey("agents.id", ondelete="CASCADE"), unique=True) # Link to the custom agent
    context_strategy_id = Column(Integer, ForeignKey("context_strategies.id", ondelete="SET NULL"), nullable=True) # Link to the context strategy
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    agent = relationship("Agent", back_populates="workflow")
    context_strategy = relationship("ContextStrategy")
    nodes = relationship("AgentWorkflowNode", back_populates="workflow")
    connections = relationship("AgentWorkflowConnection", back_populates="workflow")

class AgentWorkflowNode(Base):
    __tablename__ = "agent_workflow_nodes"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey("agent_workflows.id", ondelete="CASCADE"))
    component_id = Column(Integer, ForeignKey("agent_components.id", ondelete="CASCADE"))
    
    # Node specific details for the editor
    position_x = Column(Integer, nullable=True)
    position_y = Column(Integer, nullable=True)
    
    # Node specific configuration (overrides component defaults)
    configuration = Column(JSON, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)

    workflow = relationship("AgentWorkflow", back_populates="nodes")
    component = relationship("AgentComponent", back_populates="workflows")
    outgoing_connections = relationship("AgentWorkflowConnection", foreign_keys="AgentWorkflowConnection.source_node_id", back_populates="source_node")
    incoming_connections = relationship("AgentWorkflowConnection", foreign_keys="AgentWorkflowConnection.target_node_id", back_populates="target_node")

class AgentWorkflowConnection(Base):
    __tablename__ = "agent_workflow_connections"

    id = Column(Integer, primary_key=True, index=True)
    workflow_id = Column(Integer, ForeignKey("agent_workflows.id", ondelete="CASCADE"))
    source_node_id = Column(Integer, ForeignKey("agent_workflow_nodes.id", ondelete="CASCADE"))
    source_output_key = Column(String, nullable=False) # Key from the source node's output_schema
    target_node_id = Column(Integer, ForeignKey("agent_workflow_nodes.id", ondelete="CASCADE"))
    target_input_key = Column(String, nullable=False) # Key from the target node's input_schema
    created_at = Column(DateTime, default=datetime.utcnow)

    workflow = relationship("AgentWorkflow", back_populates="connections")
    source_node = relationship("AgentWorkflowNode", foreign_keys=[source_node_id], back_populates="outgoing_connections")
    target_node = relationship("AgentWorkflowNode", foreign_keys=[target_node_id], back_populates="incoming_connections") 