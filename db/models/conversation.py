from datetime import datetime
from typing import Optional
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Text, Enum, JSON
from sqlalchemy.orm import relationship
import enum

from ..base import Base

class ConversationType(str, enum.Enum):
    USER_AI = "user_ai" # Conversation between a single user and an AI
    USER_USER = "user_user" # Direct message between two users
    TEAM_CHAT = "team_chat" # Chat within a team
    ORGANIZATION_CHAT = "organization_chat" # Chat within an organization
    AGENT_DEBUG = "agent_debug" # Conversation for debugging a specific agent workflow

class Conversation(Base):
    __tablename__ = "conversations"

    id = Column(Integer, primary_key=True, index=True)
    uuid = Column(String, unique=True, nullable=False) # Unique identifier for the conversation
    title = Column(String, nullable=True) # Optional title for the conversation
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True) # Primary user initiating or involved
    agent_id = Column(Integer, ForeignKey("agents.id", ondelete="SET NULL"), nullable=True) # AI agent involved in user_ai conversations
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True) # Organization for group chats
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="SET NULL"), nullable=True) # Team for team chats
    conversation_type = Column(Enum(ConversationType), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    agent = relationship("Agent") # Relationship to the specific agent
    organization = relationship("Organization")
    team = relationship("Team")
    messages = relationship("Message", back_populates="conversation", order_by="Message.created_at")

class Message(Base):
    __tablename__ = "messages"

    id = Column(Integer, primary_key=True, index=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False)
    sender_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True) # User sending the message (None for AI)
    content = Column(Text, nullable=False)
    sent_at = Column(DateTime, default=datetime.utcnow)
    
    # Metadata for AI responses
    is_ai_response = Column(Boolean, default=False)
    ai_model_used = Column(String, nullable=True)
    tool_calls = Column(JSON, nullable=True) # Store information about tool calls made by the AI
    
    conversation = relationship("Conversation", back_populates="messages")
    sender = relationship("User") 