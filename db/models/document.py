from datetime import datetime
from typing import Optional
from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Integer, String, Enum, JSON, Text
from sqlalchemy.orm import relationship
import enum

from ..base import Base

class DocumentType(str, enum.Enum):
    CONTRACT = "contract"
    INVOICE = "invoice"
    RECEIPT = "receipt"
    NOTE = "note"
    EMAIL = "email"
    MEETING_MINUTES = "meeting_minutes"
    OTHER = "other"

class DocumentStatus(str, enum.Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

class Document(Base):
    __tablename__ = "documents"

    id = Column(Integer, primary_key=True, index=True)
    owner_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL")) # Personal document owner
    organization_id = Column(Integer, ForeignKey("organizations.id", ondelete="SET NULL"), nullable=True) # Document owned by an organization
    team_id = Column(Integer, ForeignKey("teams.id", ondelete="SET NULL"), nullable=True) # Document owned by a team

    title = Column(String, nullable=False)
    document_type = Column(Enum(DocumentType), nullable=False)
    status = Column(Enum(DocumentStatus), default=DocumentStatus.PENDING)
    
    # Storage
    file_path = Column(String, nullable=False)  # Path to stored file
    file_size = Column(Integer)  # Size in bytes
    mime_type = Column(String)
    
    # Processing
    processing_started_at = Column(DateTime, nullable=True)
    processing_completed_at = Column(DateTime, nullable=True)
    processing_error = Column(Text, nullable=True)
    
    # Vector store
    vector_store_id = Column(String, nullable=True)  # ID in vector store
    embedding_model = Column(String, nullable=True)  # Model used for embedding
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    owner = relationship("User", back_populates="documents")
    organization = relationship("Organization", back_populates="documents")
    team = relationship("Team", back_populates="documents")
    metadata = relationship("DocumentMetadata", back_populates="document", uselist=False)
    processing_status = relationship("DocumentProcessingStatus", back_populates="document", uselist=False)
    tasks = relationship("Task", back_populates="source_document")
    calendar_events = relationship("CalendarEvent", back_populates="source_document")

class DocumentMetadata(Base):
    __tablename__ = "document_metadata"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), unique=True)
    
    # Extracted metadata
    author = Column(String, nullable=True)
    creation_date = Column(DateTime, nullable=True)
    modification_date = Column(DateTime, nullable=True)
    page_count = Column(Integer, nullable=True)
    word_count = Column(Integer, nullable=True)
    
    # Document-specific metadata
    contract_parties = Column(JSON, nullable=True)  # For contracts
    invoice_number = Column(String, nullable=True)  # For invoices
    invoice_amount = Column(String, nullable=True)  # For invoices
    due_date = Column(DateTime, nullable=True)  # For invoices/contracts
    
    # Custom metadata
    custom_fields = Column(JSON, nullable=True)  # For any additional metadata
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="metadata")

class DocumentProcessingStatus(Base):
    __tablename__ = "document_processing_status"

    id = Column(Integer, primary_key=True, index=True)
    document_id = Column(Integer, ForeignKey("documents.id", ondelete="CASCADE"), unique=True)
    
    # Processing stages
    ocr_completed = Column(Boolean, default=False)
    text_extraction_completed = Column(Boolean, default=False)
    metadata_extraction_completed = Column(Boolean, default=False)
    entity_recognition_completed = Column(Boolean, default=False)
    task_generation_completed = Column(Boolean, default=False)
    calendar_event_generation_completed = Column(Boolean, default=False)
    
    # Processing details
    ocr_confidence = Column(Integer, nullable=True)  # 0-100
    extraction_confidence = Column(Integer, nullable=True)  # 0-100
    processing_notes = Column(Text, nullable=True)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    document = relationship("Document", back_populates="processing_status") 