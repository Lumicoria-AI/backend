# Import document models from the MongoDB models
from backend.db.mongodb.models.document import (
    DocumentType,
    DocumentStatus,
    PyObjectId,
    DocumentBase,
    DocumentCreate,
    DocumentUpdate,
    Document
)

# Define ExtractionResult model if it's not defined in the MongoDB models
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field

class ExtractionResult(BaseModel):
    """Model for document extraction results."""
    text: Optional[str] = None
    pages: Optional[int] = None
    language: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    page_content: Optional[List[Dict[str, Any]]] = None
    entities: Optional[List[Dict[str, Any]]] = None
    key_phrases: Optional[List[str]] = None
    topics: Optional[List[Dict[str, Any]]] = None
    summary: Optional[str] = None
    sentiment: Optional[Dict[str, Any]] = None
    confidence: Optional[float] = None

# Re-export everything
__all__ = [
    "DocumentType",
    "DocumentStatus",
    "PyObjectId",
    "DocumentBase",
    "DocumentCreate",
    "DocumentUpdate",
    "Document",
    "ExtractionResult"
]
