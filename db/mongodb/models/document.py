from datetime import datetime
from typing import Optional, Dict, Any, List, Annotated
from enum import Enum
from pydantic import BaseModel, Field, PlainValidator, PlainSerializer
from bson import ObjectId


def _validate_object_id(v: Any) -> ObjectId:
    if isinstance(v, ObjectId):
        return v
    if isinstance(v, str) and ObjectId.is_valid(v):
        return ObjectId(v)
    raise ValueError(f"Invalid ObjectId: {v}")


PyObjectId = Annotated[
    ObjectId,
    PlainValidator(_validate_object_id),
    PlainSerializer(lambda v: str(v), return_type=str),
]


class DocumentType(str, Enum):
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    IMAGE = "image"
    SPREADSHEET = "spreadsheet"
    PRESENTATION = "presentation"
    OTHER = "other"

class DocumentStatus(str, Enum):
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"
    ARCHIVED = "archived"

class DocumentBase(BaseModel):
    name: str
    document_type: DocumentType
    organization_id: PyObjectId
    created_by: PyObjectId
    file_url: str = ""
    file_size: int = 0
    mime_type: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentCreate(DocumentBase):
    pass

class DocumentUpdate(BaseModel):
    name: Optional[str] = None
    document_type: Optional[DocumentType] = None
    status: Optional[DocumentStatus] = None
    metadata: Optional[Dict[str, Any]] = None
    extraction_result: Optional[Dict[str, Any]] = None
    extraction_status: Optional[str] = None
    extraction_error: Optional[str] = None

class Document(DocumentBase):
    id: PyObjectId = Field(default_factory=ObjectId, alias="_id")
    status: DocumentStatus = DocumentStatus.UPLOADED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    extraction_result: Optional[Dict[str, Any]] = None
    extraction_status: Optional[str] = None
    extraction_error: Optional[str] = None

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {
            ObjectId: str,
            datetime: lambda dt: dt.isoformat()
        }
    } 