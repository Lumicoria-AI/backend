from datetime import datetime
from typing import Optional, Dict, Any, List
from enum import Enum
from pydantic import BaseModel, Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from bson import ObjectId

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

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(
        cls,
        schema_generator: Any,
        property_schema: Any,
    ) -> Any:
        return { "type": "string" }

class DocumentBase(BaseModel):
    name: str
    document_type: DocumentType
    organization_id: PyObjectId
    created_by: PyObjectId
    file_url: str
    file_size: int
    mime_type: str
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
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
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