from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, EmailStr, Field, GetCoreSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from pydantic_core import CoreSchema, core_schema
from bson import ObjectId

class NotificationType(str, Enum):
    EMAIL = "email"
    IN_APP = "in_app"
    TASK = "task"
    DOCUMENT = "document"
    WELLBEING = "wellbeing"
    SYSTEM = "system"
    AUTH = "auth"
    BILLING = "billing"

class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

class PyObjectId(ObjectId):
    @classmethod
    def __get_pydantic_core_schema__(
        cls, source_type: Any, handler: GetCoreSchemaHandler
    ) -> CoreSchema:
        return core_schema.no_info_plain_validator_function(
            cls._validate,
            serialization=core_schema.to_string_ser_schema(),
        )

    @classmethod
    def _validate(cls, v: Any) -> ObjectId:
        if isinstance(v, ObjectId):
            return v
        if isinstance(v, str) and ObjectId.is_valid(v):
            return ObjectId(v)
        raise ValueError(f"Invalid ObjectId: {v}")

class NotificationBase(BaseModel):
    user_id: Optional[str] = None
    user_email: Optional[EmailStr] = None
    notification_type: NotificationType
    title: str
    content: str
    priority: NotificationPriority = NotificationPriority.NORMAL
    metadata: Dict[str, Any] = Field(default_factory=dict)

class NotificationCreate(NotificationBase):
    pass

class NotificationUpdate(BaseModel):
    title: Optional[str] = None
    content: Optional[str] = None
    priority: Optional[NotificationPriority] = None
    metadata: Optional[Dict[str, Any]] = None
    read: Optional[bool] = None
    read_at: Optional[datetime] = None

class Notification(NotificationBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    read: bool = False
    read_at: Optional[datetime] = None

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {
            ObjectId: str,
            datetime: lambda dt: dt.isoformat()
        }
    } 