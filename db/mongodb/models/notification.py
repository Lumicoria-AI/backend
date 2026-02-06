from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, EmailStr, Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from bson import ObjectId

class NotificationType(str, Enum):
    EMAIL = "email"
    IN_APP = "in_app"
    TASK = "task"
    DOCUMENT = "document"
    WELLBEING = "wellbeing"
    SYSTEM = "system"

class NotificationPriority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"

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