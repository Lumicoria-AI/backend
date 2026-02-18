"""
Device Token model for storing FCM push notification tokens.
"""

from datetime import datetime
from typing import Optional
from enum import Enum
from pydantic import BaseModel, Field
from bson import ObjectId


class DevicePlatform(str, Enum):
    """Platform types for push notification delivery."""
    IOS = "ios"
    ANDROID = "android"
    WEB = "web"
    UNKNOWN = "unknown"


class PyObjectId(ObjectId):
    """Custom ObjectId type for Pydantic models."""
    
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v, values=None):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __get_pydantic_json_schema__(cls, core_schema, handler):
        return {"type": "string"}


class DeviceTokenBase(BaseModel):
    """Base model for device tokens."""
    user_id: str
    token: str
    platform: DevicePlatform = DevicePlatform.UNKNOWN
    device_name: Optional[str] = None
    app_version: Optional[str] = None


class DeviceTokenCreate(DeviceTokenBase):
    """Model for creating a new device token."""
    pass


class DeviceTokenUpdate(BaseModel):
    """Model for updating a device token."""
    token: Optional[str] = None
    platform: Optional[DevicePlatform] = None
    device_name: Optional[str] = None
    app_version: Optional[str] = None
    is_active: Optional[bool] = None


class DeviceToken(DeviceTokenBase):
    """Device token model with database fields."""
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    last_used: Optional[datetime] = None
    is_active: bool = True
    
    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {
            ObjectId: str,
            datetime: lambda dt: dt.isoformat()
        }
    }


class DeviceTokenResponse(BaseModel):
    """Response model for device token API endpoints."""
    id: str
    user_id: str
    platform: DevicePlatform
    device_name: Optional[str] = None
    created_at: datetime
    is_active: bool
