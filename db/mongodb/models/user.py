from datetime import datetime
from typing import Optional, Any, Annotated
from pydantic import BaseModel, EmailStr, Field, GetJsonSchemaHandler
from pydantic.json_schema import JsonSchemaValue
from bson import ObjectId

class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate
        
    @classmethod
    def validate(cls, v, info):
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

class UserBase(BaseModel):
    email: EmailStr
    full_name: str = Field(..., min_length=1, max_length=100)
    firebase_uid: Optional[str] = None
    avatar_url: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False

    model_config = {
        "json_encoders": {ObjectId: str},
        "populate_by_name": True
    }

class UserCreate(UserBase):
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    full_name: Optional[str] = Field(None, min_length=1, max_length=100)
    avatar_url: Optional[str] = None
    is_active: Optional[bool] = None

class UserProfile(BaseModel):
    user_id: PyObjectId = Field(alias="_id")
    job_title: Optional[str] = None
    company: Optional[str] = None
    timezone: str = "UTC"
    preferred_language: str = "en"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    model_config = {
        "json_encoders": {ObjectId: str},
        "populate_by_name": True
    }

class UserSettings(BaseModel):
    user_id: PyObjectId = Field(alias="_id")
    email_notifications: bool = True
    push_notifications: bool = True
    task_reminders: bool = True
    break_reminders: bool = True
    work_hours_start: str = "09:00"
    work_hours_end: str = "17:00"
    break_interval_minutes: int = 60
    break_duration_minutes: int = 5
    preferred_ai_model: str = "gemini"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    model_config = {
        "json_encoders": {ObjectId: str},
        "populate_by_name": True
    }

class User(UserBase):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    profile: Optional[UserProfile] = None
    settings: Optional[UserSettings] = None
    # Onboarding fields
    onboarding_completed: bool = False
    onboarding_completed_at: Optional[datetime] = None
    job_title: Optional[str] = None
    company: Optional[str] = None
    timezone: str = "UTC"
    preferred_language: str = "en"

    model_config = {
        "json_encoders": {ObjectId: str},
        "populate_by_name": True
    }

class UserInDB(User):
    hashed_password: str

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: Optional[User] = None

class TokenData(BaseModel):
    user_id: Optional[str] = None 