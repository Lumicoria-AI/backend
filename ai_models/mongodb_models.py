from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, constr
from bson import ObjectId

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
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

class MongoBaseModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    class Config:
        json_encoders = {ObjectId: str}
        populate_by_name = True
        arbitrary_types_allowed = True

class UserSettings(BaseModel):
    email_notifications: bool = True
    push_notifications: bool = True
    task_reminders: bool = True
    break_reminders: bool = True
    work_hours_start: str = "09:00"
    work_hours_end: str = "17:00"
    break_interval_minutes: int = 60
    break_duration_minutes: int = 5
    preferred_ai_model: str = "gemini"

class UserProfile(BaseModel):
    job_title: Optional[str] = None
    company: Optional[str] = None
    timezone: str = "UTC"
    preferred_language: str = "en"

class User(MongoBaseModel):
    email: EmailStr
    full_name: constr(min_length=1, max_length=100)
    hashed_password: str
    avatar_url: Optional[str] = None
    is_active: bool = True
    is_superuser: bool = False
    profile: UserProfile = Field(default_factory=UserProfile)
    settings: UserSettings = Field(default_factory=UserSettings)

    class Config:
        schema_extra = {
            "example": {
                "email": "user@example.com",
                "full_name": "John Doe",
                "profile": {
                    "job_title": "Developer",
                    "company": "Tech Corp",
                    "timezone": "UTC",
                    "preferred_language": "en"
                },
                "settings": {
                    "email_notifications": True,
                    "push_notifications": True,
                    "task_reminders": True,
                    "break_reminders": True,
                    "work_hours_start": "09:00",
                    "work_hours_end": "17:00",
                    "break_interval_minutes": 60,
                    "break_duration_minutes": 5,
                    "preferred_ai_model": "gemini"
                }
            }
        }

# Request/Response Models
class UserCreate(BaseModel):
    email: EmailStr
    full_name: constr(min_length=1, max_length=100)
    password: constr(min_length=8)

class UserUpdate(BaseModel):
    full_name: Optional[constr(min_length=1, max_length=100)] = None
    profile: Optional[UserProfile] = None
    settings: Optional[UserSettings] = None

class UserInDB(User):
    pass

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    user_id: Optional[str] = None 