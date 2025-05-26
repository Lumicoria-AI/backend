from sqlalchemy import Boolean, Column, String, DateTime, ForeignKey, Integer
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from pydantic import BaseModel, EmailStr, constr
from typing import Optional
from datetime import datetime

from db.base_class import Base

class User(Base):
    __tablename__ = "users"

    id = Column(String, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    hashed_password = Column(String, nullable=False)
    avatar_url = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    profile = relationship("UserProfile", back_populates="user", uselist=False)
    settings = relationship("UserSettings", back_populates="user", uselist=False)

class UserProfile(Base):
    __tablename__ = "user_profiles"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    job_title = Column(String, nullable=True)
    company = Column(String, nullable=True)
    timezone = Column(String, default="UTC")
    preferred_language = Column(String, default="en")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="profile")

class UserSettings(Base):
    __tablename__ = "user_settings"

    id = Column(String, primary_key=True, index=True)
    user_id = Column(String, ForeignKey("users.id", ondelete="CASCADE"), unique=True)
    email_notifications = Column(Boolean, default=True)
    push_notifications = Column(Boolean, default=True)
    task_reminders = Column(Boolean, default=True)
    break_reminders = Column(Boolean, default=True)
    work_hours_start = Column(String, default="09:00")
    work_hours_end = Column(String, default="17:00")
    break_interval_minutes = Column(Integer, default=60)
    break_duration_minutes = Column(Integer, default=5)
    preferred_ai_model = Column(String, default="gemini")
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

    # Relationships
    user = relationship("User", back_populates="settings")

# Pydantic models for request/response
class UserBase(BaseModel):
    email: EmailStr
    full_name: constr(min_length=1, max_length=100)

class UserCreate(UserBase):
    password: constr(min_length=8)

class UserUpdate(BaseModel):
    full_name: Optional[constr(min_length=1, max_length=100)] = None
    job_title: Optional[str] = None
    company: Optional[str] = None
    timezone: Optional[str] = None
    preferred_language: Optional[str] = None

class UserProfileUpdate(BaseModel):
    job_title: Optional[str] = None
    company: Optional[str] = None
    timezone: Optional[str] = None
    preferred_language: Optional[str] = None

class UserSettingsUpdate(BaseModel):
    email_notifications: Optional[bool] = None
    push_notifications: Optional[bool] = None
    task_reminders: Optional[bool] = None
    break_reminders: Optional[bool] = None
    work_hours_start: Optional[str] = None
    work_hours_end: Optional[str] = None
    break_interval_minutes: Optional[int] = None
    break_duration_minutes: Optional[int] = None
    preferred_ai_model: Optional[str] = None

class UserInDB(UserBase):
    id: str
    avatar_url: Optional[str] = None
    is_active: bool
    is_superuser: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class UserProfileInDB(BaseModel):
    id: str
    user_id: str
    job_title: Optional[str] = None
    company: Optional[str] = None
    timezone: str
    preferred_language: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class UserSettingsInDB(BaseModel):
    id: str
    user_id: str
    email_notifications: bool
    push_notifications: bool
    task_reminders: bool
    break_reminders: bool
    work_hours_start: str
    work_hours_end: str
    break_interval_minutes: int
    break_duration_minutes: int
    preferred_ai_model: str
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class TokenData(BaseModel):
    user_id: Optional[str] = None 