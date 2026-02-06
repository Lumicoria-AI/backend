from datetime import datetime
from typing import Optional, List
from pydantic import BaseModel, EmailStr, Field, ConfigDict
from uuid import UUID, uuid4

class UserBase(BaseModel):
    """Base user model with common attributes."""
    email: EmailStr
    full_name: str
    is_active: bool = True
    is_superuser: bool = False
    avatar_url: Optional[str] = None
    job_title: Optional[str] = None
    company: Optional[str] = None
    timezone: str = "UTC"
    preferred_language: str = "en"

    model_config = ConfigDict(
        populate_by_name=True,
        alias_generator=lambda x: "fullName" if x == "full_name" else x
    )

class UserCreate(UserBase):
    """Model for creating a new user with password."""
    password: str
    firebase_uid: Optional[str] = None
    hashed_password: Optional[str] = None  # Allow setting hashed_password

class UserCreateOAuth(UserBase):
    """Model for creating a new user from OAuth (e.g., Google)."""
    firebase_uid: str # firebase_uid is required for OAuth users
    onboarding_completed: bool = False  # Default to False for new OAuth users
    job_title: Optional[str] = None
    company: Optional[str] = None

class UserUpdate(BaseModel):
    """Model for updating an existing user."""
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    password: Optional[str] = None
    avatar_url: Optional[str] = None
    job_title: Optional[str] = None
    company: Optional[str] = None
    timezone: Optional[str] = None
    preferred_language: Optional[str] = None
    is_active: Optional[bool] = None

class UserInDB(UserBase):
    """Model for user data stored in the database."""
    id: UUID = Field(default_factory=uuid4)
    hashed_password: str
    firebase_uid: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    login_count: int = 0
    failed_login_attempts: int = 0
    last_failed_login: Optional[datetime] = None
    reset_password_token: Optional[str] = None
    reset_password_token_expires: Optional[datetime] = None
    email_verified: bool = False
    email_verification_token: Optional[str] = None
    email_verification_token_expires: Optional[datetime] = None
    two_factor_enabled: bool = False
    two_factor_secret: Optional[str] = None
    backup_codes: Optional[List[str]] = None
    preferences: dict = Field(default_factory=dict)
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)
    
    # Onboarding fields
    job_title: Optional[str] = None
    company: Optional[str] = None
    onboarding_completed: bool = False
    onboarding_completed_at: Optional[datetime] = None
    refresh_token: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)

class User(UserBase):
    """Model for user data returned to the client."""
    id: UUID
    created_at: datetime
    updated_at: Optional[datetime] = None
    last_login: Optional[datetime] = None
    email_verified: bool = False
    two_factor_enabled: bool = False
    roles: List[str] = Field(default_factory=list)
    permissions: List[str] = Field(default_factory=list)

    model_config = ConfigDict(from_attributes=True)

class UserResponse(BaseModel):
    """Model for user response data."""
    id: str
    email: str
    full_name: str
    is_active: bool
    avatar_url: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    onboarding_completed: bool = False
    job_title: Optional[str] = None
    company: Optional[str] = None
    # Additional fields that may be included but not required in responses
    onboarding_info: Optional[dict] = None

class UserProfile(BaseModel):
    """Model for user profile data."""
    user_id: str
    job_title: Optional[str] = None
    company: Optional[str] = None
    bio: Optional[str] = None
    website: Optional[str] = None
    social_links: dict = Field(default_factory=dict)
    skills: List[str] = Field(default_factory=list)
    interests: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)

class UserSettings(BaseModel):
    """Model for user settings data."""
    user_id: str
    theme: str = "light"
    email_notifications: bool = True
    push_notifications: bool = True
    newsletter_subscribed: bool = False
    task_reminders: bool = True
    break_reminders: bool = True
    work_hours_start: str = "09:00"
    work_hours_end: str = "17:00"
    break_interval_minutes: int = 60
    break_duration_minutes: int = 5
    preferred_ai_model: str = "default"
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None
    
    model_config = ConfigDict(from_attributes=True)

class Token(BaseModel):
    """Model for access token response."""
    access_token: str
    token_type: str
    user: User

class TokenResponse(BaseModel):
    """Model for token response with user data."""
    access_token: str
    refresh_token: str
    token_type: str
    user: UserResponse

class GoogleSignInRequest(BaseModel):
    """Model for Google sign-in request."""
    id_token: str