from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import Any, Optional
import firebase_admin
from firebase_admin import auth
from core.security import verify_token, rate_limit, create_access_token, get_password_hash, verify_password
from db.base import get_db
from models.user import User, UserCreate, Token, UserInDB
from pydantic import BaseModel, EmailStr, Field
import structlog
from datetime import datetime, timedelta
from fastapi.security import HTTPAuthorizationCredentials
import secrets
import uuid

logger = structlog.get_logger()
router = APIRouter()
security = HTTPBearer()

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict
    refresh_token: Optional[str] = None

class GoogleSignInRequest(BaseModel):
    id_token: str = Field(..., description="Google ID token from the client")

class UserResponse(BaseModel):
    id: int
    email: str
    full_name: str
    is_active: bool
    subscription_status: Optional[str] = None
    profile_picture: Optional[str] = None
    google_id: Optional[str] = None

@router.post("/login", response_model=Token)
async def login(
    db: AsyncSession = Depends(get_db),
    form_data: OAuth2PasswordRequestForm = Depends()
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user = await db.execute(select(User).where(User.email == form_data.username))
    user = user.scalar_one()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        user.id, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserInDB.from_orm(user)
    }

@router.post("/signup", response_model=Token)
async def signup(
    *,
    db: AsyncSession = Depends(get_db),
    user_in: UserCreate,
) -> Any:
    """
    Create new user.
    """
    user = await db.execute(select(User).where(User.email == user_in.email))
    user = user.scalar_one()
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )
    
    user_id = str(uuid.uuid4())
    user = User(
        id=user_id,
        email=user_in.email,
        full_name=user_in.full_name,
        hashed_password=get_password_hash(user_in.password),
        is_active=True,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        user.id, expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": UserInDB.from_orm(user)
    }

@router.post("/test-token", response_model=UserInDB)
async def test_token(current_user: User = Depends(get_current_user)) -> Any:
    """
    Test access token
    """
    return current_user

@router.post("/register", response_model=TokenResponse)
@rate_limit()
async def register(
    request: Request,
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
) -> Any:
    try:
        # Create user in Firebase
        firebase_user = auth.create_user(
            email=user_data.email,
            password=user_data.password,
            display_name=user_data.full_name
        )
        
        # Create user in our database
        db_user = User(
            firebase_uid=firebase_user.uid,
            email=user_data.email,
            full_name=user_data.full_name
        )
        db.add(db_user)
        await db.commit()
        await db.refresh(db_user)
        
        # Get Firebase custom token
        custom_token = auth.create_custom_token(firebase_user.uid)
        
        return {
            "access_token": custom_token.decode(),
            "token_type": "bearer",
            "user": {
                "id": db_user.id,
                "email": db_user.email,
                "full_name": db_user.full_name,
                "is_active": db_user.is_active
            }
        }
    except auth.EmailAlreadyExistsError:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email already registered"
        )
    except Exception as e:
        logger.error("Registration failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )

@router.post("/refresh", response_model=TokenResponse)
@rate_limit()
async def refresh_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Any:
    try:
        token_data = await verify_token(credentials)
        custom_token = auth.create_custom_token(token_data["uid"])
        
        return {
            "access_token": custom_token.decode(),
            "token_type": "bearer",
            "user": token_data
        }
    except Exception as e:
        logger.error("Token refresh failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@router.post("/google", response_model=TokenResponse)
@rate_limit()
async def google_sign_in(
    request: Request,
    google_data: GoogleSignInRequest,
    db: AsyncSession = Depends(get_db)
) -> Any:
    try:
        # Verify the Google ID token
        decoded_token = auth.verify_id_token(google_data.id_token)
        
        # Extract user information
        google_id = decoded_token.get("sub")
        email = decoded_token.get("email")
        name = decoded_token.get("name", "")
        picture = decoded_token.get("picture")
        
        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email not provided by Google"
            )
        
        # Check if user exists in our database
        stmt = select(User).where(User.email == email)
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            # Create new user
            user = User(
                email=email,
                full_name=name,
                profile_picture=picture,
                google_id=google_id,
                is_active=True,
                firebase_uid=google_id
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        else:
            # Update existing user's Google info if needed
            if not user.google_id:
                user.google_id = google_id
            if not user.firebase_uid:
                user.firebase_uid = google_id
            if not user.profile_picture and picture:
                user.profile_picture = picture
            if not user.full_name and name:
                user.full_name = name
            await db.commit()
        
        # Create Firebase custom token
        custom_token = auth.create_custom_token(user.firebase_uid)
        
        # Create refresh token
        refresh_token = create_access_token(
            subject=user.id,
            expires_delta=timedelta(days=30)
        )
        
        return {
            "access_token": custom_token.decode(),
            "token_type": "bearer",
            "refresh_token": refresh_token,
            "user": {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "subscription_status": user.subscription_status,
                "profile_picture": user.profile_picture
            }
        }
    except auth.InvalidIdTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google ID token"
        )
    except Exception as e:
        logger.error("Google sign in failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google sign in failed"
        )

@router.get("/me", response_model=UserResponse)
@rate_limit()
async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Any:
    try:
        token_data = await verify_token(credentials)
        
        stmt = select(User).where(User.id == token_data["uid"])
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return user
    except Exception as e:
        logger.error("Failed to get current user", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

@router.post("/logout")
@rate_limit()
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> Any:
    try:
        # In Firebase, we can't actually invalidate tokens
        # But we can revoke refresh tokens if needed
        token_data = await verify_token(credentials)
        
        # Here you might want to add the token to a blacklist in Redis
        # for additional security if needed
        
        return {"message": "Successfully logged out"}
    except Exception as e:
        logger.error("Logout failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )

print(secrets.token_urlsafe(32)) 