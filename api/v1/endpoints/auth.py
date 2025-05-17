from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from typing import Any
import firebase_admin
from firebase_admin import auth
from core.security import verify_token, rate_limit
from db.base import get_db
from models.user import User
from pydantic import BaseModel, EmailStr
import structlog

logger = structlog.get_logger()
router = APIRouter()
security = HTTPBearer()

class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: dict

class UserCreate(BaseModel):
    email: EmailStr
    password: str
    full_name: str

@router.post("/register", response_model=TokenResponse)
@rate_limit()
async def register(
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

@router.post("/login", response_model=TokenResponse)
@rate_limit()
async def login(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db)
) -> Any:
    try:
        # Verify Firebase token
        token_data = await verify_token(credentials)
        
        # Get or create user in our database
        stmt = select(User).where(User.firebase_uid == token_data["uid"])
        result = await db.execute(stmt)
        user = result.scalar_one_or_none()
        
        if not user:
            # Create user if not exists
            user = User(
                firebase_uid=token_data["uid"],
                email=token_data["email"],
                full_name=token_data.get("name", "")
            )
            db.add(user)
            await db.commit()
            await db.refresh(user)
        
        # Update last login
        user.last_login = func.now()
        await db.commit()
        
        # Create new custom token
        custom_token = auth.create_custom_token(token_data["uid"])
        
        return {
            "access_token": custom_token.decode(),
            "token_type": "bearer",
            "user": {
                "id": user.id,
                "email": user.email,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "subscription_status": user.subscription_status
            }
        }
    except Exception as e:
        logger.error("Login failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials"
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