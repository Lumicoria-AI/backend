from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, OAuth2PasswordRequestForm
from typing import Any, Optional
import firebase_admin
from firebase_admin import auth
from backend.core.security import verify_token, rate_limit, create_access_token, get_password_hash, verify_password
from backend.db.mongodb.repositories.user_repository import UserRepository, get_user_repository
from backend.models.user import UserCreate, Token, UserInDB, TokenResponse, GoogleSignInRequest, UserResponse, UserCreateOAuth
from pydantic import BaseModel, EmailStr, Field
from backend.core.config import settings  
import structlog
from datetime import datetime, timedelta
from fastapi.security import HTTPAuthorizationCredentials
import secrets
import uuid

# Import Google Auth libraries
from google.oauth2 import id_token
from google.auth.transport import requests

logger = structlog.get_logger()
router = APIRouter()
security = HTTPBearer()

# Your Google OAuth Client ID (Web application type)
GOOGLE_CLIENT_ID = "757874659613-lafmptc8bpjkktnlrn6eanh3v4778m62.apps.googleusercontent.com"

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    user_repo: UserRepository = Depends(get_user_repository),
    form_data: Optional[OAuth2PasswordRequestForm] = Depends(None)
) -> Any:
    """
    Login endpoint that accepts both form data and JSON.
    """
    try:
        content_type = request.headers.get("content-type", "").lower()
        
        if "application/json" in content_type:
            try:
                body = await request.json()
                email = body.get("email")
                password = body.get("password")
            except:
                raise HTTPException(
                    status_code=400,
                    detail="Invalid JSON format"
                )
        elif form_data:
            email = form_data.username  # OAuth2 form uses 'username' for email
            password = form_data.password
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported content type. Use application/json or application/x-www-form-urlencoded"
            )
            
        if not email or not password:
            raise HTTPException(
                status_code=400,
                detail="Email and password are required"
            )

        # Get user from database
        user = await user_repo.get_by_email(email)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # Verify password
        if not verify_password(password, user.hashed_password):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )

        # Generate tokens
        access_token = create_access_token(user.id)
        refresh_token = secrets.token_urlsafe(32)
        
        # Store refresh token
        await user_repo.update_user(user.id, {"refresh_token": refresh_token})
        
        # Return response that matches frontend's AuthResponse type
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "created_at": str(user.created_at),
                "updated_at": str(user.updated_at) if user.updated_at else None
            }
        }
    except Exception as e:
        logger.error("Login failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed"
        )

@router.post("/signup", response_model=Token)
async def signup(
    user_in: UserCreate,
    user_repository: UserRepository = Depends(get_user_repository)
) -> Any:
    """
    Create new user.
    """
    user = await user_repository.get_user_by_email(user_in.email)
    if user:
        raise HTTPException(
            status_code=400,
            detail="The user with this email already exists in the system.",
        )
    
    # Hash the password
    hashed_password = get_password_hash(user_in.password)
    
    # Create new user with hashed password
    user_data = UserCreate(
        email=user_in.email,
        full_name=user_in.full_name,
        password=user_in.password,
        hashed_password=hashed_password  # Add the hashed password
    )
    
    user = await user_repository.create_user(user_data)
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        str(user.id), expires_delta=access_token_expires
    )
    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": user
    }

@router.post("/register", response_model=TokenResponse)
@rate_limit()
async def register(
    request: Request,
    user_data: UserCreate,
    user_repository: UserRepository = Depends(get_user_repository)
) -> Any:
    try:
        # Create user in Firebase
        firebase_user = auth.create_user(
            email=user_data.email,
            password=user_data.password,
            display_name=user_data.full_name
        )
        
        # Create user in our database
        user_data.firebase_uid = firebase_user.uid
        db_user = await user_repository.create_user(user_data)
        
        # Get Firebase custom token
        custom_token = auth.create_custom_token(firebase_user.uid)            
        return {
            "access_token": custom_token.decode(),
            "token_type": "bearer",
            "user": {
                "id": str(db_user.id),
                "email": db_user.email,
                "full_name": db_user.full_name,
                "is_active": db_user.is_active,
                "created_at": db_user.created_at,
                "updated_at": db_user.updated_at
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
    user_repository: UserRepository = Depends(get_user_repository)
) -> Any:
    try:
        # Verify the Firebase ID token
        token_data = await verify_token(credentials) # Assuming verify_token verifies Firebase ID tokens
        
        # Fetch user from database using firebase_uid
        user = await user_repository.get_user_by_firebase_uid(token_data["uid"])
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        custom_token = auth.create_custom_token(token_data["uid"])
        
        return {
            "access_token": custom_token.decode(),
            "token_type": "bearer",
            "user": user # Return the full user object
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
    user_repository: UserRepository = Depends(get_user_repository)
) -> Any:
    try:
        # Verify the Google ID token using google-auth library
        idinfo = id_token.verify_oauth2_token(
            google_data.id_token, requests.Request(), GOOGLE_CLIENT_ID)

        # Extract user information from the verified token
        google_user_id = idinfo['sub']
        email = idinfo.get('email')
        name = idinfo.get('name')

        try:
            # Try to get existing Firebase user
            firebase_user = auth.get_user_by_email(email)
        except auth.UserNotFoundError:
            # Create new Firebase user if not found
            firebase_user = auth.create_user(
                uid=google_user_id,  # Use Google ID as Firebase UID
                email=email,
                display_name=name,
                email_verified=True  # Since it's verified by Google
            )

        # Check if user exists in your database using the Firebase UID
        user = await user_repository.get_user_by_firebase_uid(firebase_user.uid)
        if not user:
            # Create new user in your database using the UserCreateOAuth model
            user_data = UserCreateOAuth(
                email=email,
                full_name=name,
                firebase_uid=firebase_user.uid
            )
            user = await user_repository.create_user_oauth(user_data)
        
        # Instead of creating a custom token, we'll reuse the original ID token
        # since it's already verified and contains the correct claims
        return {
            "access_token": google_data.id_token,  # Use the original ID token
            "token_type": "bearer",
            "user": {
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "created_at": user.created_at,
                "updated_at": user.updated_at
            }
        }
    except ValueError as e:
        # Invalid token
        logger.error("Google ID token verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google ID token"
        )
    except Exception as e:
        logger.error("Google sign-in failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Google sign-in failed"
        )

@router.get("/me", response_model=UserResponse)
@rate_limit()
async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user_repository: UserRepository = Depends(get_user_repository)
) -> Any:
    try:
        # Verify the Firebase ID token
        token_data = await verify_token(credentials) # Assuming verify_token verifies Firebase ID tokens
        
        # Fetch user from database using firebase_uid
        user = await user_repository.get_user_by_firebase_uid(token_data["uid"])
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        return user
    except Exception as e:
        logger.error("Get current user failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@router.post("/logout")
@rate_limit()
async def logout(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user_repository: UserRepository = Depends(get_user_repository)
) -> Any:
    try:
        # In Firebase, revoking the server-side token isn't usually necessary for simple logout
        # The client discards the tokens.
        # If using refresh tokens, you might invalidate them here if needed.
        return {"message": "Successfully logged out"}
    except Exception as e:
        logger.error("Logout failed", error=str(e))
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

print(secrets.token_urlsafe(32))