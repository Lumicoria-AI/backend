from fastapi import APIRouter, Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, OAuth2PasswordRequestForm
from typing import Any, Optional
import asyncio
import firebase_admin
from firebase_admin import auth
from backend.core.security import verify_token, rate_limit, create_access_token, get_password_hash, verify_password
from backend.db.mongodb.repositories.user_repository import UserRepository, get_user_repository
from backend.models.user import UserCreate, Token, UserInDB, TokenResponse, GoogleSignInRequest, UserResponse, UserCreateOAuth
from backend.core.config import settings
from backend.services.notification_service import notification_service
from pydantic import BaseModel, EmailStr, Field
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

# Google OAuth Client ID loaded from centralized settings (no hardcoded value)
GOOGLE_CLIENT_ID = settings.GOOGLE_OAUTH_CLIENT_ID

@router.post("/login", response_model=TokenResponse)
async def login(
    request: Request,
    form_data: OAuth2PasswordRequestForm = Depends(),
    user_repository: UserRepository = Depends(get_user_repository)
) -> Any:
    """
    OAuth2 compatible token login, get an access token for future requests
    """
    user = await user_repository.get_user_by_email(form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )    
    if not user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")

    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
      # Include essential user info in the token for faster authentication
    user_info = {
        "email": user.email,
        "full_name": user.full_name,
        "firebase_uid": getattr(user, "firebase_uid", None),
        "is_active": user.is_active,
        # Include any other essential fields that don't change frequently
        "onboarding_completed": getattr(user, "onboarding_completed", False)
    }
    
    access_token = create_access_token(
        str(user.id), expires_delta=access_token_expires, user_data=user_info
    )
    
    # Generate or reuse refresh token
    refresh_token = getattr(user, "refresh_token", None)
    if not refresh_token:
        refresh_token = secrets.token_urlsafe(32)
        # Update user with refresh token
        await user_repository.update_user(str(user.id), {"refresh_token": refresh_token})
        # Refresh user data
        user = await user_repository.get_user_by_id(str(user.id))
    
    # Get created_at time
    created_at = getattr(user, "created_at", datetime.utcnow())
    
    # Fire-and-forget: send login security alert
    ip_address = request.client.host if hasattr(request, 'client') and request.client else None
    device = request.headers.get("User-Agent", "Unknown device")
    asyncio.ensure_future(notification_service.send_login_alert(
        user_id=str(user.id),
        email=user.email,
        name=user.full_name or "User",
        ip_address=ip_address,
        device=device,
    ))

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "avatar_url": getattr(user, "avatar_url", None),
            "onboarding_completed": getattr(user, "onboarding_completed", False),
            "created_at": created_at,
            "updated_at": getattr(user, "updated_at", None)
        }
    }

@router.post("/signup", response_model=TokenResponse)
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
    
    user = await user_repository.create_user(user_in)
    access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        str(user.id), expires_delta=access_token_expires
    )
    
    # Create a refresh token
    refresh_token = secrets.token_urlsafe(32)
    
    # Update the user's refresh token in the database
    await user_repository.update_user(str(user.id), {"refresh_token": refresh_token})
    
    # Get created_at time
    created_at = getattr(user, "created_at", datetime.utcnow())
    
    # Fire-and-forget: send welcome email + in-app notification
    asyncio.ensure_future(notification_service.send_welcome_notification(
        user_id=str(user.id),
        email=user.email,
        name=user.full_name or "User",
    ))

    return {
        "access_token": access_token,
        "refresh_token": refresh_token,
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "avatar_url": getattr(user, "avatar_url", None),
            "onboarding_completed": getattr(user, "onboarding_completed", False),
            "created_at": created_at,
            "updated_at": getattr(user, "updated_at", None)
        }
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
        
        # Create a refresh token (this is required by the TokenResponse model)
        refresh_token = secrets.token_urlsafe(32)
        
        # Update the user's refresh token in the database
        await user_repository.update_user(str(db_user.id), {"refresh_token": refresh_token})
        
        # Get created_at time
        created_at = getattr(db_user, "created_at", datetime.utcnow())
        
        # Fire-and-forget: send welcome email + in-app notification
        asyncio.ensure_future(notification_service.send_welcome_notification(
            user_id=str(db_user.id),
            email=db_user.email,
            name=db_user.full_name or "User",
        ))

        return {
            "access_token": custom_token.decode(),
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "id": str(db_user.id),
                "email": db_user.email,
                "full_name": db_user.full_name,
                "is_active": db_user.is_active,
                "avatar_url": getattr(db_user, "avatar_url", None),
                "onboarding_completed": getattr(db_user, "onboarding_completed", False),
                "created_at": created_at,
                "updated_at": getattr(db_user, "updated_at", None)
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
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user_repository: UserRepository = Depends(get_user_repository)
) -> Any:
    try:
        # Verify the token (could be Firebase ID token or our JWT)
        token_data = await verify_token(credentials)
        logger.info("Token verified for refresh", token_keys=list(token_data.keys()))
        
        # Check if we have a user_id directly (JWT) or need to use the uid (Firebase)
        user = None
        if "user_id" in token_data:
            logger.info("Using user_id to refresh token", user_id=token_data["user_id"])
            user = await user_repository.get_user_by_id(token_data["user_id"])
        elif "uid" in token_data:
            logger.info("Using uid to refresh token", uid=token_data["uid"])
            user = await user_repository.get_user_by_firebase_uid(token_data["uid"])
        
        if not user:
            logger.error("User not found during token refresh", token_data=token_data)
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
          # Create our own JWT token instead of Firebase custom token
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        # Include essential user info in the token
        user_info = {
            "email": user.email,
            "full_name": user.full_name,
            "firebase_uid": user.firebase_uid,
            "is_active": user.is_active
        }
        
        access_token = create_access_token(
            str(user.id), expires_delta=access_token_expires, user_data=user_info
        )
        
        # Generate a new refresh token or reuse existing one
        refresh_token = getattr(user, "refresh_token", None)
        if not refresh_token:
            refresh_token = secrets.token_urlsafe(32)
            # Update user with new refresh token
            await user_repository.update_user(str(user.id), {"refresh_token": refresh_token})
            # Refresh user data
            user = await user_repository.get_user_by_id(str(user.id))
        
        # Get created_at time
        created_at = getattr(user, "created_at", datetime.utcnow())
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "avatar_url": getattr(user, "avatar_url", None),
                "onboarding_completed": getattr(user, "onboarding_completed", False),
                "created_at": created_at,
                "updated_at": getattr(user, "updated_at", None)
            }
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
) -> TokenResponse:
    try:
        # Verify the Firebase ID token (sent by frontend's signInWithPopup)
        # We use Firebase Admin Verify instead of google-auth because the frontend
        # generates a Firebase ID Token (iss: https://securetoken.google.com/...)
        decoded_token = auth.verify_id_token(google_data.id_token, check_revoked=True)
        
        # Extract user information from the verified Firebase token
        firebase_uid = decoded_token['uid']
        email = decoded_token.get('email')
        name = decoded_token.get('name')
        picture = decoded_token.get('picture')
        
        logger.info("Google/Firebase sign-in verified", 
                   uid=firebase_uid, 
                   email=email, 
                   has_picture=bool(picture))

        # Check if user exists in your database using the Firebase UID
        user = await user_repository.get_user_by_firebase_uid(firebase_uid)
        
        # If not found by UID, try email (account linking)
        if not user and email:
             user = await user_repository.get_user_by_email(email)
             if user:
                 # Link existing email user to this Firebase UID
                 logger.info("Linking existing email user to Firebase/Google", user_id=str(user.id))
                 await user_repository.update_user(str(user.id), {"firebase_uid": firebase_uid})
                 if picture and not getattr(user, "avatar_url", None):
                      await user_repository.update_user(str(user.id), {"avatar_url": picture})
                 # Refresh user object
                 user = await user_repository.get_user_by_id(str(user.id))

        is_new_user = user is None
        
        if not user:
            if not email:
                 raise HTTPException(status_code=400, detail="Email not provided in token")
                 
            # Create new user in your database using the UserCreateOAuth model
            user_data = UserCreateOAuth(
                email=email,
                full_name=name or email.split('@')[0],
                firebase_uid=firebase_uid,
                avatar_url=picture,
                onboarding_completed=False
            )
            # Use the new create_user_oauth method
            user = await user_repository.create_user_oauth(user_data)
        elif picture and not getattr(user, "avatar_url", None):
            # Update existing user with Google profile picture if they don't have an avatar
            await user_repository.update_user(str(user.id), {"avatar_url": picture})
            # Refresh user data after update
            user = await user_repository.get_user_by_id(str(user.id))

        # Create a JWT token for our backend authentication using our own secret key
        # This is more reliable than using Firebase custom tokens for our backend
        access_token_expires = timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        # Include essential user info in the token
        user_info = {
            "email": user.email,
            "full_name": user.full_name,
            "firebase_uid": user.firebase_uid,
            "is_active": user.is_active
        }
        
        access_token = create_access_token(
            str(user.id), expires_delta=access_token_expires, user_data=user_info
        )
        
        # Create a refresh token (this is required by the TokenResponse model)
        refresh_token = secrets.token_urlsafe(32)
        
        # Update the user's refresh token in the database
        await user_repository.update_user(str(user.id), {"refresh_token": refresh_token})
        
        # Get the necessary fields for UserResponse
        created_at = getattr(user, "created_at", datetime.utcnow())
        
        # Fire-and-forget: welcome for new users, security alert for returning users
        if is_new_user:
            asyncio.ensure_future(notification_service.send_welcome_notification(
                user_id=str(user.id),
                email=user.email,
                name=user.full_name or "User",
            ))
        else:
            ip_address = request.client.host if hasattr(request, 'client') and request.client else None
            device = request.headers.get("User-Agent", "Unknown device")
            asyncio.ensure_future(notification_service.send_login_alert(
                user_id=str(user.id),
                email=user.email,
                name=user.full_name or "User",
                ip_address=ip_address,
                device=device,
            ))

        return {
            "access_token": access_token, # Using our own JWT token
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user": {
                "id": str(user.id),
                "email": user.email,
                "full_name": user.full_name,
                "is_active": user.is_active,
                "avatar_url": getattr(user, "avatar_url", None),
                "onboarding_completed": getattr(user, "onboarding_completed", False),
                "created_at": created_at,
                "updated_at": getattr(user, "updated_at", None)
            }
        }
    except ValueError as e:
        # Invalid token
        logger.error("Firebase ID token verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Google/Firebase ID token"
        )
    except Exception as e:
        logger.error("Google sign-in failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Google sign-in failed: {str(e)}"
        )

async def handle_onboarding_status_update(user, request, user_repository):
    """Helper function to update onboarding status in DB if needed.
    Returns information about what was updated.
    """
    result = {
        "updated": False,
        "previous_status": getattr(user, "onboarding_completed", False),
    }
    
    # Check if we need to force update the onboarding status
    if (request.headers.get("X-Onboarding-Completed") == "true" and 
        not getattr(user, "onboarding_completed", False)):
        try:
            logger.info(f"Updating onboarding status in DB for user {str(user.id)}")
            # Update the user record in the database
            await user_repository.update_user(
                str(user.id),
                {"onboarding_completed": True, "onboarding_completed_at": datetime.utcnow()}
            )
            result["updated"] = True
            logger.info(f"Successfully updated onboarding status to True for user {str(user.id)}")
        except Exception as e:
            logger.error(f"Failed to update onboarding status: {str(e)}")
            result["error"] = str(e)
    
    return result

@router.get("/me", response_model=UserResponse)
@rate_limit()
async def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user_repository: UserRepository = Depends(get_user_repository)
) -> Any:
    try:
        # Verify the token (handles both JWT and Firebase tokens)
        token_data = await verify_token(credentials)
        
        # Check if we have a user_id directly (JWT) or need to use the uid (Firebase)
        if "user_id" in token_data:
            user = await user_repository.get_user_by_id(token_data["user_id"])
        else:
            # Fallback to Firebase uid
            user = await user_repository.get_user_by_firebase_uid(token_data["uid"])
            
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
          # Log the user data for debugging
        logger.info("Retrieved current user", 
                  user_id=str(user.id), 
                  onboarding_completed=getattr(user, "onboarding_completed", False))
                  
        # Ensure the response has all required fields
        response_data = {
            "id": str(user.id),
            "email": user.email,
            "full_name": user.full_name,
            "is_active": user.is_active,
            "avatar_url": getattr(user, "avatar_url", None),
            "created_at": getattr(user, "created_at", datetime.utcnow()),
            "updated_at": getattr(user, "updated_at", None),        # Always include onboarding status - special handling
        # Some database records might not have this field, so explicitly check for its existence
        # If the user is coming from onboarding completion, we force this to be True
        "onboarding_completed": True if request.headers.get("X-Onboarding-Completed") == "true" 
                              else getattr(user, "onboarding_completed", False),
                              
        # If X-Onboarding-Completed header is set, also update the user record in DB to ensure consistency
        # This handles cases where the frontend thinks onboarding is completed but backend doesn't
        "onboarding_info": await handle_onboarding_status_update(user, request, user_repository),
        "job_title": getattr(user, "job_title", None),
        "company": getattr(user, "company", None)
        }
        
        logger.info("Returning user response", 
                   user_id=str(user.id), 
                   onboarding_completed=response_data["onboarding_completed"])
        
        return response_data
    except Exception as e:
        logger.error("Get current user failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

@router.post("/logout")
@rate_limit()
async def logout(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security),
    user_repository: UserRepository = Depends(get_user_repository)
) -> Any:
    try:
        # Verify the token to get the user
        token_data = await verify_token(credentials)
        
        # For JWT tokens with user_id, we can invalidate the refresh token
        if "user_id" in token_data:
            user_id = token_data["user_id"]
            # Clear the refresh token
            await user_repository.update_user(user_id, {"refresh_token": None})
            
        # For both JWT and Firebase, we let the client discard the tokens
        return {"message": "Successfully logged out"}
    except Exception as e:
        logger.error("Logout failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token"
        )

# print(secrets.token_urlsafe(32))  # Removed — do not leak secrets in stdout