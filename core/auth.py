"""
Authentication utilities for Lumicoria.ai
"""

from typing import Dict, Any, Optional
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
import time

from .config import settings

oauth2_scheme = OAuth2PasswordBearer(tokenUrl=f"{settings.API_V1_STR}/auth/login")

async def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, Any]:
    """
    Validate access token and return current user.
    
    Args:
        token: JWT token from request
        
    Returns:
        User data from token
        
    Raises:
        HTTPException: If token is invalid
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Decode JWT token
        payload = jwt.decode(
            token, 
            settings.SECRET_KEY, 
            algorithms=["HS256"]
        )
        
        # Extract user ID from token
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
            
        # Extract token expiration
        exp = payload.get("exp")
        if exp is None or int(time.time()) > exp:
            raise credentials_exception
            
        # Extract user data from token
        user_data = payload.get("user")
        if not user_data:
            raise credentials_exception
            
        # Add user ID to user data
        user_data["id"] = user_id
        
        # Return user data
        return user_data
        
    except JWTError:
        raise credentials_exception
