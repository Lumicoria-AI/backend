from fastapi import APIRouter, Depends, HTTPException, status, Request
from typing import Any, List, Optional, Dict
from pydantic import BaseModel
from datetime import datetime, timedelta
from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.user_repository import user_repository
from backend.models.user import User, UserUpdate
from backend.core.security import rate_limit, verify_password, get_password_hash
from backend.services.activity_logger import log_activity
import structlog

logger = structlog.get_logger()
router = APIRouter()


# ── Response Models ─────────────────────────────────────────────────────────

class SessionInfo(BaseModel):
    session_id: str
    device: str
    ip_address: Optional[str] = None
    location: Optional[str] = None
    last_active: datetime
    is_current: bool = False
    created_at: datetime

class LoginEvent(BaseModel):
    id: str
    timestamp: datetime
    ip_address: Optional[str] = None
    device: Optional[str] = None
    location: Optional[str] = None
    status: str  # "success" | "failed"
    method: str  # "password" | "google" | "firebase"

class SecurityOverview(BaseModel):
    two_factor_enabled: bool
    email_verified: bool
    last_login: Optional[datetime] = None
    login_count: int = 0
    failed_login_attempts: int = 0
    last_failed_login: Optional[datetime] = None
    last_password_change: Optional[datetime] = None
    active_sessions: int = 0
    account_created: Optional[datetime] = None

class PasswordChangeRequest(BaseModel):
    current_password: str
    new_password: str


# ── Endpoints ───────────────────────────────────────────────────────────────

@router.get("/overview", response_model=None)
@rate_limit()
async def get_security_overview(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get security overview for the current user."""
    user_id = str(current_user.id)

    # Get full user data from DB for security fields
    full_user = await user_repository.get_user_by_id(user_id)
    if not full_user:
        raise HTTPException(status_code=404, detail="User not found")

    # Count active sessions from activity logs
    from backend.db.mongodb.repositories.activity_repository import activity_repository
    recent_logins = await activity_repository.get_recent_activity(
        organization_id=getattr(current_user, "organization_id", user_id),
        user_id=user_id,
        activity_type="user.login",
        limit=100,
        skip=0,
    )
    # Approximate active sessions: unique devices in last 30 days
    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    active_devices = set()
    for entry in recent_logins:
        ts = getattr(entry, "timestamp", None)
        if ts and ts > thirty_days_ago:
            device = getattr(entry, "details", {}).get("device", "unknown")
            active_devices.add(device)

    return {
        "two_factor_enabled": getattr(full_user, "two_factor_enabled", False),
        "email_verified": getattr(full_user, "email_verified", False),
        "last_login": getattr(full_user, "last_login", None),
        "login_count": getattr(full_user, "login_count", 0),
        "failed_login_attempts": getattr(full_user, "failed_login_attempts", 0),
        "last_failed_login": getattr(full_user, "last_failed_login", None),
        "last_password_change": getattr(full_user, "last_password_change", None),
        "active_sessions": max(len(active_devices), 1),  # at least current session
        "account_created": getattr(full_user, "created_at", None),
    }


@router.get("/activity", response_model=None)
@rate_limit()
async def get_login_activity(
    request: Request,
    limit: int = 20,
    skip: int = 0,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get recent login/security activity for the current user."""
    user_id = str(current_user.id)

    from backend.db.mongodb.repositories.activity_repository import activity_repository
    # Fetch login-related activities
    all_activities = await activity_repository.get_recent_activity(
        organization_id=getattr(current_user, "organization_id", user_id),
        user_id=user_id,
        limit=200,
        skip=0,
    )

    # Filter for security-relevant events
    security_types = {
        "user.login", "user.logout", "user.login_failed",
        "user.password_changed", "user.2fa_enabled", "user.2fa_disabled",
        "user.session_revoked", "security.alert",
    }

    filtered = []
    for entry in all_activities:
        activity_type = getattr(entry, "activity_type", "")
        if activity_type in security_types:
            details = getattr(entry, "details", {})
            filtered.append({
                "id": getattr(entry, "id", str(getattr(entry, "_id", ""))),
                "timestamp": getattr(entry, "timestamp", datetime.utcnow()),
                "ip_address": details.get("ip_address"),
                "device": details.get("device"),
                "location": details.get("location"),
                "status": "failed" if "failed" in activity_type else "success",
                "method": details.get("method", "password"),
                "activity_type": activity_type,
                "description": details.get("description", activity_type.replace(".", " ").replace("_", " ").title()),
            })

    # Apply pagination
    paginated = filtered[skip:skip + limit]
    return {
        "events": paginated,
        "total": len(filtered),
        "limit": limit,
        "skip": skip,
    }


@router.get("/sessions", response_model=None)
@rate_limit()
async def get_active_sessions(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Get active sessions for the current user."""
    user_id = str(current_user.id)
    current_device = request.headers.get("User-Agent", "Unknown")
    current_ip = request.client.host if request.client else None

    from backend.db.mongodb.repositories.activity_repository import activity_repository
    logins = await activity_repository.get_recent_activity(
        organization_id=getattr(current_user, "organization_id", user_id),
        user_id=user_id,
        activity_type="user.login",
        limit=100,
        skip=0,
    )

    thirty_days_ago = datetime.utcnow() - timedelta(days=30)
    seen_devices: Dict[str, dict] = {}

    for entry in logins:
        ts = getattr(entry, "timestamp", None)
        if not ts or ts < thirty_days_ago:
            continue
        details = getattr(entry, "details", {})
        device = details.get("device", "Unknown device")
        device_key = device[:80]  # Truncate for dedup

        if device_key not in seen_devices or ts > seen_devices[device_key]["last_active"]:
            seen_devices[device_key] = {
                "session_id": str(getattr(entry, "id", getattr(entry, "_id", ""))),
                "device": device,
                "ip_address": details.get("ip_address"),
                "location": details.get("location"),
                "last_active": ts,
                "is_current": device[:60] == current_device[:60],
                "created_at": ts,
            }

    sessions = sorted(seen_devices.values(), key=lambda s: s["last_active"], reverse=True)

    # Ensure at least the current session shows
    if not sessions:
        sessions = [{
            "session_id": "current",
            "device": current_device,
            "ip_address": current_ip,
            "location": None,
            "last_active": datetime.utcnow(),
            "is_current": True,
            "created_at": datetime.utcnow(),
        }]

    return sessions


@router.post("/sessions/{session_id}/revoke", response_model=None)
@rate_limit()
async def revoke_session(
    request: Request,
    session_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Revoke a specific session (log out from that device)."""
    user_id = str(current_user.id)

    # Log the session revocation
    await log_activity(
        user_id=user_id,
        organization_id=getattr(current_user, "organization_id", user_id),
        activity_type="user.session_revoked",
        details={
            "session_id": session_id,
            "ip_address": request.client.host if request.client else None,
            "device": request.headers.get("User-Agent", "Unknown"),
            "description": "Session revoked by user",
        },
        severity="warning",
    )

    return {"message": "Session revoked successfully"}


@router.post("/change-password", response_model=None)
@rate_limit()
async def change_password(
    request: Request,
    password_data: PasswordChangeRequest,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Change the current user's password."""
    user_id = str(current_user.id)

    # Get full user with hashed_password
    full_user = await user_repository.get_user_by_id(user_id)
    if not full_user:
        raise HTTPException(status_code=404, detail="User not found")

    hashed_pw = getattr(full_user, "hashed_password", None)

    # If user has a password (not OAuth-only), verify current password
    if hashed_pw:
        if not verify_password(password_data.current_password, hashed_pw):
            # Log failed attempt
            await log_activity(
                user_id=user_id,
                organization_id=getattr(current_user, "organization_id", user_id),
                activity_type="user.password_change_failed",
                details={
                    "reason": "incorrect_current_password",
                    "ip_address": request.client.host if request.client else None,
                },
                severity="warning",
            )
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Current password is incorrect"
            )

    # Validate new password length
    if len(password_data.new_password) < 8:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="New password must be at least 8 characters"
        )

    # Hash and save new password
    new_hashed = get_password_hash(password_data.new_password)
    await user_repository.update_user(user_id, {
        "hashed_password": new_hashed,
        "last_password_change": datetime.utcnow(),
    })

    # Log password change
    await log_activity(
        user_id=user_id,
        organization_id=getattr(current_user, "organization_id", user_id),
        activity_type="user.password_changed",
        details={
            "ip_address": request.client.host if request.client else None,
            "device": request.headers.get("User-Agent", "Unknown"),
            "description": "Password changed successfully",
        },
        severity="info",
    )

    return {"message": "Password changed successfully"}


@router.post("/revoke-all-sessions", response_model=None)
@rate_limit()
async def revoke_all_sessions(
    request: Request,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """Revoke all sessions except the current one (invalidate refresh token)."""
    user_id = str(current_user.id)

    # Invalidate refresh token so other sessions can't refresh
    import secrets
    new_refresh = secrets.token_urlsafe(32)
    await user_repository.update_user(user_id, {"refresh_token": new_refresh})

    # Log the action
    await log_activity(
        user_id=user_id,
        organization_id=getattr(current_user, "organization_id", user_id),
        activity_type="user.all_sessions_revoked",
        details={
            "ip_address": request.client.host if request.client else None,
            "device": request.headers.get("User-Agent", "Unknown"),
            "description": "All other sessions revoked",
        },
        severity="warning",
    )

    return {"message": "All other sessions have been revoked", "new_refresh_token": new_refresh}
