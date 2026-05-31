"""
Invite model — used to invite non-users (or future-users) to tasks, projects,
or organizations.  Phase 5 builds the full invite flow on top of this.

Collection: `invites`
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, EmailStr, Field

from backend.models.mongodb_models import PyObjectId


class InviteStatus(str, Enum):
    PENDING = "pending"
    ACCEPTED = "accepted"
    EXPIRED = "expired"
    REVOKED = "revoked"


class InviteRole(str, Enum):
    """Role granted when the invitee accepts."""
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class InviteScope(str, Enum):
    """What the invitee is being invited to."""
    TASK = "task"
    PROJECT = "project"
    ORGANIZATION = "organization"


class Invite(BaseModel):
    """Mongo model for `invites` collection."""
    id: Optional[PyObjectId] = Field(default_factory=PyObjectId, alias="_id")
    email: EmailStr                                          # who is invited
    email_normalized: str                                    # lower-cased for lookup

    invited_by: PyObjectId                                   # inviter's user_id
    inviter_name: Optional[str] = None                       # snapshot for email
    inviter_email: Optional[str] = None                      # snapshot

    # Scope — at least one of these must be set when issuing the invite.
    scope: InviteScope = InviteScope.ORGANIZATION
    organization_id: Optional[PyObjectId] = None
    project_id: Optional[PyObjectId] = None
    task_ids: List[PyObjectId] = Field(default_factory=list)

    role: InviteRole = InviteRole.MEMBER
    token: str                                               # signed JWT-like token

    status: InviteStatus = InviteStatus.PENDING
    message: Optional[str] = None                            # optional human message

    expires_at: Optional[datetime] = None                    # default: +14 days
    created_at: datetime = Field(default_factory=datetime.utcnow)
    accepted_at: Optional[datetime] = None
    accepted_user_id: Optional[PyObjectId] = None
    revoked_at: Optional[datetime] = None

    # Reminder tracking so we don't spam pending invites.
    last_reminder_sent_at: Optional[datetime] = None
    reminder_count: int = 0

    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
        json_encoders={ObjectId: str, datetime: lambda v: v.isoformat()},
    )


class InviteCreate(BaseModel):
    """Payload for issuing an invite via the API."""
    email: EmailStr
    scope: InviteScope = InviteScope.ORGANIZATION
    organization_id: Optional[PyObjectId] = None
    project_id: Optional[PyObjectId] = None
    task_ids: List[PyObjectId] = Field(default_factory=list)
    role: InviteRole = InviteRole.MEMBER
    message: Optional[str] = None
    expires_in_days: int = 14

    model_config = ConfigDict(arbitrary_types_allowed=True)


class InviteUpdate(BaseModel):
    """Patch payload."""
    role: Optional[InviteRole] = None
    status: Optional[InviteStatus] = None
    message: Optional[str] = None
    expires_at: Optional[datetime] = None
