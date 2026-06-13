"""
Lumicoria AI — Workspace models (Teams, Projects v2, Memberships)

These Pydantic models back the new Workspace layer.  Naming convention:

    Team           — root model stored in `teams` collection
    TeamCreate     — POST body
    TeamUpdate     — PATCH body
    TeamResponse   — outbound shape (strings only, no ObjectId)

The mongodb model classes inherit MongoBaseModel (for the legacy id alias)
and play nicely with BaseRepository[T].
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from bson import ObjectId
from pydantic import BaseModel, ConfigDict, Field

from backend.models.mongodb_models import MongoBaseModel, PyObjectId


# ───────────────────────────────────────────────────────── role enums


class TeamRoleEnum(str, Enum):
    TEAM_ADMIN = "team_admin"
    EDITOR = "editor"
    OPERATOR = "operator"
    VIEWER = "viewer"


class ProjectRoleEnum(str, Enum):
    LEAD = "lead"
    EDITOR = "editor"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


class ProjectStatus(str, Enum):
    PLANNING = "planning"
    ACTIVE = "active"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    ARCHIVED = "archived"


class ProjectVisibility(str, Enum):
    PRIVATE = "private"
    TEAM = "team"
    ORG = "org"


# ─────────────────────────────────────────────────────────── Team


class Team(MongoBaseModel):
    organization_id: PyObjectId
    name: str
    slug: str
    description: Optional[str] = None
    department_tag: Optional[str] = None
    color: Optional[str] = "#6C4AB0"
    logo_url: Optional[str] = None
    cover_url: Optional[str] = None
    owner_id: PyObjectId
    admin_ids: List[PyObjectId] = Field(default_factory=list)
    member_ids: List[PyObjectId] = Field(default_factory=list)
    settings: Dict[str, Any] = Field(default_factory=dict)
    branding: Dict[str, Any] = Field(default_factory=dict)
    is_archived: bool = False
    archived_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_by: PyObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class TeamCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=120)
    slug: Optional[str] = Field(None, max_length=120, description="Auto-generated from name if omitted")
    description: Optional[str] = Field(None, max_length=2000)
    department_tag: Optional[str] = Field(None, max_length=64)
    color: Optional[str] = Field(default="#6C4AB0", max_length=16)
    logo_url: Optional[str] = None
    cover_url: Optional[str] = None
    member_ids: List[str] = Field(default_factory=list)
    admin_ids: List[str] = Field(default_factory=list)
    settings: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TeamUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=120)
    description: Optional[str] = Field(None, max_length=2000)
    department_tag: Optional[str] = Field(None, max_length=64)
    color: Optional[str] = Field(None, max_length=16)
    logo_url: Optional[str] = None
    cover_url: Optional[str] = None
    settings: Optional[Dict[str, Any]] = None
    branding: Optional[Dict[str, Any]] = None


class TeamResponse(BaseModel):
    id: str
    organization_id: str
    name: str
    slug: str
    description: Optional[str] = None
    department_tag: Optional[str] = None
    color: Optional[str] = None
    logo_url: Optional[str] = None
    cover_url: Optional[str] = None
    owner_id: str
    admin_ids: List[str] = Field(default_factory=list)
    member_ids: List[str] = Field(default_factory=list)
    settings: Dict[str, Any] = Field(default_factory=dict)
    branding: Dict[str, Any] = Field(default_factory=dict)
    is_archived: bool = False
    archived_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_by: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TeamMember(MongoBaseModel):
    team_id: PyObjectId
    user_id: PyObjectId
    organization_id: PyObjectId
    role: TeamRoleEnum = TeamRoleEnum.EDITOR
    joined_at: datetime = Field(default_factory=datetime.utcnow)
    invited_by: Optional[PyObjectId] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TeamMemberAdd(BaseModel):
    user_id: str
    role: TeamRoleEnum = TeamRoleEnum.EDITOR


class TeamMemberRoleUpdate(BaseModel):
    role: TeamRoleEnum


class TeamMemberResponse(BaseModel):
    user_id: str
    organization_id: str
    team_id: str
    role: TeamRoleEnum
    joined_at: datetime
    invited_by: Optional[str] = None
    # Hydrated user fields (optional)
    full_name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


# ─────────────────────────────────────────────────────────── ProjectV2


class ProjectV2(MongoBaseModel):
    organization_id: PyObjectId
    team_id: Optional[PyObjectId] = None
    name: str
    slug: str
    description: Optional[str] = None
    status: ProjectStatus = ProjectStatus.PLANNING
    priority: Optional[str] = "medium"
    color: Optional[str] = "#6C4AB0"
    logo_url: Optional[str] = None
    cover_image_url: Optional[str] = None
    due_date: Optional[datetime] = None
    lead_id: Optional[PyObjectId] = None
    member_ids: List[PyObjectId] = Field(default_factory=list)
    agent_keys: List[str] = Field(default_factory=list, description="Platform agent keys enabled on this project")
    custom_agent_ids: List[PyObjectId] = Field(default_factory=list)
    tag_ids: List[PyObjectId] = Field(default_factory=list)
    strict_mode: bool = False
    visibility: ProjectVisibility = ProjectVisibility.PRIVATE
    settings: Dict[str, Any] = Field(default_factory=dict)
    branding: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_archived: bool = False
    archived_at: Optional[datetime] = None
    created_by: PyObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ProjectV2Create(BaseModel):
    name: str = Field(..., min_length=1, max_length=180)
    slug: Optional[str] = None
    description: Optional[str] = Field(None, max_length=4000)
    status: ProjectStatus = ProjectStatus.PLANNING
    priority: Optional[str] = "medium"
    color: Optional[str] = "#6C4AB0"
    logo_url: Optional[str] = None
    cover_image_url: Optional[str] = None
    due_date: Optional[datetime] = None
    team_id: Optional[str] = None
    lead_id: Optional[str] = None
    member_ids: List[str] = Field(default_factory=list)
    agent_keys: List[str] = Field(default_factory=list)
    visibility: ProjectVisibility = ProjectVisibility.PRIVATE
    strict_mode: bool = False
    settings: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ProjectV2Update(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=180)
    description: Optional[str] = Field(None, max_length=4000)
    status: Optional[ProjectStatus] = None
    priority: Optional[str] = None
    color: Optional[str] = None
    logo_url: Optional[str] = None
    cover_image_url: Optional[str] = None
    due_date: Optional[datetime] = None
    team_id: Optional[str] = None
    lead_id: Optional[str] = None
    visibility: Optional[ProjectVisibility] = None
    strict_mode: Optional[bool] = None
    settings: Optional[Dict[str, Any]] = None
    branding: Optional[Dict[str, Any]] = None


class ProjectV2Response(BaseModel):
    id: str
    organization_id: str
    team_id: Optional[str] = None
    name: str
    slug: str
    description: Optional[str] = None
    status: ProjectStatus
    priority: Optional[str] = None
    color: Optional[str] = None
    logo_url: Optional[str] = None
    cover_image_url: Optional[str] = None
    due_date: Optional[datetime] = None
    lead_id: Optional[str] = None
    member_ids: List[str] = Field(default_factory=list)
    agent_keys: List[str] = Field(default_factory=list)
    custom_agent_ids: List[str] = Field(default_factory=list)
    tag_ids: List[str] = Field(default_factory=list)
    strict_mode: bool = False
    visibility: ProjectVisibility = ProjectVisibility.PRIVATE
    settings: Dict[str, Any] = Field(default_factory=dict)
    branding: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    is_archived: bool = False
    archived_at: Optional[datetime] = None
    created_by: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ProjectMember(MongoBaseModel):
    project_id: PyObjectId
    user_id: PyObjectId
    organization_id: PyObjectId
    role: ProjectRoleEnum = ProjectRoleEnum.EDITOR
    joined_at: datetime = Field(default_factory=datetime.utcnow)
    invited_by: Optional[PyObjectId] = None


class ProjectMemberAdd(BaseModel):
    user_id: str
    role: ProjectRoleEnum = ProjectRoleEnum.EDITOR


class ProjectMemberRoleUpdate(BaseModel):
    role: ProjectRoleEnum


class ProjectMemberResponse(BaseModel):
    project_id: str
    user_id: str
    organization_id: str
    role: ProjectRoleEnum
    joined_at: datetime
    invited_by: Optional[str] = None
    full_name: Optional[str] = None
    email: Optional[str] = None
    avatar_url: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)


class ProjectAgent(MongoBaseModel):
    project_id: PyObjectId
    organization_id: PyObjectId
    agent_key: Optional[str] = None        # platform agent (one of the 21 registry keys)
    custom_agent_id: Optional[PyObjectId] = None
    enabled: bool = True
    autonomy_level: str = "suggest"        # suggest | auto-propose | auto-execute
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    fallback_chain: List[str] = Field(default_factory=list)
    attached_by: PyObjectId
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


class ProjectAgentAdd(BaseModel):
    agent_key: Optional[str] = None
    custom_agent_id: Optional[str] = None
    enabled: bool = True
    autonomy_level: str = "suggest"
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    fallback_chain: List[str] = Field(default_factory=list)


class ProjectAgentResponse(BaseModel):
    id: str
    project_id: str
    organization_id: str
    agent_key: Optional[str] = None
    custom_agent_id: Optional[str] = None
    enabled: bool = True
    autonomy_level: str = "suggest"
    config_overrides: Dict[str, Any] = Field(default_factory=dict)
    fallback_chain: List[str] = Field(default_factory=list)
    attached_by: str
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


# ─────────────────────────────────────────────────────────── helpers


def slugify(name: str) -> str:
    """Lightweight slugifier for team / project slugs."""
    import re
    base = re.sub(r"[^a-zA-Z0-9]+", "-", name.strip().lower()).strip("-")
    return base[:80] or "untitled"


def stringify_oid(v: Any) -> Optional[str]:
    if v is None:
        return None
    if isinstance(v, ObjectId):
        return str(v)
    return str(v)
