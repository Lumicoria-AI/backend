"""
Lumicoria AI — Enterprise models.

ApiToken, Webhook, WebhookDelivery, SsoConfig, ScimToken, DomainClaim,
SessionPolicy, IpAllowlistEntry.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field, EmailStr


class ApiTokenScope(str, Enum):
    READ_PROJECTS = "read:projects"
    WRITE_PROJECTS = "write:projects"
    READ_TASKS = "read:tasks"
    WRITE_TASKS = "write:tasks"
    READ_AGENTS = "read:agents"
    RUN_AGENTS = "run:agents"
    WRITE_AGENTS = "write:agents"
    READ_MEMBERS = "read:members"
    WRITE_MEMBERS = "write:members"
    READ_BILLING = "read:billing"
    READ_AUDIT = "read:audit"
    ADMIN_ORG = "admin:org"


API_TOKEN_SCOPES = [s.value for s in ApiTokenScope]


class ApiTokenInDB(BaseModel):
    organization_id: str
    user_id: Optional[str] = None  # None for org-level service account
    name: str
    prefix: str  # First 6 chars displayed for identification
    token_hash: str
    scopes: List[str] = Field(default_factory=list)
    last_used_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class WebhookInDB(BaseModel):
    organization_id: str
    url: str
    events: List[str] = Field(default_factory=list)
    secret_hash: str
    enabled: bool = True
    last_delivery_at: Optional[datetime] = None
    failure_count: int = 0
    description: Optional[str] = None
    created_by: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(from_attributes=True)


class WebhookDeliveryInDB(BaseModel):
    webhook_id: str
    organization_id: str
    event: str
    payload: Dict[str, Any] = Field(default_factory=dict)
    status: str = "pending"  # pending | success | failed
    attempts: int = 0
    response_status: Optional[int] = None
    response_body: Optional[str] = None
    error: Optional[str] = None
    next_attempt_at: Optional[datetime] = None
    delivered_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(from_attributes=True)


class SsoProvider(str, Enum):
    SAML = "saml"
    OIDC = "oidc"


class SsoConfigInDB(BaseModel):
    organization_id: str
    provider: SsoProvider = SsoProvider.SAML
    metadata_xml: Optional[str] = None
    entity_id: Optional[str] = None
    acs_url: Optional[str] = None
    sso_url: Optional[str] = None
    certificate: Optional[str] = None
    attribute_map: Dict[str, str] = Field(default_factory=dict)
    default_role: str = "member"
    enabled: bool = False
    enforced_for_domains: List[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(from_attributes=True)


class ScimTokenInDB(BaseModel):
    organization_id: str
    token_hash: str
    prefix: str
    name: Optional[str] = None
    created_by: Optional[str] = None
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(from_attributes=True)


class DomainClaimInDB(BaseModel):
    organization_id: str
    domain: str
    verification_token: str
    verified_at: Optional[datetime] = None
    auto_join_role: str = "member"
    enforced: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(from_attributes=True)


class SessionPolicyInDB(BaseModel):
    organization_id: str
    idle_timeout_minutes: int = 120
    max_sessions_per_user: int = 10
    require_mfa: bool = False
    ip_allowlist_enabled: bool = False
    ip_allowlist: List[str] = Field(default_factory=list)
    data_residency: str = "us"  # us | eu | in
    cmk_enabled: bool = False
    cmk_kms_key_id: Optional[str] = None
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    updated_by: Optional[str] = None

    model_config = ConfigDict(from_attributes=True)
