"""Operator-side branding admin.

Tenants edit their portal/widget appearance + slug here.  The slug is
the public URL segment (`/portal/{slug}`); uniqueness is enforced
across all orgs, so a 409 surfaces if two tenants try to claim the
same slug.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, EmailStr, Field, field_validator

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.permission_repository import permission_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.customer_service import branding as branding_svc

router = APIRouter()


async def _require_perm(current_user: User) -> str:
    """Permission against actual org (None passes through); tenant-scope
    falls back to user_id for personal accounts."""
    user_id = str(current_user.id)
    permission_org = getattr(current_user, "organization_id", None)
    has = await permission_repository.check_permission(
        user_id=user_id,
        organization_id=permission_org,
        resource_type="AGENT",
        resource_id="customer_service",
        permission_type="EXECUTE",
    )
    if not has:
        raise HTTPException(status_code=403, detail="Permission denied")
    return permission_org or user_id


class BrandingUpsertRequest(BaseModel):
    slug: Optional[str] = Field(None, min_length=2, max_length=64)
    display_name: Optional[str] = Field(None, min_length=1, max_length=200)
    logo_url: Optional[str] = Field(None, max_length=1000)
    primary_color: Optional[str] = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")
    accent_color: Optional[str] = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")
    hero_copy: Optional[str] = Field(None, max_length=2000)
    support_email: Optional[EmailStr] = None
    sla_response_minutes: Optional[int] = Field(None, ge=1, le=10080)
    captcha_enabled: Optional[bool] = None
    public_categories: Optional[List[str]] = None

    @field_validator("slug")
    @classmethod
    def _slug_check(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not branding_svc.is_valid_slug(v.lower()):
            raise ValueError("slug must match ^[a-z0-9](?:[a-z0-9-]{1,62}[a-z0-9])?$")
        return v.lower()


@router.get("")
async def get_branding(
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_perm(current_user)
    return await branding_svc.get_for_org(org_id)


@router.put("")
async def upsert_branding(
    payload: BrandingUpsertRequest,
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_perm(current_user)
    user_id = str(current_user.id)

    fields = payload.model_dump(exclude_unset=True)
    if "support_email" in fields and fields["support_email"] is not None:
        fields["support_email"] = str(fields["support_email"]).lower()

    try:
        result = await branding_svc.upsert(org_id, fields)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="customer_service.branding_updated",
        details={"fields": list(fields.keys())},
        related_resource_type="ORG_BRANDING",
        related_resource_id=org_id,
        agent_name="Customer Service Agent",
    )
    return result
