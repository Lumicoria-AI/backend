"""Operator-facing response-template CRUD.

Replaces the hardcoded `templates = [{"id": "template_1", ...}]` mock in
`customer_service.py`.  Tenant-scoped, permission-gated.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.permission_repository import permission_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.customer_service import templates as templates_svc

router = APIRouter()
logger = structlog.get_logger(__name__)


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


class TemplateCreateRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    category: str = Field(..., min_length=1, max_length=64)
    body: str = Field(..., min_length=1, max_length=20_000)
    tone: Optional[str] = Field(None, max_length=32)
    description: Optional[str] = Field(None, max_length=500)
    variables: Optional[List[str]] = None


class TemplateUpdateRequest(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    category: Optional[str] = Field(None, min_length=1, max_length=64)
    body: Optional[str] = Field(None, min_length=1, max_length=20_000)
    tone: Optional[str] = Field(None, max_length=32)
    description: Optional[str] = Field(None, max_length=500)
    variables: Optional[List[str]] = None


@router.get("")
async def list_templates(
    category: Optional[str] = Query(None, max_length=64),
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_perm(current_user)
    return await templates_svc.list_templates(org_id, category=category)


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_template_row(
    payload: TemplateCreateRequest,
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_perm(current_user)
    user_id = str(current_user.id)

    row = await templates_svc.create_template(
        organization_id=org_id,
        name=payload.name,
        category=payload.category,
        body=payload.body,
        tone=payload.tone,
        description=payload.description,
        variables=payload.variables,
        created_by_user_id=user_id,
        created_by_agent=False,
    )

    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="customer_service.template_created",
        details={"template_id": row["id"], "name": row["name"]},
        related_resource_type="TEMPLATE",
        related_resource_id=row["id"],
        agent_name="Customer Service Agent",
    )
    return row


@router.patch("/{template_id}")
async def update_template_row(
    template_id: str,
    payload: TemplateUpdateRequest,
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_perm(current_user)
    fields = payload.model_dump(exclude_unset=True)
    updated = await templates_svc.update_template(
        organization_id=org_id,
        template_id=template_id,
        fields=fields,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Template not found")
    return updated


@router.delete("/{template_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_template_row(
    template_id: str,
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_perm(current_user)
    ok = await templates_svc.soft_delete_template(org_id, template_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Template not found")


@router.post("/{template_id}/use")
async def mark_template_used(
    template_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Bumps usage_count.  Called by frontend when an operator clicks
    a quick-reply to insert a template body."""
    org_id = await _require_perm(current_user)
    await templates_svc.increment_usage(org_id, template_id)
    return {"ok": True, "template_id": template_id}
