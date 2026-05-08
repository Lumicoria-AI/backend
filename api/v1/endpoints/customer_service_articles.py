"""Operator-facing help-center article endpoints.

Public read paths live in `customer_service_public.py` so the article
listing on the portal stays anonymous.
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
from backend.services.customer_service import articles as articles_svc

router = APIRouter()
logger = structlog.get_logger(__name__)


async def _require_perm(current_user: User) -> str:
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


class ArticleCreateRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=300)
    body: str = Field(..., min_length=1, max_length=50_000)
    summary: Optional[str] = Field(None, max_length=500)
    category: Optional[str] = Field(None, max_length=64)
    tags: Optional[List[str]] = None
    published: bool = False
    featured: bool = False


class ArticleUpdateRequest(BaseModel):
    title: Optional[str] = Field(None, min_length=1, max_length=300)
    body: Optional[str] = Field(None, min_length=1, max_length=50_000)
    summary: Optional[str] = Field(None, max_length=500)
    category: Optional[str] = Field(None, max_length=64)
    tags: Optional[List[str]] = None
    slug: Optional[str] = Field(None, max_length=120)
    published: Optional[bool] = None
    featured: Optional[bool] = None


@router.get("")
async def list_articles(
    published: Optional[bool] = Query(None),
    category: Optional[str] = Query(None, max_length=64),
    limit: int = Query(100, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_perm(current_user)
    return await articles_svc.list_articles_admin(
        org_id, published=published, category=category, limit=limit, offset=offset,
    )


@router.get("/{article_id}")
async def get_article(
    article_id: str,
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_perm(current_user)
    article = await articles_svc.get_article_admin(org_id, article_id)
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return article


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_article(
    payload: ArticleCreateRequest,
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_perm(current_user)
    user_id = str(current_user.id)

    article = await articles_svc.create_article(
        organization_id=org_id,
        title=payload.title,
        body=payload.body,
        summary=payload.summary,
        category=payload.category,
        tags=payload.tags,
        published=payload.published,
        featured=payload.featured,
        created_by_user_id=user_id,
    )
    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="customer_service.article_created",
        details={"article_id": article["id"], "slug": article["slug"]},
        related_resource_type="ARTICLE",
        related_resource_id=article["id"],
        agent_name="Customer Service Agent",
    )
    return article


@router.patch("/{article_id}")
async def update_article(
    article_id: str,
    payload: ArticleUpdateRequest,
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_perm(current_user)
    fields = payload.model_dump(exclude_unset=True)
    updated = await articles_svc.update_article(org_id, article_id, fields)
    if not updated:
        raise HTTPException(status_code=404, detail="Article not found")
    return updated


@router.delete("/{article_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_article(
    article_id: str,
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_perm(current_user)
    ok = await articles_svc.soft_delete_article(org_id, article_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Article not found")
