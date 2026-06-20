"""
Phase A4 / C — Cross-resource comments.

Mounted at `/api/v1/comments`.  Used by tasks, projects, documents, agent
runs — anywhere a threaded discussion makes sense.

Access is gated by org membership; finer-grained checks (e.g. project
membership) happen on the resource side when a write event triggers a
notification.  Read access is granted to any org member who can see the
parent resource — keeping the gate at the resource layer avoids
duplicating permission logic here.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.comments_repository import comments_repository
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.event_bus import emit

logger = structlog.get_logger(__name__)
router = APIRouter()


def _oid(value: Any) -> Optional[ObjectId]:
    if value is None:
        return None
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value))
    except Exception:
        return None


def _resolve_org_id(current_user: User) -> str:
    primary = getattr(current_user, "organization_id", None)
    if primary:
        return str(primary)
    ids = getattr(current_user, "organization_ids", None) or []
    if ids:
        return str(ids[0])
    # Personal mode — user has no org. Use the user's own id as the
    # scope key so personal comments segregate cleanly per account.
    return str(current_user.id)


async def _require_org_membership(org_id: str, current_user: User):
    # Personal mode: org_id equals the user's own id → no Organization
    # document exists. Allow the comment; the scope key (org_id ==
    # user_id) gives natural isolation. Personal tasks and personal
    # documents reach here.
    if str(org_id) == str(current_user.id):
        return None

    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(m) for m in (org.member_ids or [])]:
        raise HTTPException(status_code=403, detail="Not a member of this organization")
    return org


class CommentCreate(BaseModel):
    resource_type: str = Field(..., max_length=64)
    resource_id: str = Field(..., max_length=64)
    body: str = Field(..., min_length=1, max_length=8000)
    mentions: List[str] = Field(default_factory=list)
    parent_id: Optional[str] = None


@router.get("")
async def list_comments(
    resource_type: str = Query(...),
    resource_id: str = Query(...),
    parent_id: Optional[str] = Query(None),
    organization_id: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=500),
    skip: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_membership(org_id, current_user)
    return await comments_repository.list(
        organization_id=org_id,
        resource_type=resource_type,
        resource_id=resource_id,
        parent_id=parent_id,
        limit=limit, skip=skip,
    )


@router.post("", status_code=201)
async def create_comment(
    payload: CommentCreate,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_membership(org_id, current_user)
    row = await comments_repository.create(
        organization_id=org_id,
        resource_type=payload.resource_type,
        resource_id=payload.resource_id,
        body=payload.body,
        user_id=str(current_user.id),
        mentions=payload.mentions,
        parent_id=payload.parent_id,
    )
    await log_activity(
        user_id=str(current_user.id),
        organization_id=org_id,
        activity_type="comment.created",
        details={
            "comment_id": row["id"],
            "resource_type": payload.resource_type,
            "resource_id": payload.resource_id,
            "mentions": payload.mentions,
        },
        related_resource_type=payload.resource_type,
        related_resource_id=payload.resource_id,
    )
    await emit(
        "comment.created",
        organization_id=org_id, actor_id=str(current_user.id),
        resource_type=payload.resource_type,
        resource_id=payload.resource_id,
        payload={"comment_id": row["id"], "mentions": payload.mentions},
    )
    return row


class CommentUpdate(BaseModel):
    body: Optional[str] = Field(None, min_length=1, max_length=8000)
    mentions: Optional[List[str]] = None
    resolved: Optional[bool] = None


@router.patch("/{comment_id}")
async def update_comment(
    comment_id: str,
    payload: CommentUpdate,
    current_user: User = Depends(get_current_active_user),
):
    existing = await comments_repository.get(comment_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Comment not found")
    await _require_org_membership(existing["organization_id"], current_user)
    # Only author can edit body; anyone with org access can resolve/unresolve.
    if payload.body is not None and existing.get("user_id") != str(current_user.id):
        raise HTTPException(status_code=403, detail="Only the author can edit a comment")
    updated = await comments_repository.update(
        comment_id,
        body=payload.body,
        mentions=payload.mentions,
        resolved=payload.resolved,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Comment not found")
    await emit("comment.updated", organization_id=existing["organization_id"],
               actor_id=str(current_user.id),
               resource_type=existing["resource_type"],
               resource_id=existing["resource_id"],
               payload={"comment_id": comment_id})
    return updated


@router.delete("/{comment_id}", status_code=204)
async def delete_comment(
    comment_id: str,
    current_user: User = Depends(get_current_active_user),
):
    existing = await comments_repository.get(comment_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Comment not found")
    org = await _require_org_membership(existing["organization_id"], current_user)
    is_author = existing.get("user_id") == str(current_user.id)
    is_org_admin = _oid(current_user.id) in [_oid(a) for a in (org.admin_ids or [])]
    if not (is_author or is_org_admin):
        raise HTTPException(status_code=403, detail="Author or org admin only")
    await comments_repository.delete(comment_id)
    return None


@router.post("/{comment_id}/reactions/{emoji}")
async def react_to_comment(
    comment_id: str,
    emoji: str,
    current_user: User = Depends(get_current_active_user),
):
    existing = await comments_repository.get(comment_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Comment not found")
    await _require_org_membership(existing["organization_id"], current_user)
    return await comments_repository.react(
        comment_id, emoji=emoji, user_id=str(current_user.id), add=True,
    )


@router.delete("/{comment_id}/reactions/{emoji}")
async def unreact_to_comment(
    comment_id: str,
    emoji: str,
    current_user: User = Depends(get_current_active_user),
):
    existing = await comments_repository.get(comment_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Comment not found")
    await _require_org_membership(existing["organization_id"], current_user)
    return await comments_repository.react(
        comment_id, emoji=emoji, user_id=str(current_user.id), add=False,
    )


@router.post("/{comment_id}/resolve")
async def resolve_comment(
    comment_id: str,
    current_user: User = Depends(get_current_active_user),
):
    existing = await comments_repository.get(comment_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Comment not found")
    await _require_org_membership(existing["organization_id"], current_user)
    return await comments_repository.update(comment_id, resolved=True)


@router.post("/{comment_id}/unresolve")
async def unresolve_comment(
    comment_id: str,
    current_user: User = Depends(get_current_active_user),
):
    existing = await comments_repository.get(comment_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Comment not found")
    await _require_org_membership(existing["organization_id"], current_user)
    return await comments_repository.update(comment_id, resolved=False)


@router.get("/{comment_id}/replies")
async def list_comment_replies(
    comment_id: str,
    organization_id: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=500),
    skip: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
):
    existing = await comments_repository.get(comment_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Comment not found")
    org_id = organization_id or existing["organization_id"]
    await _require_org_membership(org_id, current_user)
    return await comments_repository.list(
        organization_id=org_id,
        resource_type=existing["resource_type"],
        resource_id=existing["resource_id"],
        parent_id=comment_id,
        limit=limit, skip=skip,
    )


@router.get("/mentions/me")
async def list_my_mentions(
    organization_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id
    if not org_id:
        try:
            org_id = _resolve_org_id(current_user)
        except HTTPException:
            org_id = None
    return await comments_repository.list_mentions_for_user(
        user_id=str(current_user.id),
        organization_id=org_id,
        limit=limit,
    )
