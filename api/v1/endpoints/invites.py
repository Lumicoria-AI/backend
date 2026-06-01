"""
Invite REST endpoints (Phase 5).

Mounted at /api/v1/invites.

Authenticated:
  POST   /invites                       — issue a new invite + send email
  GET    /invites/sent                  — invites the current user sent
  GET    /invites/received              — pending invites for this user's email
  POST   /invites/{id}/resend           — re-send the email
  DELETE /invites/{id}                  — revoke (only by inviter)

Public (no auth — the token is the credential):
  GET    /invites/by-token/{token}      — preview an invite for the accept page
  POST   /invites/by-token/{token}/accept — accept (called on signup / by logged-in user)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, status

from backend.api.deps import get_current_active_user
from backend.db.mongodb.models.invite import (
    Invite,
    InviteCreate,
    InviteRole,
    InviteScope,
    InviteStatus,
)
from backend.db.mongodb.repositories.invite_repository import invite_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.invite_service import invite_service

logger = structlog.get_logger(__name__)

router = APIRouter()


def _org_id(user: User) -> str:
    return getattr(user, "organization_id", None) or str(user.id)


def _serialize(invite: Invite) -> Dict[str, Any]:
    data = invite.model_dump(by_alias=True) if hasattr(invite, "model_dump") else dict(invite)
    for f in ("_id", "id", "invited_by", "organization_id", "project_id", "accepted_user_id"):
        if f in data and data[f] is not None:
            data[f] = str(data[f])
    if "_id" in data and "id" not in data:
        data["id"] = data.pop("_id")
    elif "_id" in data:
        data.pop("_id")
    if data.get("task_ids"):
        data["task_ids"] = [str(t) for t in data["task_ids"]]
    for enum_field in ("status", "role", "scope"):
        v = data.get(enum_field)
        if hasattr(v, "value"):
            data[enum_field] = v.value
    for dt_field in ("created_at", "expires_at", "accepted_at", "revoked_at", "last_reminder_sent_at"):
        v = data.get(dt_field)
        if isinstance(v, datetime):
            data[dt_field] = v.isoformat()
    return data


# ── Authenticated ─────────────────────────────────────────────────────


@router.post("", response_model=None, status_code=status.HTTP_201_CREATED)
@router.post("/", response_model=None, status_code=status.HTTP_201_CREATED, include_in_schema=False)
async def create_invite(
    payload: InviteCreate,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Issue a new invite and send the email."""
    # If the caller didn't pass an org id, default to their own (the
    # `_get_org_id` convention used elsewhere).
    if payload.scope == InviteScope.ORGANIZATION and not payload.organization_id:
        payload.organization_id = _org_id(current_user)
    if payload.scope in (InviteScope.PROJECT, InviteScope.TASK) and not payload.organization_id:
        payload.organization_id = _org_id(current_user)

    try:
        invite = await invite_service.issue_invite(
            payload=payload, invited_by=str(current_user.id), send_email=True,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    await log_activity(
        user_id=str(current_user.id),
        organization_id=_org_id(current_user),
        activity_type="invite.created",
        details={
            "invite_id": str(invite.id),
            "email": invite.email,
            "scope": invite.scope.value if hasattr(invite.scope, "value") else str(invite.scope),
            "role": invite.role.value if hasattr(invite.role, "value") else str(invite.role),
        },
        related_resource_type="INVITE",
        related_resource_id=str(invite.id),
    )
    return _serialize(invite)


@router.get("/sent", response_model=None)
async def list_sent_invites(
    status_filter: Optional[InviteStatus] = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
) -> List[Dict[str, Any]]:
    invites = await invite_repository.list_by_inviter(
        invited_by=str(current_user.id),
        status=status_filter,
        limit=limit,
        skip=skip,
    )
    return [_serialize(i) for i in invites]


@router.get("/received", response_model=None)
async def list_received_invites(
    current_user: User = Depends(get_current_active_user),
) -> List[Dict[str, Any]]:
    """Pending invites addressed to the current user's email.

    Used by the Profile-page banner the first time someone signs up and we
    want to surface "you were invited by X".
    """
    email = (getattr(current_user, "email", "") or "").strip().lower()
    if not email:
        return []
    invites = await invite_repository.find_pending_by_email(email)
    return [_serialize(i) for i in invites]


@router.post("/{invite_id}/resend", response_model=None)
async def resend_invite(
    invite_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    ok = await invite_service.resend_invite(invite_id, requested_by=str(current_user.id))
    if not ok:
        raise HTTPException(
            status_code=400,
            detail="Could not resend — invite missing, not pending, or not yours.",
        )
    return {"ok": True}


@router.delete("/{invite_id}", status_code=status.HTTP_204_NO_CONTENT)
async def revoke_invite(
    invite_id: str,
    current_user: User = Depends(get_current_active_user),
) -> None:
    ok = await invite_service.revoke_invite(invite_id, requested_by=str(current_user.id))
    if not ok:
        raise HTTPException(status_code=404, detail="Invite not found or not yours")
    await log_activity(
        user_id=str(current_user.id),
        organization_id=_org_id(current_user),
        activity_type="invite.revoked",
        details={"invite_id": invite_id},
        related_resource_type="INVITE",
        related_resource_id=invite_id,
    )


# ── Public (token-keyed) ─────────────────────────────────────────────


@router.get("/by-token/{token}", response_model=None, include_in_schema=True)
async def preview_invite(token: str) -> Dict[str, Any]:
    """Public endpoint used by the accept-page to render a preview before
    the user signs up.  No auth — the token is the credential."""
    invite = await invite_repository.get_by_token(token)
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")

    # Hydrate display names so the accept page can render context.
    org_name = None
    project_name = None
    if invite.organization_id:
        from backend.services.invite_service import _resolve_org_name  # type: ignore
        org_name = await _resolve_org_name(str(invite.organization_id))
    if invite.project_id:
        from backend.services.invite_service import _resolve_project_name  # type: ignore
        project_name = await _resolve_project_name(str(invite.project_id))

    tasks = []
    if invite.task_ids:
        from backend.services.invite_service import _resolve_tasks  # type: ignore
        tasks = await _resolve_tasks([str(t) for t in invite.task_ids])

    return {
        **_serialize(invite),
        "organization_name": org_name,
        "project_name": project_name,
        "task_previews": tasks,
        # Don't echo the token back — the caller already has it in the URL.
    }


@router.post("/by-token/{token}/accept", response_model=None)
async def accept_invite_by_token(
    token: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Accept by a logged-in user.

    The signup endpoint accepts pending invites automatically via
    `invite_service.accept_pending_invites_for_user`; this endpoint is
    for users who already have an account and click an invite link.
    """
    invite = await invite_repository.get_by_token(token)
    if not invite:
        raise HTTPException(status_code=404, detail="Invite not found")
    if invite.status != InviteStatus.PENDING:
        raise HTTPException(status_code=400, detail=f"Invite is {invite.status}")

    # Email must match.
    current_email = (getattr(current_user, "email", "") or "").strip().lower()
    if current_email != (invite.email_normalized or invite.email).lower():
        raise HTTPException(
            status_code=403,
            detail="This invite was sent to a different email address.",
        )

    summary = await invite_service.accept_pending_invites_for_user(
        user_id=str(current_user.id), email=current_email,
    )
    await log_activity(
        user_id=str(current_user.id),
        organization_id=_org_id(current_user),
        activity_type="invite.accepted",
        details={"invite_id": str(invite.id), **summary},
        related_resource_type="INVITE",
        related_resource_id=str(invite.id),
    )
    return {"ok": True, **summary, "invite": _serialize(invite)}
