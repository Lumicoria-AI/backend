"""
Phase A1 — Teams REST API.

Mounted under `/api/v1/organizations/{org_id}/teams/`.

This module ships the core Team surface; advanced analytics, branding uploads,
CSV import, and integration plumbing live in `teams_extended.py` (Phase A1
follow-up).

All write operations require the caller to be at least a member of the org;
team-admin promotion is required for destructive operations on the team.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.db.mongodb.repositories.team_repository import team_repository
from backend.db.mongodb.repositories.team_member_repository import team_member_repository
from backend.db.mongodb.repositories.user_repository import get_user_repository
from backend.db.mongodb.repositories.activity_repository import activity_repository
from backend.db.mongodb.repositories.invite_repository import invite_repository
from backend.db.mongodb.models.invite import InviteRole, InviteScope, InviteStatus
from backend.models.user import User
from backend.models.workspace import (
    ProjectRoleEnum,
    Team,
    TeamCreate,
    TeamMemberAdd,
    TeamMemberResponse,
    TeamMemberRoleUpdate,
    TeamResponse,
    TeamRoleEnum,
    TeamUpdate,
)
from backend.services.activity_logger import log_activity
from backend.services.billing.plan_caps import PlanCapExceeded
from backend.services.event_bus import emit
from backend.services.invite_service import invite_service

logger = structlog.get_logger(__name__)
router = APIRouter()


# ── Helpers ──────────────────────────────────────────────────────────


def _oid(value: Any) -> Optional[ObjectId]:
    if value is None:
        return None
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value))
    except Exception:
        return None


async def _require_org_membership(org_id: str, current_user: User):
    """Return the Organization the caller belongs to, or 404/403."""
    if _oid(org_id) is None:
        raise HTTPException(status_code=400, detail="Invalid organization id")
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    user_oid = _oid(current_user.id)
    member_ids = [_oid(m) for m in (org.member_ids or [])]
    if user_oid not in member_ids:
        raise HTTPException(status_code=403, detail="Not a member of this organization")
    return org


async def _require_org_admin(org_id: str, current_user: User):
    org = await _require_org_membership(org_id, current_user)
    user_oid = _oid(current_user.id)
    admin_ids = [_oid(a) for a in (org.admin_ids or [])]
    if user_oid not in admin_ids:
        raise HTTPException(status_code=403, detail="Org admin permission required")
    return org


async def _require_team_access(
    org_id: str,
    team_id: str,
    current_user: User,
) -> Team:
    await _require_org_membership(org_id, current_user)
    team = await team_repository.get_team(team_id, organization_id=org_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    member_ids = [_oid(m) for m in (team.member_ids or [])]
    if _oid(current_user.id) not in member_ids and not _caller_is_org_admin(team, current_user):
        # Org admins can always read team metadata for governance.
        raise HTTPException(status_code=403, detail="Not a member of this team")
    return team


def _caller_is_org_admin(team: Team, current_user: User) -> bool:
    # Best-effort: when the User model exposes organization_admin_ids we use it.
    admin_orgs = set(str(o) for o in (getattr(current_user, "organization_admin_ids", []) or []))
    return str(team.organization_id) in admin_orgs


async def _require_team_admin(
    org_id: str,
    team_id: str,
    current_user: User,
) -> Team:
    team = await _require_team_access(org_id, team_id, current_user)
    user_oid = _oid(current_user.id)
    admin_ids = [_oid(a) for a in (team.admin_ids or [])]
    owner_oid = _oid(team.owner_id)
    if user_oid != owner_oid and user_oid not in admin_ids:
        # Org admins can administer any team.
        if not _caller_is_org_admin(team, current_user):
            raise HTTPException(status_code=403, detail="Team admin permission required")
    return team


def _serialize_team(team: Team) -> TeamResponse:
    d = team.model_dump(mode="json") if hasattr(team, "model_dump") else team.dict()
    return TeamResponse(
        id=str(d.get("_id") or d.get("id")),
        organization_id=str(d.get("organization_id")),
        name=d["name"],
        slug=d["slug"],
        description=d.get("description"),
        department_tag=d.get("department_tag"),
        color=d.get("color"),
        logo_url=d.get("logo_url"),
        cover_url=d.get("cover_url"),
        owner_id=str(d.get("owner_id")) if d.get("owner_id") else "",
        admin_ids=[str(v) for v in (d.get("admin_ids") or [])],
        member_ids=[str(v) for v in (d.get("member_ids") or [])],
        settings=d.get("settings") or {},
        branding=d.get("branding") or {},
        is_archived=bool(d.get("is_archived")),
        archived_at=d.get("archived_at"),
        metadata=d.get("metadata") or {},
        created_by=str(d.get("created_by")) if d.get("created_by") else "",
        created_at=d.get("created_at"),
        updated_at=d.get("updated_at"),
    )


async def _hydrate_member(user_id: str, *, role: str, team_id: str, organization_id: str, joined_at: datetime, invited_by: Optional[str] = None) -> TeamMemberResponse:
    repo = await get_user_repository()
    user = None
    try:
        user = await repo.get_user_by_id(user_id)
    except Exception:
        user = None
    return TeamMemberResponse(
        user_id=str(user_id),
        organization_id=str(organization_id),
        team_id=str(team_id),
        role=TeamRoleEnum(role) if role in TeamRoleEnum._value2member_map_ else TeamRoleEnum.EDITOR,
        joined_at=joined_at or datetime.utcnow(),
        invited_by=invited_by,
        full_name=getattr(user, "full_name", None) if user else None,
        email=getattr(user, "email", None) if user else None,
        avatar_url=getattr(user, "avatar_url", None) if user else None,
    )


# ────────────────────────────────────────────────────────── Team CRUD


@router.get("", response_model=List[TeamResponse])
async def list_teams(
    org_id: str,
    include_archived: bool = Query(False),
    search: Optional[str] = Query(None, max_length=200),
    department_tag: Optional[str] = Query(None),
    only_mine: bool = Query(False, description="Only teams the caller belongs to"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_membership(org_id, current_user)
    teams = await team_repository.list_teams(
        organization_id=org_id,
        include_archived=include_archived,
        search=search,
        department_tag=department_tag,
        member_id=str(current_user.id) if only_mine else None,
        skip=skip,
        limit=limit,
    )
    return [_serialize_team(t) for t in teams]


@router.post("", response_model=TeamResponse, status_code=201)
async def create_team(
    org_id: str,
    payload: TeamCreate,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_membership(org_id, current_user)
    try:
        team = await team_repository.create_team(
            payload,
            organization_id=org_id,
            creator_id=str(current_user.id),
        )
    except PlanCapExceeded as exc:
        raise HTTPException(status_code=402, detail=exc.detail)
    # Mirror creator into normalised team_members.
    await team_member_repository.add_or_update(
        team_id=str(team.id),
        user_id=str(current_user.id),
        organization_id=org_id,
        role=TeamRoleEnum.TEAM_ADMIN,
    )
    await log_activity(
        user_id=str(current_user.id),
        organization_id=org_id,
        activity_type="team.created",
        details={"team_id": str(team.id), "name": team.name, "slug": team.slug},
        related_resource_type="team",
        related_resource_id=str(team.id),
    )
    await emit(
        "team.created",
        organization_id=org_id,
        actor_id=str(current_user.id),
        resource_type="team",
        resource_id=str(team.id),
        payload={"name": team.name, "slug": team.slug},
    )
    return _serialize_team(team)


@router.get("/{team_id}", response_model=TeamResponse)
async def get_team(
    org_id: str,
    team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_access(org_id, team_id, current_user)
    return _serialize_team(team)


@router.patch("/{team_id}", response_model=TeamResponse)
async def update_team(
    org_id: str,
    team_id: str,
    payload: TeamUpdate,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    patch = payload.model_dump(exclude_unset=True)
    updated = await team_repository.update_team(
        team_id, organization_id=org_id, patch=patch
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Team not found")
    await log_activity(
        user_id=str(current_user.id),
        organization_id=org_id,
        activity_type="team.updated",
        details={"team_id": team_id, "fields": list(patch.keys())},
        related_resource_type="team",
        related_resource_id=team_id,
    )
    await emit("team.updated", organization_id=org_id, actor_id=str(current_user.id),
               resource_type="team", resource_id=team_id, payload={"fields": list(patch.keys())})
    return _serialize_team(updated)


@router.delete("/{team_id}", status_code=204)
async def delete_team(
    org_id: str,
    team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_admin(org_id, team_id, current_user)
    if _oid(team.owner_id) != _oid(current_user.id):
        # Only owner (or org admin) can fully delete.
        if not _caller_is_org_admin(team, current_user):
            raise HTTPException(status_code=403, detail="Team owner permission required")
    deleted = await team_repository.delete_team(team_id, organization_id=org_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Team not found")
    await log_activity(
        user_id=str(current_user.id),
        organization_id=org_id,
        activity_type="team.deleted",
        details={"team_id": team_id},
        related_resource_type="team",
        related_resource_id=team_id,
    )
    await emit("team.deleted", organization_id=org_id, actor_id=str(current_user.id),
               resource_type="team", resource_id=team_id)
    return None


@router.post("/{team_id}/archive", response_model=TeamResponse)
async def archive_team(
    org_id: str,
    team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    archived = await team_repository.archive_team(team_id, organization_id=org_id, archived=True)
    if not archived:
        raise HTTPException(status_code=404, detail="Team not found")
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="team.archived",
        details={"team_id": team_id}, related_resource_type="team", related_resource_id=team_id,
    )
    return _serialize_team(archived)


@router.post("/{team_id}/restore", response_model=TeamResponse)
async def restore_team(
    org_id: str,
    team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    restored = await team_repository.archive_team(team_id, organization_id=org_id, archived=False)
    if not restored:
        raise HTTPException(status_code=404, detail="Team not found")
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="team.restored",
        details={"team_id": team_id}, related_resource_type="team", related_resource_id=team_id,
    )
    return _serialize_team(restored)


@router.post("/{team_id}/duplicate", response_model=TeamResponse, status_code=201)
async def duplicate_team(
    org_id: str,
    team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_admin(org_id, team_id, current_user)
    new = await team_repository.create_team(
        TeamCreate(
            name=f"{team.name} (copy)",
            description=team.description,
            department_tag=team.department_tag,
            color=team.color,
            logo_url=team.logo_url,
            cover_url=team.cover_url,
            settings=dict(team.settings or {}),
            metadata=dict(team.metadata or {}),
        ),
        organization_id=org_id,
        creator_id=str(current_user.id),
    )
    return _serialize_team(new)


# ────────────────────────────────────────────────────────── Members


@router.get("/{team_id}/members", response_model=List[TeamMemberResponse])
async def list_team_members(
    org_id: str,
    team_id: str,
    role: Optional[TeamRoleEnum] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_access(org_id, team_id, current_user)
    rows = await team_member_repository.list_for_team(
        team_id=team_id,
        organization_id=org_id,
        role=role,
        skip=skip,
        limit=limit,
    )
    return [
        await _hydrate_member(
            str(r.user_id),
            role=r.role.value if hasattr(r.role, "value") else str(r.role),
            team_id=team_id,
            organization_id=org_id,
            joined_at=r.joined_at,
            invited_by=str(r.invited_by) if r.invited_by else None,
        )
        for r in rows
    ]


@router.post("/{team_id}/members", response_model=TeamMemberResponse, status_code=201)
async def add_team_member(
    org_id: str,
    team_id: str,
    payload: TeamMemberAdd,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    # User must belong to the org already.
    org = await organization_repository.get_by_id(org_id)
    if _oid(payload.user_id) not in [_oid(m) for m in (org.member_ids or [])]:
        raise HTTPException(status_code=400, detail="User is not a member of this organisation")
    await team_repository.add_member(
        team_id,
        organization_id=org_id,
        user_id=payload.user_id,
        promote_to_admin=(payload.role == TeamRoleEnum.TEAM_ADMIN),
    )
    row = await team_member_repository.add_or_update(
        team_id=team_id,
        user_id=payload.user_id,
        organization_id=org_id,
        role=payload.role,
        invited_by=str(current_user.id),
    )
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="team.member_added",
        details={"team_id": team_id, "user_id": payload.user_id, "role": payload.role.value},
        related_resource_type="team", related_resource_id=team_id,
    )
    await emit("team.member_added", organization_id=org_id, actor_id=str(current_user.id),
               resource_type="team", resource_id=team_id, payload={"user_id": payload.user_id, "role": payload.role.value})
    return await _hydrate_member(
        str(row.user_id), role=row.role.value if hasattr(row.role, "value") else str(row.role),
        team_id=team_id, organization_id=org_id,
        joined_at=row.joined_at, invited_by=str(row.invited_by) if row.invited_by else None,
    )


@router.delete("/{team_id}/members/{user_id}", status_code=204)
async def remove_team_member(
    org_id: str,
    team_id: str,
    user_id: str,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_admin(org_id, team_id, current_user)
    if _oid(team.owner_id) == _oid(user_id):
        raise HTTPException(status_code=400, detail="Cannot remove the team owner")
    await team_repository.remove_member(team_id, organization_id=org_id, user_id=user_id)
    await team_member_repository.remove(team_id=team_id, user_id=user_id, organization_id=org_id)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="team.member_removed",
        details={"team_id": team_id, "user_id": user_id},
        related_resource_type="team", related_resource_id=team_id,
    )
    await emit("team.member_removed", organization_id=org_id, actor_id=str(current_user.id),
               resource_type="team", resource_id=team_id, payload={"user_id": user_id})
    return None


@router.patch("/{team_id}/members/{user_id}/role", response_model=TeamMemberResponse)
async def update_team_member_role(
    org_id: str,
    team_id: str,
    user_id: str,
    payload: TeamMemberRoleUpdate,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    if payload.role == TeamRoleEnum.TEAM_ADMIN:
        await team_repository.promote(team_id, organization_id=org_id, user_id=user_id)
    else:
        await team_repository.demote(team_id, organization_id=org_id, user_id=user_id)
    row = await team_member_repository.update_role(
        team_id=team_id, user_id=user_id, organization_id=org_id, role=payload.role,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Member not found")
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="team.member_role_changed",
        details={"team_id": team_id, "user_id": user_id, "role": payload.role.value},
        related_resource_type="team", related_resource_id=team_id,
    )
    return await _hydrate_member(
        str(row.user_id), role=row.role.value if hasattr(row.role, "value") else str(row.role),
        team_id=team_id, organization_id=org_id,
        joined_at=row.joined_at, invited_by=str(row.invited_by) if row.invited_by else None,
    )


class BulkAddMembersPayload(BaseModel):
    members: List[TeamMemberAdd] = Field(..., min_length=1, max_length=200)


@router.post("/{team_id}/members/bulk-add", response_model=List[TeamMemberResponse])
async def bulk_add_team_members(
    org_id: str,
    team_id: str,
    payload: BulkAddMembersPayload,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    org = await organization_repository.get_by_id(org_id)
    org_member_set = {str(_oid(m)) for m in (org.member_ids or [])}
    rows: List[TeamMemberResponse] = []
    for m in payload.members:
        if str(_oid(m.user_id)) not in org_member_set:
            continue
        await team_repository.add_member(
            team_id, organization_id=org_id, user_id=m.user_id,
            promote_to_admin=(m.role == TeamRoleEnum.TEAM_ADMIN),
        )
        row = await team_member_repository.add_or_update(
            team_id=team_id, user_id=m.user_id, organization_id=org_id,
            role=m.role, invited_by=str(current_user.id),
        )
        rows.append(await _hydrate_member(
            str(row.user_id), role=row.role.value if hasattr(row.role, "value") else str(row.role),
            team_id=team_id, organization_id=org_id,
            joined_at=row.joined_at, invited_by=str(row.invited_by) if row.invited_by else None,
        ))
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="team.members_bulk_added",
        details={"team_id": team_id, "count": len(rows)},
        related_resource_type="team", related_resource_id=team_id,
    )
    return rows


@router.post("/{team_id}/leave", status_code=204)
async def leave_team(
    org_id: str,
    team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_access(org_id, team_id, current_user)
    if _oid(team.owner_id) == _oid(current_user.id):
        raise HTTPException(status_code=400, detail="Team owner cannot leave — transfer ownership first")
    await team_repository.remove_member(team_id, organization_id=org_id, user_id=str(current_user.id))
    await team_member_repository.remove(team_id=team_id, user_id=str(current_user.id), organization_id=org_id)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="team.member_left",
        details={"team_id": team_id},
        related_resource_type="team", related_resource_id=team_id,
    )
    return None


class TransferOwnershipPayload(BaseModel):
    new_owner_id: str


@router.post("/{team_id}/transfer-ownership", response_model=TeamResponse)
async def transfer_team_ownership(
    org_id: str,
    team_id: str,
    payload: TransferOwnershipPayload,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_admin(org_id, team_id, current_user)
    if _oid(team.owner_id) != _oid(current_user.id):
        raise HTTPException(status_code=403, detail="Only the current owner can transfer ownership")
    # Make sure target is a member.
    if _oid(payload.new_owner_id) not in [_oid(m) for m in (team.member_ids or [])]:
        raise HTTPException(status_code=400, detail="Target user is not a team member")
    updated = await team_repository.transfer_ownership(
        team_id, organization_id=org_id, new_owner_id=payload.new_owner_id
    )
    # Promote new owner to team admin too.
    await team_repository.promote(team_id, organization_id=org_id, user_id=payload.new_owner_id)
    await team_member_repository.update_role(
        team_id=team_id, user_id=payload.new_owner_id, organization_id=org_id,
        role=TeamRoleEnum.TEAM_ADMIN,
    )
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="team.ownership_transferred",
        details={"team_id": team_id, "new_owner_id": payload.new_owner_id},
        related_resource_type="team", related_resource_id=team_id,
    )
    return _serialize_team(updated)


# ────────────────────────────────────────────────────────── Invites


@router.get("/{team_id}/invites")
async def list_team_invites(
    org_id: str,
    team_id: str,
    status_filter: Optional[InviteStatus] = Query(None, alias="status"),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    invites = await invite_repository.list_by_organization(
        organization_id=org_id, status=status_filter, limit=200,
    )
    # Filter to those scoped at this team (metadata.team_id == team_id).
    out: List[Dict[str, Any]] = []
    for inv in invites:
        meta = (getattr(inv, "metadata", None) or {})
        if str(meta.get("team_id") or "") != str(team_id):
            continue
        d = inv.model_dump(mode="json") if hasattr(inv, "model_dump") else inv.dict()
        d["id"] = str(d.pop("_id", d.get("id")))
        out.append(d)
    return out


class TeamInvitePayload(BaseModel):
    email: EmailStr
    role: InviteRole = InviteRole.MEMBER
    team_role: TeamRoleEnum = TeamRoleEnum.EDITOR
    message: Optional[str] = Field(None, max_length=2000)


@router.post("/{team_id}/invites", status_code=201)
async def create_team_invite(
    org_id: str,
    team_id: str,
    payload: TeamInvitePayload,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_admin(org_id, team_id, current_user)
    try:
        result = await invite_service.send_organization_invite(
            organization_id=org_id,
            emails=[payload.email],
            role=payload.role,
            invited_by=str(current_user.id),
            inviter_name=getattr(current_user, "full_name", "") or "",
            inviter_email=getattr(current_user, "email", "") or "",
            message=payload.message,
            metadata={"team_id": str(team.id), "team_role": payload.team_role.value},
        )
    except PlanCapExceeded as exc:
        raise HTTPException(status_code=402, detail=exc.detail)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="team.invite_sent",
        details={"team_id": team_id, "email": payload.email, "team_role": payload.team_role.value},
        related_resource_type="team", related_resource_id=team_id,
    )
    return {"team_id": team_id, "results": result}


@router.post("/{team_id}/invites/{invite_id}/resend")
async def resend_team_invite(
    org_id: str,
    team_id: str,
    invite_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    inv = await invite_repository.get_by_id(invite_id)
    if not inv or (getattr(inv, "metadata", None) or {}).get("team_id") != str(team_id):
        raise HTTPException(status_code=404, detail="Invite not found")
    await invite_service.resend_invite(invite_id)
    return {"ok": True}


@router.delete("/{team_id}/invites/{invite_id}", status_code=204)
async def revoke_team_invite(
    org_id: str,
    team_id: str,
    invite_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    inv = await invite_repository.get_by_id(invite_id)
    if not inv or (getattr(inv, "metadata", None) or {}).get("team_id") != str(team_id):
        raise HTTPException(status_code=404, detail="Invite not found")
    await invite_repository.mark_revoked(invite_id)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="team.invite_revoked",
        details={"team_id": team_id, "invite_id": invite_id},
        related_resource_type="team", related_resource_id=team_id,
    )
    return None


# ────────────────────────────────────────────────────────── Settings & Branding


@router.get("/{team_id}/settings")
async def get_team_settings(
    org_id: str,
    team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_access(org_id, team_id, current_user)
    return {"settings": team.settings or {}}


@router.patch("/{team_id}/settings")
async def update_team_settings(
    org_id: str,
    team_id: str,
    settings: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    updated = await team_repository.update_team(
        team_id, organization_id=org_id, patch={"settings": settings},
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Team not found")
    return {"settings": updated.settings or {}}


@router.get("/{team_id}/branding")
async def get_team_branding(
    org_id: str,
    team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_access(org_id, team_id, current_user)
    return {"branding": team.branding or {}, "logo_url": team.logo_url, "cover_url": team.cover_url}


@router.patch("/{team_id}/branding")
async def update_team_branding(
    org_id: str,
    team_id: str,
    branding: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    updated = await team_repository.update_team(
        team_id, organization_id=org_id, patch={"branding": branding},
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Team not found")
    return {"branding": updated.branding or {}}


# ────────────────────────────────────────────────────────── Analytics & Activity


@router.get("/{team_id}/analytics")
async def get_team_analytics_overview(
    org_id: str,
    team_id: str,
    time_range: str = Query("30d", description="1d, 7d, 30d, 90d, 1y"),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_access(org_id, team_id, current_user)
    # Aggregates from tasks + agent_runs filtered by org + team.
    # The dedicated analytics_v2.router will replace this in Phase D.
    from backend.db.mongodb.mongodb import MongoDB
    from datetime import timedelta
    days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}.get(time_range, 30)
    since = datetime.utcnow() - timedelta(days=days)
    tasks_col = await MongoDB.get_collection("tasks")
    runs_col = await MongoDB.get_collection("agent_runs")
    org_oid = _oid(org_id)

    task_match: Dict[str, Any] = {"organization_id": org_oid, "metadata.team_id": str(team_id)}
    runs_match: Dict[str, Any] = {"organization_id": org_oid, "metadata.team_id": str(team_id), "started_at": {"$gte": since}}

    tasks_total = await tasks_col.count_documents(task_match)
    tasks_completed = await tasks_col.count_documents({**task_match, "status": "completed"})
    agent_runs_total = await runs_col.count_documents(runs_match)

    return {
        "time_range": time_range,
        "since": since.isoformat() + "Z",
        "tasks": {"total": tasks_total, "completed": tasks_completed,
                  "completion_rate": (tasks_completed / tasks_total) if tasks_total else 0.0},
        "agent_runs": {"total": agent_runs_total},
    }


@router.get("/{team_id}/activity")
async def get_team_activity(
    org_id: str,
    team_id: str,
    limit: int = Query(100, ge=1, le=500),
    skip: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_access(org_id, team_id, current_user)
    # Pull activities where related_resource_type==team & resource_id==team_id
    rows = await activity_repository.list_for_resource(
        organization_id=org_id, resource_type="team", resource_id=team_id,
        limit=limit, skip=skip,
    ) if hasattr(activity_repository, "list_for_resource") else []
    if not rows:
        # Fallback: list recent activity for the org filtered to team metadata.
        from backend.db.mongodb.mongodb import MongoDB
        col = await MongoDB.get_collection("activity_logs")
        cursor = col.find({
            "organization_id": _oid(org_id),
            "$or": [
                {"related_resource_type": "team", "related_resource_id": str(team_id)},
                {"details.team_id": str(team_id)},
            ],
        }).sort("timestamp", -1).skip(skip).limit(limit)
        rows = await cursor.to_list(length=limit)
    out: List[Dict[str, Any]] = []
    for r in rows:
        d = r if isinstance(r, dict) else (r.model_dump(mode="json") if hasattr(r, "model_dump") else r.dict())
        d["id"] = str(d.pop("_id", d.get("id")))
        for k in ("organization_id", "user_id"):
            if d.get(k) is not None:
                d[k] = str(d[k])
        out.append(d)
    return out


# ────────────────────────────────────────────────────────── Permissions probe


@router.get("/{team_id}/permissions")
async def get_team_permissions(
    org_id: str,
    team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_access(org_id, team_id, current_user)
    user_oid = _oid(current_user.id)
    role = await team_member_repository.get_role(team_id=team_id, user_id=str(current_user.id), organization_id=org_id)
    is_admin = user_oid in [_oid(a) for a in (team.admin_ids or [])]
    is_owner = user_oid == _oid(team.owner_id)
    return {
        "team_id": team_id,
        "is_owner": bool(is_owner),
        "is_admin": bool(is_admin or is_owner),
        "role": role,
        "can": {
            "update": bool(is_admin or is_owner),
            "delete": bool(is_owner),
            "invite": bool(is_admin or is_owner),
            "add_member": bool(is_admin or is_owner),
            "remove_member": bool(is_admin or is_owner),
            "manage_settings": bool(is_admin or is_owner),
            "manage_branding": bool(is_admin or is_owner),
        },
    }


# ────────────────────────────────────────────────────────── Announcements (lightweight)


class AnnouncementPayload(BaseModel):
    title: str = Field(..., max_length=200)
    body: str = Field(..., max_length=4000)
    pinned: bool = False


@router.get("/{team_id}/announcements")
async def list_team_announcements(
    org_id: str,
    team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_access(org_id, team_id, current_user)
    return {"announcements": (team.settings or {}).get("announcements", [])}


@router.post("/{team_id}/announcements", status_code=201)
async def create_team_announcement(
    org_id: str,
    team_id: str,
    payload: AnnouncementPayload,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_admin(org_id, team_id, current_user)
    settings = dict(team.settings or {})
    items = list(settings.get("announcements", []))
    new_item = {
        "id": str(ObjectId()),
        "title": payload.title,
        "body": payload.body,
        "pinned": payload.pinned,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "created_by": str(current_user.id),
    }
    items.insert(0, new_item)
    settings["announcements"] = items[:50]
    await team_repository.update_team(team_id, organization_id=org_id, patch={"settings": settings})
    await emit("team.announcement_created", organization_id=org_id, actor_id=str(current_user.id),
               resource_type="team", resource_id=team_id, payload={"id": new_item["id"]})
    return new_item


@router.delete("/{team_id}/announcements/{announcement_id}", status_code=204)
async def delete_team_announcement(
    org_id: str,
    team_id: str,
    announcement_id: str,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_admin(org_id, team_id, current_user)
    settings = dict(team.settings or {})
    items = [a for a in (settings.get("announcements") or []) if a.get("id") != announcement_id]
    settings["announcements"] = items
    await team_repository.update_team(team_id, organization_id=org_id, patch={"settings": settings})
    return None


# ────────────────────────────────────────────────────────── Stub pointers (Phase B/C will deepen)


@router.get("/{team_id}/projects")
async def list_team_projects(
    org_id: str,
    team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """List projects that belong to this team.  Backed by the projects_v2
    collection introduced in Phase A2 — falls back to an empty list when
    no projects have been created against this team yet."""
    await _require_team_access(org_id, team_id, current_user)
    from backend.db.mongodb.mongodb import MongoDB
    col = await MongoDB.get_collection("projects")
    cursor = col.find({
        "organization_id": _oid(org_id), "team_id": _oid(team_id), "is_archived": False,
    }).sort("updated_at", -1).limit(200)
    rows = await cursor.to_list(length=200)
    out: List[Dict[str, Any]] = []
    for r in rows:
        r["id"] = str(r.pop("_id", r.get("id")))
        for k in ("organization_id", "team_id", "created_by", "lead_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
        for k in ("member_ids", "custom_agent_ids", "tag_ids"):
            if isinstance(r.get(k), list):
                r[k] = [str(v) for v in r[k] if v is not None]
        out.append(r)
    return out


@router.get("/{team_id}/agents")
async def list_team_agents(
    org_id: str,
    team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Aggregated list of agents enabled across this team's projects."""
    await _require_team_access(org_id, team_id, current_user)
    from backend.db.mongodb.mongodb import MongoDB
    projects = await MongoDB.get_collection("projects")
    project_ids: List[ObjectId] = []
    async for p in projects.find({"organization_id": _oid(org_id), "team_id": _oid(team_id)}, {"_id": 1}):
        project_ids.append(p["_id"])
    if not project_ids:
        return {"team_id": team_id, "agents": []}
    pa_col = await MongoDB.get_collection("project_agents")
    cursor = pa_col.find({"organization_id": _oid(org_id), "project_id": {"$in": project_ids}})
    agents = await cursor.to_list(length=2000)
    # Dedupe by agent_key/custom_agent_id
    seen = {}
    for a in agents:
        key = a.get("agent_key") or str(a.get("custom_agent_id"))
        if not key or key in seen:
            continue
        seen[key] = {
            "agent_key": a.get("agent_key"),
            "custom_agent_id": str(a["custom_agent_id"]) if a.get("custom_agent_id") else None,
            "autonomy_level": a.get("autonomy_level"),
            "enabled": bool(a.get("enabled", True)),
        }
    return {"team_id": team_id, "agents": list(seen.values())}


@router.get("/{team_id}/tasks")
async def list_team_tasks(
    org_id: str,
    team_id: str,
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_access(org_id, team_id, current_user)
    from backend.db.mongodb.mongodb import MongoDB
    col = await MongoDB.get_collection("tasks")
    query: Dict[str, Any] = {"organization_id": _oid(org_id), "metadata.team_id": str(team_id)}
    if status_filter:
        query["status"] = status_filter
    cursor = col.find(query).sort("updated_at", -1).limit(limit)
    rows = await cursor.to_list(length=limit)
    out: List[Dict[str, Any]] = []
    for r in rows:
        r["id"] = str(r.pop("_id", r.get("id")))
        for k in ("organization_id", "created_by", "assigned_to", "project_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
        out.append(r)
    return out


@router.get("/{team_id}/documents")
async def list_team_documents(
    org_id: str,
    team_id: str,
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_access(org_id, team_id, current_user)
    from backend.db.mongodb.mongodb import MongoDB
    col = await MongoDB.get_collection("documents")
    cursor = col.find({"organization_id": _oid(org_id), "metadata.team_id": str(team_id)}).sort("uploaded_at", -1).limit(limit)
    rows = await cursor.to_list(length=limit)
    for r in rows:
        r["id"] = str(r.pop("_id", r.get("id")))
        for k in ("organization_id", "uploaded_by"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


@router.get("/{team_id}/chat-channels")
async def list_team_chat_channels(
    org_id: str,
    team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_access(org_id, team_id, current_user)
    from backend.db.mongodb.mongodb import MongoDB
    col = await MongoDB.get_collection("chat_channels")
    cursor = col.find({"organization_id": _oid(org_id), "team_id": _oid(team_id)}).sort("last_message_at", -1)
    rows = await cursor.to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id", r.get("id")))
        for k in ("organization_id", "team_id", "project_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


@router.get("/{team_id}/reminders")
async def list_team_reminders(
    org_id: str,
    team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_access(org_id, team_id, current_user)
    from backend.db.mongodb.mongodb import MongoDB
    col = await MongoDB.get_collection("reminders")
    cursor = col.find({"organization_id": _oid(org_id), "resource_type": "team", "resource_id": team_id}).sort("due_at", 1)
    rows = await cursor.to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id", r.get("id")))
        for k in ("organization_id", "user_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows
