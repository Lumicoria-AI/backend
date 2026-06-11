"""
Phase A2 — Projects v2 REST API (org-scoped, normalised tasks).

Mounted under `/api/v1/organizations/{org_id}/projects/` (and re-mounted
under `/teams/{team_id}/projects/` via the workspace surface).

Core surface (CRUD + members + agents + pointers + analytics + settings +
branding).  Advanced extensions (templates, bulk ops, imports, exports,
views, automations) live in `projects_v2_extended.py`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.db.mongodb.repositories.project_v2_repository import project_v2_repository
from backend.db.mongodb.repositories.project_member_repository import project_member_repository
from backend.db.mongodb.repositories.project_agent_repository import project_agent_repository
from backend.db.mongodb.repositories.team_repository import team_repository
from backend.db.mongodb.repositories.user_repository import get_user_repository
from backend.models.user import User
from backend.models.workspace import (
    ProjectAgentAdd,
    ProjectAgentResponse,
    ProjectMemberAdd,
    ProjectMemberResponse,
    ProjectMemberRoleUpdate,
    ProjectRoleEnum,
    ProjectStatus,
    ProjectV2,
    ProjectV2Create,
    ProjectV2Response,
    ProjectV2Update,
    ProjectVisibility,
)
from backend.services.activity_logger import log_activity
from backend.services.billing.plan_caps import PlanCapExceeded
from backend.services.event_bus import emit

logger = structlog.get_logger(__name__)
router = APIRouter()


# ── helpers ──────────────────────────────────────────────────────────


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
    if _oid(org_id) is None:
        raise HTTPException(status_code=400, detail="Invalid organization id")
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(m) for m in (org.member_ids or [])]:
        raise HTTPException(status_code=403, detail="Not a member of this organization")
    return org


async def _require_project_access(
    org_id: str, project_id: str, current_user: User,
) -> ProjectV2:
    await _require_org_membership(org_id, current_user)
    project = await project_v2_repository.get_project(project_id, organization_id=org_id)
    if not project:
        raise HTTPException(status_code=404, detail="Project not found")
    user_oid = _oid(current_user.id)
    member_ids = [_oid(m) for m in (project.member_ids or [])]
    # Project visibility lets org-wide visible projects be read by any org member.
    if project.visibility in (ProjectVisibility.ORG.value, ProjectVisibility.ORG):
        return project
    if user_oid not in member_ids:
        raise HTTPException(status_code=403, detail="Not a project member")
    return project


async def _require_project_role(
    org_id: str,
    project_id: str,
    current_user: User,
    *,
    min_role: ProjectRoleEnum,
) -> ProjectV2:
    project = await _require_project_access(org_id, project_id, current_user)
    role = await project_member_repository.get_role(
        project_id=project_id, user_id=str(current_user.id), organization_id=org_id,
    )
    rank = {ProjectRoleEnum.VIEWER.value: 1, ProjectRoleEnum.REVIEWER.value: 2,
            ProjectRoleEnum.EDITOR.value: 3, ProjectRoleEnum.LEAD.value: 4}
    if not role or rank.get(role, 0) < rank[min_role.value]:
        # Project lead by lead_id also counts.
        if _oid(project.lead_id) != _oid(current_user.id):
            raise HTTPException(status_code=403, detail=f"Project role {min_role.value} or higher required")
    return project


def _serialize_project(p: ProjectV2) -> ProjectV2Response:
    d = p.model_dump(mode="json") if hasattr(p, "model_dump") else p.dict()
    return ProjectV2Response(
        id=str(d.get("_id") or d.get("id")),
        organization_id=str(d["organization_id"]),
        team_id=str(d["team_id"]) if d.get("team_id") else None,
        name=d["name"],
        slug=d["slug"],
        description=d.get("description"),
        status=ProjectStatus(d.get("status", "planning")) if isinstance(d.get("status"), str) else d.get("status", ProjectStatus.PLANNING),
        priority=d.get("priority"),
        color=d.get("color"),
        cover_image_url=d.get("cover_image_url"),
        due_date=d.get("due_date"),
        lead_id=str(d["lead_id"]) if d.get("lead_id") else None,
        member_ids=[str(v) for v in (d.get("member_ids") or [])],
        agent_keys=list(d.get("agent_keys") or []),
        custom_agent_ids=[str(v) for v in (d.get("custom_agent_ids") or [])],
        tag_ids=[str(v) for v in (d.get("tag_ids") or [])],
        strict_mode=bool(d.get("strict_mode")),
        visibility=ProjectVisibility(d.get("visibility", "private")) if isinstance(d.get("visibility"), str) else d.get("visibility", ProjectVisibility.PRIVATE),
        settings=d.get("settings") or {},
        branding=d.get("branding") or {},
        metadata=d.get("metadata") or {},
        is_archived=bool(d.get("is_archived")),
        archived_at=d.get("archived_at"),
        created_by=str(d.get("created_by")) if d.get("created_by") else "",
        created_at=d.get("created_at"),
        updated_at=d.get("updated_at"),
    )


def _serialize_agent(a) -> ProjectAgentResponse:
    d = a.model_dump(mode="json") if hasattr(a, "model_dump") else a.dict()
    return ProjectAgentResponse(
        id=str(d.get("_id") or d.get("id")),
        project_id=str(d["project_id"]),
        organization_id=str(d["organization_id"]),
        agent_key=d.get("agent_key"),
        custom_agent_id=str(d["custom_agent_id"]) if d.get("custom_agent_id") else None,
        enabled=bool(d.get("enabled", True)),
        autonomy_level=d.get("autonomy_level") or "suggest",
        config_overrides=d.get("config_overrides") or {},
        fallback_chain=list(d.get("fallback_chain") or []),
        attached_by=str(d.get("attached_by")) if d.get("attached_by") else "",
        created_at=d.get("created_at") or datetime.utcnow(),
        updated_at=d.get("updated_at") or datetime.utcnow(),
    )


async def _hydrate_member(
    user_id: str, *, role: str, project_id: str, organization_id: str,
    joined_at: datetime, invited_by: Optional[str] = None,
) -> ProjectMemberResponse:
    repo = await get_user_repository()
    user = None
    try:
        user = await repo.get_user_by_id(user_id)
    except Exception:
        user = None
    return ProjectMemberResponse(
        project_id=str(project_id), user_id=str(user_id),
        organization_id=str(organization_id),
        role=ProjectRoleEnum(role) if role in ProjectRoleEnum._value2member_map_ else ProjectRoleEnum.EDITOR,
        joined_at=joined_at or datetime.utcnow(),
        invited_by=invited_by,
        full_name=getattr(user, "full_name", None) if user else None,
        email=getattr(user, "email", None) if user else None,
        avatar_url=getattr(user, "avatar_url", None) if user else None,
    )


# ── Project CRUD ─────────────────────────────────────────────────────


@router.get("", response_model=List[ProjectV2Response])
async def list_projects(
    org_id: str,
    team_id: Optional[str] = Query(None),
    status: Optional[ProjectStatus] = Query(None),
    only_mine: bool = Query(False),
    include_archived: bool = Query(False),
    search: Optional[str] = Query(None, max_length=200),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_membership(org_id, current_user)
    projects = await project_v2_repository.list_projects(
        organization_id=org_id,
        team_id=team_id,
        member_id=str(current_user.id) if only_mine else None,
        status=status.value if status else None,
        include_archived=include_archived,
        search=search, skip=skip, limit=limit,
    )
    return [_serialize_project(p) for p in projects]


@router.post("", response_model=ProjectV2Response, status_code=201)
async def create_project(
    org_id: str,
    payload: ProjectV2Create,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_membership(org_id, current_user)
    # Team must exist + caller must be a member if team_id supplied
    if payload.team_id:
        team = await team_repository.get_team(payload.team_id, organization_id=org_id)
        if not team:
            raise HTTPException(status_code=400, detail="Unknown team_id")
        if _oid(current_user.id) not in [_oid(m) for m in (team.member_ids or [])]:
            raise HTTPException(status_code=403, detail="Not a member of the target team")
    try:
        project = await project_v2_repository.create_project(
            payload, organization_id=org_id, creator_id=str(current_user.id),
        )
    except PlanCapExceeded as exc:
        raise HTTPException(status_code=402, detail=exc.detail)
    # Mirror creator into normalised project_members.
    await project_member_repository.add_or_update(
        project_id=str(project.id), user_id=str(current_user.id),
        organization_id=org_id, role=ProjectRoleEnum.LEAD,
    )
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="project.created",
        details={"project_id": str(project.id), "name": project.name, "team_id": payload.team_id},
        related_resource_type="project", related_resource_id=str(project.id),
    )
    await emit(
        "project.created", organization_id=org_id, actor_id=str(current_user.id),
        team_id=payload.team_id, project_id=str(project.id),
        resource_type="project", resource_id=str(project.id),
        payload={"name": project.name, "slug": project.slug},
    )
    return _serialize_project(project)


@router.get("/{project_id}", response_model=ProjectV2Response)
async def get_project(
    org_id: str,
    project_id: str,
    current_user: User = Depends(get_current_active_user),
):
    project = await _require_project_access(org_id, project_id, current_user)
    return _serialize_project(project)


@router.patch("/{project_id}", response_model=ProjectV2Response)
async def update_project(
    org_id: str,
    project_id: str,
    payload: ProjectV2Update,
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.EDITOR)
    patch = payload.model_dump(exclude_unset=True)
    # Coerce enums to values
    for k in ("status", "visibility"):
        v = patch.get(k)
        if v is not None and hasattr(v, "value"):
            patch[k] = v.value
    updated = await project_v2_repository.update_project(
        project_id, organization_id=org_id, patch=patch,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="project.updated",
        details={"project_id": project_id, "fields": list(patch.keys())},
        related_resource_type="project", related_resource_id=project_id,
    )
    await emit("project.updated", organization_id=org_id, actor_id=str(current_user.id),
               project_id=project_id, resource_type="project", resource_id=project_id,
               payload={"fields": list(patch.keys())})
    return _serialize_project(updated)


@router.delete("/{project_id}", status_code=204)
async def delete_project(
    org_id: str,
    project_id: str,
    current_user: User = Depends(get_current_active_user),
):
    project = await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.LEAD)
    deleted = await project_v2_repository.delete_project(project_id, organization_id=org_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Project not found")
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="project.deleted",
        details={"project_id": project_id, "name": project.name},
        related_resource_type="project", related_resource_id=project_id,
    )
    await emit("project.deleted", organization_id=org_id, actor_id=str(current_user.id),
               project_id=project_id, resource_type="project", resource_id=project_id)
    return None


@router.post("/{project_id}/archive", response_model=ProjectV2Response)
async def archive_project(
    org_id: str, project_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.LEAD)
    updated = await project_v2_repository.archive_project(project_id, organization_id=org_id, archived=True)
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="project.archived",
        details={"project_id": project_id},
        related_resource_type="project", related_resource_id=project_id,
    )
    return _serialize_project(updated)


@router.post("/{project_id}/restore", response_model=ProjectV2Response)
async def restore_project(
    org_id: str, project_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.LEAD)
    updated = await project_v2_repository.archive_project(project_id, organization_id=org_id, archived=False)
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")
    return _serialize_project(updated)


@router.post("/{project_id}/duplicate", response_model=ProjectV2Response, status_code=201)
async def duplicate_project(
    org_id: str, project_id: str,
    current_user: User = Depends(get_current_active_user),
):
    src = await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.EDITOR)
    payload = ProjectV2Create(
        name=f"{src.name} (copy)",
        description=src.description,
        status=ProjectStatus.PLANNING,
        priority=src.priority,
        color=src.color,
        cover_image_url=src.cover_image_url,
        team_id=str(src.team_id) if src.team_id else None,
        lead_id=str(src.lead_id) if src.lead_id else None,
        member_ids=[str(m) for m in (src.member_ids or [])],
        agent_keys=list(src.agent_keys or []),
        visibility=src.visibility,
        strict_mode=bool(src.strict_mode),
        settings=dict(src.settings or {}),
        metadata=dict(src.metadata or {}),
    )
    new = await project_v2_repository.create_project(
        payload, organization_id=org_id, creator_id=str(current_user.id),
    )
    return _serialize_project(new)


class TransferTeamPayload(BaseModel):
    team_id: Optional[str] = None  # None = move to org-level


@router.post("/{project_id}/transfer-to-team", response_model=ProjectV2Response)
async def transfer_project_to_team(
    org_id: str, project_id: str,
    payload: TransferTeamPayload,
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.LEAD)
    if payload.team_id:
        team = await team_repository.get_team(payload.team_id, organization_id=org_id)
        if not team:
            raise HTTPException(status_code=400, detail="Unknown team_id")
    updated = await project_v2_repository.update_project(
        project_id, organization_id=org_id, patch={"team_id": payload.team_id},
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")
    return _serialize_project(updated)


# ── Project members ──────────────────────────────────────────────────


@router.get("/{project_id}/members", response_model=List[ProjectMemberResponse])
async def list_project_members(
    org_id: str, project_id: str,
    role: Optional[ProjectRoleEnum] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(200, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_access(org_id, project_id, current_user)
    rows = await project_member_repository.list_for_project(
        project_id=project_id, organization_id=org_id, role=role, skip=skip, limit=limit,
    )
    return [
        await _hydrate_member(
            str(r.user_id),
            role=r.role.value if hasattr(r.role, "value") else str(r.role),
            project_id=project_id, organization_id=org_id,
            joined_at=r.joined_at, invited_by=str(r.invited_by) if r.invited_by else None,
        )
        for r in rows
    ]


@router.post("/{project_id}/members", response_model=ProjectMemberResponse, status_code=201)
async def add_project_member(
    org_id: str, project_id: str,
    payload: ProjectMemberAdd,
    current_user: User = Depends(get_current_active_user),
):
    project = await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.LEAD)
    org = await organization_repository.get_by_id(org_id)
    if _oid(payload.user_id) not in [_oid(m) for m in (org.member_ids or [])]:
        raise HTTPException(status_code=400, detail="User is not a member of this organisation")
    await project_v2_repository.add_member(project_id, organization_id=org_id, user_id=payload.user_id)
    row = await project_member_repository.add_or_update(
        project_id=project_id, user_id=payload.user_id, organization_id=org_id,
        role=payload.role, invited_by=str(current_user.id),
    )
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="project.member_added",
        details={"project_id": project_id, "user_id": payload.user_id, "role": payload.role.value},
        related_resource_type="project", related_resource_id=project_id,
    )
    await emit("project.member_added", organization_id=org_id, actor_id=str(current_user.id),
               project_id=project_id, resource_type="project", resource_id=project_id,
               payload={"user_id": payload.user_id, "role": payload.role.value})
    return await _hydrate_member(
        str(row.user_id), role=row.role.value if hasattr(row.role, "value") else str(row.role),
        project_id=project_id, organization_id=org_id,
        joined_at=row.joined_at, invited_by=str(row.invited_by) if row.invited_by else None,
    )


@router.delete("/{project_id}/members/{user_id}", status_code=204)
async def remove_project_member(
    org_id: str, project_id: str, user_id: str,
    current_user: User = Depends(get_current_active_user),
):
    project = await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.LEAD)
    if _oid(project.lead_id) == _oid(user_id):
        raise HTTPException(status_code=400, detail="Cannot remove the project lead")
    await project_v2_repository.remove_member(project_id, organization_id=org_id, user_id=user_id)
    await project_member_repository.remove(project_id=project_id, user_id=user_id, organization_id=org_id)
    return None


@router.patch("/{project_id}/members/{user_id}/role", response_model=ProjectMemberResponse)
async def update_project_member_role(
    org_id: str, project_id: str, user_id: str,
    payload: ProjectMemberRoleUpdate,
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.LEAD)
    row = await project_member_repository.update_role(
        project_id=project_id, user_id=user_id, organization_id=org_id, role=payload.role,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Member not found")
    return await _hydrate_member(
        str(row.user_id), role=row.role.value if hasattr(row.role, "value") else str(row.role),
        project_id=project_id, organization_id=org_id,
        joined_at=row.joined_at, invited_by=str(row.invited_by) if row.invited_by else None,
    )


# ── Project agents ───────────────────────────────────────────────────


@router.get("/{project_id}/agents", response_model=List[ProjectAgentResponse])
async def list_project_agents(
    org_id: str, project_id: str,
    enabled_only: bool = Query(False),
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_access(org_id, project_id, current_user)
    rows = await project_agent_repository.list_for_project(
        project_id=project_id, organization_id=org_id, enabled_only=enabled_only,
    )
    return [_serialize_agent(r) for r in rows]


@router.post("/{project_id}/agents", response_model=ProjectAgentResponse, status_code=201)
async def attach_project_agent(
    org_id: str, project_id: str,
    payload: ProjectAgentAdd,
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.EDITOR)
    row = await project_agent_repository.attach(
        project_id=project_id, organization_id=org_id, payload=payload,
        actor_id=str(current_user.id),
    )
    # Mirror into project.agent_keys / custom_agent_ids
    if payload.agent_key:
        await project_v2_repository.attach_agent_key(
            project_id, organization_id=org_id, agent_key=payload.agent_key,
        )
    elif payload.custom_agent_id:
        await project_v2_repository.attach_custom_agent(
            project_id, organization_id=org_id, custom_agent_id=payload.custom_agent_id,
        )
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="project.agent_attached",
        details={"project_id": project_id, "agent_key": payload.agent_key,
                 "custom_agent_id": payload.custom_agent_id},
        related_resource_type="project", related_resource_id=project_id,
    )
    return _serialize_agent(row)


@router.delete("/{project_id}/agents/{agent_ref}", status_code=204)
async def detach_project_agent(
    org_id: str, project_id: str, agent_ref: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.EDITOR)
    # Resolve whether agent_ref is platform key or custom agent ObjectId
    existing = await project_agent_repository.get_by_ref(
        project_id=project_id, organization_id=org_id, agent_ref=agent_ref,
    )
    if not existing:
        raise HTTPException(status_code=404, detail="Agent not attached to this project")
    if existing.agent_key:
        await project_agent_repository.detach(
            project_id=project_id, organization_id=org_id, agent_key=existing.agent_key,
        )
        await project_v2_repository.detach_agent_key(
            project_id, organization_id=org_id, agent_key=existing.agent_key,
        )
    else:
        await project_agent_repository.detach(
            project_id=project_id, organization_id=org_id,
            custom_agent_id=str(existing.custom_agent_id),
        )
        await project_v2_repository.detach_custom_agent(
            project_id, organization_id=org_id, custom_agent_id=str(existing.custom_agent_id),
        )
    return None


class ProjectAgentConfigPatch(BaseModel):
    enabled: Optional[bool] = None
    autonomy_level: Optional[str] = None
    config_overrides: Optional[Dict[str, Any]] = None
    fallback_chain: Optional[List[str]] = None


@router.patch("/{project_id}/agents/{agent_ref}/config", response_model=ProjectAgentResponse)
async def update_project_agent_config(
    org_id: str, project_id: str, agent_ref: str,
    payload: ProjectAgentConfigPatch,
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.EDITOR)
    patch = payload.model_dump(exclude_unset=True)
    updated = await project_agent_repository.update(
        project_id=project_id, organization_id=org_id, agent_ref=agent_ref, patch=patch,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Agent not attached to this project")
    return _serialize_agent(updated)


# ── Settings & branding ──────────────────────────────────────────────


@router.get("/{project_id}/settings")
async def get_project_settings(
    org_id: str, project_id: str,
    current_user: User = Depends(get_current_active_user),
):
    project = await _require_project_access(org_id, project_id, current_user)
    return {"settings": project.settings or {}}


@router.patch("/{project_id}/settings")
async def update_project_settings(
    org_id: str, project_id: str,
    settings: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.LEAD)
    updated = await project_v2_repository.update_project(
        project_id, organization_id=org_id, patch={"settings": settings},
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"settings": updated.settings or {}}


@router.post("/{project_id}/strict-mode/enable", response_model=ProjectV2Response)
async def enable_project_strict_mode(
    org_id: str, project_id: str,
    current_user: User = Depends(get_current_active_user),
):
    # Strict mode requires Business+ plan caps; the cap check is best-effort
    # for now and will be replaced by the proper permission gate in Phase E.
    await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.LEAD)
    updated = await project_v2_repository.update_project(
        project_id, organization_id=org_id, patch={"strict_mode": True},
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")
    return _serialize_project(updated)


@router.post("/{project_id}/strict-mode/disable", response_model=ProjectV2Response)
async def disable_project_strict_mode(
    org_id: str, project_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.LEAD)
    updated = await project_v2_repository.update_project(
        project_id, organization_id=org_id, patch={"strict_mode": False},
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")
    return _serialize_project(updated)


@router.get("/{project_id}/branding")
async def get_project_branding(
    org_id: str, project_id: str,
    current_user: User = Depends(get_current_active_user),
):
    project = await _require_project_access(org_id, project_id, current_user)
    return {
        "branding": project.branding or {},
        "color": project.color,
        "cover_image_url": project.cover_image_url,
    }


@router.patch("/{project_id}/branding")
async def update_project_branding(
    org_id: str, project_id: str,
    branding: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_role(org_id, project_id, current_user, min_role=ProjectRoleEnum.LEAD)
    updated = await project_v2_repository.update_project(
        project_id, organization_id=org_id, patch={"branding": branding},
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Project not found")
    return {"branding": updated.branding or {}}


# ── Resource pointers (Tasks / Documents / Chat / KB) ────────────────


@router.get("/{project_id}/tasks")
async def list_project_tasks(
    org_id: str, project_id: str,
    status_filter: Optional[str] = Query(None, alias="status"),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_access(org_id, project_id, current_user)
    from backend.db.mongodb.mongodb import MongoDB
    col = await MongoDB.get_collection("tasks")
    query: Dict[str, Any] = {
        "organization_id": _oid(org_id),
        "project_id": _oid(project_id),
    }
    if status_filter:
        query["status"] = status_filter
    cursor = col.find(query).sort("updated_at", -1).skip(skip).limit(limit)
    rows = await cursor.to_list(length=limit)
    for r in rows:
        r["id"] = str(r.pop("_id", r.get("id")))
        for k in ("organization_id", "project_id", "created_by", "assigned_to"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


@router.get("/{project_id}/documents")
async def list_project_documents(
    org_id: str, project_id: str,
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_access(org_id, project_id, current_user)
    from backend.db.mongodb.mongodb import MongoDB
    col = await MongoDB.get_collection("documents")
    cursor = col.find({
        "organization_id": _oid(org_id),
        "$or": [
            {"project_id": _oid(project_id)},
            {"metadata.project_id": str(project_id)},
        ],
    }).sort("uploaded_at", -1).limit(limit)
    rows = await cursor.to_list(length=limit)
    for r in rows:
        r["id"] = str(r.pop("_id", r.get("id")))
        for k in ("organization_id", "uploaded_by", "project_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


@router.get("/{project_id}/chat-channels")
async def list_project_chat_channels(
    org_id: str, project_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_access(org_id, project_id, current_user)
    from backend.db.mongodb.mongodb import MongoDB
    col = await MongoDB.get_collection("chat_channels")
    cursor = col.find({
        "organization_id": _oid(org_id),
        "project_id": _oid(project_id),
    }).sort("last_message_at", -1)
    rows = await cursor.to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id", r.get("id")))
        for k in ("organization_id", "team_id", "project_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
    return rows


# ── Activity & analytics ─────────────────────────────────────────────


@router.get("/{project_id}/activity")
async def get_project_activity(
    org_id: str, project_id: str,
    limit: int = Query(100, ge=1, le=500),
    skip: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_access(org_id, project_id, current_user)
    from backend.db.mongodb.mongodb import MongoDB
    from backend.db.serializers import stringify_rows
    col = await MongoDB.get_collection("activity_logs")
    cursor = col.find({
        "organization_id": _oid(org_id),
        "$or": [
            {"related_resource_type": "project", "related_resource_id": str(project_id)},
            {"details.project_id": str(project_id)},
        ],
    }).sort("timestamp", -1).skip(skip).limit(limit)
    rows = await cursor.to_list(length=limit)
    return stringify_rows(rows)


@router.get("/{project_id}/analytics")
async def get_project_analytics(
    org_id: str, project_id: str,
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
):
    await _require_project_access(org_id, project_id, current_user)
    days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}.get(time_range, 30)
    since = datetime.utcnow() - timedelta(days=days)
    from backend.db.mongodb.mongodb import MongoDB
    tasks_col = await MongoDB.get_collection("tasks")
    runs_col = await MongoDB.get_collection("agent_runs")
    docs_col = await MongoDB.get_collection("documents")
    org_oid = _oid(org_id)
    proj_oid = _oid(project_id)

    tasks_total = await tasks_col.count_documents({"organization_id": org_oid, "project_id": proj_oid})
    tasks_completed = await tasks_col.count_documents({"organization_id": org_oid, "project_id": proj_oid, "status": "completed"})
    tasks_overdue = await tasks_col.count_documents({
        "organization_id": org_oid, "project_id": proj_oid,
        "status": {"$nin": ["completed", "cancelled", "archived"]},
        "due_date": {"$lt": datetime.utcnow()},
    })
    runs_total = await runs_col.count_documents({
        "organization_id": org_oid, "project_id": proj_oid,
        "started_at": {"$gte": since},
    })
    docs_total = await docs_col.count_documents({
        "organization_id": org_oid,
        "$or": [
            {"project_id": proj_oid},
            {"metadata.project_id": str(project_id)},
        ],
    })

    return {
        "time_range": time_range,
        "since": since.isoformat() + "Z",
        "tasks": {
            "total": tasks_total,
            "completed": tasks_completed,
            "overdue": tasks_overdue,
            "completion_rate": (tasks_completed / tasks_total) if tasks_total else 0.0,
        },
        "agent_runs": {"total": runs_total},
        "documents": {"total": docs_total},
    }


@router.get("/{project_id}/permissions")
async def get_project_permissions(
    org_id: str, project_id: str,
    current_user: User = Depends(get_current_active_user),
):
    project = await _require_project_access(org_id, project_id, current_user)
    role = await project_member_repository.get_role(
        project_id=project_id, user_id=str(current_user.id), organization_id=org_id,
    )
    is_lead = _oid(project.lead_id) == _oid(current_user.id) or role == ProjectRoleEnum.LEAD.value
    rank = {ProjectRoleEnum.VIEWER.value: 1, ProjectRoleEnum.REVIEWER.value: 2,
            ProjectRoleEnum.EDITOR.value: 3, ProjectRoleEnum.LEAD.value: 4}
    role_rank = rank.get(role, 0)
    return {
        "project_id": project_id,
        "is_lead": bool(is_lead),
        "role": role,
        "can": {
            "read": True,
            "update": bool(role_rank >= 3 or is_lead),
            "delete": bool(is_lead),
            "manage_members": bool(is_lead),
            "manage_agents": bool(role_rank >= 3 or is_lead),
            "run_agents": bool(role_rank >= 3 or is_lead),
            "create_task": bool(role_rank >= 3 or is_lead),
            "upload_doc": bool(role_rank >= 3 or is_lead),
            "manage_settings": bool(is_lead),
            "manage_strict_mode": bool(is_lead),
        },
    }
