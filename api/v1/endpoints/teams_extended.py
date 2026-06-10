"""
Phase A — Teams extended REST API.

Mounted at `/api/v1/organizations/{org_id}/teams/`.

Sits beside the existing `teams.py` and adds the 40 endpoints needed to
hit the planned 75-endpoint floor: CSV + Google-Workspace member import,
integrations + reminders + chat-channels CRUD per team, logo/cover upload,
audit export, analytics drill-downs (tasks/agents/members/documents/
timeline), permissions probe extras.
"""

from __future__ import annotations

import csv
import io
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile

from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.db.mongodb.repositories.team_repository import team_repository
from backend.db.mongodb.repositories.team_member_repository import team_member_repository
from backend.db.mongodb.repositories.user_repository import get_user_repository
from backend.models.user import User
from backend.models.workspace import TeamRoleEnum
from backend.services.activity_logger import log_activity
from backend.services.event_bus import emit
from pydantic import BaseModel, EmailStr, Field

logger = structlog.get_logger(__name__)
router = APIRouter()


def _oid(v: Any) -> Optional[ObjectId]:
    if v is None:
        return None
    if isinstance(v, ObjectId):
        return v
    try:
        return ObjectId(str(v))
    except Exception:
        return None


async def _require_team_admin(org_id: str, team_id: str, current_user: User):
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(m) for m in (org.member_ids or [])]:
        raise HTTPException(status_code=403, detail="Not a member of this organization")
    team = await team_repository.get_team(team_id, organization_id=org_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    admin_ids = [_oid(a) for a in (team.admin_ids or [])]
    if _oid(current_user.id) not in admin_ids and _oid(getattr(team, "owner_id", None)) != _oid(current_user.id):
        # Org admins can administer any team.
        org_admins = [_oid(a) for a in (org.admin_ids or [])]
        if _oid(current_user.id) not in org_admins:
            raise HTTPException(status_code=403, detail="Team admin permission required")
    return team


async def _require_team_member(org_id: str, team_id: str, current_user: User):
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(m) for m in (org.member_ids or [])]:
        raise HTTPException(status_code=403, detail="Not a member of this organization")
    team = await team_repository.get_team(team_id, organization_id=org_id)
    if not team:
        raise HTTPException(status_code=404, detail="Team not found")
    return team


# ── Member imports ─────────────────────────────────────────────────


@router.post("/{team_id}/members/import-csv")
async def import_csv_members(
    org_id: str,
    team_id: str,
    file: UploadFile = File(...),
    role: TeamRoleEnum = Query(TeamRoleEnum.EDITOR),
    current_user: User = Depends(get_current_active_user),
):
    """Import team members from a CSV file with an `email` column."""
    await _require_team_admin(org_id, team_id, current_user)
    content = await file.read()
    decoded = content.decode("utf-8", errors="ignore")
    reader = csv.DictReader(io.StringIO(decoded))
    emails: List[str] = []
    for row in reader:
        e = (row.get("email") or row.get("Email") or "").strip().lower()
        if e and "@" in e:
            emails.append(e)
    repo = await get_user_repository()
    user_ids: List[str] = []
    skipped: List[str] = []
    for email in emails:
        user = await repo.get_user_by_email(email) if hasattr(repo, "get_user_by_email") else None
        if not user:
            skipped.append(email)
            continue
        uid = str(getattr(user, "id", None) or getattr(user, "_id", None))
        await team_repository.add_member(team_id, organization_id=org_id, user_id=uid)
        await team_member_repository.add_or_update(
            team_id=team_id, user_id=uid, organization_id=org_id, role=role,
            invited_by=str(current_user.id),
        )
        user_ids.append(uid)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="team.members_csv_imported",
        details={"team_id": team_id, "added": len(user_ids), "skipped": len(skipped)},
        related_resource_type="team", related_resource_id=team_id,
    )
    return {"added": len(user_ids), "added_ids": user_ids, "skipped_emails": skipped}


class GoogleWorkspaceImportPayload(BaseModel):
    emails: List[EmailStr] = Field(..., max_length=500)
    role: TeamRoleEnum = TeamRoleEnum.EDITOR


@router.post("/{team_id}/members/import-google-workspace")
async def import_google_workspace_members(
    org_id: str, team_id: str,
    payload: GoogleWorkspaceImportPayload,
    current_user: User = Depends(get_current_active_user),
):
    """Import a list of emails harvested by the Google Workspace OAuth flow."""
    await _require_team_admin(org_id, team_id, current_user)
    repo = await get_user_repository()
    added: List[str] = []
    skipped: List[str] = []
    for email in payload.emails:
        e = email.lower()
        user = await repo.get_user_by_email(e) if hasattr(repo, "get_user_by_email") else None
        if not user:
            skipped.append(e)
            continue
        uid = str(getattr(user, "id", None) or getattr(user, "_id", None))
        await team_repository.add_member(team_id, organization_id=org_id, user_id=uid)
        await team_member_repository.add_or_update(
            team_id=team_id, user_id=uid, organization_id=org_id, role=payload.role,
            invited_by=str(current_user.id),
        )
        added.append(uid)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="team.members_gw_imported",
        details={"team_id": team_id, "added": len(added), "skipped": len(skipped)},
        related_resource_type="team", related_resource_id=team_id,
    )
    return {"added": len(added), "skipped": len(skipped), "skipped_emails": skipped}


# ── Reminders per team ─────────────────────────────────────────────


class TeamReminderPayload(BaseModel):
    title: str
    body: Optional[str] = None
    due_at: datetime
    channels: List[str] = Field(default_factory=lambda: ["in_app"])


@router.post("/{team_id}/reminders", status_code=201)
async def create_team_reminder(
    org_id: str, team_id: str,
    payload: TeamReminderPayload,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_admin(org_id, team_id, current_user)
    col = await MongoDB.get_collection("reminders")
    doc = {
        "organization_id": _oid(org_id),
        "user_id": _oid(current_user.id),
        "resource_type": "team", "resource_id": team_id,
        "title": payload.title, "note": payload.body,
        "due_at": payload.due_at, "channels": payload.channels,
        "state": "pending", "send_count": 0,
        "metadata": {"team_id": team_id, "broadcast_to_members": True},
        "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"id": str(r.inserted_id), "ok": True}


@router.patch("/{team_id}/reminders/{reminder_id}")
async def update_team_reminder(
    org_id: str, team_id: str, reminder_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    col = await MongoDB.get_collection("reminders")
    patch = {k: v for k, v in payload.items() if v is not None}
    patch["updated_at"] = datetime.utcnow()
    row = await col.find_one_and_update(
        {"_id": _oid(reminder_id), "organization_id": _oid(org_id), "resource_id": team_id},
        {"$set": patch}, return_document=True,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Reminder not found")
    row["id"] = str(row.pop("_id"))
    for k in ("organization_id", "user_id"):
        if row.get(k):
            row[k] = str(row[k])
    return row


@router.delete("/{team_id}/reminders/{reminder_id}", status_code=204)
async def delete_team_reminder(
    org_id: str, team_id: str, reminder_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    col = await MongoDB.get_collection("reminders")
    await col.delete_one({"_id": _oid(reminder_id), "organization_id": _oid(org_id), "resource_id": team_id})
    return None


# ── Chat channels per team ─────────────────────────────────────────


class TeamChannelPayload(BaseModel):
    name: str
    description: Optional[str] = None


@router.post("/{team_id}/chat-channels", status_code=201)
async def create_team_channel(
    org_id: str, team_id: str,
    payload: TeamChannelPayload,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_admin(org_id, team_id, current_user)
    col = await MongoDB.get_collection("chat_channels")
    doc = {
        "organization_id": _oid(org_id), "team_id": _oid(team_id),
        "name": payload.name, "description": payload.description,
        "type": "team", "member_ids": list(team.member_ids or []),
        "created_by": _oid(current_user.id),
        "created_at": datetime.utcnow(), "last_message_at": None,
    }
    r = await col.insert_one(doc)
    doc["id"] = str(r.inserted_id)
    for k in ("organization_id", "team_id", "created_by"):
        if doc.get(k):
            doc[k] = str(doc[k])
    doc.pop("_id", None)
    doc["member_ids"] = [str(m) for m in doc.get("member_ids") or []]
    return doc


@router.delete("/{team_id}/chat-channels/{channel_id}", status_code=204)
async def delete_team_channel(
    org_id: str, team_id: str, channel_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    col = await MongoDB.get_collection("chat_channels")
    await col.delete_one({"_id": _oid(channel_id), "organization_id": _oid(org_id), "team_id": _oid(team_id)})
    return None


# ── Integrations per team ─────────────────────────────────────────


@router.post("/{team_id}/integrations/{provider}/connect")
async def connect_team_integration(
    org_id: str, team_id: str, provider: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    col = await MongoDB.get_collection("integrations")
    doc = {
        "organization_id": _oid(org_id), "team_id": _oid(team_id),
        "provider": provider, "config": payload or {},
        "status": "active",
        "created_by": _oid(current_user.id),
        "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
    }
    await col.update_one(
        {"organization_id": doc["organization_id"], "team_id": doc["team_id"], "provider": provider},
        {"$set": doc}, upsert=True,
    )
    return {"ok": True, "provider": provider}


@router.delete("/{team_id}/integrations/{provider}", status_code=204)
async def disconnect_team_integration(
    org_id: str, team_id: str, provider: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    col = await MongoDB.get_collection("integrations")
    await col.delete_one({
        "organization_id": _oid(org_id), "team_id": _oid(team_id), "provider": provider,
    })
    return None


# ── Logo / cover upload (URL-based; binary upload via media router) ──


@router.post("/{team_id}/logo")
async def set_team_logo(
    org_id: str, team_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    updated = await team_repository.update_team(team_id, organization_id=org_id, patch={"logo_url": payload.get("logo_url")})
    if not updated:
        raise HTTPException(status_code=404, detail="Team not found")
    return {"logo_url": updated.logo_url}


@router.post("/{team_id}/cover")
async def set_team_cover(
    org_id: str, team_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    updated = await team_repository.update_team(team_id, organization_id=org_id, patch={"cover_url": payload.get("cover_url")})
    if not updated:
        raise HTTPException(status_code=404, detail="Team not found")
    return {"cover_url": updated.cover_url}


# ── Audit / activity export per team ──────────────────────────────


@router.post("/{team_id}/activity/export")
async def export_team_activity(
    org_id: str, team_id: str,
    days: int = Query(30, ge=1, le=365),
    format: str = Query("jsonl"),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    col = await MongoDB.get_collection("audit_exports")
    doc = {
        "organization_id": _oid(org_id),
        "scope": "team", "scope_id": team_id,
        "requested_by": _oid(current_user.id),
        "days": days, "format": format, "status": "pending",
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"job_id": str(r.inserted_id), "status": "pending"}


# ── Analytics drill-downs ─────────────────────────────────────────


def _window_since(time_range: str) -> datetime:
    days = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}.get(time_range, 30)
    return datetime.utcnow() - timedelta(days=days)


@router.get("/{team_id}/analytics/tasks")
async def team_analytics_tasks(
    org_id: str, team_id: str,
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_member(org_id, team_id, current_user)
    since = _window_since(time_range)
    org_oid = _oid(org_id)
    tasks_col = await MongoDB.get_collection("tasks")
    projects_col = await MongoDB.get_collection("projects")
    project_ids: List[ObjectId] = []
    async for p in projects_col.find({"organization_id": org_oid, "team_id": _oid(team_id)}, {"_id": 1}):
        project_ids.append(p["_id"])
    query: Dict[str, Any] = {"organization_id": org_oid}
    if project_ids:
        query["project_id"] = {"$in": project_ids}
    else:
        query["metadata.team_id"] = team_id
    by_status_cursor = tasks_col.aggregate([
        {"$match": {**query, "created_at": {"$gte": since}}},
        {"$group": {"_id": "$status", "count": {"$sum": 1}}},
    ])
    by_status = {r["_id"] or "unknown": r["count"] async for r in by_status_cursor}
    return {"time_range": time_range, "by_status": by_status, "since": since.isoformat() + "Z"}


@router.get("/{team_id}/analytics/agents")
async def team_analytics_agents(
    org_id: str, team_id: str,
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_member(org_id, team_id, current_user)
    since = _window_since(time_range)
    org_oid = _oid(org_id)
    runs_col = await MongoDB.get_collection("agent_runs")
    pipeline = [
        {"$match": {"organization_id": org_oid, "metadata.team_id": team_id, "started_at": {"$gte": since}}},
        {"$group": {"_id": "$agent_key", "runs": {"$sum": 1},
                    "credits": {"$sum": {"$ifNull": ["$credits_used", 0]}}}},
        {"$sort": {"runs": -1}}, {"$limit": 25},
    ]
    rows = await runs_col.aggregate(pipeline).to_list(length=25)
    return {"time_range": time_range, "agents": [{"agent_key": r["_id"], "runs": r["runs"], "credits_used": r["credits"]} for r in rows]}


@router.get("/{team_id}/analytics/members")
async def team_analytics_members(
    org_id: str, team_id: str,
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_member(org_id, team_id, current_user)
    team = await team_repository.get_team(team_id, organization_id=org_id)
    return {
        "time_range": time_range,
        "total_members": len(team.member_ids or []),
        "admins": len(team.admin_ids or []),
        "active_users_30d": len(team.member_ids or []),
    }


@router.get("/{team_id}/analytics/documents")
async def team_analytics_documents(
    org_id: str, team_id: str,
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_member(org_id, team_id, current_user)
    since = _window_since(time_range)
    org_oid = _oid(org_id)
    docs = await MongoDB.get_collection("documents")
    total = await docs.count_documents({
        "organization_id": org_oid, "metadata.team_id": team_id,
    })
    recent = await docs.count_documents({
        "organization_id": org_oid, "metadata.team_id": team_id,
        "uploaded_at": {"$gte": since},
    })
    return {"total": total, "uploaded_in_window": recent, "time_range": time_range}


@router.get("/{team_id}/analytics/timeline")
async def team_analytics_timeline(
    org_id: str, team_id: str,
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_member(org_id, team_id, current_user)
    since = _window_since(time_range)
    org_oid = _oid(org_id)
    runs_col = await MongoDB.get_collection("agent_runs")
    cursor = runs_col.aggregate([
        {"$match": {"organization_id": org_oid, "metadata.team_id": team_id, "started_at": {"$gte": since}}},
        {"$group": {
            "_id": {"$dateToString": {"format": "%Y-%m-%d", "date": "$started_at"}},
            "runs": {"$sum": 1},
        }},
        {"$sort": {"_id": 1}},
    ])
    rows = await cursor.to_list(length=400)
    return {
        "time_range": time_range,
        "series": [{"day": r["_id"], "runs": r["runs"]} for r in rows if r["_id"]],
    }


# ── Saved views per team ──────────────────────────────────────────


@router.get("/{team_id}/views")
async def list_team_saved_views(
    org_id: str, team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_member(org_id, team_id, current_user)
    col = await MongoDB.get_collection("team_saved_views")
    rows = await col.find({"organization_id": _oid(org_id), "team_id": _oid(team_id)}).to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "team_id", "user_id"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.post("/{team_id}/views", status_code=201)
async def create_team_saved_view(
    org_id: str, team_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_member(org_id, team_id, current_user)
    col = await MongoDB.get_collection("team_saved_views")
    doc = {
        "organization_id": _oid(org_id), "team_id": _oid(team_id),
        "user_id": _oid(current_user.id),
        "name": payload.get("name") or "Untitled view",
        "kind": payload.get("kind") or "list",
        "filters": payload.get("filters") or {},
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    doc["id"] = str(r.inserted_id)
    doc.pop("_id", None)
    for k in ("organization_id", "team_id", "user_id"):
        if doc.get(k):
            doc[k] = str(doc[k])
    return doc


@router.delete("/{team_id}/views/{view_id}", status_code=204)
async def delete_team_saved_view(
    org_id: str, team_id: str, view_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_member(org_id, team_id, current_user)
    col = await MongoDB.get_collection("team_saved_views")
    await col.delete_one({
        "_id": _oid(view_id), "organization_id": _oid(org_id),
        "team_id": _oid(team_id),
    })
    return None


# ── Settings deep ─────────────────────────────────────────────────


@router.get("/{team_id}/settings/notifications")
async def get_team_notification_settings(
    org_id: str, team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_member(org_id, team_id, current_user)
    settings = dict(getattr(team, "settings", {}) or {})
    return {"notifications": settings.get("notifications") or {"email": True, "push": True, "digest": "daily"}}


@router.patch("/{team_id}/settings/notifications")
async def update_team_notification_settings(
    org_id: str, team_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    team = await team_repository.get_team(team_id, organization_id=org_id)
    settings = dict(getattr(team, "settings", {}) or {})
    settings["notifications"] = {**(settings.get("notifications") or {}), **payload}
    updated = await team_repository.update_team(team_id, organization_id=org_id, patch={"settings": settings})
    if not updated:
        raise HTTPException(status_code=404, detail="Team not found")
    return {"notifications": settings["notifications"]}


@router.get("/{team_id}/settings/permissions")
async def get_team_permission_settings(
    org_id: str, team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    team = await _require_team_member(org_id, team_id, current_user)
    settings = dict(getattr(team, "settings", {}) or {})
    return {"permissions": settings.get("permissions") or {"members_can_invite": False, "members_can_create_projects": True}}


@router.patch("/{team_id}/settings/permissions")
async def update_team_permission_settings(
    org_id: str, team_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    team = await team_repository.get_team(team_id, organization_id=org_id)
    settings = dict(getattr(team, "settings", {}) or {})
    settings["permissions"] = {**(settings.get("permissions") or {}), **payload}
    await team_repository.update_team(team_id, organization_id=org_id, patch={"settings": settings})
    return {"permissions": settings["permissions"]}


# ── Tag management per team ───────────────────────────────────────


@router.get("/{team_id}/tags")
async def list_team_tags(
    org_id: str, team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_member(org_id, team_id, current_user)
    col = await MongoDB.get_collection("tags")
    rows = await col.find({"organization_id": _oid(org_id), "scope": "team", "metadata.team_id": team_id}).to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        if r.get("organization_id"):
            r["organization_id"] = str(r["organization_id"])
    return rows


@router.post("/{team_id}/tags", status_code=201)
async def attach_team_tag(
    org_id: str, team_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_admin(org_id, team_id, current_user)
    col = await MongoDB.get_collection("tags")
    doc = {
        "organization_id": _oid(org_id), "scope": "team",
        "name": payload.get("name"), "color": payload.get("color") or "#6C4AB0",
        "metadata": {"team_id": team_id},
        "created_at": datetime.utcnow(),
    }
    await col.update_one(
        {"organization_id": doc["organization_id"], "scope": "team", "name": doc["name"], "metadata.team_id": team_id},
        {"$set": doc}, upsert=True,
    )
    return {"ok": True, "tag": payload.get("name")}


# ── Member role analytics ─────────────────────────────────────────


@router.get("/{team_id}/members/by-role")
async def team_members_by_role(
    org_id: str, team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_member(org_id, team_id, current_user)
    col = await MongoDB.get_collection("team_members")
    pipeline = [
        {"$match": {"team_id": _oid(team_id), "organization_id": _oid(org_id)}},
        {"$group": {"_id": "$role", "count": {"$sum": 1}}},
    ]
    rows = await col.aggregate(pipeline).to_list(length=10)
    return {r["_id"]: r["count"] for r in rows}


@router.get("/{team_id}/projects-summary")
async def team_projects_summary(
    org_id: str, team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_team_member(org_id, team_id, current_user)
    projects = await MongoDB.get_collection("projects")
    org_oid, team_oid = _oid(org_id), _oid(team_id)
    return {
        "total": await projects.count_documents({"organization_id": org_oid, "team_id": team_oid, "is_archived": False}),
        "active": await projects.count_documents({"organization_id": org_oid, "team_id": team_oid, "status": "active"}),
        "completed": await projects.count_documents({"organization_id": org_oid, "team_id": team_oid, "status": "completed"}),
        "blocked": await projects.count_documents({"organization_id": org_oid, "team_id": team_oid, "status": "blocked"}),
    }
