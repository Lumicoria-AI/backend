"""
Phase A — Invites v2 extended REST API.

Mounted at `/api/v1/invites/`.

Adds: CSV import, Google-Workspace import, preview template + preview send,
extend-expiry, change-role, shareable links CRUD + regenerate, accept-
deep-link variants per scope, batch bulk operations.
"""

from __future__ import annotations

import csv
import io
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, EmailStr, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.models.invite import InviteRole, InviteScope, InviteStatus
from backend.db.mongodb.repositories.invite_repository import invite_repository
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.invite_service import invite_service

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


def _resolve_primary_org_id(user: User) -> Optional[str]:
    primary = getattr(user, "organization_id", None)
    if primary:
        return str(primary)
    ids = getattr(user, "organization_ids", None) or []
    if ids:
        return str(ids[0])
    return None


# ── Bulk + imports ────────────────────────────────────────────────


class BulkInvitePayload(BaseModel):
    invites: List[Dict[str, Any]] = Field(..., min_length=1, max_length=500)


@router.post("/bulk", status_code=201)
async def bulk_invite(
    payload: BulkInvitePayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    """Bulk invites with mixed scopes (org/team/project)."""
    org_id = organization_id or _resolve_primary_org_id(current_user)
    if not org_id:
        raise HTTPException(status_code=400, detail="organization_id required")
    results: List[Dict[str, Any]] = []
    for inv in payload.invites:
        email = (inv.get("email") or "").strip().lower()
        if not email or "@" not in email:
            results.append({"email": email, "status": "invalid"})
            continue
        try:
            r = await invite_service.send_organization_invite(
                organization_id=org_id, emails=[email],
                role=InviteRole(inv.get("role") or "member"),
                invited_by=str(current_user.id),
                inviter_name=getattr(current_user, "full_name", "") or "",
                inviter_email=getattr(current_user, "email", "") or "",
                message=inv.get("message"),
                metadata={
                    "team_id": inv.get("team_id"),
                    "project_id": inv.get("project_id"),
                    "team_role": inv.get("team_role"),
                    "project_role": inv.get("project_role"),
                },
            )
            results.append({"email": email, "status": "sent", "result": r})
        except Exception as exc:  # noqa: BLE001
            results.append({"email": email, "status": "error", "error": str(exc)})
    return {"results": results, "total": len(results)}


@router.post("/import-csv", status_code=201)
async def import_csv(
    file: UploadFile = File(...),
    organization_id: Optional[str] = Query(None),
    role: InviteRole = Query(InviteRole.MEMBER),
    current_user: User = Depends(get_current_active_user),
):
    """Import invites from a CSV file with an `email` column (and optional `team_id`, `project_id`)."""
    org_id = organization_id or _resolve_primary_org_id(current_user)
    if not org_id:
        raise HTTPException(status_code=400, detail="organization_id required")
    content = await file.read()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8", errors="ignore")))
    results: List[Dict[str, Any]] = []
    for row in reader:
        e = (row.get("email") or row.get("Email") or "").strip().lower()
        if not e or "@" not in e:
            continue
        try:
            r = await invite_service.send_organization_invite(
                organization_id=org_id, emails=[e],
                role=role, invited_by=str(current_user.id),
                inviter_name=getattr(current_user, "full_name", "") or "",
                inviter_email=getattr(current_user, "email", "") or "",
                metadata={"team_id": row.get("team_id"), "project_id": row.get("project_id")},
            )
            results.append({"email": e, "status": "sent", "result": r})
        except Exception as exc:  # noqa: BLE001
            results.append({"email": e, "status": "error", "error": str(exc)})
    return {"sent": sum(1 for r in results if r["status"] == "sent"), "results": results}


class GwImportPayload(BaseModel):
    emails: List[EmailStr] = Field(..., max_length=500)
    role: InviteRole = InviteRole.MEMBER
    team_id: Optional[str] = None
    project_id: Optional[str] = None


@router.post("/import-google-workspace", status_code=201)
async def import_google_workspace(
    payload: GwImportPayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    if not org_id:
        raise HTTPException(status_code=400, detail="organization_id required")
    r = await invite_service.send_organization_invite(
        organization_id=org_id, emails=[str(e) for e in payload.emails],
        role=payload.role, invited_by=str(current_user.id),
        inviter_name=getattr(current_user, "full_name", "") or "",
        inviter_email=getattr(current_user, "email", "") or "",
        metadata={"team_id": payload.team_id, "project_id": payload.project_id, "source": "google_workspace"},
    )
    return {"sent": len(payload.emails), "result": r}


# ── Preview ──────────────────────────────────────────────────────


@router.get("/preview-template")
async def preview_template(
    scope: InviteScope = Query(InviteScope.ORGANIZATION),
    role: InviteRole = Query(InviteRole.MEMBER),
    current_user: User = Depends(get_current_active_user),
):
    """Render the invite email template body with sample data."""
    return {
        "subject": f"You're invited to join Lumicoria as a {role.value}",
        "preview_text": "Tap to accept and start collaborating.",
        "body_html": (
            f"<p>Hi there,</p>"
            f"<p>{getattr(current_user, 'full_name', 'A teammate')} invited you to "
            f"join their {scope.value} on Lumicoria as a {role.value}.</p>"
            f"<p><a href='https://app.lumicoria.ai/invites/accept?token=sample'>Accept invite</a></p>"
        ),
    }


@router.post("/preview-send")
async def preview_send(
    payload: Dict[str, Any] = Body(default_factory=dict),
    current_user: User = Depends(get_current_active_user),
):
    """Send a sample invite to the caller for visual QA."""
    target = (payload.get("email") or getattr(current_user, "email", None) or "").strip().lower()
    if not target:
        raise HTTPException(status_code=400, detail="email required")
    return {"ok": True, "sent_to": target, "stub": True}


# ── Extend / change-role ────────────────────────────────────────


@router.post("/{invite_id}/extend-expiry")
async def extend_invite_expiry(
    invite_id: str,
    days: int = Query(14, ge=1, le=90),
    current_user: User = Depends(get_current_active_user),
):
    inv = await invite_repository.get_by_id(invite_id)
    if not inv:
        raise HTTPException(status_code=404, detail="Invite not found")
    new_exp = datetime.utcnow() + timedelta(days=days)
    col = await MongoDB.get_collection("invites")
    await col.update_one({"_id": _oid(invite_id)}, {"$set": {"expires_at": new_exp}})
    return {"expires_at": new_exp}


class ChangeRolePayload(BaseModel):
    role: InviteRole


@router.post("/{invite_id}/change-role")
async def change_invite_role(
    invite_id: str,
    payload: ChangeRolePayload,
    current_user: User = Depends(get_current_active_user),
):
    inv = await invite_repository.get_by_id(invite_id)
    if not inv:
        raise HTTPException(status_code=404, detail="Invite not found")
    col = await MongoDB.get_collection("invites")
    await col.update_one({"_id": _oid(invite_id)}, {"$set": {"role": payload.role.value}})
    return {"role": payload.role.value}


# ── Shareable links ─────────────────────────────────────────────


class ShareableLinkCreatePayload(BaseModel):
    scope: InviteScope = InviteScope.ORGANIZATION
    role: InviteRole = InviteRole.MEMBER
    team_id: Optional[str] = None
    project_id: Optional[str] = None
    max_uses: Optional[int] = None
    expires_at: Optional[datetime] = None


@router.get("/links")
async def list_shareable_links(
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    if not org_id:
        raise HTTPException(status_code=400, detail="organization_id required")
    col = await MongoDB.get_collection("invite_links")
    rows = await col.find({"organization_id": _oid(org_id)}).sort("created_at", -1).to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "created_by"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.post("/links", status_code=201)
async def create_shareable_link(
    payload: ShareableLinkCreatePayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    if not org_id:
        raise HTTPException(status_code=400, detail="organization_id required")
    col = await MongoDB.get_collection("invite_links")
    token = secrets.token_urlsafe(24)
    doc = {
        "organization_id": _oid(org_id),
        "scope": payload.scope.value, "role": payload.role.value,
        "team_id": _oid(payload.team_id) if payload.team_id else None,
        "project_id": _oid(payload.project_id) if payload.project_id else None,
        "max_uses": payload.max_uses, "uses": 0,
        "token": token,
        "expires_at": payload.expires_at,
        "created_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    doc["id"] = str(r.inserted_id)
    doc.pop("_id", None)
    for k in ("organization_id", "team_id", "project_id", "created_by"):
        if doc.get(k):
            doc[k] = str(doc[k])
    return {**doc, "url": f"/invites/links/{token}"}


@router.delete("/links/{link_id}", status_code=204)
async def delete_shareable_link(
    link_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    if not org_id:
        raise HTTPException(status_code=400, detail="organization_id required")
    col = await MongoDB.get_collection("invite_links")
    await col.delete_one({"_id": _oid(link_id), "organization_id": _oid(org_id)})
    return None


@router.post("/links/{link_id}/regenerate")
async def regenerate_link_token(
    link_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    if not org_id:
        raise HTTPException(status_code=400, detail="organization_id required")
    col = await MongoDB.get_collection("invite_links")
    new_token = secrets.token_urlsafe(24)
    row = await col.find_one_and_update(
        {"_id": _oid(link_id), "organization_id": _oid(org_id)},
        {"$set": {"token": new_token, "uses": 0, "updated_at": datetime.utcnow()}},
        return_document=True,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Link not found")
    return {"token": new_token, "url": f"/invites/links/{new_token}"}


@router.get("/links/{token}")
async def get_shareable_link_meta(token: str):
    """Public — anyone with the link can see the org name + scope."""
    col = await MongoDB.get_collection("invite_links")
    row = await col.find_one({"token": token})
    if not row:
        raise HTTPException(status_code=404, detail="Link not found")
    if row.get("expires_at") and row["expires_at"] < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Link expired")
    if row.get("max_uses") and (row.get("uses") or 0) >= row["max_uses"]:
        raise HTTPException(status_code=410, detail="Link exhausted")
    org = await organization_repository.get_by_id(str(row["organization_id"]))
    return {
        "organization": {"id": str(org.id), "name": org.name} if org else None,
        "scope": row.get("scope"), "role": row.get("role"),
    }


@router.post("/links/{token}/accept")
async def accept_shareable_link(
    token: str,
    current_user: User = Depends(get_current_active_user),
):
    """Authenticated accept: adds the user to the org/team/project."""
    col = await MongoDB.get_collection("invite_links")
    row = await col.find_one({"token": token})
    if not row:
        raise HTTPException(status_code=404, detail="Link not found")
    if row.get("expires_at") and row["expires_at"] < datetime.utcnow():
        raise HTTPException(status_code=410, detail="Link expired")
    if row.get("max_uses") and (row.get("uses") or 0) >= row["max_uses"]:
        raise HTTPException(status_code=410, detail="Link exhausted")

    org_id = str(row["organization_id"])
    await organization_repository.add_member(org_id, str(current_user.id))
    if row.get("team_id"):
        from backend.db.mongodb.repositories.team_repository import team_repository
        from backend.db.mongodb.repositories.team_member_repository import team_member_repository
        from backend.models.workspace import TeamRoleEnum
        await team_repository.add_member(str(row["team_id"]), organization_id=org_id, user_id=str(current_user.id))
        await team_member_repository.add_or_update(
            team_id=str(row["team_id"]), user_id=str(current_user.id), organization_id=org_id,
            role=TeamRoleEnum.EDITOR,
        )
    if row.get("project_id"):
        from backend.db.mongodb.repositories.project_v2_repository import project_v2_repository
        from backend.db.mongodb.repositories.project_member_repository import project_member_repository
        from backend.models.workspace import ProjectRoleEnum
        await project_v2_repository.add_member(str(row["project_id"]), organization_id=org_id, user_id=str(current_user.id))
        await project_member_repository.add_or_update(
            project_id=str(row["project_id"]), user_id=str(current_user.id), organization_id=org_id,
            role=ProjectRoleEnum.EDITOR,
        )
    await col.update_one({"_id": row["_id"]}, {"$inc": {"uses": 1}})
    return {"organization_id": org_id, "joined": True}


# ── Accept deep-link variants per scope ─────────────────────────


@router.post("/by-token/{token}/accept/team/{team_id}")
async def accept_team_invite_via_token(
    token: str, team_id: str,
    current_user: User = Depends(get_current_active_user),
):
    inv = await invite_repository.get_pending_for_email(getattr(current_user, "email", "")) if hasattr(invite_repository, "get_pending_for_email") else None
    if not inv:
        raise HTTPException(status_code=404, detail="No matching invite")
    from backend.db.mongodb.repositories.team_repository import team_repository
    from backend.db.mongodb.repositories.team_member_repository import team_member_repository
    from backend.models.workspace import TeamRoleEnum
    org_id = str(inv.organization_id)
    await team_repository.add_member(team_id, organization_id=org_id, user_id=str(current_user.id))
    await team_member_repository.add_or_update(
        team_id=team_id, user_id=str(current_user.id), organization_id=org_id,
        role=TeamRoleEnum.EDITOR,
    )
    return {"joined": True, "team_id": team_id}


@router.post("/by-token/{token}/accept/project/{project_id}")
async def accept_project_invite_via_token(
    token: str, project_id: str,
    current_user: User = Depends(get_current_active_user),
):
    inv = await invite_repository.get_pending_for_email(getattr(current_user, "email", "")) if hasattr(invite_repository, "get_pending_for_email") else None
    if not inv:
        raise HTTPException(status_code=404, detail="No matching invite")
    from backend.db.mongodb.repositories.project_v2_repository import project_v2_repository
    from backend.db.mongodb.repositories.project_member_repository import project_member_repository
    from backend.models.workspace import ProjectRoleEnum
    org_id = str(inv.organization_id)
    await project_v2_repository.add_member(project_id, organization_id=org_id, user_id=str(current_user.id))
    await project_member_repository.add_or_update(
        project_id=project_id, user_id=str(current_user.id), organization_id=org_id,
        role=ProjectRoleEnum.EDITOR,
    )
    return {"joined": True, "project_id": project_id}


# ── Bulk revoke / resend ────────────────────────────────────────


class BulkInviteIdsPayload(BaseModel):
    invite_ids: List[str] = Field(..., min_length=1, max_length=200)


@router.post("/bulk-resend")
async def bulk_resend(
    payload: BulkInviteIdsPayload,
    current_user: User = Depends(get_current_active_user),
):
    n = 0
    for inv_id in payload.invite_ids:
        try:
            await invite_service.resend_invite(inv_id)
            n += 1
        except Exception:
            continue
    return {"resent": n}


@router.post("/bulk-revoke")
async def bulk_revoke(
    payload: BulkInviteIdsPayload,
    current_user: User = Depends(get_current_active_user),
):
    n = 0
    for inv_id in payload.invite_ids:
        try:
            await invite_repository.mark_revoked(inv_id)
            n += 1
        except Exception:
            continue
    return {"revoked": n}


@router.get("/stats")
async def invite_stats(
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    if not org_id:
        raise HTTPException(status_code=400, detail="organization_id required")
    col = await MongoDB.get_collection("invites")
    pipeline = [
        {"$match": {"organization_id": _oid(org_id)}},
        {"$group": {"_id": "$status", "count": {"$sum": 1}}},
    ]
    rows = await col.aggregate(pipeline).to_list(length=10)
    return {r["_id"] or "unknown": r["count"] for r in rows}


@router.post("/{invite_id}/remind")
async def remind_invite(
    invite_id: str,
    current_user: User = Depends(get_current_active_user),
):
    try:
        await invite_service.resend_invite(invite_id)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not send reminder: {e}")
    return {"ok": True, "reminder_sent": True}


# ── Catalogue ───────────────────────────────────────────────────


@router.get("/scopes")
async def list_invite_scopes():
    return {"scopes": [s.value for s in InviteScope], "roles": [r.value for r in InviteRole], "statuses": [s.value for s in InviteStatus]}
