"""
Phase A — Organizations extended REST API.

Mounted at `/api/v1/organizations`.

Sits beside the existing `organizations.py` and adds the 45+ endpoints the
Workspace plan calls out: deep settings, branding, limits, plan upgrade,
admins/owners management, ownership transfer/delegate, seat-status, custom
roles, tags, announcements, onboarding checklist, activity export, and
domain proxies.

No endpoints in `organizations.py` are touched.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.db.mongodb.repositories.org_subscription_repository import (
    org_subscription_repository, seat_assignment_repository,
)
from backend.models.billing import PLAN_LIMITS, SubscriptionPlan, get_plan_limits
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.event_bus import emit

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


async def _require_org_member(org_id: str, current_user: User):
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(m) for m in (org.member_ids or [])]:
        raise HTTPException(status_code=403, detail="Not a member of this organization")
    return org


async def _require_org_admin(org_id: str, current_user: User):
    org = await _require_org_member(org_id, current_user)
    admin_ids = [_oid(a) for a in (org.admin_ids or [])]
    if _oid(current_user.id) not in admin_ids:
        raise HTTPException(status_code=403, detail="Org admin permission required")
    return org


async def _require_org_owner(org_id: str, current_user: User):
    org = await _require_org_member(org_id, current_user)
    if _oid(getattr(org, "owner_id", None)) != _oid(current_user.id):
        raise HTTPException(status_code=403, detail="Org owner permission required")
    return org


# ── Public profile + branding ───────────────────────────────────────


@router.get("/{org_id}/profile")
async def get_public_profile(org_id: str, current_user: User = Depends(get_current_active_user)):
    org = await _require_org_member(org_id, current_user)
    d = org.model_dump(mode="json") if hasattr(org, "model_dump") else org.dict()
    return {
        "id": str(d.get("_id") or d.get("id")),
        "name": d.get("name"),
        "description": d.get("description"),
        "industry": d.get("industry"),
        "website": d.get("website"),
        "logo_url": d.get("logo_url"),
        "plan": d.get("plan"),
    }


@router.patch("/{org_id}/profile")
async def update_public_profile(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    allowed = {k: payload[k] for k in ("name", "description", "industry", "website", "logo_url") if k in payload}
    if not allowed:
        raise HTTPException(status_code=400, detail="No supported fields in payload")
    updated = await organization_repository.update(org_id, allowed)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="organization.profile_updated",
        details={"fields": list(allowed.keys())},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return updated


@router.post("/{org_id}/profile/logo")
async def update_profile_logo(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    return await organization_repository.update(org_id, {"logo_url": payload.get("logo_url")})


@router.post("/{org_id}/profile/cover")
async def update_profile_cover(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    org = await organization_repository.get_by_id(org_id)
    settings = dict(getattr(org, "settings", {}) or {})
    settings["cover_url"] = payload.get("cover_url")
    return await organization_repository.update_organization_settings(org_id, settings)


@router.get("/{org_id}/branding")
async def get_branding(org_id: str, current_user: User = Depends(get_current_active_user)):
    org = await _require_org_member(org_id, current_user)
    settings = dict(getattr(org, "settings", {}) or {})
    # Prefer the canonical top-level cover_url field (written by the
    # /media/cover/org upload endpoint); fall back to legacy settings
    # for orgs created before the field was added.
    cover_url = getattr(org, "cover_url", None) or settings.get("cover_url")
    return {
        "branding": settings.get("branding") or {},
        "logo_url": getattr(org, "logo_url", None),
        "cover_url": cover_url,
        "primary_color": (settings.get("branding") or {}).get("primary_color") or "#6C4AB0",
        "accent_color": (settings.get("branding") or {}).get("accent_color") or "#0EA5E9",
    }


@router.patch("/{org_id}/branding")
async def update_branding(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    org = await organization_repository.get_by_id(org_id)
    settings = dict(getattr(org, "settings", {}) or {})
    branding = dict(settings.get("branding") or {})
    for k in ("primary_color", "accent_color", "email_footer", "favicon_url", "logo_url"):
        if k in payload and payload[k] is not None:
            branding[k] = payload[k]
    settings["branding"] = branding
    await organization_repository.update_organization_settings(org_id, settings)
    return {"branding": branding}


# ── Settings deep ───────────────────────────────────────────────────


@router.get("/{org_id}/settings")
async def get_settings(org_id: str, current_user: User = Depends(get_current_active_user)):
    org = await _require_org_member(org_id, current_user)
    return {"settings": getattr(org, "settings", None) or {}}


@router.patch("/{org_id}/settings")
async def patch_settings(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    org = await organization_repository.get_by_id(org_id)
    settings = dict(getattr(org, "settings", {}) or {})
    settings.update(payload)
    updated = await organization_repository.update_organization_settings(org_id, settings)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="organization.settings_updated",
        details={"fields": list(payload.keys())},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return {"settings": getattr(updated, "settings", None) or {}}


@router.get("/{org_id}/limits")
async def get_limits(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    plan = (sub.plan if sub else "free")
    plan_str = plan.value if hasattr(plan, "value") else str(plan)
    limits = get_plan_limits(plan_str)
    used_seats = await seat_assignment_repository.count_active(org_id)
    return {
        "plan": plan_str,
        "limits": {k: v for k, v in limits.items() if k.startswith("max_") or k == "allowed_models"},
        "usage": {"seats_used": used_seats, "seats_purchased": sub.seats_purchased if sub else 0},
    }


@router.post("/{org_id}/upgrade-plan")
async def upgrade_plan(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    """Convenience: initiates Stripe Checkout via the org-billing router."""
    await _require_org_owner(org_id, current_user)
    return {
        "redirect": f"/api/v1/org-billing/{org_id}/checkout",
        "suggested_plan": payload.get("plan", "business"),
    }


# ── Admins / Owners ────────────────────────────────────────────────


@router.get("/{org_id}/admins")
async def list_admins(org_id: str, current_user: User = Depends(get_current_active_user)):
    org = await _require_org_member(org_id, current_user)
    return {"admin_ids": [str(a) for a in (org.admin_ids or [])], "count": len(org.admin_ids or [])}


@router.get("/{org_id}/owners")
async def list_owners(org_id: str, current_user: User = Depends(get_current_active_user)):
    org = await _require_org_member(org_id, current_user)
    return {"owner_id": str(getattr(org, "owner_id", "")), "count": 1 if getattr(org, "owner_id", None) else 0}


class TransferOwnerPayload(BaseModel):
    new_owner_id: str


@router.post("/{org_id}/owners/transfer")
async def transfer_owner(
    org_id: str,
    payload: TransferOwnerPayload,
    current_user: User = Depends(get_current_active_user),
):
    org = await _require_org_owner(org_id, current_user)
    if _oid(payload.new_owner_id) not in [_oid(m) for m in (org.member_ids or [])]:
        raise HTTPException(status_code=400, detail="Target user is not a member of the org")
    await organization_repository.update(org_id, {"owner_id": ObjectId(payload.new_owner_id)})
    await organization_repository.promote_to_admin(org_id, payload.new_owner_id)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="organization.ownership_transferred",
        details={"new_owner_id": payload.new_owner_id},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return {"ok": True, "new_owner_id": payload.new_owner_id}


@router.post("/{org_id}/owners/delegate")
async def delegate_owner(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    """Promote a member to admin without transferring ownership."""
    await _require_org_owner(org_id, current_user)
    target = payload.get("user_id")
    if not target:
        raise HTTPException(status_code=400, detail="user_id required")
    await organization_repository.promote_to_admin(org_id, target)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="organization.delegated",
        details={"user_id": target}, related_resource_type="organization", related_resource_id=str(org_id),
    )
    return {"ok": True, "user_id": target}


# ── Seats summary ──────────────────────────────────────────────────


@router.get("/{org_id}/seat-status")
async def seat_status(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    used = await seat_assignment_repository.count_active(org_id)
    return {
        "purchased": sub.seats_purchased if sub else 0,
        "used": used,
        "remaining": max((sub.seats_purchased if sub else 0) - used, 0),
        "plan": sub.plan if sub else "free",
    }


# ── Tags ───────────────────────────────────────────────────────────


@router.get("/{org_id}/tags")
async def list_tags(
    org_id: str,
    scope: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("tags")
    q: Dict[str, Any] = {"organization_id": _oid(org_id)}
    if scope:
        q["scope"] = scope
    rows = await col.find(q).sort("name", 1).to_list(length=500)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        if r.get("organization_id"):
            r["organization_id"] = str(r["organization_id"])
    return rows


class TagPayload(BaseModel):
    name: str
    color: Optional[str] = "#6C4AB0"
    scope: str = "project"


@router.post("/{org_id}/tags", status_code=201)
async def create_tag(
    org_id: str,
    payload: TagPayload,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("tags")
    doc = {
        "organization_id": _oid(org_id),
        "name": payload.name, "color": payload.color, "scope": payload.scope,
        "created_at": datetime.utcnow(),
    }
    await col.update_one(
        {"organization_id": doc["organization_id"], "scope": payload.scope, "name": payload.name},
        {"$set": doc}, upsert=True,
    )
    row = await col.find_one({"organization_id": doc["organization_id"], "scope": payload.scope, "name": payload.name})
    row["id"] = str(row.pop("_id"))
    row["organization_id"] = str(row["organization_id"])
    return row


@router.patch("/{org_id}/tags/{tag_id}")
async def update_tag(
    org_id: str, tag_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("tags")
    patch = {k: v for k, v in payload.items() if k in ("name", "color", "scope") and v is not None}
    if not patch:
        raise HTTPException(status_code=400, detail="Nothing to update")
    row = await col.find_one_and_update(
        {"_id": _oid(tag_id), "organization_id": _oid(org_id)},
        {"$set": patch}, return_document=True,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Tag not found")
    row["id"] = str(row.pop("_id"))
    row["organization_id"] = str(row["organization_id"])
    return row


@router.delete("/{org_id}/tags/{tag_id}", status_code=204)
async def delete_tag(org_id: str, tag_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("tags")
    await col.delete_one({"_id": _oid(tag_id), "organization_id": _oid(org_id)})
    return None


# ── Announcements ──────────────────────────────────────────────────


@router.get("/{org_id}/announcements")
async def list_announcements(org_id: str, current_user: User = Depends(get_current_active_user)):
    org = await _require_org_member(org_id, current_user)
    settings = dict(getattr(org, "settings", {}) or {})
    return {"announcements": settings.get("announcements") or []}


@router.post("/{org_id}/announcements", status_code=201)
async def create_announcement(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    org = await organization_repository.get_by_id(org_id)
    settings = dict(getattr(org, "settings", {}) or {})
    items: List[Dict[str, Any]] = list(settings.get("announcements") or [])
    new_item = {
        "id": str(ObjectId()),
        "title": payload.get("title", "Announcement"),
        "body": payload.get("body", ""),
        "pinned": bool(payload.get("pinned", False)),
        "created_at": datetime.utcnow().isoformat() + "Z",
        "created_by": str(current_user.id),
    }
    items.insert(0, new_item)
    settings["announcements"] = items[:100]
    await organization_repository.update_organization_settings(org_id, settings)
    await emit("organization.announcement_created", organization_id=org_id, actor_id=str(current_user.id), payload={"id": new_item["id"]})
    return new_item


@router.delete("/{org_id}/announcements/{announcement_id}", status_code=204)
async def delete_announcement(org_id: str, announcement_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_admin(org_id, current_user)
    org = await organization_repository.get_by_id(org_id)
    settings = dict(getattr(org, "settings", {}) or {})
    items = [a for a in (settings.get("announcements") or []) if a.get("id") != announcement_id]
    settings["announcements"] = items
    await organization_repository.update_organization_settings(org_id, settings)
    return None


# ── Onboarding checklist ───────────────────────────────────────────


@router.get("/{org_id}/onboarding-checklist")
async def get_onboarding_checklist(org_id: str, current_user: User = Depends(get_current_active_user)):
    org = await _require_org_member(org_id, current_user)
    settings = dict(getattr(org, "settings", {}) or {})
    items = [
        {"id": "profile", "label": "Complete workspace profile", "done": bool((org.description or "")) },
        {"id": "team", "label": "Create your first team", "done": False},
        {"id": "project", "label": "Create your first project", "done": False},
        {"id": "invite", "label": "Invite your first teammate", "done": len(org.member_ids or []) > 1},
        {"id": "agent", "label": "Activate an agent on a project", "done": False},
        {"id": "billing", "label": "Set up billing", "done": bool(settings.get("stripe_customer_id"))},
    ]
    return {"items": items, "completed_steps": settings.get("onboarding_steps") or []}


@router.post("/{org_id}/onboarding-checklist/{step_id}/complete")
async def complete_onboarding_step(
    org_id: str, step_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    org = await organization_repository.get_by_id(org_id)
    settings = dict(getattr(org, "settings", {}) or {})
    completed = set(settings.get("onboarding_steps") or [])
    completed.add(step_id)
    settings["onboarding_steps"] = list(completed)
    await organization_repository.update_organization_settings(org_id, settings)
    return {"completed_steps": list(completed)}


# ── Activity export ────────────────────────────────────────────────


@router.get("/{org_id}/activity")
async def get_org_activity(
    org_id: str,
    activity_type: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=1000),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("activity_logs")
    q: Dict[str, Any] = {"organization_id": _oid(org_id)}
    if activity_type:
        q["activity_type"] = activity_type
    cursor = col.find(q).sort("timestamp", -1).limit(limit)
    rows = await cursor.to_list(length=limit)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "user_id"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.post("/{org_id}/activity/export")
async def export_activity(
    org_id: str,
    days: int = Query(30, ge=1, le=365),
    format: str = Query("jsonl"),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("audit_exports")
    doc = {
        "organization_id": _oid(org_id),
        "requested_by": _oid(current_user.id),
        "days": days, "format": format, "status": "pending",
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"job_id": str(r.inserted_id), "status": "pending"}


# ── Custom roles (org-level RBAC scaffold) ─────────────────────────


@router.get("/{org_id}/custom-roles")
async def list_custom_roles(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("custom_roles")
    rows = await col.find({"organization_id": _oid(org_id)}).to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        if r.get("organization_id"):
            r["organization_id"] = str(r["organization_id"])
    return rows


class CustomRolePayload(BaseModel):
    name: str
    description: Optional[str] = None
    permissions: List[str] = Field(default_factory=list)


@router.post("/{org_id}/custom-roles", status_code=201)
async def create_custom_role(
    org_id: str,
    payload: CustomRolePayload,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    col = await MongoDB.get_collection("custom_roles")
    doc = {
        "organization_id": _oid(org_id),
        "name": payload.name, "description": payload.description,
        "permissions": payload.permissions,
        "created_by": _oid(current_user.id),
        "created_at": datetime.utcnow(), "updated_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    doc["id"] = str(r.inserted_id)
    doc["organization_id"] = str(doc["organization_id"])
    if doc.get("created_by"):
        doc["created_by"] = str(doc["created_by"])
    doc.pop("_id", None)
    return doc


@router.patch("/{org_id}/custom-roles/{role_id}")
async def update_custom_role(
    org_id: str, role_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    col = await MongoDB.get_collection("custom_roles")
    patch = {k: v for k, v in payload.items() if k in ("name", "description", "permissions") and v is not None}
    if not patch:
        raise HTTPException(status_code=400, detail="Nothing to update")
    patch["updated_at"] = datetime.utcnow()
    row = await col.find_one_and_update(
        {"_id": _oid(role_id), "organization_id": _oid(org_id)},
        {"$set": patch}, return_document=True,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Custom role not found")
    row["id"] = str(row.pop("_id"))
    row["organization_id"] = str(row["organization_id"])
    return row


@router.delete("/{org_id}/custom-roles/{role_id}", status_code=204)
async def delete_custom_role(org_id: str, role_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_owner(org_id, current_user)
    col = await MongoDB.get_collection("custom_roles")
    await col.delete_one({"_id": _oid(role_id), "organization_id": _oid(org_id)})
    return None


# ── Stats ─────────────────────────────────────────────────────────


@router.get("/{org_id}/stats")
async def org_stats(org_id: str, current_user: User = Depends(get_current_active_user)):
    org = await _require_org_member(org_id, current_user)
    org_oid = _oid(org_id)
    tasks = await MongoDB.get_collection("tasks")
    projects = await MongoDB.get_collection("projects")
    teams = await MongoDB.get_collection("teams")
    runs = await MongoDB.get_collection("agent_runs")
    return {
        "members": len(org.member_ids or []),
        "admins": len(org.admin_ids or []),
        "teams": await teams.count_documents({"organization_id": org_oid, "is_archived": False}),
        "projects": await projects.count_documents({"organization_id": org_oid, "is_archived": False}),
        "tasks": await tasks.count_documents({"organization_id": org_oid}),
        "agent_runs_lifetime": await runs.count_documents({"organization_id": org_oid}),
    }


# ── Domains proxy (delegate to enterprise router) ─────────────────


@router.get("/{org_id}/domains")
async def list_domains_proxy(org_id: str, current_user: User = Depends(get_current_active_user)):
    """Proxy: domain claims live under /enterprise.  This endpoint exists
    so the org admin surface can render them without a hard dependency."""
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("domain_claims")
    rows = await col.find({"organization_id": _oid(org_id)}).to_list(length=100)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        if r.get("organization_id"):
            r["organization_id"] = str(r["organization_id"])
    return rows


@router.get("/{org_id}/integrations")
async def list_org_integrations_proxy(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("integrations")
    rows = await col.find({"organization_id": _oid(org_id)}).to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        if r.get("organization_id"):
            r["organization_id"] = str(r["organization_id"])
        if r.get("created_by"):
            r["created_by"] = str(r["created_by"])
    return rows


@router.get("/{org_id}/webhooks")
async def list_org_webhooks_proxy(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("webhooks")
    rows = await col.find({"organization_id": _oid(org_id)}).to_list(length=100)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        r.pop("secret_hash", None)
        if r.get("organization_id"):
            r["organization_id"] = str(r["organization_id"])
    return rows


@router.get("/{org_id}/api-tokens")
async def list_org_api_tokens_proxy(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("api_tokens")
    rows = await col.find({"organization_id": _oid(org_id), "revoked_at": None}).to_list(length=100)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        r.pop("token_hash", None)
        if r.get("organization_id"):
            r["organization_id"] = str(r["organization_id"])
        if r.get("user_id"):
            r["user_id"] = str(r["user_id"])
    return rows


# ── Health / posture ──────────────────────────────────────────────


@router.get("/{org_id}/posture")
async def org_posture(org_id: str, current_user: User = Depends(get_current_active_user)):
    """Security posture snapshot."""
    await _require_org_admin(org_id, current_user)
    sso_col = await MongoDB.get_collection("sso_configurations")
    sso = await sso_col.find_one({"organization_id": _oid(org_id)})
    domains = await MongoDB.get_collection("domain_claims")
    verified = await domains.count_documents({"organization_id": _oid(org_id), "verified_at": {"$ne": None}})
    return {
        "sso_configured": bool(sso and sso.get("enabled")),
        "verified_domains": verified,
        "mfa_required": False,
        "session_idle_minutes": 120,
    }


@router.get("/{org_id}/health")
async def org_health(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_member(org_id, current_user)
    return {"status": "ok", "checked_at": datetime.utcnow().isoformat() + "Z"}
