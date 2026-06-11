"""
Phase A — Integrations v2.

Mounted at `/api/v1/integrations-v2`.

Slack, Microsoft Teams, Google Workspace, Notion, Linear, Jira, Asana,
Trello, Monday, GitHub, Figma, Salesforce, HubSpot — each per scope
(org / team / project), with OAuth start/callback, status, post, sync,
disconnect.  Connection state is stored in `integrations` keyed by
(organization_id, scope_type, scope_id, provider).
"""

from __future__ import annotations

import secrets
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity

logger = structlog.get_logger(__name__)
router = APIRouter()

SUPPORTED_PROVIDERS = [
    "slack", "microsoft_teams", "google_workspace", "notion", "linear",
    "jira", "asana", "trello", "monday", "github", "figma", "salesforce",
    "hubspot", "zapier", "outlook",
]


def _oid(v: Any) -> Optional[ObjectId]:
    if v is None:
        return None
    if isinstance(v, ObjectId):
        return v
    try:
        return ObjectId(str(v))
    except Exception:
        return None


def _resolve_primary_org_id(user: User) -> str:
    primary = getattr(user, "organization_id", None)
    if primary:
        return str(primary)
    ids = getattr(user, "organization_ids", None) or []
    if ids:
        return str(ids[0])
    raise HTTPException(status_code=400, detail="User has no organization context")


def _serialize(row: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(row)
    d["id"] = str(d.pop("_id"))
    d.pop("oauth_token", None)
    d.pop("oauth_refresh_token", None)
    for k in ("organization_id", "scope_id", "created_by"):
        if d.get(k):
            d[k] = str(d[k])
    return d


async def _filter_query(scope_type: str, scope_id: str, organization_id: str) -> Dict[str, Any]:
    return {
        "organization_id": _oid(organization_id),
        "scope_type": scope_type,
        "scope_id": scope_id,
    }


# ── Catalogue ────────────────────────────────────────────────────


@router.get("/catalogue")
async def integration_catalogue():
    return {
        "providers": [
            {"key": "slack", "name": "Slack", "category": "chat"},
            {"key": "microsoft_teams", "name": "Microsoft Teams", "category": "chat"},
            {"key": "google_workspace", "name": "Google Workspace", "category": "productivity"},
            {"key": "outlook", "name": "Outlook", "category": "productivity"},
            {"key": "notion", "name": "Notion", "category": "docs"},
            {"key": "linear", "name": "Linear", "category": "issues"},
            {"key": "jira", "name": "Jira", "category": "issues"},
            {"key": "asana", "name": "Asana", "category": "tasks"},
            {"key": "trello", "name": "Trello", "category": "tasks"},
            {"key": "monday", "name": "Monday.com", "category": "tasks"},
            {"key": "github", "name": "GitHub", "category": "code"},
            {"key": "figma", "name": "Figma", "category": "design"},
            {"key": "salesforce", "name": "Salesforce", "category": "crm"},
            {"key": "hubspot", "name": "HubSpot", "category": "crm"},
            {"key": "zapier", "name": "Zapier", "category": "automation"},
        ],
    }


# ── Generic CRUD per scope ────────────────────────────────────────


@router.get("/{scope_type}/{scope_id}")
async def list_integrations_for_scope(
    scope_type: str, scope_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    if scope_type not in ("org", "team", "project"):
        raise HTTPException(status_code=400, detail="Invalid scope_type")
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("integrations")
    cursor = col.find(await _filter_query(scope_type, scope_id, org_id))
    rows = await cursor.to_list(length=100)
    return [_serialize(r) for r in rows]


class OAuthStartPayload(BaseModel):
    return_url: Optional[str] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


@router.post("/{provider}/{scope_type}/{scope_id}/oauth/start")
async def oauth_start(
    provider: str, scope_type: str, scope_id: str,
    payload: OAuthStartPayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"Unsupported provider {provider}")
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("integration_oauth_states")
    state = secrets.token_urlsafe(32)
    doc = {
        "state": state, "provider": provider,
        "organization_id": _oid(org_id),
        "scope_type": scope_type, "scope_id": scope_id,
        "user_id": _oid(current_user.id),
        "return_url": payload.return_url,
        "extra": payload.extra,
        "created_at": datetime.utcnow(),
    }
    await col.insert_one(doc)
    return {
        "state": state,
        "authorize_url_hint": f"/integrations/oauth/{provider}/authorize?state={state}",
    }


class OAuthCallbackPayload(BaseModel):
    state: str
    code: Optional[str] = None
    access_token: Optional[str] = None
    refresh_token: Optional[str] = None
    workspace_metadata: Dict[str, Any] = Field(default_factory=dict)


@router.post("/{provider}/oauth/callback")
async def oauth_callback(provider: str, payload: OAuthCallbackPayload):
    col = await MongoDB.get_collection("integration_oauth_states")
    state_row = await col.find_one({"state": payload.state, "provider": provider})
    if not state_row:
        raise HTTPException(status_code=404, detail="Unknown state")
    integrations = await MongoDB.get_collection("integrations")
    doc = {
        "organization_id": state_row["organization_id"],
        "scope_type": state_row["scope_type"],
        "scope_id": state_row["scope_id"],
        "provider": provider,
        "oauth_token": payload.access_token or payload.code,
        "oauth_refresh_token": payload.refresh_token,
        "workspace_metadata": payload.workspace_metadata,
        "status": "active", "sync_status": "idle",
        "created_by": state_row.get("user_id"),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await integrations.update_one(
        {"organization_id": doc["organization_id"],
         "scope_type": doc["scope_type"],
         "scope_id": doc["scope_id"],
         "provider": provider},
        {"$set": doc}, upsert=True,
    )
    await col.delete_one({"_id": state_row["_id"]})
    return {"ok": True, "return_url": state_row.get("return_url")}


@router.delete("/{provider}/{scope_type}/{scope_id}", status_code=204)
async def disconnect_integration(
    provider: str, scope_type: str, scope_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("integrations")
    await col.delete_one({
        "organization_id": _oid(org_id),
        "scope_type": scope_type, "scope_id": scope_id,
        "provider": provider,
    })
    return None


@router.post("/{provider}/{scope_type}/{scope_id}/sync")
async def sync_integration(
    provider: str, scope_type: str, scope_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("integrations")
    row = await col.find_one_and_update(
        {"organization_id": _oid(org_id),
         "scope_type": scope_type, "scope_id": scope_id,
         "provider": provider},
        {"$set": {"sync_status": "syncing", "last_sync_at": datetime.utcnow()}},
        return_document=True,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Integration not connected")
    return {"ok": True, "sync_status": "syncing"}


@router.get("/{provider}/{scope_type}/{scope_id}/status")
async def integration_status(
    provider: str, scope_type: str, scope_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("integrations")
    row = await col.find_one({
        "organization_id": _oid(org_id),
        "scope_type": scope_type, "scope_id": scope_id,
        "provider": provider,
    })
    if not row:
        return {"connected": False}
    return {"connected": True, **_serialize(row)}


class PostPayload(BaseModel):
    target: Optional[str] = None
    message: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


@router.post("/{provider}/{scope_type}/{scope_id}/post")
async def integration_post(
    provider: str, scope_type: str, scope_id: str,
    payload: PostPayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    integrations = await MongoDB.get_collection("integrations")
    row = await integrations.find_one({
        "organization_id": _oid(org_id),
        "scope_type": scope_type, "scope_id": scope_id,
        "provider": provider,
    })
    if not row:
        raise HTTPException(status_code=404, detail="Integration not connected")
    # Persist the outbound message; the per-provider sender lives in
    # backend/integrations/*.py and runs out of band.
    outbox = await MongoDB.get_collection("integration_outbox")
    doc = {
        "organization_id": _oid(org_id),
        "scope_type": scope_type, "scope_id": scope_id,
        "provider": provider,
        "target": payload.target, "message": payload.message,
        "metadata": payload.metadata, "status": "queued",
        "created_at": datetime.utcnow(),
    }
    r = await outbox.insert_one(doc)
    return {"queued_id": str(r.inserted_id), "status": "queued"}


@router.get("/{provider}/{scope_type}/{scope_id}/health")
async def integration_health(
    provider: str, scope_type: str, scope_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    status = await integration_status(provider, scope_type, scope_id, organization_id, current_user)  # type: ignore[arg-type]
    return {
        "ok": status.get("connected"),
        "status": status.get("sync_status", "unknown"),
        "last_sync_at": status.get("last_sync_at"),
    }


@router.post("/{provider}/{scope_type}/{scope_id}/test")
async def integration_test(
    provider: str, scope_type: str, scope_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    return await integration_post(
        provider, scope_type, scope_id,
        PostPayload(target=None, message="Lumicoria test event"),
        organization_id, current_user,
    )  # type: ignore[arg-type]


@router.get("/me/connected")
async def my_connected_integrations(
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    """All integrations across every scope in the active org."""
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("integrations")
    cursor = col.find({"organization_id": _oid(org_id)})
    rows = await cursor.to_list(length=500)
    return [_serialize(r) for r in rows]


@router.get("/{provider}/scopes-available")
async def scopes_available(provider: str):
    # Static map of OAuth scopes each provider requests.
    scopes = {
        "slack": ["channels:read", "chat:write", "users:read"],
        "google_workspace": ["openid", "email", "profile", "calendar.events", "drive.readonly"],
        "github": ["repo", "read:user", "user:email"],
        "notion": ["read_content", "update_content"],
    }
    return {"provider": provider, "scopes": scopes.get(provider, [])}


@router.get("/outbox")
async def list_outbox(
    organization_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("integration_outbox")
    q: Dict[str, Any] = {"organization_id": _oid(org_id)}
    if status:
        q["status"] = status
    cursor = col.find(q).sort("created_at", -1).limit(limit)
    rows = await cursor.to_list(length=limit)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id",):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.post("/outbox/{outbox_id}/retry")
async def retry_outbox(
    outbox_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("integration_outbox")
    await col.update_one(
        {"_id": _oid(outbox_id)},
        {"$set": {"status": "queued", "retried_at": datetime.utcnow()}},
    )
    return {"ok": True}


@router.delete("/outbox/{outbox_id}", status_code=204)
async def delete_outbox(
    outbox_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("integration_outbox")
    await col.delete_one({"_id": _oid(outbox_id)})
    return None
