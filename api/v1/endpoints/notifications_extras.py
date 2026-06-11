"""
Phase C — Notifications + prefs depth.

Mounted at `/api/v1/notifications-v2`.

Sits beside `notifications.router` (5 routes) + `notification_rules.router`
(5 routes) and adds the depth the plan calls for: rule CRUD, snooze,
one-tap unsubscribe, digest preview, device CRUD, topic subscribe/
unsubscribe, broadcast, per-resource subscriptions.
"""

from __future__ import annotations

import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.models.user import User

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


def _resolve_primary_org_id(user: User) -> str:
    primary = getattr(user, "organization_id", None)
    if primary:
        return str(primary)
    ids = getattr(user, "organization_ids", None) or []
    if ids:
        return str(ids[0])
    raise HTTPException(status_code=400, detail="User has no organization context")


# ── Rules CRUD ────────────────────────────────────────────────────


class RuleCreate(BaseModel):
    name: str
    event_type: str
    conditions: List[Dict[str, Any]] = Field(default_factory=list)
    channels: List[str] = Field(default_factory=lambda: ["in_app"])
    enabled: bool = True


@router.get("/rules")
async def list_rules(current_user: User = Depends(get_current_active_user)):
    col = await MongoDB.get_collection("notification_rules")
    cursor = col.find({"user_id": _oid(current_user.id)}).sort("created_at", -1)
    rows = await cursor.to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        if r.get("user_id"):
            r["user_id"] = str(r["user_id"])
    return rows


@router.post("/rules", status_code=201)
async def create_rule(payload: RuleCreate, current_user: User = Depends(get_current_active_user)):
    col = await MongoDB.get_collection("notification_rules")
    doc = {
        "user_id": _oid(current_user.id),
        "name": payload.name,
        "event_type": payload.event_type,
        "conditions": payload.conditions,
        "channels": payload.channels,
        "enabled": payload.enabled,
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"id": str(r.inserted_id)}


@router.patch("/rules/{rule_id}")
async def update_rule(
    rule_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("notification_rules")
    patch = {k: v for k, v in payload.items()
             if k in ("name", "event_type", "conditions", "channels", "enabled") and v is not None}
    patch["updated_at"] = datetime.utcnow()
    row = await col.find_one_and_update(
        {"_id": _oid(rule_id), "user_id": _oid(current_user.id)},
        {"$set": patch}, return_document=True,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Rule not found")
    row["id"] = str(row.pop("_id"))
    return row


@router.delete("/rules/{rule_id}", status_code=204)
async def delete_rule(
    rule_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("notification_rules")
    await col.delete_one({"_id": _oid(rule_id), "user_id": _oid(current_user.id)})
    return None


@router.post("/rules/{rule_id}/enable")
async def enable_rule(
    rule_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("notification_rules")
    await col.update_one({"_id": _oid(rule_id), "user_id": _oid(current_user.id)},
                         {"$set": {"enabled": True}})
    return {"ok": True}


@router.post("/rules/{rule_id}/disable")
async def disable_rule(
    rule_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("notification_rules")
    await col.update_one({"_id": _oid(rule_id), "user_id": _oid(current_user.id)},
                         {"$set": {"enabled": False}})
    return {"ok": True}


# ── Snooze / unsubscribe ─────────────────────────────────────────


@router.post("/{notification_id}/snooze")
async def snooze_notification(
    notification_id: str,
    minutes: int = Query(60, ge=5, le=10080),
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("notifications")
    await col.update_one(
        {"_id": _oid(notification_id), "user_id": str(current_user.id)},
        {"$set": {"snoozed_until": datetime.utcnow() + timedelta(minutes=minutes)}},
    )
    return {"ok": True}


@router.post("/unsubscribe/{token}")
async def unsubscribe(token: str):
    col = await MongoDB.get_collection("unsubscribe_tokens")
    row = await col.find_one({"token": token})
    if not row:
        raise HTTPException(status_code=404, detail="Invalid token")
    prefs_col = await MongoDB.get_collection("notification_preferences")
    await prefs_col.update_many(
        {"user_id": row.get("user_id"), "category": row.get("category")},
        {"$set": {"enabled": False}},
    )
    return {"ok": True, "category": row.get("category")}


# ── Digest preview ───────────────────────────────────────────────


@router.get("/digest/preview")
async def digest_preview(
    cadence: str = Query("daily"),
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("notifications")
    since = datetime.utcnow() - timedelta(days=1 if cadence == "daily" else 7)
    cursor = col.find({
        "user_id": str(current_user.id),
        "created_at": {"$gte": since},
    }).limit(50)
    rows = await cursor.to_list(length=50)
    for r in rows:
        r["id"] = str(r.pop("_id"))
    return {"cadence": cadence, "since": since.isoformat() + "Z", "items": rows}


# ── Device tokens ────────────────────────────────────────────────


class DeviceTokenCreate(BaseModel):
    token: str
    platform: str = Field(..., description="ios | android | web")
    label: Optional[str] = None


@router.get("/devices")
async def list_devices(current_user: User = Depends(get_current_active_user)):
    col = await MongoDB.get_collection("device_tokens")
    cursor = col.find({"user_id": _oid(current_user.id)})
    rows = await cursor.to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        if r.get("user_id"):
            r["user_id"] = str(r["user_id"])
    return rows


@router.post("/devices", status_code=201)
async def register_device(payload: DeviceTokenCreate, current_user: User = Depends(get_current_active_user)):
    col = await MongoDB.get_collection("device_tokens")
    doc = {
        "user_id": _oid(current_user.id),
        "token": payload.token, "platform": payload.platform,
        "label": payload.label, "created_at": datetime.utcnow(),
    }
    await col.update_one(
        {"user_id": doc["user_id"], "token": payload.token},
        {"$set": doc}, upsert=True,
    )
    return {"ok": True}


@router.delete("/devices/{device_id}", status_code=204)
async def delete_device(
    device_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("device_tokens")
    await col.delete_one({"_id": _oid(device_id), "user_id": _oid(current_user.id)})
    return None


@router.post("/devices/{device_id}/test-push")
async def test_push(
    device_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("device_tokens")
    row = await col.find_one({"_id": _oid(device_id), "user_id": _oid(current_user.id)})
    if not row:
        raise HTTPException(status_code=404, detail="Device not found")
    return {"ok": True, "queued": True, "token_prefix": (row.get("token") or "")[:8]}


# ── Topics / subscriptions ──────────────────────────────────────


@router.get("/topics")
async def list_topics():
    return {
        "topics": [
            {"key": "billing", "label": "Billing"},
            {"key": "agents", "label": "Agent runs"},
            {"key": "tasks", "label": "Tasks"},
            {"key": "documents", "label": "Documents"},
            {"key": "security", "label": "Security"},
            {"key": "system", "label": "System"},
        ],
    }


@router.post("/topics/{topic}/subscribe")
async def subscribe_topic(
    topic: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("topic_subscriptions")
    await col.update_one(
        {"user_id": _oid(current_user.id), "topic": topic},
        {"$set": {"subscribed_at": datetime.utcnow()}},
        upsert=True,
    )
    return {"ok": True}


@router.delete("/topics/{topic}", status_code=204)
async def unsubscribe_topic(
    topic: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("topic_subscriptions")
    await col.delete_one({"user_id": _oid(current_user.id), "topic": topic})
    return None


# ── Broadcast (admin-only) ──────────────────────────────────────


class BroadcastPayload(BaseModel):
    title: str
    content: str
    target_user_ids: Optional[List[str]] = None


@router.post("/broadcast/org/{org_id}")
async def broadcast_to_org(
    org_id: str,
    payload: BroadcastPayload,
    current_user: User = Depends(get_current_active_user),
):
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(a) for a in (org.admin_ids or [])]:
        raise HTTPException(status_code=403, detail="Org admin permission required")
    col = await MongoDB.get_collection("notifications")
    targets = payload.target_user_ids or [str(m) for m in (org.member_ids or [])]
    now = datetime.utcnow()
    docs = [{
        "user_id": uid, "title": payload.title, "content": payload.content,
        "notification_type": "system", "priority": "normal",
        "metadata": {"broadcast": True, "org_id": org_id},
        "created_at": now, "read": False,
    } for uid in targets]
    if docs:
        await col.insert_many(docs)
    return {"sent": len(docs)}


# ── Per-resource subscriptions ─────────────────────────────────


@router.get("/subscriptions")
async def list_subscriptions(
    resource_type: Optional[str] = Query(None),
    resource_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("resource_subscriptions")
    q: Dict[str, Any] = {"user_id": _oid(current_user.id)}
    if resource_type:
        q["resource_type"] = resource_type
    if resource_id:
        q["resource_id"] = resource_id
    cursor = col.find(q)
    rows = await cursor.to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        if r.get("user_id"):
            r["user_id"] = str(r["user_id"])
    return rows


@router.post("/subscriptions", status_code=201)
async def create_subscription(
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("resource_subscriptions")
    await col.update_one(
        {"user_id": _oid(current_user.id),
         "resource_type": payload.get("resource_type"),
         "resource_id": payload.get("resource_id")},
        {"$set": {"created_at": datetime.utcnow(),
                  "channels": payload.get("channels") or ["in_app"]}},
        upsert=True,
    )
    return {"ok": True}


@router.delete("/subscriptions/{sub_id}", status_code=204)
async def delete_subscription(
    sub_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("resource_subscriptions")
    await col.delete_one({"_id": _oid(sub_id), "user_id": _oid(current_user.id)})
    return None


# ── Catalog discovery ──────────────────────────────────────────


@router.get("/event-catalogue")
async def event_catalogue():
    return {
        "events": [
            "task.created", "task.completed", "task.assigned",
            "project.created", "project.member_added", "team.member_added",
            "agent.run_completed", "agent.run_failed",
            "document.uploaded", "invite.accepted", "org.seat_assigned",
            "comment.created", "automation.test",
            "billing.payment_succeeded", "billing.payment_failed",
        ],
        "channels": ["email", "push", "in_app"],
    }


@router.get("/unread/by-category")
async def unread_by_category(current_user: User = Depends(get_current_active_user)):
    col = await MongoDB.get_collection("notifications")
    pipeline = [
        {"$match": {"user_id": str(current_user.id), "read": False}},
        {"$group": {"_id": "$notification_type", "count": {"$sum": 1}}},
    ]
    rows = await col.aggregate(pipeline).to_list(length=20)
    return {r["_id"] or "unknown": r["count"] for r in rows}


# ── Bulk operations ────────────────────────────────────────────


@router.post("/mark-all-read")
async def mark_all_read(current_user: User = Depends(get_current_active_user)):
    col = await MongoDB.get_collection("notifications")
    result = await col.update_many(
        {"user_id": str(current_user.id), "read": False},
        {"$set": {"read": True, "read_at": datetime.utcnow()}},
    )
    return {"updated": result.modified_count}


@router.post("/bulk-delete")
async def bulk_delete(
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("notifications")
    ids = payload.get("notification_ids") or []
    oids = [_oid(i) for i in ids if _oid(i)]
    result = await col.delete_many({"_id": {"$in": oids}, "user_id": str(current_user.id)})
    return {"deleted": result.deleted_count}
