"""
Phase C — Comments depth.

Mounted at `/api/v1/comments-v2`.

Adds reviews (request / approve / reject / pending), resource sharing,
comment counts per resource, and bulk operations on top of the existing
`comments.router` (10 routes).
"""

from __future__ import annotations

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


def _resolve_primary_org_id(user: User) -> str:
    primary = getattr(user, "organization_id", None)
    if primary:
        return str(primary)
    ids = getattr(user, "organization_ids", None) or []
    if ids:
        return str(ids[0])
    raise HTTPException(status_code=400, detail="User has no organization context")


# ── Reviews ───────────────────────────────────────────────────────


class ReviewRequestPayload(BaseModel):
    resource_type: str
    resource_id: str
    reviewer_ids: List[str] = Field(..., min_length=1, max_length=20)
    notes: Optional[str] = None


@router.post("/reviews/request", status_code=201)
async def request_review(
    payload: ReviewRequestPayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("reviews")
    doc = {
        "organization_id": _oid(org_id),
        "resource_type": payload.resource_type,
        "resource_id": payload.resource_id,
        "reviewer_ids": [_oid(uid) for uid in payload.reviewer_ids],
        "notes": payload.notes,
        "requested_by": _oid(current_user.id),
        "status": "pending",
        "decisions": [],
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="review.requested",
        details={"review_id": str(r.inserted_id),
                 "resource_type": payload.resource_type,
                 "resource_id": payload.resource_id},
        related_resource_type=payload.resource_type,
        related_resource_id=payload.resource_id,
    )
    await emit("review.requested", organization_id=org_id, actor_id=str(current_user.id),
               resource_type=payload.resource_type, resource_id=payload.resource_id,
               payload={"review_id": str(r.inserted_id)})
    return {"review_id": str(r.inserted_id)}


@router.post("/reviews/{review_id}/approve")
async def approve_review(
    review_id: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("reviews")
    row = await col.find_one({"_id": _oid(review_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Review not found")
    if _oid(current_user.id) not in (row.get("reviewer_ids") or []):
        raise HTTPException(status_code=403, detail="Not a reviewer on this review")
    decision = {
        "reviewer_id": _oid(current_user.id),
        "decision": "approve", "notes": payload.get("notes"),
        "at": datetime.utcnow(),
    }
    await col.update_one(
        {"_id": _oid(review_id)},
        {"$push": {"decisions": decision},
         "$set": {"status": "approved", "completed_at": datetime.utcnow()}},
    )
    await emit("review.approved", organization_id=str(row.get("organization_id")) if row.get("organization_id") else None,
               actor_id=str(current_user.id),
               resource_type=row.get("resource_type"), resource_id=row.get("resource_id"),
               payload={"review_id": review_id})
    return {"ok": True}


@router.post("/reviews/{review_id}/reject")
async def reject_review(
    review_id: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("reviews")
    row = await col.find_one({"_id": _oid(review_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Review not found")
    if _oid(current_user.id) not in (row.get("reviewer_ids") or []):
        raise HTTPException(status_code=403, detail="Not a reviewer on this review")
    decision = {
        "reviewer_id": _oid(current_user.id),
        "decision": "reject", "notes": payload.get("notes"),
        "at": datetime.utcnow(),
    }
    await col.update_one(
        {"_id": _oid(review_id)},
        {"$push": {"decisions": decision},
         "$set": {"status": "rejected", "completed_at": datetime.utcnow()}},
    )
    return {"ok": True}


@router.get("/reviews/pending")
async def list_pending_reviews(
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("reviews")
    cursor = col.find({
        "organization_id": _oid(org_id),
        "status": "pending",
        "reviewer_ids": _oid(current_user.id),
    }).sort("created_at", -1)
    rows = await cursor.to_list(length=100)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "requested_by"):
            if r.get(k):
                r[k] = str(r[k])
        r["reviewer_ids"] = [str(x) for x in (r.get("reviewer_ids") or [])]
    return rows


@router.get("/reviews/{review_id}")
async def get_review(
    review_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("reviews")
    row = await col.find_one({"_id": _oid(review_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Review not found")
    row["id"] = str(row.pop("_id"))
    for k in ("organization_id", "requested_by"):
        if row.get(k):
            row[k] = str(row[k])
    row["reviewer_ids"] = [str(x) for x in (row.get("reviewer_ids") or [])]
    return row


# ── Resource sharing ─────────────────────────────────────────────


class ResourceSharePayload(BaseModel):
    resource_type: str
    resource_id: str
    target_user_ids: List[str] = Field(default_factory=list)
    target_team_ids: List[str] = Field(default_factory=list)
    can_edit: bool = False


@router.post("/shares", status_code=201)
async def share_resource(
    payload: ResourceSharePayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("resource_shares")
    doc = {
        "organization_id": _oid(org_id),
        "resource_type": payload.resource_type,
        "resource_id": payload.resource_id,
        "target_user_ids": [_oid(u) for u in payload.target_user_ids],
        "target_team_ids": [_oid(t) for t in payload.target_team_ids],
        "can_edit": payload.can_edit,
        "shared_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"share_id": str(r.inserted_id)}


@router.get("/shares")
async def list_shares(
    resource_type: Optional[str] = Query(None),
    resource_id: Optional[str] = Query(None),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("resource_shares")
    q: Dict[str, Any] = {"organization_id": _oid(org_id)}
    if resource_type:
        q["resource_type"] = resource_type
    if resource_id:
        q["resource_id"] = resource_id
    rows = await col.find(q).to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "shared_by"):
            if r.get(k):
                r[k] = str(r[k])
        r["target_user_ids"] = [str(x) for x in (r.get("target_user_ids") or [])]
        r["target_team_ids"] = [str(x) for x in (r.get("target_team_ids") or [])]
    return rows


@router.delete("/shares/{share_id}", status_code=204)
async def delete_share(
    share_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("resource_shares")
    await col.delete_one({"_id": _oid(share_id), "shared_by": _oid(current_user.id)})
    return None


# ── Counts + summaries ──────────────────────────────────────────


@router.get("/counts")
async def comment_counts(
    resource_type: str = Query(...),
    resource_ids: List[str] = Query(...),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("comments")
    pipeline = [
        {"$match": {
            "organization_id": _oid(org_id),
            "resource_type": resource_type,
            "resource_id": {"$in": resource_ids},
        }},
        {"$group": {"_id": "$resource_id", "count": {"$sum": 1}}},
    ]
    rows = await col.aggregate(pipeline).to_list(length=500)
    return {r["_id"]: r["count"] for r in rows}


@router.get("/recent")
async def recent_comments(
    resource_type: Optional[str] = Query(None),
    organization_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("comments")
    q: Dict[str, Any] = {"organization_id": _oid(org_id)}
    if resource_type:
        q["resource_type"] = resource_type
    cursor = col.find(q).sort("created_at", -1).limit(limit)
    rows = await cursor.to_list(length=limit)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "user_id", "parent_id"):
            if r.get(k):
                r[k] = str(r[k])
        r["mentions"] = [str(m) for m in (r.get("mentions") or [])]
    return rows


@router.get("/unread/mentions")
async def unread_mentions(
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("comments")
    handle = (getattr(current_user, "email", "") or "").split("@")[0]
    count = await col.count_documents({
        "organization_id": _oid(org_id),
        "mentions": handle,
    }) if handle else 0
    return {"unread": count}


# ── Bulk operations ─────────────────────────────────────────────


class BulkResolvePayload(BaseModel):
    comment_ids: List[str] = Field(..., min_length=1, max_length=200)
    resolved: bool = True


@router.post("/bulk-resolve")
async def bulk_resolve(
    payload: BulkResolvePayload,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("comments")
    oids = [_oid(c) for c in payload.comment_ids if _oid(c)]
    result = await col.update_many(
        {"_id": {"$in": oids}},
        {"$set": {"resolved": payload.resolved}},
    )
    return {"updated": result.modified_count}


@router.delete("/bulk")
async def bulk_delete_comments(
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("comments")
    ids = payload.get("comment_ids") or []
    oids = [_oid(c) for c in ids if _oid(c)]
    result = await col.delete_many({
        "_id": {"$in": oids},
        "user_id": _oid(current_user.id),
    })
    return {"deleted": result.deleted_count}


# ── Per-resource subscription helpers ──────────────────────────


@router.post("/watch")
async def watch_resource(
    payload: Dict[str, Any] = Body(...),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("resource_subscriptions")
    await col.update_one(
        {"user_id": _oid(current_user.id),
         "organization_id": _oid(org_id),
         "resource_type": payload.get("resource_type"),
         "resource_id": payload.get("resource_id")},
        {"$set": {"channels": payload.get("channels") or ["in_app"],
                  "created_at": datetime.utcnow()}},
        upsert=True,
    )
    return {"ok": True}


@router.delete("/watch", status_code=204)
async def unwatch_resource(
    resource_type: str = Query(...),
    resource_id: str = Query(...),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("resource_subscriptions")
    await col.delete_one({
        "user_id": _oid(current_user.id),
        "organization_id": _oid(org_id),
        "resource_type": resource_type, "resource_id": resource_id,
    })
    return None
