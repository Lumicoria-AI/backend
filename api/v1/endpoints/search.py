"""
Phase A — Federated search router.

Mounted at `/api/v1/search`.

Thin federated search across the workspace's most-used collections.
Reuses MongoDB text indexes when available; falls back to case-insensitive
regex matching.  Each hit is normalised to a common envelope so the
frontend command palette / search bar can render results uniformly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Body, Depends, HTTPException, Query

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


async def _require_org_member(org_id: str, current_user: User):
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(m) for m in (org.member_ids or [])]:
        raise HTTPException(status_code=403, detail="Not a member of this organization")
    return org


def _make_envelope(kind: str, doc: Dict[str, Any]) -> Dict[str, Any]:
    title = (doc.get("title") or doc.get("name") or doc.get("subject")
             or doc.get("filename") or "Untitled")
    subtitle = (doc.get("description") or doc.get("body") or doc.get("status") or "")[:160]
    return {
        "kind": kind,
        "id": str(doc.get("_id") or doc.get("id")),
        "title": title,
        "subtitle": subtitle,
        "url_hint": _suggest_url(kind, doc),
    }


def _suggest_url(kind: str, doc: Dict[str, Any]) -> Optional[str]:
    if kind == "project":
        return f"/workspace/projects/{doc.get('_id') or doc.get('id')}"
    if kind == "team":
        return f"/workspace/teams/{doc.get('_id') or doc.get('id')}"
    if kind == "task":
        return f"/tasks?task={doc.get('_id') or doc.get('id')}"
    if kind == "document":
        return f"/documents?doc={doc.get('_id') or doc.get('id')}"
    return None


async def _search_collection(
    *, col_name: str, kind: str, query: str, org_oid: ObjectId, limit: int,
) -> List[Dict[str, Any]]:
    col = await MongoDB.get_collection(col_name)
    # Try text index first.
    try:
        cursor = col.find({
            "organization_id": org_oid, "$text": {"$search": query},
        }).limit(limit)
        rows = await cursor.to_list(length=limit)
        if rows:
            return [_make_envelope(kind, r) for r in rows]
    except Exception:
        pass
    # Fallback regex.
    rgx = {"$regex": query, "$options": "i"}
    fallback_q = {
        "organization_id": org_oid,
        "$or": [
            {"name": rgx}, {"title": rgx}, {"description": rgx},
            {"subject": rgx}, {"filename": rgx},
        ],
    }
    cursor = col.find(fallback_q).limit(limit)
    rows = await cursor.to_list(length=limit)
    return [_make_envelope(kind, r) for r in rows]


# ── Search endpoints ──────────────────────────────────────────────


@router.get("")
async def federated_search(
    q: str = Query(..., min_length=1, max_length=200),
    organization_id: Optional[str] = Query(None),
    limit_per_kind: int = Query(8, ge=1, le=25),
    current_user: User = Depends(get_current_active_user),
):
    """Federated search across projects, teams, tasks, documents."""
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    org_oid = _oid(org_id)
    results: Dict[str, List[Dict[str, Any]]] = {}
    pairs = (
        ("projects", "project"),
        ("teams", "team"),
        ("tasks", "task"),
        ("documents", "document"),
    )
    for col_name, kind in pairs:
        try:
            results[kind] = await _search_collection(
                col_name=col_name, kind=kind, query=q,
                org_oid=org_oid, limit=limit_per_kind,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("search.collection_failed", kind=kind, error=str(exc))
            results[kind] = []
    total = sum(len(v) for v in results.values())
    return {"query": q, "total": total, "results": results}


@router.get("/suggest")
async def search_suggestions(
    q: str = Query("", max_length=120),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    """Typeahead suggestions for the command palette."""
    if not q:
        return {"suggestions": []}
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    org_oid = _oid(org_id)
    suggestions: List[Dict[str, Any]] = []
    rgx = {"$regex": f"^{q}", "$options": "i"}
    # Projects + teams as title-prefix suggestions; cheap.
    projects = await MongoDB.get_collection("projects")
    async for p in projects.find(
        {"organization_id": org_oid, "name": rgx}, {"_id": 1, "name": 1}
    ).limit(5):
        suggestions.append({"kind": "project", "id": str(p["_id"]), "name": p["name"]})
    teams = await MongoDB.get_collection("teams")
    async for t in teams.find(
        {"organization_id": org_oid, "name": rgx}, {"_id": 1, "name": 1}
    ).limit(5):
        suggestions.append({"kind": "team", "id": str(t["_id"]), "name": t["name"]})
    return {"suggestions": suggestions[:10]}


# ── Saved searches ────────────────────────────────────────────────


@router.get("/saved")
async def list_saved_searches(
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("saved_searches")
    cursor = col.find({"user_id": _oid(current_user.id), "organization_id": _oid(org_id)}).sort("created_at", -1)
    rows = await cursor.to_list(length=100)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("user_id", "organization_id"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.post("/saved", status_code=201)
async def create_saved_search(
    payload: Dict[str, Any] = Body(...),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("saved_searches")
    doc = {
        "user_id": _oid(current_user.id),
        "organization_id": _oid(org_id),
        "name": payload.get("name") or "Untitled search",
        "query": payload.get("query") or "",
        "filters": payload.get("filters") or {},
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"id": str(r.inserted_id)}


@router.delete("/saved/{search_id}", status_code=204)
async def delete_saved_search(
    search_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("saved_searches")
    await col.delete_one({"_id": _oid(search_id), "user_id": _oid(current_user.id)})
    return None


# ── Recent searches ──────────────────────────────────────────────


@router.get("/recent")
async def list_recent_searches(
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("recent_searches")
    cursor = col.find({"user_id": _oid(current_user.id)}).sort("touched_at", -1).limit(20)
    rows = await cursor.to_list(length=20)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        if r.get("user_id"):
            r["user_id"] = str(r["user_id"])
    return rows


@router.post("/recent")
async def touch_recent_search(
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("recent_searches")
    await col.update_one(
        {"user_id": _oid(current_user.id), "query": payload.get("query") or ""},
        {"$set": {"touched_at": datetime.utcnow()},
         "$setOnInsert": {"created_at": datetime.utcnow()}},
        upsert=True,
    )
    return {"ok": True}


@router.delete("/recent", status_code=204)
async def clear_recent_searches(current_user: User = Depends(get_current_active_user)):
    col = await MongoDB.get_collection("recent_searches")
    await col.delete_many({"user_id": _oid(current_user.id)})
    return None
