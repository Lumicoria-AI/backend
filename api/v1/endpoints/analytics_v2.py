"""
Phase D — Analytics v2 REST API.

Mounted at `/api/v1/analytics-v2`.

Surfaces per-level metrics computed by `backend/services/analytics_v2.py`.
The legacy `/api/v1/analytics` router stays in place for the existing
Dashboard payload; this router adds the deeper drill-downs needed by the
Workspace surface (org / team / project / user / agent / cost).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.models.user import User
from backend.services import analytics_v2 as svc
from backend.services.activity_logger import log_activity

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


def _resolve_org_id(current_user: User) -> str:
    primary = getattr(current_user, "organization_id", None)
    if primary:
        return str(primary)
    ids = getattr(current_user, "organization_ids", None) or []
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


# ── Org level ────────────────────────────────────────────────────────


@router.get("/org/{org_id}/overview")
async def org_overview(
    org_id: str,
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    return await svc.org_overview(org_id, time_range=time_range)


@router.get("/org/{org_id}/throughput")
async def org_throughput(
    org_id: str,
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    return await svc.org_task_throughput(org_id, time_range=time_range)


@router.get("/org/{org_id}/cycle-time")
async def org_cycle_time(
    org_id: str,
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    return await svc.org_cycle_time(org_id, time_range=time_range)


@router.get("/org/{org_id}/cost")
async def org_cost(
    org_id: str,
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    return await svc.org_cost(org_id, time_range=time_range)


@router.get("/org/{org_id}/seat-forecast")
async def org_seat_forecast(
    org_id: str,
    horizon_days: int = Query(90, ge=7, le=365),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_member(org_id, current_user)
    return await svc.org_seat_forecast(org_id, horizon_days=horizon_days)


# ── Team level ───────────────────────────────────────────────────────


@router.get("/team/{team_id}/overview")
async def team_overview(
    team_id: str,
    organization_id: Optional[str] = Query(None),
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    return await svc.team_overview(org_id, team_id, time_range=time_range)


# ── Project level ────────────────────────────────────────────────────


@router.get("/project/{project_id}/burnup")
async def project_burnup(
    project_id: str,
    organization_id: Optional[str] = Query(None),
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    return await svc.project_burnup(org_id, project_id, time_range=time_range)


@router.get("/project/{project_id}/throughput")
async def project_throughput(
    project_id: str,
    organization_id: Optional[str] = Query(None),
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    return await svc.project_throughput(org_id, project_id, time_range=time_range)


# ── User level ───────────────────────────────────────────────────────


@router.get("/user/{user_id}")
async def user_summary(
    user_id: str,
    organization_id: Optional[str] = Query(None),
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
):
    if organization_id:
        await _require_org_member(organization_id, current_user)
    # Users can always pull their own summary.
    if str(current_user.id) != str(user_id) and organization_id is None:
        raise HTTPException(status_code=400, detail="organization_id required when querying another user")
    return await svc.user_summary(user_id, organization_id, time_range=time_range)


@router.get("/me")
async def me_summary(
    organization_id: Optional[str] = Query(None),
    time_range: str = Query("30d"),
    current_user: User = Depends(get_current_active_user),
):
    return await svc.user_summary(str(current_user.id), organization_id, time_range=time_range)


# ── Activity / audit export ─────────────────────────────────────────


@router.get("/org/{org_id}/audit/recent")
async def org_audit_recent(
    org_id: str,
    limit: int = Query(200, ge=1, le=1000),
    skip: int = Query(0, ge=0),
    severity: Optional[str] = Query(None),
    activity_type: Optional[str] = Query(None),
    resource_type: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    """Org-wide audit log read.  Admin-only.  Streaming export lives in the
    sister `/export` endpoint below."""
    org = await _require_org_member(org_id, current_user)
    if _oid(current_user.id) not in [_oid(a) for a in (org.admin_ids or [])]:
        raise HTTPException(status_code=403, detail="Org admin permission required")
    from backend.db.mongodb.mongodb import MongoDB
    col = await MongoDB.get_collection("activity_logs")
    query: Dict[str, Any] = {"organization_id": _oid(org_id)}
    if severity:
        query["severity"] = severity
    if activity_type:
        query["activity_type"] = activity_type
    if resource_type:
        query["related_resource_type"] = resource_type
    cursor = col.find(query).sort("timestamp", -1).skip(skip).limit(limit)
    rows = await cursor.to_list(length=limit)
    out: List[Dict[str, Any]] = []
    for r in rows:
        r["id"] = str(r.pop("_id", r.get("id")))
        for k in ("organization_id", "user_id"):
            if r.get(k) is not None:
                r[k] = str(r[k])
        out.append(r)
    return out


@router.post("/org/{org_id}/audit/export")
async def org_audit_export(
    org_id: str,
    days: int = Query(30, ge=1, le=365),
    format: str = Query("jsonl", description="csv | jsonl"),
    current_user: User = Depends(get_current_active_user),
):
    """Enqueue a signed export job.  Returns a job id immediately; the
    worker writes the file to MinIO and posts the signed URL back via
    notification when done."""
    org = await _require_org_member(org_id, current_user)
    if _oid(current_user.id) not in [_oid(a) for a in (org.admin_ids or [])]:
        raise HTTPException(status_code=403, detail="Org admin permission required")
    from backend.db.mongodb.mongodb import MongoDB
    col = await MongoDB.get_collection("audit_exports")
    doc = {
        "organization_id": _oid(org_id),
        "requested_by": _oid(current_user.id),
        "days": int(days),
        "format": format,
        "status": "pending",
        "created_at": __import__("datetime").datetime.utcnow(),
    }
    result = await col.insert_one(doc)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="org.audit_export_requested",
        details={"days": days, "format": format, "job_id": str(result.inserted_id)},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return {"job_id": str(result.inserted_id), "status": "pending"}


@router.get("/org/{org_id}/audit/exports/{job_id}")
async def get_audit_export(
    org_id: str,
    job_id: str,
    current_user: User = Depends(get_current_active_user),
):
    org = await _require_org_member(org_id, current_user)
    if _oid(current_user.id) not in [_oid(a) for a in (org.admin_ids or [])]:
        raise HTTPException(status_code=403, detail="Org admin permission required")
    from backend.db.mongodb.mongodb import MongoDB
    col = await MongoDB.get_collection("audit_exports")
    row = await col.find_one({"_id": _oid(job_id), "organization_id": _oid(org_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Export job not found")
    row["id"] = str(row.pop("_id"))
    for k in ("organization_id", "requested_by"):
        if row.get(k) is not None:
            row[k] = str(row[k])
    return row
