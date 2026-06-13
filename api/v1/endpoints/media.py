"""
Phase A — Media router.

Mounted at `/api/v1/media`.

Avatar / cover uploads scoped to user / org / team / project, plus a
read-side library and a signed-URL resolver.  Storage flows through
`backend/services/storage_service.py` so MinIO + R2 dual-write is
automatic.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Body, Depends, File, HTTPException, Query, UploadFile

from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.models.user import User

logger = structlog.get_logger(__name__)
router = APIRouter()

ALLOWED_MIMETYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/gif", "image/webp", "image/svg+xml",
}


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


def _build_key(scope: str, scope_id: str, filename: str) -> str:
    safe = filename.replace("/", "_")
    return f"avatars/{scope}/{scope_id}/{safe}"


async def _upload_and_record(
    *, scope: str, scope_id: str, file: UploadFile,
    organization_id: str, user_id: str,
) -> Dict[str, Any]:
    if file.content_type not in ALLOWED_MIMETYPES:
        raise HTTPException(status_code=400, detail=f"Unsupported mime-type: {file.content_type}")
    content = await file.read()
    if len(content) > 5 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 5MB)")
    key = _build_key(scope, scope_id, file.filename or "upload.bin")
    try:
        from backend.services.storage_service import storage_service
        # NOTE: storage_service.upload_file's signature is (file_content,
        # key, content_type) — pass keyword args so the positional order
        # bug (botocore rejecting bytes as Key) can never recur.
        await storage_service.upload_file(
            file_content=content, key=key, content_type=file.content_type,
        )
        url = storage_service.get_public_url(key)
    except Exception as exc:  # noqa: BLE001
        logger.exception("media.upload_failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Storage upload failed")

    col = await MongoDB.get_collection("media_assets")
    doc = {
        "organization_id": _oid(organization_id),
        "scope": scope, "scope_id": scope_id,
        "key": key, "url": url, "mime_type": file.content_type,
        "size": len(content), "filename": file.filename,
        "uploaded_by": _oid(user_id),
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"id": str(r.inserted_id), "url": url, "key": key, "size": len(content)}


# ── Avatar uploads per scope ──────────────────────────────────────


@router.post("/avatar/user")
async def upload_user_avatar(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
):
    org_id = _resolve_primary_org_id(current_user)
    return await _upload_and_record(
        scope="user", scope_id=str(current_user.id),
        file=file, organization_id=org_id, user_id=str(current_user.id),
    )


@router.post("/avatar/org/{org_id}")
async def upload_org_avatar(
    org_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
):
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(a) for a in (org.admin_ids or [])]:
        raise HTTPException(status_code=403, detail="Org admin permission required")
    res = await _upload_and_record(
        scope="org", scope_id=org_id, file=file,
        organization_id=org_id, user_id=str(current_user.id),
    )
    await organization_repository.update(org_id, {"logo_url": res["url"]})
    return res


@router.post("/avatar/team/{team_id}")
async def upload_team_avatar(
    team_id: str,
    file: UploadFile = File(...),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    res = await _upload_and_record(
        scope="team", scope_id=team_id, file=file,
        organization_id=org_id, user_id=str(current_user.id),
    )
    try:
        from backend.db.mongodb.repositories.team_repository import team_repository
        await team_repository.update_team(
            team_id, organization_id=org_id, patch={"logo_url": res["url"]},
        )
    except Exception:
        pass
    return res


@router.post("/avatar/project/{project_id}")
async def upload_project_avatar(
    project_id: str,
    file: UploadFile = File(...),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    res = await _upload_and_record(
        scope="project", scope_id=project_id, file=file,
        organization_id=org_id, user_id=str(current_user.id),
    )
    try:
        from backend.db.mongodb.repositories.project_v2_repository import project_v2_repository
        await project_v2_repository.update_project(
            project_id, organization_id=org_id, patch={"logo_url": res["url"]},
        )
    except Exception:
        pass
    return res


# ── Cover uploads ────────────────────────────────────────────────


@router.post("/cover/team/{team_id}")
async def upload_team_cover(
    team_id: str,
    file: UploadFile = File(...),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    res = await _upload_and_record(
        scope="team_cover", scope_id=team_id, file=file,
        organization_id=org_id, user_id=str(current_user.id),
    )
    # Persist directly onto the team row so cover survives reads via /teams.
    try:
        from backend.db.mongodb.repositories.team_repository import team_repository
        await team_repository.update_team(
            team_id, organization_id=org_id, patch={"cover_url": res["url"]},
        )
    except Exception:
        # Cover upload still succeeded — surface the URL to the caller so the
        # frontend can patch the team via /teams as a fallback.
        pass
    return res


@router.post("/cover/project/{project_id}")
async def upload_project_cover(
    project_id: str,
    file: UploadFile = File(...),
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    res = await _upload_and_record(
        scope="project_cover", scope_id=project_id, file=file,
        organization_id=org_id, user_id=str(current_user.id),
    )
    try:
        from backend.db.mongodb.repositories.project_v2_repository import project_v2_repository
        await project_v2_repository.update_project(
            project_id, organization_id=org_id, patch={"cover_image_url": res["url"]},
        )
    except Exception:
        pass
    return res


@router.post("/cover/org/{org_id}")
async def upload_org_cover(
    org_id: str,
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
):
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(a) for a in (org.admin_ids or [])]:
        raise HTTPException(status_code=403, detail="Org admin permission required")
    res = await _upload_and_record(
        scope="org_cover", scope_id=org_id, file=file,
        organization_id=org_id, user_id=str(current_user.id),
    )
    try:
        await organization_repository.update(org_id, {"cover_url": res["url"]})
    except Exception:
        pass
    return res


# ── Library + signed URL + delete ────────────────────────────────


@router.get("/library")
async def list_media(
    organization_id: Optional[str] = Query(None),
    scope: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    col = await MongoDB.get_collection("media_assets")
    q: Dict[str, Any] = {"organization_id": _oid(org_id)}
    if scope:
        q["scope"] = scope
    cursor = col.find(q).sort("created_at", -1).limit(limit)
    rows = await cursor.to_list(length=limit)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "uploaded_by"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.get("/{asset_id}/signed-url")
async def get_signed_url(
    asset_id: str,
    expires_seconds: int = Query(3600, ge=60, le=86400),
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("media_assets")
    row = await col.find_one({"_id": _oid(asset_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Asset not found")
    try:
        from backend.services.storage_service import storage_service
        url = await storage_service.get_presigned_url(row["key"], expiry=expires_seconds)
    except Exception as exc:  # noqa: BLE001
        logger.exception("media.signed_url_failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Could not sign URL")
    return {"url": url, "expires_in": expires_seconds}


@router.delete("/{asset_id}", status_code=204)
async def delete_media(
    asset_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("media_assets")
    row = await col.find_one({"_id": _oid(asset_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Asset not found")
    try:
        from backend.services.storage_service import storage_service
        await storage_service.delete_file(row["key"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("media.delete_storage_failed", error=str(exc))
    await col.delete_one({"_id": _oid(asset_id)})
    return None


# ── Resize / crop (queue-only stubs that record intent) ───────────


@router.post("/resize")
async def queue_resize(
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("media_jobs")
    doc = {
        "user_id": _oid(current_user.id),
        "kind": "resize",
        "asset_id": _oid(payload.get("asset_id")) if payload.get("asset_id") else None,
        "params": payload,
        "status": "queued",
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"job_id": str(r.inserted_id), "status": "queued"}


@router.post("/crop")
async def queue_crop(
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("media_jobs")
    doc = {
        "user_id": _oid(current_user.id),
        "kind": "crop",
        "asset_id": _oid(payload.get("asset_id")) if payload.get("asset_id") else None,
        "params": payload,
        "status": "queued",
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"job_id": str(r.inserted_id), "status": "queued"}
