"""Operator-side branding admin.

Tenants edit their portal/widget appearance + slug here.  The slug is
the public URL segment (`/portal/{slug}`); uniqueness is enforced
across all orgs, so a 409 surfaces if two tenants try to claim the
same slug.

Also surfaces the **meeting (Jitsi) branding** fields and the binary
logo/favicon upload endpoints used by the /settings/meeting-branding
page — uploads go through storage_service (S3/MinIO) and persist a
public-readable URL on OrgBrandingSQL.meeting_logo_url / _favicon_url.
"""

from __future__ import annotations

import imghdr
import time
from typing import List, Optional

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from pydantic import BaseModel, EmailStr, Field, field_validator

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.permission_repository import permission_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.customer_service import branding as branding_svc
from backend.services.storage_service import storage_service

router = APIRouter()

# ── Upload guards ─────────────────────────────────────────────────────
_LOGO_MAX_BYTES = 1 * 1024 * 1024       # 1 MB
_FAVICON_MAX_BYTES = 200 * 1024         # 200 KB
_ALLOWED_IMAGE_TYPES = {"png", "jpeg", "jpg", "webp", "gif"}
# SVG isn't detected by imghdr; we accept it via the content-type sniff.
_ALLOWED_CONTENT_TYPES = {
    "image/png", "image/jpeg", "image/jpg", "image/webp",
    "image/gif", "image/svg+xml",
}


async def _require_perm(current_user: User) -> str:
    """Permission against actual org (None passes through); tenant-scope
    falls back to user_id for personal accounts."""
    user_id = str(current_user.id)
    permission_org = getattr(current_user, "organization_id", None)
    has = await permission_repository.check_permission(
        user_id=user_id,
        organization_id=permission_org,
        resource_type="AGENT",
        resource_id="customer_service",
        permission_type="EXECUTE",
    )
    if not has:
        raise HTTPException(status_code=403, detail="Permission denied")
    return permission_org or user_id


class BrandingUpsertRequest(BaseModel):
    slug: Optional[str] = Field(None, min_length=2, max_length=64)
    display_name: Optional[str] = Field(None, min_length=1, max_length=200)
    logo_url: Optional[str] = Field(None, max_length=1000)
    primary_color: Optional[str] = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")
    accent_color: Optional[str] = Field(None, pattern=r"^#[0-9a-fA-F]{6}$")
    hero_copy: Optional[str] = Field(None, max_length=2000)
    support_email: Optional[EmailStr] = None
    sla_response_minutes: Optional[int] = Field(None, ge=1, le=10080)
    captcha_enabled: Optional[bool] = None
    public_categories: Optional[List[str]] = None
    # Meeting (Jitsi) branding
    meeting_app_name: Optional[str] = Field(None, max_length=120)
    meeting_logo_url: Optional[str] = Field(None, max_length=1000)
    meeting_favicon_url: Optional[str] = Field(None, max_length=1000)
    meeting_watermark_link: Optional[str] = Field(None, max_length=512)
    meeting_welcome_message: Optional[str] = Field(None, max_length=2000)

    @field_validator("slug")
    @classmethod
    def _slug_check(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return v
        if not branding_svc.is_valid_slug(v.lower()):
            raise ValueError("slug must match ^[a-z0-9](?:[a-z0-9-]{1,62}[a-z0-9])?$")
        return v.lower()


@router.get("")
async def get_branding(
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_perm(current_user)
    return await branding_svc.get_for_org(org_id)


@router.put("")
async def upsert_branding(
    payload: BrandingUpsertRequest,
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_perm(current_user)
    user_id = str(current_user.id)

    fields = payload.model_dump(exclude_unset=True)
    if "support_email" in fields and fields["support_email"] is not None:
        fields["support_email"] = str(fields["support_email"]).lower()

    try:
        result = await branding_svc.upsert(org_id, fields)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="customer_service.branding_updated",
        details={"fields": list(fields.keys())},
        related_resource_type="ORG_BRANDING",
        related_resource_id=org_id,
        agent_name="Customer Service Agent",
    )
    return result


# ── Meeting branding: logo + favicon upload ───────────────────────────


async def _store_branding_image(
    *,
    file: UploadFile,
    org_id: str,
    kind: str,           # "logo" | "favicon"
    max_bytes: int,
) -> str:
    """Validate + upload a branding image to S3/MinIO; return the URL.

    Layout: ``org-branding/{org_id}/{kind}-{timestamp}.{ext}``.
    The timestamp suffix means each upload busts CDN cache cleanly —
    JitsiEmbed always loads the latest version when an org updates it.
    """
    # Read into memory — branding images are small (≤1 MB).
    body = await file.read()
    if len(body) == 0:
        raise HTTPException(status_code=400, detail=f"{kind} is empty")
    if len(body) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"{kind} exceeds {max_bytes // 1024} KB",
        )

    content_type = (file.content_type or "").lower()
    if content_type not in _ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"unsupported {kind} content type: {content_type!r}",
        )

    # Cross-validate against the file's actual bytes for raster formats.
    if content_type != "image/svg+xml":
        sniffed = imghdr.what(None, body)
        if sniffed not in _ALLOWED_IMAGE_TYPES:
            raise HTTPException(
                status_code=415,
                detail=f"{kind} bytes don't match a supported image type",
            )

    ext = {
        "image/png": "png",
        "image/jpeg": "jpg",
        "image/jpg": "jpg",
        "image/webp": "webp",
        "image/gif": "gif",
        "image/svg+xml": "svg",
    }[content_type]

    object_key = f"org-branding/{org_id}/{kind}-{int(time.time())}.{ext}"
    try:
        await storage_service.upload_file(
            file_content=body,
            key=object_key,
            content_type=content_type,
        )
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=502,
            detail=f"{kind} upload failed: {exc}",
        ) from exc

    # Return a public-readable URL the frontend can serve.
    return storage_service.get_public_url(object_key)


@router.post("/logo")
async def upload_meeting_logo(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
):
    """Upload an organization meeting logo (square, ≤1 MB). Returns
    `{"logo_url": ..., "uploaded_at": ...}` and persists the URL on
    OrgBrandingSQL.meeting_logo_url."""
    org_id = await _require_perm(current_user)
    user_id = str(current_user.id)

    logo_url = await _store_branding_image(
        file=file, org_id=org_id, kind="logo", max_bytes=_LOGO_MAX_BYTES,
    )

    try:
        await branding_svc.upsert(org_id, {"meeting_logo_url": logo_url})
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="meeting_branding.logo_uploaded",
        details={"logo_url": logo_url},
        related_resource_type="ORG_BRANDING",
        related_resource_id=org_id,
    )
    return {"logo_url": logo_url, "uploaded_at": int(time.time())}


@router.post("/favicon")
async def upload_meeting_favicon(
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
):
    """Upload an organization meeting favicon (≤200 KB)."""
    org_id = await _require_perm(current_user)
    user_id = str(current_user.id)

    favicon_url = await _store_branding_image(
        file=file, org_id=org_id, kind="favicon", max_bytes=_FAVICON_MAX_BYTES,
    )

    try:
        await branding_svc.upsert(org_id, {"meeting_favicon_url": favicon_url})
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))

    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="meeting_branding.favicon_uploaded",
        details={"favicon_url": favicon_url},
        related_resource_type="ORG_BRANDING",
        related_resource_id=org_id,
    )
    return {"favicon_url": favicon_url, "uploaded_at": int(time.time())}


@router.delete("/logo")
async def remove_meeting_logo(
    current_user: User = Depends(get_current_active_user),
):
    """Clear the meeting logo override (reverts to the default logo_url)."""
    org_id = await _require_perm(current_user)
    user_id = str(current_user.id)
    await branding_svc.upsert(org_id, {"meeting_logo_url": None})
    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="meeting_branding.logo_cleared",
        details={},
        related_resource_type="ORG_BRANDING",
        related_resource_id=org_id,
    )
    return {"ok": True}
