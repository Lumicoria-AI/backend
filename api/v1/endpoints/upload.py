"""
Reusable file upload endpoint — uploads to MinIO + R2 dual-write storage.
Used by blog, documents, chat, and other features.
"""

import uuid
from fastapi import APIRouter, UploadFile, File, Query, Depends, HTTPException, status
from pydantic import BaseModel
import structlog

from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.services.storage_service import storage_service

logger = structlog.get_logger(__name__)

router = APIRouter()

ALLOWED_CONTENT_TYPES = {
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/svg+xml",
    "video/mp4",
    "video/webm",
    "application/pdf",
}

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


class UploadResponse(BaseModel):
    key: str
    url: str
    content_type: str
    size: int


@router.post("", response_model=UploadResponse)
async def upload_file(
    file: UploadFile = File(...),
    folder: str = Query(default="uploads", max_length=100),
    current_user: User = Depends(get_current_active_user),
):
    """Upload a file to storage. Returns the object key and a presigned URL."""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type '{file.content_type}' not allowed. Accepted: {', '.join(sorted(ALLOWED_CONTENT_TYPES))}",
        )

    file_bytes = await file.read()
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE // (1024 * 1024)} MB.",
        )

    safe_filename = file.filename or "file"
    key = f"{folder}/{uuid.uuid4()}_{safe_filename}"

    try:
        await storage_service.upload_file(file_bytes, key, file.content_type)
        # blog/ prefix has public-read policy — return permanent URL
        if folder == "blog":
            url = storage_service.get_public_url(key)
        else:
            url = await storage_service.get_presigned_url(key)
    except Exception as e:
        logger.error("upload_failed", error=str(e), key=key)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload file.",
        )

    return UploadResponse(
        key=key,
        url=url,
        content_type=file.content_type,
        size=len(file_bytes),
    )
