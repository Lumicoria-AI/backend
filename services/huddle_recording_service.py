"""
Lumicoria Huddle — recording storage.

Strategy (Phase 1):
  - The host's browser captures the meeting via MediaRecorder + getDisplayMedia
    and streams 1MB Blob chunks to `POST /huddles/{id}/recording/chunks`.
  - Each chunk is appended to MinIO (via storage_service.upload_file) under
    a per-chunk key `huddle-recordings/{org_id}/{huddle_id}/chunk-{n}.{ext}`.
  - `POST /huddles/{id}/recording/finish` concatenates all chunks in order
    into a single object and writes a single `recording_url` + signed playback
    URL back to the HuddleSQL row.
  - Retention is scheduled via Celery beat — see `tasks/huddle_tasks.py`.

This keeps Phase 1 production-grade without requiring a self-hosted Jitsi.
Phase 2 swaps the browser-side recorder for Jibri.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import structlog
from sqlalchemy import select, update as sa_update

from backend.db.postgres import get_async_sessionmaker
from backend.db.postgres_models import HuddleSQL
from backend.services.activity_logger import log_activity
from backend.services.storage_service import storage_service

logger = structlog.get_logger(__name__)

CHUNK_KEY_PREFIX = "huddle-recordings"


# ── Chunk upload ────────────────────────────────────────────────────────

async def start_recording(
    huddle_id: str,
    *,
    requesting_user_id: str,
) -> Dict[str, Any]:
    """Flip recording_enabled=True on the huddle row and emit an
    activity log entry. The browser begins streaming chunks after."""
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        q = select(HuddleSQL).where(HuddleSQL.id == huddle_id, HuddleSQL.deleted_at.is_(None))
        row = (await session.execute(q)).scalar_one_or_none()
        if not row or row.host_user_id != requesting_user_id:
            return {"ok": False, "error": "not_found_or_forbidden"}
        row.recording_enabled = True
        row.updated_at = datetime.utcnow()
        await session.commit()
        await session.refresh(row)
        org_id = row.organization_id

    await log_activity(
        user_id=requesting_user_id,
        organization_id=org_id,
        activity_type="huddle.recording_started",
        details={"huddle_id": huddle_id},
        related_resource_type="huddle",
        related_resource_id=huddle_id,
        agent_name="Huddle",
    )
    return {"ok": True}


async def upload_chunk(
    huddle_id: str,
    *,
    chunk_index: int,
    blob: bytes,
    content_type: str = "video/webm",
    requesting_user_id: str,
) -> Dict[str, Any]:
    """Upload one Blob chunk from the browser MediaRecorder."""
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        q = select(HuddleSQL).where(HuddleSQL.id == huddle_id, HuddleSQL.deleted_at.is_(None))
        row = (await session.execute(q)).scalar_one_or_none()
        if not row or row.host_user_id != requesting_user_id:
            return {"ok": False, "error": "not_found_or_forbidden"}
        org_id = row.organization_id

    ext = "webm" if "webm" in content_type else "mp4" if "mp4" in content_type else "bin"
    key = f"{CHUNK_KEY_PREFIX}/{org_id}/{huddle_id}/chunk-{chunk_index:06d}.{ext}"
    upload = await storage_service.upload_file(blob, key, content_type=content_type)
    return {
        "ok": True,
        "key": key,
        "size": upload.get("size"),
        "chunk_index": chunk_index,
    }


async def finish_recording(
    huddle_id: str,
    *,
    total_chunks: int,
    content_type: str = "video/webm",
    requesting_user_id: str,
) -> Dict[str, Any]:
    """Concatenate uploaded chunks into a single object, persist
    `recording_url` + expiry, and return a playback signed URL.

    NOTE: For Phase 1 we keep chunks individually and synthesise a manifest
    object containing the ordered chunk keys. Playback uses a single signed
    URL that the frontend stitches via the MediaSource Extensions API. This
    avoids the cost of downloading + re-uploading a single large file on the
    backend. Phase 2 (Jibri-based) writes a single MP4 directly.
    """
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        q = select(HuddleSQL).where(HuddleSQL.id == huddle_id, HuddleSQL.deleted_at.is_(None))
        row = (await session.execute(q)).scalar_one_or_none()
        if not row or row.host_user_id != requesting_user_id:
            return {"ok": False, "error": "not_found_or_forbidden"}
        org_id = row.organization_id
        retention = int(row.recording_retention_days or 30)

    # Write a tiny manifest.
    ext = "webm" if "webm" in content_type else "mp4" if "mp4" in content_type else "bin"
    manifest_key = f"{CHUNK_KEY_PREFIX}/{org_id}/{huddle_id}/manifest.json"
    chunk_keys: List[str] = [
        f"{CHUNK_KEY_PREFIX}/{org_id}/{huddle_id}/chunk-{i:06d}.{ext}"
        for i in range(total_chunks)
    ]
    import json
    manifest = {
        "huddle_id": huddle_id,
        "content_type": content_type,
        "total_chunks": total_chunks,
        "chunks": chunk_keys,
        "created_at": datetime.utcnow().isoformat(),
    }
    await storage_service.upload_file(
        json.dumps(manifest).encode("utf-8"),
        manifest_key,
        content_type="application/json",
    )

    expires_at = datetime.utcnow() + timedelta(days=retention)
    total_size = 0  # exact size sum requires per-chunk read; skip for Phase 1

    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        await session.execute(
            sa_update(HuddleSQL)
            .where(HuddleSQL.id == huddle_id)
            .values(
                recording_object_key=manifest_key,
                recording_mime=content_type,
                recording_expires_at=expires_at,
                recording_size_bytes=total_size,
                updated_at=datetime.utcnow(),
            )
        )
        await session.commit()

    playback = await get_recording_url(huddle_id, requesting_user_id=requesting_user_id)

    await log_activity(
        user_id=requesting_user_id,
        organization_id=org_id,
        activity_type="huddle.recording_stored",
        details={
            "huddle_id": huddle_id,
            "manifest_key": manifest_key,
            "total_chunks": total_chunks,
            "retention_days": retention,
        },
        related_resource_type="huddle",
        related_resource_id=huddle_id,
        agent_name="Huddle",
    )
    return {
        "ok": True,
        "manifest_key": manifest_key,
        "playback_url": playback.get("playback_url"),
        "chunk_urls": playback.get("chunk_urls"),
        "expires_at": expires_at.isoformat(),
    }


async def get_recording_url(
    huddle_id: str,
    *,
    requesting_user_id: str,
    expiry_seconds: int = 60 * 60 * 24,
) -> Dict[str, Any]:
    """Return signed playback URLs (one per chunk, manifest tells order)."""
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        q = select(HuddleSQL).where(HuddleSQL.id == huddle_id, HuddleSQL.deleted_at.is_(None))
        row = (await session.execute(q)).scalar_one_or_none()
        if not row:
            return {"ok": False, "error": "not_found"}
        # Allow host OR any participant
        from backend.db.postgres_models import HuddleParticipantSQL
        if row.host_user_id != requesting_user_id:
            pq = select(HuddleParticipantSQL).where(
                HuddleParticipantSQL.huddle_id == huddle_id,
                HuddleParticipantSQL.user_id == requesting_user_id,
            ).limit(1)
            if not (await session.execute(pq)).scalar_one_or_none():
                return {"ok": False, "error": "forbidden"}
        if not row.recording_object_key:
            return {"ok": False, "error": "no_recording"}
        if row.recording_expires_at and row.recording_expires_at < datetime.utcnow():
            return {"ok": False, "error": "expired"}

        manifest_key = row.recording_object_key
        content_type = row.recording_mime or "video/webm"

    # Download manifest to get chunk list
    import json
    raw = await storage_service.download_file(manifest_key)
    manifest = json.loads(raw.decode("utf-8"))
    chunk_keys: List[str] = manifest.get("chunks", [])

    # Generate a signed URL per chunk
    chunk_urls = [
        await storage_service.get_presigned_url(k, expiry=expiry_seconds)
        for k in chunk_keys
    ]
    manifest_url = await storage_service.get_presigned_url(manifest_key, expiry=expiry_seconds)
    return {
        "ok": True,
        "manifest_url": manifest_url,
        "playback_url": chunk_urls[0] if chunk_urls else None,
        "chunk_urls": chunk_urls,
        "content_type": content_type,
    }


# ── Retention sweep (called from Celery) ───────────────────────────────

async def expire_recording(huddle_id: str) -> Dict[str, Any]:
    """Delete all chunk objects + manifest, clear recording_url on the row."""
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        q = select(HuddleSQL).where(HuddleSQL.id == huddle_id)
        row = (await session.execute(q)).scalar_one_or_none()
        if not row or not row.recording_object_key:
            return {"ok": False, "error": "no_recording"}
        manifest_key = row.recording_object_key

    import json
    try:
        raw = await storage_service.download_file(manifest_key)
        manifest = json.loads(raw.decode("utf-8"))
        for k in manifest.get("chunks", []):
            try:
                await storage_service.delete_file(k)
            except Exception:
                pass
        await storage_service.delete_file(manifest_key)
    except Exception as e:
        logger.warning("huddle_expire_failed", huddle_id=huddle_id, error=str(e))
        return {"ok": False, "error": str(e)}

    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        await session.execute(
            sa_update(HuddleSQL)
            .where(HuddleSQL.id == huddle_id)
            .values(
                recording_object_key=None,
                recording_url=None,
                recording_expires_at=None,
                recording_size_bytes=None,
                updated_at=datetime.utcnow(),
            )
        )
        await session.commit()
    return {"ok": True}
