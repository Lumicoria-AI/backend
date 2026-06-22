"""Download Gmail attachments to MinIO in parallel.

For each GmailMessageRef with ``has_attachments``:
  1. For each attachment_id, fetch raw bytes via ``client.get_attachment``
     (which already has 429-aware retry).
  2. Reject by size + MIME guard (skip executables, archives, oversize).
  3. Upload to ``storage_service`` at a stable per-user key:
        users/{user_id}/gmail/{message_id}/{attachment_id}/<filename>
     Phase 6 will swap ``upload`` for ``upload_encrypted`` so bytes are
     CMK-wrapped before they land in MinIO.
  4. Stash the (message_id, attachment_id, key, mime, size, filename)
     tuple in ``state.meta["attachment_blobs"]`` for the ingest node
     to read.

A 5-wide ``asyncio.Semaphore`` caps concurrent downloads. The Gmail
quota (250/100s per user) easily accommodates 5 simultaneous
attachment GETs even on a heavy mailbox.
"""

from __future__ import annotations

import asyncio
import mimetypes
from typing import Any, Dict, List, Tuple

import structlog

from ..state import BrainState, GmailMessageRef
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


_CONCURRENCY = 5
_DEFAULT_MAX_BYTES = 25 * 1024 * 1024  # 25 MB
_SKIP_MIMES = {
    "application/zip",
    "application/x-zip-compressed",
    "application/x-rar-compressed",
    "application/x-7z-compressed",
    "application/x-tar",
    "application/x-msdownload",  # .exe
    "application/x-sh",
    "application/octet-stream",  # unknown — let MIME sniff fail safe
}


@traced_node("download_attachments")
async def download_attachments(state: BrainState) -> Dict[str, Any]:
    client = state.meta.get("google_client")
    if client is None or not state.emails:
        return {
            "__payload_summary": {
                "messages_with_attachments": 0,
                "downloaded": 0,
                "skipped": 0,
            },
            "__eval_score": 1.0,
        }

    prefs = state.meta.get("brain_prefs") or {}
    max_total = int(prefs.get("max_attachments_per_run", 20))
    max_bytes = int(prefs.get("max_attachment_bytes", _DEFAULT_MAX_BYTES))

    # Flatten the (msg_id, attachment_id, filename) work list.
    work: List[Tuple[str, str, str]] = []
    for m in state.emails:
        if not m.has_attachments:
            continue
        for aid in m.attachment_ids:
            work.append((m.message_id, aid, _filename_guess(m.subject or "attachment")))
            if len(work) >= max_total:
                break
        if len(work) >= max_total:
            break

    if not work:
        return {
            "__payload_summary": {
                "messages_with_attachments": sum(
                    1 for m in state.emails if m.has_attachments
                ),
                "downloaded": 0,
                "skipped": 0,
            },
            "__eval_score": 1.0,
        }

    semaphore = asyncio.Semaphore(_CONCURRENCY)

    async def _one(item: Tuple[str, str, str]) -> Dict[str, Any] | None:
        msg_id, att_id, fname = item
        async with semaphore:
            try:
                data = await client.get_attachment(msg_id, att_id)
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "download_attachments.get_failed",
                    message_id=msg_id, attachment_id=att_id, error=str(exc),
                )
                return None
            if not data:
                return None
            if len(data) > max_bytes:
                logger.info(
                    "download_attachments.oversize_skipped",
                    message_id=msg_id, size=len(data),
                )
                return None

            mime, _ = mimetypes.guess_type(fname)
            mime = mime or "application/octet-stream"
            if mime in _SKIP_MIMES:
                logger.info(
                    "download_attachments.mime_blocked",
                    message_id=msg_id, mime=mime,
                )
                return None

            key = f"users/{state.user_id}/gmail/{msg_id}/{att_id}/{fname}"
            # CMK-wrap before upload. The org_id is what scopes the
            # encryption KEK — in personal mode we fall back to the
            # user_id (same convention used elsewhere in the brain).
            org_id_for_cmk = state.organization_id or state.user_id
            try:
                from backend.services.storage_service import storage_service
                upload_result = await storage_service.upload_encrypted(
                    file_content=data,
                    key=key,
                    content_type=mime,
                    organization_id=org_id_for_cmk,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "download_attachments.upload_failed",
                    key=key, error=str(exc),
                )
                return None

            return {
                "message_id": msg_id,
                "attachment_id": att_id,
                "filename": fname,
                "mime_type": mime,
                "size": len(data),
                "minio_key": key,
                "encrypted": bool(upload_result.get("encrypted")),
            }

    results = await asyncio.gather(*[_one(w) for w in work])
    blobs: List[Dict[str, Any]] = [r for r in results if r is not None]

    # Stash for the ingest node. state.meta is a dict so this passes
    # through LangGraph without serialisation.
    new_meta = {**state.meta, "attachment_blobs": blobs}

    eval_score = (len(blobs) / len(work)) if work else 1.0

    # Per-attachment activity log — IDs only, never the bytes / filename
    # content beyond the bucket key. Used by the org audit feed.
    try:
        from backend.services.activity_logger import log_activity
        for b in blobs:
            await log_activity(
                user_id=state.user_id,
                organization_id=state.organization_id,
                activity_type="brain.attachment_ingested",
                details={
                    "run_id": state.run_id,
                    "message_id": b.get("message_id"),
                    "attachment_id": b.get("attachment_id"),
                    "mime_type": b.get("mime_type"),
                    "size": b.get("size"),
                    "encrypted": bool(b.get("encrypted")),
                },
                related_resource_type="BRAIN_RUN",
                related_resource_id=state.run_id,
            )
    except Exception:
        pass

    return {
        "meta": new_meta,
        "__payload_summary": {
            "messages_with_attachments": sum(
                1 for m in state.emails if m.has_attachments
            ),
            "attempted": len(work),
            "downloaded": len(blobs),
            "skipped": len(work) - len(blobs),
            "encrypted": sum(1 for b in blobs if b.get("encrypted")),
        },
        "__eval_score": round(eval_score, 3),
    }


def _filename_guess(seed: str) -> str:
    """Gmail's API doesn't always carry the original filename on the
    attachment metadata — we synthesise one from the subject so the
    MinIO key + downstream parser routing has something to work with."""
    import re
    base = re.sub(r"[^A-Za-z0-9_.-]+", "-", seed)[:60].strip("-") or "attachment"
    return base + ".bin"
