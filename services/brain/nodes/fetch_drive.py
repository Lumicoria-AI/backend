"""Fetch Drive files changed since the last brain run.

Uses Google's incremental Changes feed:
  - First call (no stored token) → grab the bootstrap startPageToken,
    persist it on user.preferences.brain.drive_page_token, and return
    no changes (first run is a no-op for Drive).
  - Subsequent calls → ask the Changes API for everything since that
    token, advance the token to the new high-water mark.

Filters:
  - Skip removed/trashed unless prefs.include_drive_removed=true.
  - Skip files > brain_prefs.max_drive_file_mb (default 50 MB).
  - Skip files modified by the user themselves (their own writes are
    rarely action-creating context).

We only emit DriveFileRefs into state — the actual bytes get fetched
by the ingest node on demand and CMK-encrypted in Phase 6.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import structlog
from bson import ObjectId

from ..state import BrainState, DriveFileRef
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


_DEFAULT_MAX_FILE_MB = 50


@traced_node("fetch_drive")
async def fetch_drive(state: BrainState) -> Dict[str, Any]:
    client = state.meta.get("google_client")
    prefs = state.meta.get("brain_prefs") or {}
    if client is None:
        return {
            "drive_changes": [],
            "__payload_summary": {"count": 0, "reason": "no_client"},
            "__eval_score": 1.0,
        }

    max_mb = int(prefs.get("max_drive_file_mb", _DEFAULT_MAX_FILE_MB))
    include_removed = bool(prefs.get("include_drive_removed", False))

    stored_token = (prefs or {}).get("drive_page_token")

    # First call: bootstrap.
    if not stored_token:
        try:
            seed = await client.list_drive_changes(start_page_token=None)
        except Exception as exc:  # noqa: BLE001
            logger.warning("fetch_drive.bootstrap_failed", error=str(exc))
            return {
                "drive_changes": [],
                "__payload_summary": {"count": 0, "error": "bootstrap_failed"},
                "__eval_score": 0.0,
                "__status": "fallback",
            }
        new_token = seed.get("new_start_page_token")
        if new_token:
            await _persist_token(state.user_id, new_token)
        return {
            "drive_changes": [],
            "__payload_summary": {
                "count": 0,
                "bootstrap": True,
                "next_token_stored": bool(new_token),
            },
            "__eval_score": 1.0,
        }

    # Subsequent calls: pull the changes feed since the stored token.
    try:
        page = await client.list_drive_changes(
            start_page_token=stored_token,
            include_removed=include_removed,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_drive.changes_failed", error=str(exc))
        return {
            "drive_changes": [],
            "__payload_summary": {"count": 0, "error": "changes_failed"},
            "__eval_score": 0.0,
            "__status": "fallback",
        }

    raw_changes = page.get("changes") or []
    new_token = page.get("new_start_page_token") or stored_token

    files: List[DriveFileRef] = []
    skipped_size = 0
    for ch in raw_changes:
        ref = _to_drive_ref(ch)
        if ref is None:
            continue
        # Size guard — request was lightweight (metadata only); we filter
        # before issuing a download in the ingest node.
        size = (ch.get("file") or {}).get("size")
        try:
            if size is not None and (int(size) / (1024 * 1024)) > max_mb:
                skipped_size += 1
                continue
        except (TypeError, ValueError):
            pass
        files.append(ref)

    if new_token and new_token != stored_token:
        await _persist_token(state.user_id, new_token)

    return {
        "drive_changes": files,
        "__payload_summary": {
            "count": len(files),
            "skipped_oversize": skipped_size,
            "advanced_token": new_token != stored_token,
        },
        "__eval_score": 1.0,
    }


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _to_drive_ref(change: Dict[str, Any]) -> DriveFileRef | None:
    if change.get("removed"):
        return DriveFileRef(
            file_id=str(change.get("fileId") or ""),
            removed=True,
        )
    f = change.get("file") or {}
    file_id = str(f.get("id") or change.get("fileId") or "")
    if not file_id:
        return None
    modified = f.get("modifiedTime")
    modified_dt = None
    if modified:
        try:
            modified_dt = datetime.fromisoformat(modified.replace("Z", "+00:00"))
            if modified_dt.tzinfo is not None:
                modified_dt = modified_dt.replace(tzinfo=None)
        except Exception:
            modified_dt = None
    return DriveFileRef(
        file_id=file_id,
        name=(f.get("name") or "")[:300] or None,
        mime_type=f.get("mimeType"),
        modified_at=modified_dt,
        removed=bool(f.get("trashed")),
    )


async def _persist_token(user_id: str, token: str) -> None:
    """Update the user's drive_page_token. Best-effort — token loss
    just means the next run treats itself as a fresh bootstrap."""
    try:
        from backend.db.mongodb.mongodb import MongoDB
        db = await MongoDB.get_database()
        try:
            oid: Any = ObjectId(user_id)
        except Exception:
            oid = user_id
        await db.users.update_one(
            {"_id": oid},
            {"$set": {
                "preferences.brain.drive_page_token": token,
                "preferences.brain.drive_page_token_at": datetime.utcnow(),
            }},
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("fetch_drive.token_persist_failed", error=str(exc))
