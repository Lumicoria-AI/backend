"""Download Gmail attachments in parallel, with a semaphore.

Phase 2 stub. Phase 3 will:
  - For each GmailMessageRef with has_attachments, iterate attachment_ids.
  - `asyncio.Semaphore(5)` caps concurrent downloads — protects the
    Gmail per-user quota and keeps VM memory bounded for large PDFs.
  - Per-attachment retry x3 via `google_workspace_client.get_attachment`
    (which already has tenacity backoff on 429/5xx).
  - Size guard: skip > settings.BRAIN_MAX_ATTACHMENT_BYTES (default 25MB).
  - Type guard: skip MIME types we can't parse (zips, executables).
  - Upload bytes via `storage_service.upload_encrypted` (Phase 6 wires
    CMK) at key `users/{user_id}/gmail/{message_id}/{attachment_id}/{filename}`.
  - Stash MinIO keys in state.meta["attachment_keys"] for the ingest
    node to consume.
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState
from ..tracing import traced_node


@traced_node("download_attachments")
async def download_attachments(state: BrainState) -> Dict[str, Any]:
    total_attachments = sum(len(m.attachment_ids) for m in state.emails)
    return {
        "__payload_summary": {
            "messages_with_attachments": sum(
                1 for m in state.emails if m.has_attachments
            ),
            "attachments_total": total_attachments,
            "downloaded": 0,
            "skipped": 0,
        },
        "__eval_score": 1.0,
    }
