"""Fetch the user's Gmail messages since the last brain run.

Pipeline:
  1. Resolve window: morning mode → last 24 h; evening mode → 12 h.
  2. List message ids since the window start (paginated, label-filtered).
  3. Batch-fetch each id via ``client.get_message`` under a 3-wide
     ``asyncio.Semaphore`` so we stay under Gmail's 250-req-per-100s
     per-user quota even on a very busy inbox.
  4. Map raw Gmail payloads into ``GmailMessageRef`` objects — we keep
     subject + snippet + label_ids + attachment_ids in state; the body
     is fetched again by the ingest node when needed so the LangGraph
     state payload stays bounded.

The payload_summary records counts so the trace row is searchable.
"""

from __future__ import annotations

import asyncio
import base64
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from typing import Any, Dict, List, Optional

import structlog

from ..state import BrainState, GmailMessageRef
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


_FETCH_CONCURRENCY = 3


@traced_node("fetch_gmail")
async def fetch_gmail(state: BrainState) -> Dict[str, Any]:
    client = state.meta.get("google_client")
    if client is None:
        return {
            "emails": [],
            "__payload_summary": {"count": 0, "reason": "no_client"},
            "__eval_score": 1.0,
        }

    prefs = state.meta.get("brain_prefs") or {}
    max_emails = int(prefs.get("max_emails_per_run", 100))
    exclude_labels = list(prefs.get("mailbox_labels_exclude") or [])
    include_labels = list(prefs.get("mailbox_labels_include") or [])

    # Window — morning gets a bigger reach back.
    hours_back = 24 if state.mode == "morning" else 12
    after_dt = datetime.utcnow() - timedelta(hours=hours_back)
    after_epoch = int(after_dt.timestamp())

    # 1. Lightweight id list, paginated under the hood.
    try:
        id_refs: List[Dict[str, str]] = await client.list_message_ids_since(
            after_epoch_seconds=after_epoch,
            label_ids=include_labels or None,
            exclude_label_ids=exclude_labels or None,
            max_results=max_emails,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("fetch_gmail.list_failed", error=str(exc))
        return {
            "emails": [],
            "__payload_summary": {"count": 0, "error": "list_failed"},
            "__eval_score": 0.0,
            "__status": "fallback",
        }

    if not id_refs:
        return {
            "emails": [],
            "__payload_summary": {
                "count": 0,
                "window_hours": hours_back,
                "after": after_dt.isoformat() + "Z",
            },
            "__eval_score": 1.0,
        }

    # 2. Batch fetch each id with concurrency cap.
    semaphore = asyncio.Semaphore(_FETCH_CONCURRENCY)

    async def _get_one(ref: Dict[str, str]) -> Optional[GmailMessageRef]:
        async with semaphore:
            try:
                msg = await client.get_message(ref["id"], fmt="full")
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "fetch_gmail.get_message_failed",
                    message_id=ref.get("id"), error=str(exc),
                )
                return None
            if msg is None:
                return None
            return _to_gmail_ref(msg, ref.get("threadId"))

    results = await asyncio.gather(*[_get_one(r) for r in id_refs])
    emails: List[GmailMessageRef] = [r for r in results if r is not None]

    payload = {
        "count": len(emails),
        "requested": len(id_refs),
        "window_hours": hours_back,
        "with_attachments": sum(1 for e in emails if e.has_attachments),
        "labels_excluded": exclude_labels,
    }

    eval_score = len(emails) / len(id_refs) if id_refs else 1.0

    return {
        "emails": emails,
        "__payload_summary": payload,
        "__eval_score": round(eval_score, 3),
    }


# ─────────────────────────────────────────────────────────────────────
# Gmail payload → GmailMessageRef mapper
# ─────────────────────────────────────────────────────────────────────


def _to_gmail_ref(msg: Dict[str, Any], thread_id: Optional[str]) -> GmailMessageRef:
    """Walk the Gmail v1 message payload into a flat ref. The full body
    isn't kept on state — it's re-fetched at ingest time when needed
    so the LangGraph state stays manageable on huge mailboxes."""
    headers = {
        h.get("name", "").lower(): h.get("value", "")
        for h in (msg.get("payload", {}).get("headers") or [])
    }

    received_at = _parse_received(headers.get("date"))
    label_ids = msg.get("labelIds") or []

    attachment_ids: List[str] = []
    _walk_attachments(msg.get("payload", {}), attachment_ids)

    return GmailMessageRef(
        message_id=msg.get("id", ""),
        thread_id=thread_id or msg.get("threadId"),
        subject=(headers.get("subject") or "")[:300] or None,
        from_addr=(headers.get("from") or "")[:240] or None,
        received_at=received_at,
        label_ids=list(label_ids),
        has_attachments=bool(attachment_ids),
        attachment_ids=attachment_ids,
        snippet=(msg.get("snippet") or "")[:500] or None,
    )


def _parse_received(date_str: Optional[str]) -> Optional[datetime]:
    if not date_str:
        return None
    try:
        dt = parsedate_to_datetime(date_str)
        if dt.tzinfo is not None:
            return dt.replace(tzinfo=None)
        return dt
    except Exception:
        return None


def _walk_attachments(part: Dict[str, Any], collector: List[str]) -> None:
    """Recurse the MIME part tree; collect attachmentId for each leaf
    part with a non-empty filename + body.attachmentId."""
    if not isinstance(part, dict):
        return
    body = part.get("body") or {}
    filename = part.get("filename") or ""
    if filename and body.get("attachmentId"):
        collector.append(body["attachmentId"])
    for sub in (part.get("parts") or []):
        _walk_attachments(sub, collector)
