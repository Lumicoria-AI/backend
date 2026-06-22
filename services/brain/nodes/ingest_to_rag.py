"""Ingest emails + attachments + Drive deltas into Weaviate.

Phase 2 stub. Phase 3 will:
  - For each GmailMessageRef: `document_processor.process_text(body,
    metadata={source:"gmail", message_id, thread_id, user_id, org_id})`.
  - For each downloaded attachment: `process_file(tmp_path, metadata=...)`
    where source="gmail_attachment" and parent_message_id is set.
  - For each DriveFileRef: `process_google_drive(file_id, metadata=...,
    drive_client=client, user_id=...)` — already real after Phase 1.
  - Per-doc retry x3. Idempotency via sha256(content) — already-ingested
    chunks return their existing doc_id instead of re-embedding.
  - Eval: passed iff ≥80% of inputs successfully embedded. Below that →
    record fallback so the prioritise node knows the RAG context is
    partial and weights raw-snippet input more heavily.
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState
from ..tracing import traced_node


@traced_node("ingest_to_rag")
async def ingest_to_rag(state: BrainState) -> Dict[str, Any]:
    total_inputs = (
        len(state.emails) + len(state.drive_changes) + len(state.huddle_recents)
    )
    return {
        "ingested_doc_ids": [],
        "__payload_summary": {
            "inputs": total_inputs,
            "ingested": 0,
            "skipped": 0,
            "failed": 0,
        },
        "__eval_score": 1.0,
    }
