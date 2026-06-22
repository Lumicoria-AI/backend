"""Close out the run — write activity-log entries for visibility.

The Postgres BrainRun + BrainTrace rows are written by the runner
*after* this node returns, in a batched commit. This node records
app-level activity (the one shown in /workspace/admin/audit), so a
human auditing the org can see "brain ran for user X at hh:mm,
created Y tasks, sent digest to email".

Care is taken to log IDs + counts only — never the email body, never
the proposal content. That's the compliance bar for Phase 6 (CMK)
and we may as well enforce it now.
"""

from __future__ import annotations

from typing import Any, Dict

import structlog

from ..state import BrainState
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


@traced_node("audit")
async def audit(state: BrainState) -> Dict[str, Any]:
    try:
        from backend.services.activity_logger import log_activity
    except Exception:
        log_activity = None  # type: ignore[assignment]

    org_id = state.organization_id or state.user_id

    # Compose a single rolled-up event for the run.
    details: Dict[str, Any] = {
        "run_id": state.run_id,
        "mode": state.mode,
        "skip_reason": state.skip_reason,
        "emails_processed": len(state.emails),
        "attachments_processed": len((state.meta or {}).get("attachment_blobs") or []),
        "ingested_doc_ids": len(state.ingested_doc_ids),
        "tasks_created": len(state.created_task_ids),
        "proposals_drafted": sum(
            1 for s in (state.proposal_status_by_task or {}).values()
            if s in ("pending_review", "approved")
        ),
        "delivery_channels": list(state.delivery_channels or []),
        "fallback_count": state.fallback_count,
    }

    activity_type = (
        "brain.run_skipped" if state.skip_reason
        else "brain.run_completed"
    )

    if log_activity is not None:
        try:
            await log_activity(
                user_id=state.user_id,
                organization_id=org_id,
                activity_type=activity_type,
                details=details,
                related_resource_type="BRAIN_RUN",
                related_resource_id=state.run_id,
                source="brain",
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("audit.activity_log_failed", error=str(exc))

    if state.fallback_count > 0:
        # Surface degradation as its own activity row so the admin
        # filter for "brain degraded today" is one query.
        if log_activity is not None:
            try:
                await log_activity(
                    user_id=state.user_id,
                    organization_id=org_id,
                    activity_type="brain.degraded",
                    details={
                        "run_id": state.run_id,
                        "fallback_count": state.fallback_count,
                        "mode": state.mode,
                    },
                    related_resource_type="BRAIN_RUN",
                    related_resource_id=state.run_id,
                    source="brain",
                )
            except Exception:
                pass

    return {
        "__payload_summary": details,
        "__eval_score": 1.0,
    }
