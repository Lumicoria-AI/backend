"""Persist ranked actions as Lumicoria tasks.

For each ``RankedAction``:
  - Build a ``TaskCreate`` payload with metadata.source="brain",
    metadata.run_id, evidence_message_ids/event_ids/file_ids — so the
    frontend can render a "this came from your inbox" badge and the
    digest email can deep-link back to the originating email/file.
  - Call ``task_repository.create_task`` directly (bypasses the API's
    extra org-membership middleware; the brain is already an
    authenticated server-side flow).
  - Stamp ``created_by = user_id``; ``assigned_to = user_id`` so the
    task lands in the user's own queue.
  - Idempotency: skip duplicates when a task with the same
    ``metadata.brain_evidence_key`` was already created during this
    digest_run_id — happens on retries.

Per-task failure doesn't abort the run; we log + continue. The
returned ``created_task_ids`` feeds ``fire_agents`` next.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List

import structlog
from bson import ObjectId

from ..state import BrainState, RankedAction
from ..tracing import traced_node

logger = structlog.get_logger(__name__)


@traced_node("create_tasks")
async def create_tasks(state: BrainState) -> Dict[str, Any]:
    if not state.ranked_actions:
        return {
            "created_task_ids": [],
            "__payload_summary": {
                "ranked_actions": 0, "created": 0, "skipped_duplicates": 0,
                "failed": 0,
            },
            "__eval_score": 1.0,
        }

    try:
        from backend.db.mongodb.repositories.task_repository import task_repository
        from backend.models.mongodb_models import (
            TaskCreate, TaskPriority, TaskStatus, AssigneeKind,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("create_tasks.imports_failed", error=str(exc))
        return {
            "created_task_ids": [],
            "__payload_summary": {"error": "imports_failed"},
            "__eval_score": 0.0,
            "__status": "fallback",
        }

    try:
        uid_oid: Any = ObjectId(state.user_id)
    except Exception:
        uid_oid = state.user_id

    org_id_for_create = state.organization_id or state.user_id

    created_ids: List[str] = []
    failed = 0
    skipped_dupes = 0

    # Quick dedupe against this run — pre-load existing brain tasks for
    # this run_id once, then check titles in-memory.
    seen_keys = await _existing_keys_for_run(state.user_id, state.run_id)

    for action in state.ranked_actions:
        evidence_key = _evidence_key(action)
        if evidence_key in seen_keys:
            skipped_dupes += 1
            continue
        seen_keys.add(evidence_key)

        try:
            priority_enum = _coerce_priority(action.priority)
        except Exception:
            priority_enum = TaskPriority.MEDIUM

        try:
            payload = TaskCreate(
                title=action.title[:200] or "Brain-generated task",
                description=action.description[:1500] if action.description else None,
                status=TaskStatus.TODO,
                priority=priority_enum,
                due_date=action.due_date,
                assigned_to=uid_oid,
                assigned_to_agent=action.assigned_to_agent,
                assignee_kind=(
                    AssigneeKind.AGENT
                    if action.assigned_to_agent
                    else AssigneeKind.USER
                ),
                metadata={
                    "source": "brain",
                    "brain_mode": state.mode,
                    "brain_run_id": state.run_id,
                    "brain_confidence": float(action.confidence or 0.0),
                    "evidence_message_ids": list(action.evidence_message_ids),
                    "evidence_event_ids": list(action.evidence_event_ids),
                    "evidence_file_ids": list(action.evidence_file_ids),
                    "brain_evidence_key": evidence_key,
                },
            )
            created = await task_repository.create_task(
                task_data=payload,
                creator_id=state.user_id,
                organization_id=org_id_for_create,
            )
            tid = str(getattr(created, "id", "") or "")
            if tid:
                created_ids.append(tid)
                # One audit row per created task — title is the brain's
                # extracted title (already filtered for PII by the
                # classify + prioritise stages).
                try:
                    from backend.services.activity_logger import log_activity
                    await log_activity(
                        user_id=state.user_id,
                        organization_id=state.organization_id,
                        activity_type="brain.task_created",
                        details={
                            "run_id": state.run_id,
                            "task_id": tid,
                            "title": action.title[:120],
                            "priority": action.priority,
                            "assigned_to_agent": action.assigned_to_agent,
                            "confidence": float(action.confidence or 0.0),
                        },
                        related_resource_type="TASK",
                        related_resource_id=tid,
                    )
                except Exception:
                    pass
            else:
                failed += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            logger.warning(
                "create_tasks.create_failed",
                title=action.title[:60], error=str(exc),
            )

    return {
        "created_task_ids": created_ids,
        "__payload_summary": {
            "ranked_actions": len(state.ranked_actions),
            "created": len(created_ids),
            "skipped_duplicates": skipped_dupes,
            "failed": failed,
        },
        "__eval_score": (
            (len(created_ids) + skipped_dupes) / len(state.ranked_actions)
            if state.ranked_actions else 1.0
        ),
    }


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


def _evidence_key(action: RankedAction) -> str:
    """Stable identity for an action — used to dedupe inside one run."""
    parts: List[str] = [action.title.strip().lower()[:80]]
    parts.extend(sorted(action.evidence_message_ids))
    parts.extend(sorted(action.evidence_event_ids))
    parts.extend(sorted(action.evidence_file_ids))
    return "|".join(parts)


def _coerce_priority(p: str):
    """Map our RankedAction priority literal → TaskPriority enum."""
    from backend.models.mongodb_models import TaskPriority
    mapping = {
        "critical": TaskPriority.CRITICAL,
        "high": TaskPriority.HIGH,
        "medium": TaskPriority.MEDIUM,
        "low": TaskPriority.LOW,
    }
    return mapping.get((p or "medium").lower(), TaskPriority.MEDIUM)


async def _existing_keys_for_run(user_id: str, run_id: str) -> set[str]:
    """Pull `metadata.brain_evidence_key` for any task already created
    in this run — covers retry safety + idempotent re-fires."""
    try:
        from backend.db.mongodb.mongodb import MongoDB
        db = await MongoDB.get_database()
        try:
            uid_oid: Any = ObjectId(user_id)
        except Exception:
            uid_oid = user_id
        cursor = db.tasks.find(
            {
                "created_by": uid_oid,
                "metadata.brain_run_id": run_id,
            },
            projection={"metadata.brain_evidence_key": 1},
        )
        keys: set[str] = set()
        async for t in cursor:
            md = t.get("metadata") or {}
            key = md.get("brain_evidence_key")
            if key:
                keys.add(str(key))
        return keys
    except Exception:
        return set()
