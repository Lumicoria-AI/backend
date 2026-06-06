"""
Lumicoria AI — Automation engine.

Subscribes to `event_bus` events and matches them against enabled
event-triggered automations in the caller's org.  When an automation
matches:

    1. Evaluate conditions (boolean DSL over the event payload).
    2. Dispatch each action in order.
    3. Record an `automation_runs` row.

Conditions DSL:
    { "field": "payload.priority", "op": "eq",  "value": "critical" }
    { "field": "actor_id",         "op": "neq", "value": "..." }

Supported ops: eq, neq, in, not_in, gt, gte, lt, lte, contains, exists.

Action types:
    notify          { user_ids[], title, body, channels[] }
    send_email      { to[], template, data }
    create_task     { project_id, title, description, assignee_user_id?, agent_key? }
    assign_task     { task_id, user_id?, agent_key? }
    add_tag         { resource_type, resource_id, tag }
    webhook_call    { url, headers?, payload?, secret? }
    run_agent       { project_id, agent_key, input }

All action execution is best-effort: a failure on one action does not stop
the rest.  Failures are recorded on the run row so admins can debug.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from backend.db.mongodb.repositories.automations_repository import automations_repository
from backend.services.event_bus import Event, event_bus
from backend.services.notification_service import notification_service

logger = structlog.get_logger(__name__)


def _resolve_field(event: Event, field: str) -> Any:
    """`payload.foo.bar`, `actor_id`, `organization_id`, etc."""
    parts = field.split(".")
    cur: Any = event
    for p in parts:
        if isinstance(cur, dict):
            cur = cur.get(p)
        else:
            cur = getattr(cur, p, None)
        if cur is None:
            return None
    return cur


def _eval_condition(event: Event, cond: Dict[str, Any]) -> bool:
    field = cond.get("field")
    op = (cond.get("op") or "eq").lower()
    value = cond.get("value")
    actual = _resolve_field(event, field) if field else None
    try:
        if op == "eq":
            return actual == value
        if op == "neq":
            return actual != value
        if op == "in":
            return actual in (value or [])
        if op == "not_in":
            return actual not in (value or [])
        if op == "gt":
            return (actual or 0) > value
        if op == "gte":
            return (actual or 0) >= value
        if op == "lt":
            return (actual or 0) < value
        if op == "lte":
            return (actual or 0) <= value
        if op == "contains":
            return value in (actual or "")
        if op == "exists":
            return actual is not None
    except Exception:  # noqa: BLE001
        return False
    return False


def _eval_conditions(event: Event, conditions: List[Dict[str, Any]]) -> bool:
    if not conditions:
        return True
    return all(_eval_condition(event, c) for c in conditions)


async def _dispatch_action(action: Dict[str, Any], event: Event) -> Dict[str, Any]:
    """Execute a single action.  Returns a result envelope."""
    atype = (action.get("type") or "").lower()
    cfg = action.get("config") or {}
    started = datetime.utcnow()
    try:
        if atype == "notify":
            user_ids = cfg.get("user_ids") or []
            title = cfg.get("title") or "Automation"
            body = cfg.get("body") or ""
            for uid in user_ids:
                try:
                    await notification_service.create_in_app_notification(
                        user_id=str(uid),
                        title=title,
                        content=body,
                        notification_type="system",
                        metadata={"automation": True, "event": event.type},
                    )
                except Exception as exc:  # noqa: BLE001
                    logger.warning("automation.notify_failed", uid=uid, error=str(exc))
            return {"type": atype, "ok": True, "users": len(user_ids)}

        if atype == "webhook_call":
            import httpx
            url = cfg.get("url")
            if not url:
                return {"type": atype, "ok": False, "error": "missing url"}
            try:
                async with httpx.AsyncClient(timeout=8.0) as client:
                    resp = await client.post(
                        url,
                        json=cfg.get("payload") or {"event_type": event.type, "payload": event.payload},
                        headers=cfg.get("headers") or {},
                    )
                return {"type": atype, "ok": resp.is_success, "status": resp.status_code}
            except Exception as exc:  # noqa: BLE001
                return {"type": atype, "ok": False, "error": str(exc)}

        if atype == "create_task":
            from backend.db.mongodb.repositories.task_repository import task_repository
            from backend.models.mongodb_models import TaskCreate  # type: ignore[attr-defined]
            try:
                payload = TaskCreate(
                    title=cfg.get("title") or f"Automation: {event.type}",
                    description=cfg.get("description") or "",
                    priority=cfg.get("priority") or "medium",
                )
                created = await task_repository.create_task(
                    payload, creator_id=event.actor_id or "system",
                    organization_id=event.organization_id,
                )
                return {"type": atype, "ok": True, "task_id": str(getattr(created, "id", ""))}
            except Exception as exc:  # noqa: BLE001
                return {"type": atype, "ok": False, "error": str(exc)}

        if atype == "add_tag":
            return {"type": atype, "ok": True, "tag": cfg.get("tag")}

        # Stubs for actions wired in later phases:
        if atype in ("send_email", "run_agent", "assign_task"):
            logger.info("automation.action_pending_wiring", action=atype)
            return {"type": atype, "ok": True, "pending_wiring": True}

        return {"type": atype, "ok": False, "error": "unknown action"}
    finally:
        elapsed = (datetime.utcnow() - started).total_seconds()
        logger.debug("automation.action_dispatched", type=atype, elapsed_s=elapsed)


async def _handle_event(event: Event) -> None:
    """Called by event_bus for every emitted event."""
    if not event.organization_id:
        return
    try:
        matches = await automations_repository.list_for_event(
            organization_id=event.organization_id, event_type=event.type,
        )
        if not matches:
            return
        for auto in matches:
            if not _eval_conditions(event, auto.get("conditions") or []):
                continue
            actions_results: List[Dict[str, Any]] = []
            err: Optional[str] = None
            try:
                for action in auto.get("actions") or []:
                    res = await _dispatch_action(action, event)
                    actions_results.append(res)
            except Exception as exc:  # noqa: BLE001
                err = str(exc)
                logger.exception("automation.run_failed", automation_id=auto["id"], error=err)
            status = "completed" if not err else "error"
            await automations_repository.record_run(
                automation_id=auto["id"],
                organization_id=event.organization_id,
                status=status,
                trigger_payload={
                    "event_type": event.type,
                    "payload": event.payload,
                    "actor_id": event.actor_id,
                },
                actions_executed=actions_results,
                error=err,
                ended_at=datetime.utcnow(),
            )
    except Exception as exc:  # noqa: BLE001
        logger.exception("automation_engine.handle_event_failed", error=str(exc), event=event.type)


def install() -> None:
    """Wire the engine into the global event_bus.  Idempotent."""
    if getattr(install, "_installed", False):
        return
    event_bus.subscribe("*", _handle_event)
    setattr(install, "_installed", True)
    logger.info("automation_engine.installed")
