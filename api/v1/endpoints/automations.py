"""
Phase C — Automations REST API.

Mounted at `/api/v1/automations`.

Org / team / project-scoped automation rules.  Triggers may be event-based,
schedule-based (cron), or manual.  The `backend/services/automation_engine.py`
worker subscribes to event_bus and dispatches matching rules.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.automations_repository import automations_repository
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.event_bus import emit

logger = structlog.get_logger(__name__)
router = APIRouter()


def _oid(v: Any) -> Optional[ObjectId]:
    if v is None:
        return None
    if isinstance(v, ObjectId):
        return v
    try:
        return ObjectId(str(v))
    except Exception:
        return None


def _resolve_org_id(current_user: User) -> str:
    primary = getattr(current_user, "organization_id", None)
    if primary:
        return str(primary)
    ids = getattr(current_user, "organization_ids", None) or []
    if ids:
        return str(ids[0])
    raise HTTPException(status_code=400, detail="User has no organization context")


async def _require_org_member(org_id: str, current_user: User):
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(m) for m in (org.member_ids or [])]:
        raise HTTPException(status_code=403, detail="Not a member of this organization")
    return org


class AutomationTrigger(BaseModel):
    type: str = Field(..., description="event | schedule | manual")
    config: Dict[str, Any] = Field(default_factory=dict)


class AutomationCondition(BaseModel):
    field: str
    op: str = "eq"
    value: Any = None


class AutomationAction(BaseModel):
    type: str
    config: Dict[str, Any] = Field(default_factory=dict)


class AutomationCreate(BaseModel):
    name: str = Field(..., max_length=200)
    description: Optional[str] = Field(None, max_length=2000)
    trigger: AutomationTrigger
    conditions: List[AutomationCondition] = Field(default_factory=list)
    actions: List[AutomationAction] = Field(default_factory=list)
    enabled: bool = True
    team_id: Optional[str] = None
    project_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AutomationUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    trigger: Optional[AutomationTrigger] = None
    conditions: Optional[List[AutomationCondition]] = None
    actions: Optional[List[AutomationAction]] = None
    enabled: Optional[bool] = None


@router.get("")
async def list_automations(
    organization_id: Optional[str] = Query(None),
    enabled: Optional[bool] = Query(None),
    project_id: Optional[str] = Query(None),
    team_id: Optional[str] = Query(None),
    trigger_type: Optional[str] = Query(None),
    limit: int = Query(200, ge=1, le=500),
    skip: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    return await automations_repository.list(
        organization_id=org_id, enabled=enabled,
        project_id=project_id, team_id=team_id,
        trigger_type=trigger_type, limit=limit, skip=skip,
    )


@router.post("", status_code=201)
async def create_automation(
    payload: AutomationCreate,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    row = await automations_repository.create(
        organization_id=org_id,
        name=payload.name,
        description=payload.description,
        trigger=payload.trigger.model_dump(),
        conditions=[c.model_dump() for c in payload.conditions],
        actions=[a.model_dump() for a in payload.actions],
        enabled=payload.enabled,
        team_id=payload.team_id, project_id=payload.project_id,
        created_by=str(current_user.id),
        metadata=payload.metadata,
    )
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="automation.created",
        details={"automation_id": row["id"], "name": row["name"]},
        related_resource_type="automation", related_resource_id=row["id"],
    )
    return row


@router.get("/{automation_id}")
async def get_automation(
    automation_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    row = await automations_repository.get(automation_id, organization_id=org_id)
    if not row:
        raise HTTPException(status_code=404, detail="Automation not found")
    return row


@router.patch("/{automation_id}")
async def update_automation(
    automation_id: str,
    payload: AutomationUpdate,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    patch = payload.model_dump(exclude_unset=True)
    if "trigger" in patch and patch["trigger"]:
        patch["trigger"] = AutomationTrigger(**patch["trigger"]).model_dump()
    if "conditions" in patch and patch["conditions"] is not None:
        patch["conditions"] = [AutomationCondition(**c).model_dump() for c in patch["conditions"]]
    if "actions" in patch and patch["actions"] is not None:
        patch["actions"] = [AutomationAction(**a).model_dump() for a in patch["actions"]]
    row = await automations_repository.update(automation_id, org_id, patch=patch)
    if not row:
        raise HTTPException(status_code=404, detail="Automation not found")
    return row


@router.delete("/{automation_id}", status_code=204)
async def delete_automation(
    automation_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    await automations_repository.delete(automation_id, organization_id=org_id)
    return None


@router.post("/{automation_id}/enable")
async def enable_automation(
    automation_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    row = await automations_repository.update(automation_id, org_id, patch={"enabled": True})
    if not row:
        raise HTTPException(status_code=404, detail="Automation not found")
    return row


@router.post("/{automation_id}/disable")
async def disable_automation(
    automation_id: str,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    row = await automations_repository.update(automation_id, org_id, patch={"enabled": False})
    if not row:
        raise HTTPException(status_code=404, detail="Automation not found")
    return row


class TestRunPayload(BaseModel):
    event_payload: Dict[str, Any] = Field(default_factory=dict)


@router.post("/{automation_id}/test-run")
async def test_run_automation(
    automation_id: str,
    payload: TestRunPayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    """Synthetic event injected into the automation_engine for this rule
    only.  Useful for dry-runs from the UI."""
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    auto = await automations_repository.get(automation_id, organization_id=org_id)
    if not auto:
        raise HTTPException(status_code=404, detail="Automation not found")
    trigger_cfg = auto.get("trigger", {}) or {}
    event_type = (trigger_cfg.get("config") or {}).get("event_type") or "automation.test"
    await emit(event_type, organization_id=org_id, actor_id=str(current_user.id),
               payload=payload.event_payload, source="ui")
    return {"ok": True, "event": event_type}


@router.get("/runs")
async def list_runs(
    organization_id: Optional[str] = Query(None),
    automation_id: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    skip: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_org_id(current_user)
    await _require_org_member(org_id, current_user)
    return await automations_repository.list_runs(
        organization_id=org_id, automation_id=automation_id, status=status,
        limit=limit, skip=skip,
    )


@router.get("/event-catalogue")
async def list_event_catalogue():
    """Catalogue of event types automations can trigger on."""
    return {
        "events": [
            {"type": "task.created", "scope": "task"},
            {"type": "task.updated", "scope": "task"},
            {"type": "task.completed", "scope": "task"},
            {"type": "task.assigned", "scope": "task"},
            {"type": "task.proposal_ready", "scope": "task"},
            {"type": "project.created", "scope": "project"},
            {"type": "project.member_added", "scope": "project"},
            {"type": "team.created", "scope": "team"},
            {"type": "team.member_added", "scope": "team"},
            {"type": "agent.run_completed", "scope": "agent"},
            {"type": "agent.run_failed", "scope": "agent"},
            {"type": "document.uploaded", "scope": "document"},
            {"type": "invite.accepted", "scope": "invite"},
            {"type": "org.seat_assigned", "scope": "org"},
            {"type": "comment.created", "scope": "comment"},
            {"type": "automation.test", "scope": "internal"},
        ],
        "actions": [
            {"type": "notify", "description": "Create an in-app notification"},
            {"type": "send_email", "description": "Send a templated email (Phase C+)"},
            {"type": "create_task", "description": "Open a new task"},
            {"type": "assign_task", "description": "Re-assign a task (Phase C+)"},
            {"type": "add_tag", "description": "Tag a resource"},
            {"type": "webhook_call", "description": "POST to an external URL"},
            {"type": "run_agent", "description": "Trigger an agent run (Phase C+)"},
        ],
        "condition_ops": ["eq", "neq", "in", "not_in", "gt", "gte", "lt", "lte", "contains", "exists"],
    }
