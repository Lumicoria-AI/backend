from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Query, Body
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from datetime import datetime, timedelta
from enum import Enum

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.task_repository import task_repository
from backend.db.postgres import get_optional_async_db
from backend.db.postgres_repositories.task_repository import PostgresTaskRepository
from backend.core.config import settings
from sqlalchemy.ext.asyncio import AsyncSession
from backend.models.user import User
from backend.models.task import Task, TaskCreate, TaskUpdate, TaskStatus, TaskPriority
from backend.services.activity_logger import log_activity
from backend.services.calendar_service import calendar_service
from backend.services.task_action_tokens import (
    TaskAction,
    TaskActionTokenError,
    decode_action_token,
)

router = APIRouter()


def _get_org_id(user: User) -> str:
    """Safely extract organization_id from the user, falling back to user.id."""
    return getattr(user, "organization_id", None) or str(user.id)


def _stringify_objectids(value: Any) -> Any:
    """Recursively walk dicts / lists and turn every `ObjectId` into a str.

    Used to JSON-serialise tasks for the frontend.  Phase 1 added new
    `ObjectId`-typed fields (`calendar_event_id`, `invite_id`) and Phase 5
    nested ObjectIds inside `status_history` / `assignment_history` entries
    (`changed_by`, `assigned_to`).  Touching them by name is brittle — a
    recursive cleanup is bulletproof and ~zero overhead at this scale.
    """
    from bson import ObjectId as _OID
    if isinstance(value, _OID):
        return str(value)
    if isinstance(value, list):
        return [_stringify_objectids(v) for v in value]
    if isinstance(value, dict):
        return {k: _stringify_objectids(v) for k, v in value.items()}
    return value


class TaskResponse(BaseModel):
    id: str
    title: str
    name: Optional[str] = None  # Alias for title (backwards compat)
    description: Optional[str] = None
    status: TaskStatus
    due_date: Optional[datetime] = None
    priority: Optional[str] = None
    organization_id: Optional[str] = None
    created_by: Optional[str] = None
    assigned_to: Optional[str] = None
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    progress: Optional[int] = None
    document_id: Optional[str] = None  # Link to source document if applicable
    agent_id: Optional[str] = None  # Link to agent that created the task if applicable

class TaskSummaryResponse(BaseModel):
    total_tasks: int
    tasks_by_status: Dict[TaskStatus, int]
    tasks_by_priority: Dict[Optional[int], int]
    overdue_tasks_count: int
    upcoming_tasks_count: int
    # Add other relevant summary fields from TaskRepository.get_task_stats if needed

@router.post("/", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
async def create_task(
    task_in: TaskCreate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    Create a new task.
    """
    try:
        task = await task_repository.create_task(
            task_data=task_in,
            creator_id=current_user.id,
            organization_id=_get_org_id(current_user)
        )

        # Dual-write to Postgres if enabled
        if settings.POSTGRES_ENABLED and settings.POSTGRES_DUAL_WRITE and db is not None:
            try:
                repo = PostgresTaskRepository(db)
                await repo.create_task(
                    task_data=task_in.model_dump() if hasattr(task_in, "model_dump") else task_in.dict(),
                    creator_id=current_user.id,
                    organization_id=_get_org_id(current_user)
                )
            except Exception:
                pass

        await log_activity(
            user_id=str(current_user.id),
            organization_id=_get_org_id(current_user),
            activity_type="task.created",
            details={"title": task_in.title, "status": task_in.status.value if task_in.status else "todo", "priority": str(task_in.priority.value) if task_in.priority else "medium"},
            related_resource_type="TASK",
            related_resource_id=str(task.id) if hasattr(task, "id") else None,
        )

        # ── Phase 2: auto-create a Lumicoria calendar event when the task
        # has a due_date.  Failure-tolerant — never breaks the task POST.
        try:
            if getattr(task, "due_date", None):
                event = await calendar_service.create_event_for_task(
                    task, owner_user_id=str(current_user.id)
                )
                if event:
                    await task_repository.update_task(
                        task_id=str(task.id),
                        organization_id=_get_org_id(current_user),
                        update_data={"calendar_event_id": str(event.id)},
                    )
        except Exception:  # pragma: no cover — observability handled by service
            pass

        # Phase 5: serialise ObjectIds before returning so FastAPI's encoder
        # doesn't choke on calendar_event_id / status_history.changed_by etc.
        t_dict = task.model_dump() if hasattr(task, "model_dump") else task.dict()
        if "_id" in t_dict:
            t_dict["id"] = str(t_dict.pop("_id"))
        elif "id" in t_dict:
            t_dict["id"] = str(t_dict["id"])
        return _stringify_objectids(t_dict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{task_id}", response_model=None)
async def get_task(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    Get a task by ID.
    """
    task = await task_repository.get_task_by_id(
        task_id=task_id,
        organization_id=_get_org_id(current_user)
    )
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Task not found"
        )
    if str(task.organization_id) != _get_org_id(current_user):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Not authorized to access this task")
    # Serialize ObjectIds (Phase 5 — same recursive cleanup as list_tasks)
    t_dict = task.model_dump() if hasattr(task, "model_dump") else task.dict()
    if "_id" in t_dict:
        t_dict["id"] = str(t_dict.pop("_id"))
    elif "id" in t_dict:
        t_dict["id"] = str(t_dict["id"])
    return _stringify_objectids(t_dict)

@router.put("/{task_id}", response_model=None)
async def update_task(
    task_id: str,
    task_in: TaskUpdate,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    Update a task.
    """
    existing_task = await task_repository.get_task_by_id(task_id, organization_id=_get_org_id(current_user))
    if not existing_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found or not authorized")

    task = await task_repository.update_task(
        task_id=task_id,
        organization_id=_get_org_id(current_user),
        update_data=task_in.dict(exclude_unset=True),
        changed_by=str(current_user.id),
        changed_by_name=getattr(current_user, "full_name", None),
    )
    if settings.POSTGRES_ENABLED and settings.POSTGRES_DUAL_WRITE and db is not None:
        try:
            repo = PostgresTaskRepository(db)
            await repo.update_task(
                task_id=task_id,
                organization_id=_get_org_id(current_user),
                update_data=task_in.dict(exclude_unset=True)
            )
        except Exception:
            pass
    if not task:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update task"
        )

    await log_activity(
        user_id=str(current_user.id),
        organization_id=_get_org_id(current_user),
        activity_type="task.updated",
        details={"task_id": task_id, "updated_fields": list(task_in.dict(exclude_unset=True).keys())},
        related_resource_type="TASK",
        related_resource_id=task_id,
    )

    # ── Phase 5: notify the task creator when someone else updates status ──
    try:
        patched_fields = set(task_in.dict(exclude_unset=True).keys())
        if "status" in patched_fields:
            creator_id = str(getattr(existing_task, "created_by", "") or "")
            if creator_id and creator_id != str(current_user.id):
                from backend.services.notification_service import notification_service
                from backend.db.mongodb.models.notification import (
                    NotificationPriority, NotificationType,
                )
                actor_name = getattr(current_user, "full_name", None) or getattr(current_user, "email", "A teammate")
                new_status = task_in.status.value if hasattr(task_in.status, "value") else str(task_in.status)
                task_title = getattr(task, "title", "a task")
                label = {
                    "completed":   "marked complete",
                    "in_progress": "started",
                    "blocked":     "blocked",
                    "cancelled":   "cancelled",
                    "deferred":    "deferred",
                    "todo":        "reopened",
                }.get(new_status, "updated")
                try:
                    await notification_service.create_in_app_notification(
                        user_id=creator_id,
                        title=f"{actor_name} {label} a task",
                        content=task_title,
                        notification_type=NotificationType.TASK,
                        priority=(
                            NotificationPriority.HIGH
                            if new_status == "completed" else NotificationPriority.MEDIUM
                        ),
                        metadata={
                            "task_id": task_id,
                            "actor_id": str(current_user.id),
                            "actor_name": actor_name,
                            "new_status": new_status,
                            "action": "status_changed",
                        },
                    )
                except Exception:
                    pass
    except Exception:
        pass

    # ── Phase 2: mirror task changes onto the linked calendar event ────
    try:
        patched_fields = set(task_in.dict(exclude_unset=True).keys())
        # If status changed to completed → mark event done.
        if "status" in patched_fields and getattr(task, "status", None) == TaskStatus.COMPLETED:
            await calendar_service.mark_event_completed_for_task(task_id)
        # If anything that affects the calendar changed, re-sync.
        if patched_fields & {"due_date", "title", "description", "priority", "status"}:
            await calendar_service.update_event_for_task(
                task, owner_user_id=str(current_user.id)
            )
            # Make sure the task carries the calendar_event_id back.
            if not getattr(task, "calendar_event_id", None) and getattr(task, "due_date", None):
                from backend.db.mongodb.repositories.calendar_repository import calendar_repository
                ev = await calendar_repository.get_by_task_id(task_id)
                if ev:
                    await task_repository.update_task(
                        task_id=task_id,
                        organization_id=_get_org_id(current_user),
                        update_data={"calendar_event_id": str(ev.id)},
                    )
    except Exception:
        pass

    # Phase 5: serialise ObjectIds before returning.
    t_dict = task.model_dump() if hasattr(task, "model_dump") else task.dict()
    if "_id" in t_dict:
        t_dict["id"] = str(t_dict.pop("_id"))
    elif "id" in t_dict:
        t_dict["id"] = str(t_dict["id"])
    return _stringify_objectids(t_dict)

@router.delete("/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_task(
    task_id: str,
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> None:
    """
    Delete a task.
    """
    existing_task = await task_repository.get_task_by_id(task_id, organization_id=_get_org_id(current_user))
    if not existing_task:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Task not found or not authorized")

    deleted = await task_repository.delete_task(
        task_id=task_id,
        organization_id=_get_org_id(current_user)
    )
    if settings.POSTGRES_ENABLED and settings.POSTGRES_DUAL_WRITE and db is not None:
        try:
            repo = PostgresTaskRepository(db)
            await repo.delete_task(
                task_id=task_id,
                organization_id=_get_org_id(current_user)
            )
        except Exception:
            pass
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete task"
        )

    await log_activity(
        user_id=str(current_user.id),
        organization_id=_get_org_id(current_user),
        activity_type="task.deleted",
        details={"task_id": task_id},
        related_resource_type="TASK",
        related_resource_id=task_id,
    )

    # ── Phase 2: also remove the linked calendar event ─────────────────
    try:
        await calendar_service.delete_event_for_task(task_id)
    except Exception:
        pass

    return None

@router.get("/", response_model=None)
async def list_tasks(
    status: Optional[TaskStatus] = Query(None),
    assigned_to: Optional[str] = Query(None),
    document_id: Optional[str] = Query(None),
    agent_id: Optional[str] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000), # Increased default limit
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    List tasks for the current user or organization with filters.
    Always reads from MongoDB (source of truth). Postgres is write-only for now.
    """
    tasks = await task_repository.get_organization_tasks(
        organization_id=_get_org_id(current_user),
        status=status,
        assigned_to=assigned_to,
        document_id=document_id,
        agent_id=agent_id,
        skip=skip,
        limit=limit
    )

    # Serialize tasks to dicts for the frontend
    result = []
    for t in tasks:
        t_dict = t.model_dump() if hasattr(t, "model_dump") else (t.dict() if hasattr(t, "dict") else t)
        # Ensure id is a string
        if "_id" in t_dict:
            t_dict["id"] = str(t_dict.pop("_id"))
        elif "id" in t_dict:
            t_dict["id"] = str(t_dict["id"])
        # Phase 5: recursive ObjectId/datetime → str cleanup.  Catches the
        # new fields (calendar_event_id, invite_id) AND the nested ObjectIds
        # inside status_history / assignment_history that the old hand-coded
        # field list missed.
        t_dict = _stringify_objectids(t_dict)
        # Extract document_id from metadata for convenience
        if not t_dict.get("document_id") and isinstance(t_dict.get("metadata"), dict):
            t_dict["document_id"] = t_dict["metadata"].get("document_id")
        result.append(t_dict)
    return result

@router.get("/upcoming", response_model=List[TaskResponse])
async def list_upcoming_tasks(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    List upcoming tasks for the current user (due in next 7 days by default).
    """
    tasks = await task_repository.get_upcoming_tasks(
        organization_id=_get_org_id(current_user),
        user_id=current_user.id
    )
    return tasks

@router.get("/analytics", response_model=Dict[str, Any])
async def get_task_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"), # e.g., "7d", "30d", "1y"
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> Any:
    """
    Get task analytics for the current user or organization.
    """
    analytics = await task_repository.get_task_analytics(
        organization_id=_get_org_id(current_user),
        user_id=current_user.id,
        time_range=time_range
    )
    return analytics

@router.get("/summary", response_model=TaskSummaryResponse)
async def get_task_summary(
    current_user: User = Depends(get_current_active_user),
    db: AsyncSession | None = Depends(get_optional_async_db)
) -> TaskSummaryResponse:
    """
    Get summary statistics for tasks in the organization.
    """
    summary_data = await task_repository.get_task_stats(
        organization_id=_get_org_id(current_user)
    )
    return TaskSummaryResponse(**summary_data)


# ── Phase 5: assign by user_id OR email (auto-invite when not a user) ─────

class AssignTaskRequest(BaseModel):
    user_id: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = "member"  # "admin" | "member" | "viewer"


@router.post("/{task_id}/assign", response_model=None)
async def assign_task(
    task_id: str,
    payload: AssignTaskRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Assign a task to either a known user or an email address.

    When `email` is provided and no Lumicoria user owns that email, an
    invite is issued automatically + the task is stamped with
    `assigned_to_email` and `assignee_kind="email_invite"`.  When the
    invitee accepts (or signs up), `assigned_to` is swapped in by the
    invite-acceptance hook.
    """
    org_id = _get_org_id(current_user)

    # Ensure the task exists and the caller has access.
    task = await task_repository.get_task_by_id(task_id, organization_id=org_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")

    if payload.user_id:
        await task_repository.update_task(
            task_id=task_id,
            organization_id=org_id,
            update_data={
                "assigned_to": payload.user_id,
                "assigned_to_email": None,
                "assignee_kind": "user",
            },
            changed_by=str(current_user.id),
            changed_by_name=getattr(current_user, "full_name", None),
        )
        await log_activity(
            user_id=str(current_user.id),
            organization_id=org_id,
            activity_type="task.assigned",
            details={"task_id": task_id, "user_id": payload.user_id},
            related_resource_type="TASK",
            related_resource_id=task_id,
        )

        # ── Notify the new assignee (Phase 5) ───────────────────────────
        # Don't ping yourself when you self-assign.
        if str(payload.user_id) != str(current_user.id):
            try:
                from backend.services.notification_service import notification_service
                from backend.db.mongodb.models.notification import NotificationPriority, NotificationType
                assigner_name = getattr(current_user, "full_name", None) or getattr(current_user, "email", "Someone")
                task_title = getattr(task, "title", "a task")
                await notification_service.create_in_app_notification(
                    user_id=str(payload.user_id),
                    title="You were assigned a task",
                    content=f"{assigner_name} assigned you: {task_title}",
                    notification_type=NotificationType.TASK,
                    priority=NotificationPriority.HIGH,
                    metadata={
                        "task_id": task_id,
                        "assigner_id": str(current_user.id),
                        "assigner_name": assigner_name,
                        "action": "assigned",
                    },
                )
                # Push too — fire-and-forget; never fails the request
                try:
                    from backend.services.push_notification_service import push_notification_service
                    await push_notification_service.send_to_user(
                        user_id=str(payload.user_id),
                        title="Assigned to a task",
                        body=f"{assigner_name}: {task_title[:60]}",
                        data={"type": "task_assigned", "task_id": task_id},
                    )
                except Exception:
                    pass
            except Exception as e:
                import structlog as _sl
                _sl.get_logger().warning("assign_notification_failed", error=str(e))

        return {"assigned": True, "via": "user_id", "user_id": payload.user_id}

    if payload.email:
        from backend.services.invite_service import invite_service
        from backend.db.mongodb.models.invite import InviteRole
        role = (payload.role or "member").lower()
        try:
            role_enum = InviteRole(role)
        except ValueError:
            role_enum = InviteRole.MEMBER
        try:
            result = await invite_service.assign_task_by_email_or_invite(
                task_id=task_id,
                email=payload.email,
                organization_id=org_id,
                invited_by=str(current_user.id),
                role=role_enum,
            )
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

        await log_activity(
            user_id=str(current_user.id),
            organization_id=org_id,
            activity_type="task.assigned_by_email" if result.get("via") == "invite" else "task.assigned",
            details={
                "task_id": task_id,
                "email": payload.email,
                "via": result.get("via"),
                "invite_id": result.get("invite_id"),
            },
            related_resource_type="TASK",
            related_resource_id=task_id,
        )
        return result

    raise HTTPException(
        status_code=400,
        detail="Provide either `user_id` or `email` to assign this task.",
    )



# ── Phase 4: signed-token "Mark complete / Mark started" from emails ──────

def _action_result_page(
    *,
    title: str,
    headline: str,
    body: str,
    accent: str = "#6C4AB0",
    next_url: str = "https://lumicoria.ai/tasks",
    cta: str = "Open Lumicoria",
) -> HTMLResponse:
    """Render a small, brand-on, no-gradient response page after an email click.

    Public-facing (no login required) and self-contained — works inline in
    any mail client preview window or browser.
    """
    safe_headline = (headline or "").replace("<", "&lt;").replace(">", "&gt;")
    safe_body = (body or "").replace("<", "&lt;").replace(">", "&gt;")
    html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} — Lumicoria.ai</title>
<style>
  body {{ margin:0; font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
          background:#F8FAFC; color:#1E293B; }}
  .wrap {{ max-width:480px; margin:80px auto; padding:32px 28px; background:#fff;
           border:1px solid #E2E8F0; border-radius:16px; }}
  .accent {{ width:48px; height:48px; border-radius:12px; background:{accent};
             color:#fff; display:flex; align-items:center; justify-content:center;
             font-size:24px; line-height:48px; text-align:center; margin-bottom:20px; }}
  h1 {{ font-size:22px; margin:0 0 12px; color:#0F172A; }}
  p {{ margin:0 0 18px; font-size:14px; line-height:1.6; color:#475569; }}
  .cta {{ display:inline-block; padding:10px 18px; border-radius:9999px;
          background:#0F172A; color:#fff; font-size:13px; font-weight:500;
          text-decoration:none; }}
  .brand {{ margin-top:28px; font-size:11px; color:#94A3B8; text-align:center; }}
</style></head>
<body>
  <div class="wrap">
    <div class="accent">✓</div>
    <h1>{safe_headline}</h1>
    <p>{safe_body}</p>
    <a class="cta" href="{next_url}">{cta} →</a>
    <div class="brand">Lumicoria.ai · task action</div>
  </div>
</body></html>"""
    return HTMLResponse(content=html)


@router.get("/action", include_in_schema=False)
async def task_action_from_email(
    token: str = Query(..., description="Signed task-action token from an email"),
) -> HTMLResponse:
    """One-click in-email handler for "Mark complete" / "Mark started" buttons.

    No login required — the JWT itself is the credential (HS256-signed,
    scoped to one user + one task + one action, 7-day expiry).  Returns
    a small HTML confirmation page that works in any mail client.
    """
    # 1. Decode + validate the token
    try:
        claims = decode_action_token(token)
    except TaskActionTokenError as e:
        if e.reason == "expired":
            return _action_result_page(
                title="Link expired",
                headline="This link has expired",
                body="Open Lumicoria to mark this task — links are valid for 7 days.",
                accent="#EF4444",
            )
        return _action_result_page(
            title="Invalid link",
            headline="We couldn't verify this link",
            body="The action link is invalid. Open Lumicoria to manage your task.",
            accent="#EF4444",
        )

    user_id = claims["user_id"]
    task_id = claims["task_id"]
    action = claims["action"]

    # 2. Look the task up (no org check — the token is the proof of intent)
    task = await task_repository.get_task_by_id(task_id)
    if not task:
        return _action_result_page(
            title="Task not found",
            headline="Task not found",
            body="This task may have been deleted. Open Lumicoria to see your current tasks.",
            accent="#EF4444",
        )

    # Tenant guard: the token's `sub` must match the task's `created_by`
    # OR `assigned_to` — anyone else clicking the link is rejected even
    # if the JWT itself verified (defense in depth).
    creator_id = str(getattr(task, "created_by", "") or "")
    assignee_id = str(getattr(task, "assigned_to", "") or "")
    if user_id not in {creator_id, assignee_id}:
        return _action_result_page(
            title="Not allowed",
            headline="This link is not for this account",
            body="Sign in to Lumicoria to manage this task.",
            accent="#EF4444",
        )

    # 3. Map action → status update
    org_id = str(getattr(task, "organization_id", "") or user_id)
    current_status = (
        task.status.value if hasattr(task.status, "value") else str(task.status or "")
    )

    if action == TaskAction.COMPLETE.value:
        if current_status == TaskStatus.COMPLETED.value:
            return _action_result_page(
                title="Already done",
                headline="This task is already complete",
                body="Nice work. You can review it any time in Lumicoria.",
            )
        await task_repository.update_task(
            task_id=task_id,
            organization_id=org_id,
            update_data={
                "status": TaskStatus.COMPLETED.value,
                "completed_at": datetime.utcnow(),
                "progress": 100,
            },
        )
        # Mirror to calendar (failure-tolerant)
        try:
            await calendar_service.mark_event_completed_for_task(task_id)
        except Exception:
            pass
        await log_activity(
            user_id=user_id,
            organization_id=org_id,
            activity_type="task.completed_from_email",
            details={"task_id": task_id},
            related_resource_type="TASK",
            related_resource_id=task_id,
        )
        return _action_result_page(
            title="Marked complete",
            headline="Task marked complete",
            body=f"“{getattr(task, 'title', 'Task')}” is done. Lumicoria has updated your dashboard.",
            accent="#10B981",
        )

    if action == TaskAction.START.value:
        if current_status in (TaskStatus.IN_PROGRESS.value, TaskStatus.COMPLETED.value):
            return _action_result_page(
                title="Already in progress",
                headline="You're already on it",
                body="This task is already underway. Keep going.",
            )
        await task_repository.update_task(
            task_id=task_id,
            organization_id=org_id,
            update_data={"status": TaskStatus.IN_PROGRESS.value},
        )
        await log_activity(
            user_id=user_id,
            organization_id=org_id,
            activity_type="task.started_from_email",
            details={"task_id": task_id},
            related_resource_type="TASK",
            related_resource_id=task_id,
        )
        return _action_result_page(
            title="Marked started",
            headline="Got it — task in progress",
            body=f"“{getattr(task, 'title', 'Task')}” is now marked as in progress.",
            accent="#3B82F6",
        )

    if action == TaskAction.SNOOZE.value:
        # Push due_date by 1 day, capped to +14 days from now overall.
        current_due = getattr(task, "due_date", None)
        new_due = (current_due or datetime.utcnow()) + timedelta(days=1)
        max_due = datetime.utcnow() + timedelta(days=14)
        if new_due > max_due:
            new_due = max_due
        await task_repository.update_task(
            task_id=task_id,
            organization_id=org_id,
            update_data={"due_date": new_due},
        )
        await log_activity(
            user_id=user_id,
            organization_id=org_id,
            activity_type="task.snoozed_from_email",
            details={"task_id": task_id, "new_due_date": new_due.isoformat() + "Z"},
            related_resource_type="TASK",
            related_resource_id=task_id,
        )
        return _action_result_page(
            title="Snoozed",
            headline="Snoozed by a day",
            body=f"Due date now {new_due.strftime('%a, %b %d')}.",
            accent="#F59E0B",
        )

    return _action_result_page(
        title="Unsupported",
        headline="Unsupported action",
        body="That action isn't recognised. Open Lumicoria to manage the task.",
        accent="#EF4444",
    )

