"""
Lumicoria Huddle — live meeting REST API.

Mounted at /api/v1/huddles (see backend/api/v1/api.py).

Endpoints:
  POST   /                             create
  GET    /                             list mine
  GET    /{huddle_id}                  detail (auth OR ?share_token=)
  PATCH  /{huddle_id}                  update
  POST   /{huddle_id}/start            move scheduled → live
  POST   /{huddle_id}/end              end + trigger summary
  POST   /{huddle_id}/join             register a participant
  POST   /{huddle_id}/leave            mark left
  POST   /{huddle_id}/transcript       append a transcript chunk
  GET    /{huddle_id}/transcript       fetch full transcript
  POST   /{huddle_id}/agents           attach an AI agent
  DELETE /{huddle_id}/agents/{key}     detach an AI agent
  POST   /{huddle_id}/invite           send invite emails
  GET    /share/{share_token}          public read for invitees (no auth)
  POST   /{huddle_id}/recording/start  flag recording on
  POST   /{huddle_id}/recording/chunk  upload a recorded chunk
  POST   /{huddle_id}/recording/finish concatenate + persist
  GET    /{huddle_id}/recording        signed playback URLs
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Body, Depends, File, Form, HTTPException, Query, UploadFile, status
from pydantic import BaseModel, EmailStr, Field

from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.services import (
    huddle_recording_service as recording,
    huddle_service as svc,
)
from backend.services.huddle_plan_guard import enforce_can_create, enforce_can_join

logger = structlog.get_logger(__name__)
router = APIRouter()


# ── Pydantic shapes ────────────────────────────────────────────────────

class CreateHuddleRequest(BaseModel):
    title: Optional[str] = None
    meeting_type: str = Field("instant", description="instant | scheduled | recurring")
    team_id: Optional[str] = None
    project_id: Optional[str] = None
    agent_keys: List[str] = Field(default_factory=list)
    custom_agent_ids: List[str] = Field(default_factory=list)
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None
    recording_enabled: bool = False
    recording_retention_days: int = 30
    lobby_enabled: bool = False
    require_sso: bool = False
    e2ee_enabled: bool = False
    data_residency: str = "us"
    metadata: Dict[str, Any] = Field(default_factory=dict)


class PatchHuddleRequest(BaseModel):
    title: Optional[str] = None
    agent_keys: Optional[List[str]] = None
    custom_agent_ids: Optional[List[str]] = None
    recording_enabled: Optional[bool] = None
    recording_retention_days: Optional[int] = None
    lobby_enabled: Optional[bool] = None
    require_sso: Optional[bool] = None
    e2ee_enabled: Optional[bool] = None
    scheduled_start: Optional[datetime] = None
    scheduled_end: Optional[datetime] = None


class JoinRequest(BaseModel):
    guest_name: Optional[str] = None
    guest_email: Optional[EmailStr] = None
    role: str = "participant"


class TranscriptChunkRequest(BaseModel):
    text: str
    speaker_name: Optional[str] = "Speaker"


class AttachAgentRequest(BaseModel):
    agent_key: Optional[str] = None
    custom_agent_id: Optional[str] = None


class InviteRequest(BaseModel):
    emails: List[EmailStr] = Field(default_factory=list)
    message: Optional[str] = None


class EndHuddleRequest(BaseModel):
    final_transcript: Optional[str] = None


class FinishRecordingRequest(BaseModel):
    total_chunks: int
    content_type: str = "video/webm"


# ── Plan lookup helper ─────────────────────────────────────────────────

async def _user_plan(user_id: str) -> str:
    try:
        from backend.services.billing_service import get_user_subscription
        sub = await get_user_subscription(user_id)
        plan = getattr(sub, "plan", None)
        return getattr(plan, "value", str(plan or "free"))
    except Exception:
        return "free"


def _resolve_org_id(current_user: User) -> str:
    org_id = getattr(current_user, "organization_id", None) or getattr(current_user, "active_organization_id", None)
    if not org_id:
        org_id = str(current_user.id)
    return str(org_id)


# ── Endpoints ──────────────────────────────────────────────────────────

@router.post("/", status_code=status.HTTP_201_CREATED)
async def create_huddle_endpoint(
    payload: CreateHuddleRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    user_id = str(current_user.id)
    org_id = _resolve_org_id(current_user)
    plan = await _user_plan(user_id)

    await enforce_can_create(
        plan=plan,
        organization_id=org_id,
        host_user_id=user_id,
        recording_enabled=payload.recording_enabled,
        recording_retention_days=payload.recording_retention_days,
        agent_count=len(payload.agent_keys) + len(payload.custom_agent_ids),
        require_sso=payload.require_sso,
        meeting_type=payload.meeting_type,
    )

    result = await svc.create_huddle(
        host_user_id=user_id,
        organization_id=org_id,
        title=payload.title,
        meeting_type=payload.meeting_type,
        team_id=payload.team_id,
        project_id=payload.project_id,
        agent_keys=payload.agent_keys,
        custom_agent_ids=payload.custom_agent_ids,
        scheduled_start=payload.scheduled_start,
        scheduled_end=payload.scheduled_end,
        recording_enabled=payload.recording_enabled,
        recording_retention_days=payload.recording_retention_days,
        lobby_enabled=payload.lobby_enabled,
        require_sso=payload.require_sso,
        e2ee_enabled=payload.e2ee_enabled,
        data_residency=payload.data_residency,
        metadata=payload.metadata,
    )
    return result


@router.get("/")
async def list_huddles_endpoint(
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    statuses = [s.strip() for s in status_filter.split(",")] if status_filter else None
    items = await svc.list_huddles(
        user_id=str(current_user.id),
        organization_id=_resolve_org_id(current_user),
        statuses=statuses,
        limit=limit,
    )
    return {"items": items, "total": len(items)}


@router.get("/{huddle_id}")
async def get_huddle_endpoint(
    huddle_id: str,
    share_token: Optional[str] = Query(None),
    current_user: Optional[User] = Depends(get_current_active_user),
) -> Dict[str, Any]:
    result = await svc.get_huddle(
        huddle_id,
        requesting_user_id=str(current_user.id) if current_user else None,
        share_token=share_token,
    )
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Huddle not found")
    return result


@router.get("/share/{share_token}")
async def get_huddle_public(share_token: str) -> Dict[str, Any]:
    """No-auth public view for invitees. Looks up by share_token directly."""
    from sqlalchemy import select
    from backend.db.postgres import get_async_sessionmaker
    from backend.db.postgres_models import HuddleSQL

    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        q = select(HuddleSQL).where(
            HuddleSQL.share_token == share_token,
            HuddleSQL.deleted_at.is_(None),
        )
        row = (await session.execute(q)).scalar_one_or_none()
        if not row:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Invalid link")
        return svc._serialize_huddle(row, public=True)


@router.patch("/{huddle_id}")
async def patch_huddle_endpoint(
    huddle_id: str,
    payload: PatchHuddleRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    result = await svc.patch_huddle(
        huddle_id,
        requesting_user_id=str(current_user.id),
        patch=payload.model_dump(exclude_unset=True),
    )
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Not found or forbidden")
    return result


@router.post("/{huddle_id}/start")
async def start_huddle_endpoint(
    huddle_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    result = await svc.start_huddle(huddle_id, requesting_user_id=str(current_user.id))
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Not found or forbidden")
    return result


@router.post("/{huddle_id}/end")
async def end_huddle_endpoint(
    huddle_id: str,
    payload: EndHuddleRequest = Body(default_factory=EndHuddleRequest),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    result = await svc.end_huddle(
        huddle_id,
        requesting_user_id=str(current_user.id),
        final_transcript=payload.final_transcript,
    )
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Not found or forbidden")
    return result


@router.post("/{huddle_id}/join")
async def join_huddle_endpoint(
    huddle_id: str,
    payload: JoinRequest = Body(default_factory=JoinRequest),
    current_user: Optional[User] = Depends(get_current_active_user),
) -> Dict[str, Any]:
    user_id: Optional[str] = str(current_user.id) if current_user else None
    plan = await _user_plan(user_id) if user_id else "free"
    existing = await svc.list_participants(huddle_id)
    open_count = sum(1 for p in existing if not p.get("left_at"))
    enforce_can_join(plan=plan, current_participants=open_count)

    p = await svc.add_participant(
        huddle_id,
        user_id=user_id,
        guest_name=payload.guest_name if not user_id else None,
        guest_email=payload.guest_email if not user_id else None,
        role=payload.role,
    )
    if p is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Huddle not found")
    return p


@router.post("/{huddle_id}/leave")
async def leave_huddle_endpoint(
    huddle_id: str,
    current_user: Optional[User] = Depends(get_current_active_user),
) -> Dict[str, Any]:
    user_id = str(current_user.id) if current_user else None
    ok = await svc.remove_participant(huddle_id, user_id=user_id)
    return {"ok": ok}


@router.post("/{huddle_id}/transcript")
async def append_transcript_endpoint(
    huddle_id: str,
    payload: TranscriptChunkRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    return await svc.append_transcript_chunk(
        huddle_id,
        text=payload.text,
        speaker_name=payload.speaker_name or "Speaker",
        speaker_user_id=str(current_user.id),
        user_id_for_activity=str(current_user.id),
    )


@router.get("/{huddle_id}/transcript")
async def get_transcript_endpoint(
    huddle_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    chunks = await svc.get_transcript(huddle_id)
    return {"chunks": chunks, "total": len(chunks)}


@router.post("/{huddle_id}/agents")
async def attach_agent_endpoint(
    huddle_id: str,
    payload: AttachAgentRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    if not payload.agent_key and not payload.custom_agent_id:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="agent_key or custom_agent_id required")
    result = await svc.attach_agent(
        huddle_id,
        agent_key=payload.agent_key,
        custom_agent_id=payload.custom_agent_id,
        requesting_user_id=str(current_user.id),
    )
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Not found or forbidden")
    return result


@router.delete("/{huddle_id}/agents/{agent_ref}")
async def detach_agent_endpoint(
    huddle_id: str,
    agent_ref: str,
    custom: bool = Query(False, description="Pass true when agent_ref is a custom_agent_id"),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    result = await svc.detach_agent(
        huddle_id,
        agent_key=None if custom else agent_ref,
        custom_agent_id=agent_ref if custom else None,
        requesting_user_id=str(current_user.id),
    )
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Not found or forbidden")
    return result


@router.post("/{huddle_id}/invite")
async def invite_endpoint(
    huddle_id: str,
    payload: InviteRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Send simple invite emails containing the share URL. Reuses
    email_service for sending; falls back to returning the link if
    email delivery isn't configured."""
    huddle = await svc.get_huddle(huddle_id, requesting_user_id=str(current_user.id))
    if not huddle:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Not found or forbidden")

    base = "https://lumicoria.ai"
    try:
        from backend.core.config import settings as _settings
        base = getattr(_settings, "FRONTEND_URL", base) or base
    except Exception:
        pass
    share_url = f"{base}/huddles/join/{huddle['share_token']}"
    delivered: List[str] = []
    skipped: List[str] = []
    try:
        from backend.services.email_service import get_email_service
        email_service = await get_email_service()
        for to in payload.emails:
            try:
                await email_service.send(
                    to=to,
                    subject=f"You're invited: {huddle['title']}",
                    template_name="huddle_invite",
                    template_data={
                        "title": huddle["title"],
                        "host_name": getattr(current_user, "full_name", None) or getattr(current_user, "email", "Your colleague"),
                        "share_url": share_url,
                        "message": payload.message or "",
                    },
                    tags=["huddle", "invite"],
                )
                delivered.append(to)
            except Exception:
                skipped.append(to)
    except Exception:
        # Email service unavailable — return the link for the caller to share.
        skipped = list(payload.emails)
    return {
        "share_url": share_url,
        "delivered": delivered,
        "skipped": skipped,
    }


# ── Recording ──────────────────────────────────────────────────────────

@router.post("/{huddle_id}/recording/start")
async def start_recording_endpoint(
    huddle_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    user_id = str(current_user.id)
    plan = await _user_plan(user_id)
    from backend.services.huddle_plan_guard import caps_for
    if not caps_for(plan).get("recording_allowed"):
        raise HTTPException(
            status.HTTP_402_PAYMENT_REQUIRED,
            detail={"code": "upgrade_required", "message": "Recording requires Professional+", "current_plan": plan},
        )
    return await recording.start_recording(huddle_id, requesting_user_id=user_id)


@router.post("/{huddle_id}/recording/chunk")
async def upload_chunk_endpoint(
    huddle_id: str,
    chunk_index: int = Form(...),
    content_type: str = Form("video/webm"),
    file: UploadFile = File(...),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    body = await file.read()
    return await recording.upload_chunk(
        huddle_id,
        chunk_index=chunk_index,
        blob=body,
        content_type=content_type or file.content_type or "video/webm",
        requesting_user_id=str(current_user.id),
    )


@router.post("/{huddle_id}/recording/finish")
async def finish_recording_endpoint(
    huddle_id: str,
    payload: FinishRecordingRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    return await recording.finish_recording(
        huddle_id,
        total_chunks=payload.total_chunks,
        content_type=payload.content_type,
        requesting_user_id=str(current_user.id),
    )


@router.get("/{huddle_id}/recording")
async def get_recording_endpoint(
    huddle_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    return await recording.get_recording_url(huddle_id, requesting_user_id=str(current_user.id))


# ── Calendar export ────────────────────────────────────────────────────

class CalendarExportRequest(BaseModel):
    attendees: List[EmailStr] = Field(default_factory=list)
    description: Optional[str] = None
    calendar_id: str = "primary"


@router.post("/{huddle_id}/calendar-export")
async def calendar_export_endpoint(
    huddle_id: str,
    payload: CalendarExportRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Push the huddle into the host's Google Calendar as an event with
    a Lumicoria Huddle join URL. Requires the org to have the Google
    Workspace integration connected."""
    huddle = await svc.get_huddle(huddle_id, requesting_user_id=str(current_user.id))
    if not huddle:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Not found or forbidden")
    if not huddle.get("scheduled_start") or not huddle.get("scheduled_end"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Huddle has no scheduled_start / scheduled_end")

    base = "https://lumicoria.ai"
    try:
        from backend.core.config import settings as _settings
        base = getattr(_settings, "FRONTEND_URL", base) or base
    except Exception:
        pass
    share_url = f"{base}/huddles/join/{huddle['share_token']}"
    description_text = payload.description or "Lumicoria Huddle — live meeting with AI agents."
    description_full = (
        f"{description_text}\n\n"
        f"Join the meeting: {share_url}\n\n"
        f"This meeting includes Lumicoria AI agents that capture decisions and action items in real time."
    )

    # Try to call Google Workspace integration. If the user hasn't
    # connected the integration we just return the link so they can
    # paste it manually.
    try:
        from backend.integrations.google_workspace import GoogleWorkspaceIntegration
        from backend.integrations.google_workspace_client import GoogleWorkspaceClient  # type: ignore
        client = GoogleWorkspaceClient(user_id=str(current_user.id))
        integ = GoogleWorkspaceIntegration(client=client)
        event = await integ.create_calendar_event(
            summary=huddle.get("title") or "Lumicoria Huddle",
            description=description_full,
            start_time=datetime.fromisoformat(huddle["scheduled_start"].replace("Z", "+00:00")),
            end_time=datetime.fromisoformat(huddle["scheduled_end"].replace("Z", "+00:00")),
            attendees=list(payload.attendees),
            calendar_id=payload.calendar_id,
            location=share_url,
        )
        return {
            "ok": True,
            "share_url": share_url,
            "event": event,
        }
    except Exception as e:
        logger.warning("huddle_calendar_export_failed", huddle_id=huddle_id, error=str(e))
        return {
            "ok": False,
            "share_url": share_url,
            "error": "Google Workspace integration unavailable — share the link manually.",
        }
