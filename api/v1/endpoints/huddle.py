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
from backend.services.jitsi_jwt import domain as jitsi_domain, sign_room_jwt
from backend.services import huddle_analytics, huddle_tts

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


def _enrich_with_jitsi(huddle: Dict[str, Any], user: Optional[User]) -> Dict[str, Any]:
    """Attach jitsi_domain + jitsi_jwt to the response so the frontend
    can embed self-hosted Jitsi (when configured) without an extra call."""
    if not huddle:
        return huddle
    huddle["jitsi_domain"] = jitsi_domain()
    is_host = bool(user) and str(getattr(user, "id", "")) == huddle.get("host_user_id")
    huddle["jitsi_jwt"] = sign_room_jwt(
        room=huddle.get("room_name") or "*",
        user_id=str(getattr(user, "id", "")) if user else None,
        display_name=getattr(user, "full_name", None) or getattr(user, "email", None) if user else None,
        email=getattr(user, "email", None) if user else None,
        moderator=is_host,
        allow_recording=bool(huddle.get("recording_enabled")),
    )
    return huddle


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
    return _enrich_with_jitsi(result, current_user)


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
    return _enrich_with_jitsi(result, current_user)


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
        public_view = svc._serialize_huddle(row, public=True)
        public_view["jitsi_domain"] = jitsi_domain()
        public_view["jitsi_jwt"] = sign_room_jwt(
            room=row.room_name, display_name="Guest",
        )
        return public_view


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
    return _enrich_with_jitsi(result, current_user)


@router.post("/{huddle_id}/start")
async def start_huddle_endpoint(
    huddle_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    result = await svc.start_huddle(huddle_id, requesting_user_id=str(current_user.id))
    if not result:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Not found or forbidden")
    return _enrich_with_jitsi(result, current_user)


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


# ── Jibri handoff (server-side recording) ──────────────────────────────

class JibriWebhookRequest(BaseModel):
    """Payload Jibri POSTs when it finishes recording a room. The
    Jibri service is configured via `docker/jitsi/docker-compose.yml`
    with `JIBRI_FINALIZE_RECORDING_SCRIPT_PATH` invoking curl against
    this endpoint."""
    huddle_id: str
    room_name: str
    object_key: str  # MinIO key where Jibri uploaded the MP4
    mime: str = "video/mp4"
    size_bytes: Optional[int] = None
    duration_sec: Optional[int] = None
    signature: Optional[str] = None  # HMAC of payload signed by Jibri shared secret


@router.post("/jibri/webhook")
async def jibri_webhook_endpoint(payload: JibriWebhookRequest = Body(...)) -> Dict[str, Any]:
    """Called by Jibri after server-side recording finalises. We verify
    the HMAC and stamp the recording_url onto the HuddleSQL row.

    HMAC: `hmac.sha256(JITSI_APP_SECRET, f"{huddle_id}.{object_key}")`.
    """
    import hmac, hashlib
    from datetime import timedelta
    from sqlalchemy import select, update as sa_update
    from backend.db.postgres import get_async_sessionmaker
    from backend.db.postgres_models import HuddleSQL
    from backend.core.config import settings as _settings

    if payload.signature and _settings.JITSI_APP_SECRET:
        expected = hmac.new(
            _settings.JITSI_APP_SECRET.encode("utf-8"),
            f"{payload.huddle_id}.{payload.object_key}".encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        if not hmac.compare_digest(expected, payload.signature):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid signature")

    factory = get_async_sessionmaker()
    async with factory() as session:
        q = select(HuddleSQL).where(HuddleSQL.id == payload.huddle_id, HuddleSQL.deleted_at.is_(None))
        row = (await session.execute(q)).scalar_one_or_none()
        if not row:
            raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Huddle not found")
        retention = int(row.recording_retention_days or 30)
        expires_at = datetime.utcnow() + timedelta(days=retention)
        await session.execute(
            sa_update(HuddleSQL)
            .where(HuddleSQL.id == payload.huddle_id)
            .values(
                recording_mode="jibri",
                recording_object_key=payload.object_key,
                recording_mime=payload.mime,
                recording_size_bytes=payload.size_bytes,
                recording_expires_at=expires_at,
                updated_at=datetime.utcnow(),
            )
        )
        await session.commit()
        org_id = row.organization_id

    # Fire huddle.recording_ready webhook
    try:
        from backend.services.huddle_events import emit_webhook, fire_and_forget
        fire_and_forget(emit_webhook(org_id, "huddle.recording_ready", {
            "huddle_id": payload.huddle_id,
            "object_key": payload.object_key,
            "mime": payload.mime,
            "size_bytes": payload.size_bytes,
            "duration_sec": payload.duration_sec,
            "recording_mode": "jibri",
        }))
    except Exception:
        pass

    return {"ok": True}


# ── Phase 3 — TTS / analytics / calendar back-sync / ICS ───────────────

class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    voice: str = "warm"
    quality: str = "standard"


@router.post("/{huddle_id}/tts")
async def tts_endpoint(
    huddle_id: str,
    payload: TTSRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Synthesize speech for the virtual agent feature. Returns the audio
    bytes inline so the frontend can play them via Web Audio + inject
    into the host's mic stream."""
    from fastapi.responses import Response
    huddle = await svc.get_huddle(huddle_id, requesting_user_id=str(current_user.id))
    if not huddle:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Not found or forbidden")
    try:
        audio, mime = await huddle_tts.synthesize(payload.text, voice=payload.voice, quality=payload.quality)
    except RuntimeError as e:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(e))
    return Response(content=audio, media_type=mime)


@router.get("/tts/voices")
async def tts_voices_endpoint(current_user: User = Depends(get_current_active_user)) -> Dict[str, Any]:
    return {"voices": huddle_tts.VOICE_CATALOG}


@router.get("/{huddle_id}/analytics")
async def analytics_endpoint(
    huddle_id: str,
    recompute: bool = Query(False),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    huddle = await svc.get_huddle(huddle_id, requesting_user_id=str(current_user.id))
    if not huddle:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Not found or forbidden")
    if recompute or not (huddle.get("metadata") or {}).get("speaker_analytics"):
        analytics = await huddle_analytics.persist_for_huddle(huddle_id)
    else:
        analytics = (huddle.get("metadata") or {}).get("speaker_analytics")
    return {"huddle_id": huddle_id, "analytics": analytics}


@router.get("/{huddle_id}/ics")
async def ics_endpoint(
    huddle_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Generate an ICS file so Outlook / Apple Calendar can subscribe."""
    from fastapi.responses import Response
    huddle = await svc.get_huddle(huddle_id, requesting_user_id=str(current_user.id))
    if not huddle:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Not found or forbidden")

    start = huddle.get("scheduled_start") or huddle.get("started_at") or huddle.get("created_at")
    end = huddle.get("scheduled_end") or huddle.get("ended_at") or start
    if not start:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="Huddle has no time data")

    def _ics_dt(iso: str) -> str:
        d = datetime.fromisoformat(iso.replace("Z", "+00:00"))
        return d.strftime("%Y%m%dT%H%M%SZ")

    base = "https://lumicoria.ai"
    try:
        from backend.core.config import settings as _settings
        base = getattr(_settings, "FRONTEND_URL", base) or base
    except Exception:
        pass
    share_url = f"{base}/huddles/join/{huddle['share_token']}"

    title = (huddle.get("title") or "Lumicoria Huddle").replace("\n", " ")
    description = (
        "Lumicoria Huddle — join the live room.\\n"
        f"Join: {share_url}\\n\\n"
        "AI agents will capture decisions + action items automatically."
    )
    uid = f"huddle-{huddle_id}@lumicoria.ai"
    now_stamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")

    ics = (
        "BEGIN:VCALENDAR\r\n"
        "PRODID:-//Lumicoria//Huddle//EN\r\n"
        "VERSION:2.0\r\n"
        "CALSCALE:GREGORIAN\r\n"
        "METHOD:PUBLISH\r\n"
        "BEGIN:VEVENT\r\n"
        f"UID:{uid}\r\n"
        f"DTSTAMP:{now_stamp}\r\n"
        f"DTSTART:{_ics_dt(start)}\r\n"
        f"DTEND:{_ics_dt(end)}\r\n"
        f"SUMMARY:{title}\r\n"
        f"DESCRIPTION:{description}\r\n"
        f"LOCATION:{share_url}\r\n"
        f"URL;VALUE=URI:{share_url}\r\n"
        "END:VEVENT\r\n"
        "END:VCALENDAR\r\n"
    )
    return Response(
        content=ics,
        media_type="text/calendar; charset=utf-8",
        headers={"Content-Disposition": f'attachment; filename="huddle-{huddle_id}.ics"'},
    )


@router.post("/sync-calendar")
async def sync_calendar_endpoint(
    days_ahead: int = Query(14, ge=1, le=60),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Pull Google Calendar events that mention a Lumicoria join URL and
    create scheduled HuddleSQL rows for them so they show up in our UI.

    Returns the list of newly-created huddles. Idempotent — events
    already linked to a huddle (matched by metadata.calendar_event_id)
    are skipped."""
    user_id = str(current_user.id)
    org_id = _resolve_org_id(current_user)

    base = "https://lumicoria.ai"
    try:
        from backend.core.config import settings as _settings
        base = getattr(_settings, "FRONTEND_URL", base) or base
    except Exception:
        pass

    try:
        from backend.integrations.google_workspace import GoogleWorkspaceIntegration
        from backend.integrations.google_workspace_client import GoogleWorkspaceClient  # type: ignore
        client = GoogleWorkspaceClient(user_id=user_id)
        integ = GoogleWorkspaceIntegration(client=client)
        events = await integ.get_upcoming_events(days_ahead=days_ahead)
    except Exception as e:
        logger.warning("huddle_calendar_sync_failed", error=str(e))
        return {"ok": False, "error": "Google Workspace integration unavailable.", "created": []}

    created: List[Dict[str, Any]] = []
    skipped = 0
    for ev in (events or []):
        ev_id = ev.get("id") or ev.get("event_id")
        if not ev_id:
            continue
        # Skip events not pointing at Lumicoria
        text = " ".join([
            ev.get("description") or "",
            ev.get("location") or "",
            ev.get("hangoutLink") or "",
            ev.get("summary") or "",
        ])
        if "lumicoria" not in text.lower():
            continue
        # Already mirrored?
        existing = await svc.list_huddles(
            user_id=user_id, organization_id=org_id, limit=200,
            statuses=["scheduled", "live"],
        )
        if any(h.get("metadata", {}).get("calendar_event_id") == ev_id for h in existing):
            skipped += 1
            continue
        start = ev.get("start", {}).get("dateTime") or ev.get("start", {}).get("date")
        end = ev.get("end", {}).get("dateTime") or ev.get("end", {}).get("date")
        if not start or not end:
            continue
        h = await svc.create_huddle(
            host_user_id=user_id,
            organization_id=org_id,
            title=ev.get("summary") or "Calendar meeting",
            meeting_type="scheduled",
            scheduled_start=datetime.fromisoformat(start.replace("Z", "+00:00")),
            scheduled_end=datetime.fromisoformat(end.replace("Z", "+00:00")),
            agent_keys=["meeting"],
            metadata={
                "calendar_event_id": ev_id,
                "calendar_source": "google",
                "original_location": ev.get("location"),
            },
        )
        created.append(h)
    return {"ok": True, "created": created, "skipped": skipped}


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
