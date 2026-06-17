"""
Lumicoria Huddle — live meeting orchestration.

This module owns the lifecycle of a Huddle room:
  - create (instant / scheduled)
  - participant join / leave bookkeeping
  - transcript chunk ingestion
  - end-of-call: trigger MeetingAgent for the post-call summary

Persistence is in Postgres (HuddleSQL + HuddleParticipantSQL + HuddleTranscriptChunkSQL).
Real-time fan-out uses the existing realtime publish helper on the
`rt:huddle:{huddle_id}` topic.

We avoid a separate repository class to match the inline-SQL style already
used in `api/v1/endpoints/meeting.py`.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import and_, func, select, update as sa_update

from backend.db.postgres import get_async_sessionmaker
from backend.db.postgres_models import (
    HuddleParticipantSQL,
    HuddleSQL,
    HuddleTranscriptChunkSQL,
    MeetingSQL,
)
from backend.services.activity_logger import log_activity

logger = structlog.get_logger(__name__)


# ── Public lifecycle ─────────────────────────────────────────────────────

async def create_huddle(
    *,
    host_user_id: str,
    organization_id: str,
    title: Optional[str] = None,
    meeting_type: str = "instant",
    team_id: Optional[str] = None,
    project_id: Optional[str] = None,
    agent_keys: Optional[List[str]] = None,
    custom_agent_ids: Optional[List[str]] = None,
    scheduled_start: Optional[datetime] = None,
    scheduled_end: Optional[datetime] = None,
    recording_enabled: bool = False,
    recording_retention_days: int = 30,
    lobby_enabled: bool = False,
    require_sso: bool = False,
    e2ee_enabled: bool = False,
    data_residency: str = "us",
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a huddle row. Status starts `live` for instant meetings or
    `scheduled` when `scheduled_start` is in the future."""
    is_scheduled = meeting_type == "scheduled" or (scheduled_start and scheduled_start > datetime.utcnow())
    status = "scheduled" if is_scheduled else "live"

    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        row = HuddleSQL(
            host_user_id=host_user_id,
            organization_id=organization_id,
            team_id=team_id,
            project_id=project_id,
            title=title or _default_title(meeting_type, host_user_id),
            meeting_type=meeting_type,
            status=status,
            scheduled_start=scheduled_start,
            scheduled_end=scheduled_end,
            started_at=None if is_scheduled else datetime.utcnow(),
            agent_keys=agent_keys or [],
            custom_agent_ids=custom_agent_ids or [],
            recording_enabled=recording_enabled,
            recording_retention_days=recording_retention_days,
            lobby_enabled=lobby_enabled,
            require_sso=require_sso,
            e2ee_enabled=e2ee_enabled,
            data_residency=data_residency,
            meta=metadata or {},
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        result = _serialize_huddle(row)

    await log_activity(
        user_id=host_user_id,
        organization_id=organization_id,
        activity_type="huddle.created",
        details={
            "huddle_id": result["id"],
            "title": result["title"],
            "meeting_type": meeting_type,
            "team_id": team_id,
            "project_id": project_id,
        },
        related_resource_type="huddle",
        related_resource_id=result["id"],
        agent_name="Huddle",
    )
    return result


async def list_huddles(
    *,
    user_id: str,
    organization_id: str,
    statuses: Optional[List[str]] = None,
    limit: int = 50,
) -> List[Dict[str, Any]]:
    """Return huddles I host plus huddles I joined as a participant."""
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        # Huddles I host in this org
        q = (
            select(HuddleSQL)
            .where(
                HuddleSQL.organization_id == organization_id,
                HuddleSQL.deleted_at.is_(None),
            )
            .order_by(HuddleSQL.created_at.desc())
            .limit(limit)
        )
        if statuses:
            q = q.where(HuddleSQL.status.in_(statuses))
        rows = (await session.execute(q)).scalars().all()
        # Filter to "mine" — host or participant
        out: List[Dict[str, Any]] = []
        for r in rows:
            if r.host_user_id == user_id:
                out.append(_serialize_huddle(r))
                continue
            # Check participant table
            pq = select(HuddleParticipantSQL).where(
                HuddleParticipantSQL.huddle_id == r.id,
                HuddleParticipantSQL.user_id == user_id,
            ).limit(1)
            if (await session.execute(pq)).scalar_one_or_none():
                out.append(_serialize_huddle(r))
        return out


async def get_huddle(
    huddle_id: str,
    *,
    requesting_user_id: Optional[str] = None,
    share_token: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Fetch a huddle. Auth: either a logged-in user OR a valid share_token."""
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        q = select(HuddleSQL).where(
            HuddleSQL.id == huddle_id,
            HuddleSQL.deleted_at.is_(None),
        )
        row = (await session.execute(q)).scalar_one_or_none()
        if not row:
            return None
        if share_token and row.share_token == share_token:
            return _serialize_huddle(row, public=True)
        if requesting_user_id and (row.host_user_id == requesting_user_id):
            return _serialize_huddle(row)
        if requesting_user_id:
            pq = select(HuddleParticipantSQL).where(
                HuddleParticipantSQL.huddle_id == huddle_id,
                HuddleParticipantSQL.user_id == requesting_user_id,
            ).limit(1)
            if (await session.execute(pq)).scalar_one_or_none():
                return _serialize_huddle(row)
        return None


async def patch_huddle(
    huddle_id: str,
    *,
    requesting_user_id: str,
    patch: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Host or org admin can update title, agent_keys, recording_enabled,
    lobby_enabled, e2ee_enabled, scheduled_start, scheduled_end."""
    allowed = {
        "title", "agent_keys", "custom_agent_ids", "recording_enabled",
        "recording_retention_days", "lobby_enabled", "require_sso",
        "e2ee_enabled", "scheduled_start", "scheduled_end",
    }
    clean = {k: v for k, v in patch.items() if k in allowed and v is not None}
    if not clean:
        return await get_huddle(huddle_id, requesting_user_id=requesting_user_id)

    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        q = select(HuddleSQL).where(
            HuddleSQL.id == huddle_id,
            HuddleSQL.deleted_at.is_(None),
        )
        row = (await session.execute(q)).scalar_one_or_none()
        if not row or row.host_user_id != requesting_user_id:
            return None
        for k, v in clean.items():
            setattr(row, k, v)
        row.updated_at = datetime.utcnow()
        await session.commit()
        await session.refresh(row)
        return _serialize_huddle(row)


async def start_huddle(huddle_id: str, *, requesting_user_id: str) -> Optional[Dict[str, Any]]:
    """Move a scheduled huddle to live state."""
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        q = select(HuddleSQL).where(HuddleSQL.id == huddle_id, HuddleSQL.deleted_at.is_(None))
        row = (await session.execute(q)).scalar_one_or_none()
        if not row or row.host_user_id != requesting_user_id:
            return None
        if row.status not in {"scheduled", "live"}:
            return None
        row.status = "live"
        row.started_at = row.started_at or datetime.utcnow()
        row.updated_at = datetime.utcnow()
        await session.commit()
        await session.refresh(row)
        result = _serialize_huddle(row)

    await log_activity(
        user_id=requesting_user_id,
        organization_id=result["organization_id"],
        activity_type="huddle.started",
        details={"huddle_id": huddle_id, "title": result["title"]},
        related_resource_type="huddle",
        related_resource_id=huddle_id,
        agent_name="Huddle",
    )
    return result


async def end_huddle(
    huddle_id: str,
    *,
    requesting_user_id: str,
    final_transcript: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Mark huddle ended, persist transcript, fire post-call summary."""
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        q = select(HuddleSQL).where(HuddleSQL.id == huddle_id, HuddleSQL.deleted_at.is_(None))
        row = (await session.execute(q)).scalar_one_or_none()
        if not row:
            return None
        if row.host_user_id != requesting_user_id:
            # Allow co-hosts later; for now only host can end.
            return None
        if row.status == "ended":
            return _serialize_huddle(row)
        row.status = "ended"
        row.ended_at = datetime.utcnow()
        if row.started_at:
            row.duration_sec = int((row.ended_at - row.started_at).total_seconds())
        if final_transcript is not None:
            row.transcript_text = final_transcript
        else:
            # Stitch chunks if no final body provided
            cq = (
                select(HuddleTranscriptChunkSQL)
                .where(HuddleTranscriptChunkSQL.huddle_id == huddle_id)
                .order_by(HuddleTranscriptChunkSQL.ts.asc())
            )
            chunks = (await session.execute(cq)).scalars().all()
            row.transcript_text = "\n".join(
                f"{c.speaker_name}: {c.text}" for c in chunks if c.text
            )
        row.updated_at = datetime.utcnow()
        await session.commit()
        await session.refresh(row)
        result = _serialize_huddle(row)

    await log_activity(
        user_id=requesting_user_id,
        organization_id=result["organization_id"],
        activity_type="huddle.ended",
        details={
            "huddle_id": huddle_id,
            "duration_sec": result.get("duration_sec"),
            "participant_count_peak": result.get("participant_count_peak"),
        },
        related_resource_type="huddle",
        related_resource_id=huddle_id,
        agent_name="Huddle",
    )

    # Fire MeetingAgent post-call summary asynchronously (don't block end()).
    if result.get("transcript_text"):
        asyncio.create_task(_run_post_call_summary(huddle_id, result))
    return result


async def _run_post_call_summary(huddle_id: str, huddle: Dict[str, Any]) -> None:
    """Background — run MeetingAgent on the accumulated transcript and link
    the resulting `meetings` row back to this huddle."""
    try:
        from backend.agents.meeting_agent import MeetingAgent
        agent = MeetingAgent()
        result = await agent.process_async({
            "transcript": huddle.get("transcript_text") or "",
            "meeting_type": "general",
            "metadata": {
                "title": huddle.get("title"),
                "participants": huddle.get("participant_user_ids") or [],
                "type": "general",
                "date": (huddle.get("ended_at") or datetime.utcnow().isoformat()),
            },
            "context": {},
        })

        # Save MeetingSQL row and link huddle.processed_meeting_id
        session_factory = get_async_sessionmaker()
        async with session_factory() as session:
            meeting = MeetingSQL(
                user_id=huddle["host_user_id"],
                organization_id=huddle.get("organization_id"),
                title=huddle.get("title"),
                meeting_type="general",
                transcript=huddle.get("transcript_text") or "",
                summary=result.get("summary", ""),
                raw_response=result.get("raw_response"),
                model_used=result.get("model_used"),
                action_items=result.get("action_items", []),
                decisions=result.get("decisions", []),
                key_points=result.get("key_points", []),
                follow_ups=result.get("follow_ups", []),
                questions=result.get("questions", []),
                concerns=result.get("concerns", []),
                participants=huddle.get("participant_user_ids") or [],
                source="huddle",
                processed_at=datetime.utcnow(),
            )
            session.add(meeting)
            await session.commit()
            await session.refresh(meeting)

            await session.execute(
                sa_update(HuddleSQL)
                .where(HuddleSQL.id == huddle_id)
                .values(processed_meeting_id=meeting.id, updated_at=datetime.utcnow())
            )
            await session.commit()

            logger.info("huddle_post_call_processed", huddle_id=huddle_id, meeting_id=meeting.id)
    except Exception as e:
        logger.warning("huddle_post_call_failed", huddle_id=huddle_id, error=str(e))


# ── Participants ─────────────────────────────────────────────────────────

async def add_participant(
    huddle_id: str,
    *,
    user_id: Optional[str] = None,
    guest_name: Optional[str] = None,
    guest_email: Optional[str] = None,
    role: str = "participant",
    agent_key: Optional[str] = None,
    custom_agent_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Register a participant join. Idempotent — if same user already
    has an open row (no left_at), updates joined_at."""
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        # Find existing open row for the same identity
        filt = [HuddleParticipantSQL.huddle_id == huddle_id, HuddleParticipantSQL.left_at.is_(None)]
        if user_id:
            filt.append(HuddleParticipantSQL.user_id == user_id)
        elif agent_key:
            filt.append(HuddleParticipantSQL.agent_key == agent_key)
        elif guest_email:
            filt.append(HuddleParticipantSQL.guest_email == guest_email)
        existing = (await session.execute(select(HuddleParticipantSQL).where(and_(*filt)).limit(1))).scalar_one_or_none()
        if existing:
            existing.joined_at = datetime.utcnow()
            await session.commit()
            return _serialize_participant(existing)

        p = HuddleParticipantSQL(
            huddle_id=huddle_id,
            user_id=user_id,
            guest_name=guest_name,
            guest_email=guest_email,
            role=role,
            agent_key=agent_key,
            custom_agent_id=custom_agent_id,
        )
        session.add(p)

        # Bump participant_count_peak if needed
        q_open = select(func.count()).select_from(HuddleParticipantSQL).where(
            HuddleParticipantSQL.huddle_id == huddle_id,
            HuddleParticipantSQL.left_at.is_(None),
        )
        live_count = (await session.execute(q_open)).scalar() or 0
        new_count = int(live_count) + 1
        await session.execute(
            sa_update(HuddleSQL)
            .where(HuddleSQL.id == huddle_id, HuddleSQL.participant_count_peak < new_count)
            .values(participant_count_peak=new_count, updated_at=datetime.utcnow())
        )
        await session.commit()
        await session.refresh(p)

        # Look up org for activity log
        org_id = (await session.execute(
            select(HuddleSQL.organization_id).where(HuddleSQL.id == huddle_id)
        )).scalar_one_or_none()

    if user_id and org_id:
        await log_activity(
            user_id=user_id,
            organization_id=org_id,
            activity_type="huddle.participant_joined",
            details={"huddle_id": huddle_id, "role": role},
            related_resource_type="huddle",
            related_resource_id=huddle_id,
            agent_name="Huddle",
        )
    return _serialize_participant(p)


async def remove_participant(
    huddle_id: str,
    *,
    user_id: Optional[str] = None,
    guest_email: Optional[str] = None,
    agent_key: Optional[str] = None,
) -> bool:
    """Stamp `left_at` on the participant's open row."""
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        filt = [HuddleParticipantSQL.huddle_id == huddle_id, HuddleParticipantSQL.left_at.is_(None)]
        if user_id:
            filt.append(HuddleParticipantSQL.user_id == user_id)
        elif agent_key:
            filt.append(HuddleParticipantSQL.agent_key == agent_key)
        elif guest_email:
            filt.append(HuddleParticipantSQL.guest_email == guest_email)
        else:
            return False
        row = (await session.execute(select(HuddleParticipantSQL).where(and_(*filt)).limit(1))).scalar_one_or_none()
        if not row:
            return False
        row.left_at = datetime.utcnow()
        await session.commit()

        org_id = (await session.execute(
            select(HuddleSQL.organization_id).where(HuddleSQL.id == huddle_id)
        )).scalar_one_or_none()

    if user_id and org_id:
        await log_activity(
            user_id=user_id,
            organization_id=org_id,
            activity_type="huddle.participant_left",
            details={"huddle_id": huddle_id},
            related_resource_type="huddle",
            related_resource_id=huddle_id,
            agent_name="Huddle",
        )
    return True


async def list_participants(huddle_id: str) -> List[Dict[str, Any]]:
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        q = (
            select(HuddleParticipantSQL)
            .where(HuddleParticipantSQL.huddle_id == huddle_id)
            .order_by(HuddleParticipantSQL.joined_at.asc())
        )
        rows = (await session.execute(q)).scalars().all()
        return [_serialize_participant(r) for r in rows]


# ── Transcript chunks ───────────────────────────────────────────────────

async def append_transcript_chunk(
    huddle_id: str,
    *,
    text: str,
    speaker_name: str = "Speaker",
    speaker_user_id: Optional[str] = None,
    user_id_for_activity: Optional[str] = None,
) -> Dict[str, Any]:
    """Persist a transcript chunk. Returns the chunk row (so caller can
    pass `id` to the live agent dispatcher in Phase 1.5)."""
    if not text or not text.strip():
        return {}
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        chunk = HuddleTranscriptChunkSQL(
            huddle_id=huddle_id,
            speaker_user_id=speaker_user_id,
            speaker_name=speaker_name,
            text=text.strip(),
        )
        session.add(chunk)
        await session.commit()
        await session.refresh(chunk)
        result = _serialize_chunk(chunk)

        org_id = (await session.execute(
            select(HuddleSQL.organization_id).where(HuddleSQL.id == huddle_id)
        )).scalar_one_or_none()

    if user_id_for_activity and org_id:
        # Only log every Nth chunk to avoid drowning the audit log
        # (callers can throttle; here we log every chunk).
        await log_activity(
            user_id=user_id_for_activity,
            organization_id=org_id,
            activity_type="huddle.transcript_chunk_received",
            details={"huddle_id": huddle_id, "chunk_id": result["id"], "chars": len(result["text"])},
            related_resource_type="huddle",
            related_resource_id=huddle_id,
            agent_name="Huddle",
        )
    return result


async def get_transcript(huddle_id: str) -> List[Dict[str, Any]]:
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        q = (
            select(HuddleTranscriptChunkSQL)
            .where(HuddleTranscriptChunkSQL.huddle_id == huddle_id)
            .order_by(HuddleTranscriptChunkSQL.ts.asc())
        )
        rows = (await session.execute(q)).scalars().all()
        return [_serialize_chunk(r) for r in rows]


# ── Agents in the call ─────────────────────────────────────────────────

async def attach_agent(
    huddle_id: str,
    *,
    agent_key: Optional[str] = None,
    custom_agent_id: Optional[str] = None,
    requesting_user_id: str,
) -> Optional[Dict[str, Any]]:
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        q = select(HuddleSQL).where(HuddleSQL.id == huddle_id, HuddleSQL.deleted_at.is_(None))
        row = (await session.execute(q)).scalar_one_or_none()
        if not row or row.host_user_id != requesting_user_id:
            return None
        keys = list(row.agent_keys or [])
        cids = list(row.custom_agent_ids or [])
        if agent_key and agent_key not in keys:
            keys.append(agent_key)
        if custom_agent_id and custom_agent_id not in cids:
            cids.append(custom_agent_id)
        row.agent_keys = keys
        row.custom_agent_ids = cids
        row.updated_at = datetime.utcnow()
        await session.commit()
        # Add as participant
    if agent_key or custom_agent_id:
        await add_participant(
            huddle_id,
            agent_key=agent_key,
            custom_agent_id=custom_agent_id,
            role="agent",
        )
    return await get_huddle(huddle_id, requesting_user_id=requesting_user_id)


async def detach_agent(
    huddle_id: str,
    *,
    agent_key: Optional[str] = None,
    custom_agent_id: Optional[str] = None,
    requesting_user_id: str,
) -> Optional[Dict[str, Any]]:
    session_factory = get_async_sessionmaker()
    async with session_factory() as session:
        q = select(HuddleSQL).where(HuddleSQL.id == huddle_id, HuddleSQL.deleted_at.is_(None))
        row = (await session.execute(q)).scalar_one_or_none()
        if not row or row.host_user_id != requesting_user_id:
            return None
        keys = [k for k in (row.agent_keys or []) if k != agent_key] if agent_key else list(row.agent_keys or [])
        cids = [c for c in (row.custom_agent_ids or []) if c != custom_agent_id] if custom_agent_id else list(row.custom_agent_ids or [])
        row.agent_keys = keys
        row.custom_agent_ids = cids
        row.updated_at = datetime.utcnow()
        await session.commit()
    if agent_key:
        await remove_participant(huddle_id, agent_key=agent_key)
    return await get_huddle(huddle_id, requesting_user_id=requesting_user_id)


# ── Helpers ────────────────────────────────────────────────────────────

def _default_title(meeting_type: str, host_user_id: str) -> str:
    if meeting_type == "scheduled":
        return "Scheduled meeting"
    if meeting_type == "recurring":
        return "Recurring meeting"
    return "Instant meeting"


def _serialize_huddle(row: HuddleSQL, public: bool = False) -> Dict[str, Any]:
    base = {
        "id": row.id,
        "room_name": row.room_name,
        "title": row.title,
        "meeting_type": row.meeting_type,
        "status": row.status,
        "team_id": row.team_id,
        "project_id": row.project_id,
        "scheduled_start": row.scheduled_start.isoformat() if row.scheduled_start else None,
        "scheduled_end": row.scheduled_end.isoformat() if row.scheduled_end else None,
        "started_at": row.started_at.isoformat() if row.started_at else None,
        "ended_at": row.ended_at.isoformat() if row.ended_at else None,
        "duration_sec": row.duration_sec,
        "participant_count_peak": row.participant_count_peak,
        "agent_keys": list(row.agent_keys or []),
        "custom_agent_ids": list(row.custom_agent_ids or []),
        "recording_enabled": bool(row.recording_enabled),
        "recording_url": row.recording_url,
        "recording_retention_days": row.recording_retention_days,
        "recording_expires_at": row.recording_expires_at.isoformat() if row.recording_expires_at else None,
        "lobby_enabled": bool(row.lobby_enabled),
        "require_sso": bool(row.require_sso),
        "e2ee_enabled": bool(row.e2ee_enabled),
        "data_residency": row.data_residency,
        "metadata": dict(row.meta or {}),
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }
    if public:
        # Public view (token-gated) — strip identifying fields
        return {k: base[k] for k in ("id", "room_name", "title", "status", "agent_keys", "started_at", "lobby_enabled", "e2ee_enabled")}
    base["host_user_id"] = row.host_user_id
    base["organization_id"] = row.organization_id
    base["share_token"] = row.share_token
    base["transcript_text"] = row.transcript_text
    base["processed_meeting_id"] = row.processed_meeting_id
    return base


def _serialize_participant(row: HuddleParticipantSQL) -> Dict[str, Any]:
    return {
        "id": row.id,
        "huddle_id": row.huddle_id,
        "user_id": row.user_id,
        "guest_name": row.guest_name,
        "guest_email": row.guest_email,
        "agent_key": row.agent_key,
        "custom_agent_id": row.custom_agent_id,
        "role": row.role,
        "joined_at": row.joined_at.isoformat() if row.joined_at else None,
        "left_at": row.left_at.isoformat() if row.left_at else None,
    }


def _serialize_chunk(row: HuddleTranscriptChunkSQL) -> Dict[str, Any]:
    return {
        "id": row.id,
        "huddle_id": row.huddle_id,
        "speaker_user_id": row.speaker_user_id,
        "speaker_name": row.speaker_name,
        "text": row.text,
        "ts": row.ts.isoformat() if row.ts else None,
        "agent_responses": list(row.agent_responses or []),
    }
