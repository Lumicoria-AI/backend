"""
Lumicoria Huddle — real-time room WebSocket.

Connect: ws://host:port/api/v1/ws/huddle/{huddle_id}?token=JWT
Public:  ws://host:port/api/v1/ws/huddle/share/{share_token}

Server → client message envelope:
  {
    "type":  "transcript_chunk"          // chunk persisted
           | "agent_response"            // an attached agent answered
           | "participant_joined"        // someone joined
           | "participant_left"          // someone left
           | "huddle_ended"              // host ended the call
           | "ping"                      // heartbeat
           | "connected"                 // initial ack
    "huddle_id": "...",
    ...payload depending on type
  }

Client → server:
  {"type": "pong"}                       // heartbeat reply
  {"type": "subscribe"}                  // re-subscribe (no-op for now)
"""

from __future__ import annotations

import asyncio
import json
from typing import Optional

import structlog
from fastapi import APIRouter, Query, WebSocket, WebSocketDisconnect
from sqlalchemy import select

from backend.core.security import verify_token
from backend.db.postgres import get_async_sessionmaker
from backend.db.postgres_models import HuddleSQL, HuddleParticipantSQL
from backend.services.notification_service import connection_manager
from backend.services.realtime import realtime

logger = structlog.get_logger(__name__)
router = APIRouter()

HEARTBEAT_SEC = 25


async def _auth_user(token: Optional[str]) -> Optional[str]:
    if not token:
        return None
    try:
        payload = await verify_token(token)
        if payload:
            return payload.get("user_id") or payload.get("sub")
    except Exception as exc:
        logger.warning("huddle_ws_auth_failed", error=str(exc))
    return None


async def _huddle_for_id(huddle_id: str):
    factory = get_async_sessionmaker()
    async with factory() as session:
        q = select(HuddleSQL).where(HuddleSQL.id == huddle_id, HuddleSQL.deleted_at.is_(None))
        return (await session.execute(q)).scalar_one_or_none()


async def _huddle_for_share(share_token: str):
    factory = get_async_sessionmaker()
    async with factory() as session:
        q = select(HuddleSQL).where(HuddleSQL.share_token == share_token, HuddleSQL.deleted_at.is_(None))
        return (await session.execute(q)).scalar_one_or_none()


async def _is_participant(huddle_id: str, user_id: str) -> bool:
    factory = get_async_sessionmaker()
    async with factory() as session:
        h = (await session.execute(
            select(HuddleSQL.host_user_id).where(HuddleSQL.id == huddle_id)
        )).scalar_one_or_none()
        if h == user_id:
            return True
        p = (await session.execute(
            select(HuddleParticipantSQL).where(
                HuddleParticipantSQL.huddle_id == huddle_id,
                HuddleParticipantSQL.user_id == user_id,
            ).limit(1)
        )).scalar_one_or_none()
        return p is not None


async def _send_heartbeat(websocket: WebSocket) -> None:
    while True:
        try:
            await asyncio.sleep(HEARTBEAT_SEC)
            await websocket.send_json({"type": "ping"})
        except Exception:
            break


async def _serve_room(websocket: WebSocket, huddle_id: str, user_id_for_connection: str) -> None:
    """Accept the socket, register with the local ConnectionManager keyed
    by `huddle_id`, subscribe the broker, and forward messages until the
    client disconnects."""
    # We register using a synthetic "user" id of `huddle:{huddle_id}` so
    # ConnectionManager treats this as a separate channel. The broker
    # delivers `rt:huddle:{id}` messages to this synthetic id.
    channel_id = f"huddle:{huddle_id}:{user_id_for_connection}"
    # The outer endpoint already accepted the socket. Register without
    # re-accepting — connect() would call websocket.accept() a second
    # time and Starlette raises a hard ASGI error.
    connection_manager.register(websocket, channel_id)
    await realtime.subscribe_huddle(huddle_id)

    await websocket.send_json({
        "type": "connected",
        "huddle_id": huddle_id,
        "channel_id": channel_id,
    })

    heartbeat = asyncio.create_task(_send_heartbeat(websocket))

    try:
        while True:
            data = await websocket.receive_text()
            try:
                msg = json.loads(data)
            except Exception:
                continue
            mtype = msg.get("type")
            if mtype == "pong":
                continue
            if mtype == "subscribe":
                # No-op: we already subscribed on connect.
                await websocket.send_json({"type": "subscribed", "huddle_id": huddle_id})
    except WebSocketDisconnect:
        logger.info("huddle_ws_disconnect", huddle_id=huddle_id, channel_id=channel_id)
    except Exception as exc:  # noqa: BLE001
        logger.warning("huddle_ws_error", huddle_id=huddle_id, error=str(exc))
    finally:
        heartbeat.cancel()
        try:
            connection_manager.disconnect(websocket, channel_id)
        except Exception:
            pass


@router.websocket("/huddle/{huddle_id}")
async def websocket_huddle(
    websocket: WebSocket,
    huddle_id: str,
    token: Optional[str] = Query(None),
):
    user_id = await _auth_user(token)
    if not user_id:
        await websocket.close(code=4001, reason="Unauthorized")
        return

    if not await _is_participant(huddle_id, user_id):
        # Lenient: still let host attach later in the call; for now reject.
        await websocket.close(code=4003, reason="Not a participant")
        return

    await websocket.accept()
    await _serve_room(websocket, huddle_id, user_id)


@router.websocket("/huddle/share/{share_token}")
async def websocket_huddle_public(
    websocket: WebSocket,
    share_token: str,
):
    """Guest websocket — joining via the public share link before signing
    in. Auth is via the unguessable share_token."""
    h = await _huddle_for_share(share_token)
    if not h:
        await websocket.close(code=4004, reason="Invalid link")
        return
    await websocket.accept()
    await _serve_room(websocket, h.id, f"guest-{share_token[:8]}")
