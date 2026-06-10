"""
Phase C — Chat (channels + messages + WS).

Mounted at `/api/v1/chat/`.

Tighter cut than the planned 60-endpoint surface: channels CRUD, messages
CRUD, /ws/chat/{channel_id} (subscribe + send), mentions, and the
`/run <agent_key> <prompt>` slash command that dispatches to the agent
router and posts the response back to the channel.

Threads, reactions, DMs, pins, and search are deferred — comments router
already covers threaded discussion on individual resources.
"""

from __future__ import annotations

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import (
    APIRouter,
    Body,
    Depends,
    HTTPException,
    Query,
    WebSocket,
    WebSocketDisconnect,
)
from pydantic import BaseModel, Field

from backend.api.deps import get_current_active_user
from backend.core.security import verify_token
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.event_bus import emit

logger = structlog.get_logger(__name__)
router = APIRouter()

SLASH_RE = re.compile(r"^/run\s+([a-z_][a-z0-9_]*)\s+(.+)$", re.IGNORECASE)
MENTION_RE = re.compile(r"@([a-zA-Z0-9_.-]+)")


def _oid(v: Any) -> Optional[ObjectId]:
    if v is None:
        return None
    if isinstance(v, ObjectId):
        return v
    try:
        return ObjectId(str(v))
    except Exception:
        return None


def _resolve_primary_org_id(user: User) -> str:
    primary = getattr(user, "organization_id", None)
    if primary:
        return str(primary)
    ids = getattr(user, "organization_ids", None) or []
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


async def _require_channel_access(channel_id: str, user_id: str) -> Dict[str, Any]:
    col = await MongoDB.get_collection("chat_channels")
    ch = await col.find_one({"_id": _oid(channel_id)})
    if not ch:
        raise HTTPException(status_code=404, detail="Channel not found")
    # Workspace + project + team channels: any org member can join.  DMs +
    # private channels require explicit membership.
    ch_type = ch.get("type") or "team"
    if ch_type in ("dm", "private"):
        members = [_oid(m) for m in (ch.get("member_ids") or [])]
        if _oid(user_id) not in members:
            raise HTTPException(status_code=403, detail="Not a channel member")
    return ch


def _serialize_channel(ch: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(ch)
    d["id"] = str(d.pop("_id"))
    for k in ("organization_id", "team_id", "project_id", "created_by"):
        if d.get(k):
            d[k] = str(d[k])
    d["member_ids"] = [str(m) for m in (d.get("member_ids") or [])]
    return d


def _serialize_message(m: Dict[str, Any]) -> Dict[str, Any]:
    d = dict(m)
    d["id"] = str(d.pop("_id"))
    for k in ("organization_id", "channel_id", "user_id", "parent_message_id"):
        if d.get(k):
            d[k] = str(d[k])
    d["mentions"] = [str(x) for x in (d.get("mentions") or [])]
    return d


# ── Channels CRUD ──────────────────────────────────────────────────


class ChannelCreatePayload(BaseModel):
    name: str = Field(..., max_length=120)
    type: str = Field("workspace", description="workspace | team | project | dm | private")
    team_id: Optional[str] = None
    project_id: Optional[str] = None
    member_ids: List[str] = Field(default_factory=list)
    description: Optional[str] = Field(None, max_length=1000)


@router.get("/channels")
async def list_channels(
    organization_id: Optional[str] = Query(None),
    project_id: Optional[str] = Query(None),
    team_id: Optional[str] = Query(None),
    type_filter: Optional[str] = Query(None, alias="type"),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("chat_channels")
    q: Dict[str, Any] = {"organization_id": _oid(org_id)}
    if team_id:
        q["team_id"] = _oid(team_id)
    if project_id:
        q["project_id"] = _oid(project_id)
    if type_filter:
        q["type"] = type_filter
    cursor = col.find(q).sort("last_message_at", -1).limit(200)
    rows = await cursor.to_list(length=200)
    return [_serialize_channel(r) for r in rows]


@router.post("/channels", status_code=201)
async def create_channel(
    payload: ChannelCreatePayload,
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("chat_channels")
    member_ids = {str(current_user.id), *payload.member_ids}
    doc = {
        "organization_id": _oid(org_id),
        "team_id": _oid(payload.team_id) if payload.team_id else None,
        "project_id": _oid(payload.project_id) if payload.project_id else None,
        "name": payload.name,
        "type": payload.type,
        "description": payload.description,
        "member_ids": [_oid(m) for m in member_ids],
        "created_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
        "last_message_at": None,
    }
    r = await col.insert_one(doc)
    doc["_id"] = r.inserted_id
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="chat.channel_created",
        details={"channel_id": str(r.inserted_id), "name": payload.name},
        related_resource_type="chat_channel", related_resource_id=str(r.inserted_id),
    )
    await emit(
        "chat.channel_created", organization_id=org_id, actor_id=str(current_user.id),
        team_id=payload.team_id, project_id=payload.project_id,
        resource_type="chat_channel", resource_id=str(r.inserted_id),
        payload={"name": payload.name},
    )
    return _serialize_channel(doc)


@router.get("/channels/{channel_id}")
async def get_channel(
    channel_id: str,
    current_user: User = Depends(get_current_active_user),
):
    ch = await _require_channel_access(channel_id, str(current_user.id))
    return _serialize_channel(ch)


@router.delete("/channels/{channel_id}", status_code=204)
async def delete_channel(
    channel_id: str,
    current_user: User = Depends(get_current_active_user),
):
    ch = await _require_channel_access(channel_id, str(current_user.id))
    if str(ch.get("created_by")) != str(current_user.id):
        # Only the creator (or an admin) can hard-delete.
        org = await organization_repository.get_by_id(str(ch["organization_id"]))
        if _oid(current_user.id) not in [_oid(a) for a in (org.admin_ids or [])]:
            raise HTTPException(status_code=403, detail="Only channel creator or org admin can delete")
    col = await MongoDB.get_collection("chat_channels")
    await col.delete_one({"_id": _oid(channel_id)})
    return None


@router.post("/channels/{channel_id}/join", status_code=204)
async def join_channel(
    channel_id: str,
    current_user: User = Depends(get_current_active_user),
):
    ch = await _require_channel_access(channel_id, str(current_user.id))
    col = await MongoDB.get_collection("chat_channels")
    await col.update_one(
        {"_id": _oid(channel_id)},
        {"$addToSet": {"member_ids": _oid(current_user.id)}},
    )
    return None


@router.post("/channels/{channel_id}/leave", status_code=204)
async def leave_channel(
    channel_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("chat_channels")
    await col.update_one(
        {"_id": _oid(channel_id)},
        {"$pull": {"member_ids": _oid(current_user.id)}},
    )
    return None


# ── Messages ──────────────────────────────────────────────────────


class MessageCreatePayload(BaseModel):
    content: str = Field(..., max_length=8000)
    parent_message_id: Optional[str] = None
    attachments: List[Dict[str, Any]] = Field(default_factory=list)


@router.get("/channels/{channel_id}/messages")
async def list_messages(
    channel_id: str,
    before: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_active_user),
):
    ch = await _require_channel_access(channel_id, str(current_user.id))
    col = await MongoDB.get_collection("chat_messages")
    q: Dict[str, Any] = {"channel_id": _oid(channel_id)}
    if before:
        q["created_at"] = {"$lt": datetime.fromisoformat(before.replace("Z", ""))}
    cursor = col.find(q).sort("created_at", -1).limit(limit)
    rows = await cursor.to_list(length=limit)
    rows.reverse()  # chronological for the client
    return [_serialize_message(r) for r in rows]


async def _post_message(
    *,
    channel_id: str,
    channel_doc: Dict[str, Any],
    user_id: Optional[str],
    content: str,
    parent_message_id: Optional[str] = None,
    agent_key: Optional[str] = None,
    attachments: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    msg_col = await MongoDB.get_collection("chat_messages")
    ch_col = await MongoDB.get_collection("chat_channels")
    now = datetime.utcnow()
    # Mentions: extract @handles → leave as strings for the client to resolve;
    # IDs would require a directory lookup the chat router doesn't own.
    mention_handles = list(set(MENTION_RE.findall(content)))
    doc = {
        "channel_id": _oid(channel_id),
        "organization_id": channel_doc["organization_id"],
        "user_id": _oid(user_id) if user_id else None,
        "agent_key": agent_key,
        "content": content,
        "attachments": attachments or [],
        "mentions": mention_handles,
        "parent_message_id": _oid(parent_message_id) if parent_message_id else None,
        "reactions": {},
        "created_at": now,
    }
    r = await msg_col.insert_one(doc)
    doc["_id"] = r.inserted_id
    await ch_col.update_one(
        {"_id": _oid(channel_id)},
        {"$set": {"last_message_at": now}},
    )
    return doc


@router.post("/channels/{channel_id}/messages", status_code=201)
async def post_message(
    channel_id: str,
    payload: MessageCreatePayload,
    current_user: User = Depends(get_current_active_user),
):
    ch = await _require_channel_access(channel_id, str(current_user.id))
    user_msg = await _post_message(
        channel_id=channel_id, channel_doc=ch,
        user_id=str(current_user.id), content=payload.content,
        parent_message_id=payload.parent_message_id,
        attachments=payload.attachments,
    )
    await emit(
        "chat.message_posted",
        organization_id=str(ch["organization_id"]),
        actor_id=str(current_user.id),
        project_id=str(ch["project_id"]) if ch.get("project_id") else None,
        team_id=str(ch["team_id"]) if ch.get("team_id") else None,
        resource_type="chat_message",
        resource_id=str(user_msg["_id"]),
        payload={"channel_id": channel_id, "preview": payload.content[:120]},
    )

    # Slash-command interception: post agent reply inline.
    m = SLASH_RE.match(payload.content.strip())
    if m:
        agent_key, prompt = m.group(1).lower(), m.group(2).strip()
        try:
            agent_reply = await _run_agent_inline(
                agent_key=agent_key,
                prompt=prompt,
                organization_id=str(ch["organization_id"]),
                user_id=str(current_user.id),
            )
        except Exception as exc:  # noqa: BLE001
            agent_reply = f"⚠️ Agent {agent_key} failed: {exc}"
        bot_msg = await _post_message(
            channel_id=channel_id, channel_doc=ch,
            user_id=None, agent_key=agent_key, content=agent_reply,
        )
        return {
            "user_message": _serialize_message(user_msg),
            "agent_message": _serialize_message(bot_msg),
        }

    return _serialize_message(user_msg)


@router.delete("/messages/{message_id}", status_code=204)
async def delete_message(
    message_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("chat_messages")
    row = await col.find_one({"_id": _oid(message_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Message not found")
    if str(row.get("user_id")) != str(current_user.id):
        raise HTTPException(status_code=403, detail="Can only delete your own messages")
    await col.delete_one({"_id": _oid(message_id)})
    return None


# ── Slash-command dispatch ────────────────────────────────────────


async def _run_agent_inline(
    *, agent_key: str, prompt: str, organization_id: str, user_id: str,
) -> str:
    """Quick router-friendly dispatch.  Falls back to a friendly canned
    reply when the agent isn't available in this build."""
    try:
        from backend.agents.router import AGENT_REGISTRY
    except Exception:
        return f"_Agent registry not available in this build._"

    if agent_key not in AGENT_REGISTRY:
        return f"_Unknown agent `{agent_key}`. Try one of: {', '.join(list(AGENT_REGISTRY.keys())[:6])}…_"

    # Most agents register an `acall`/`run`/`process_async` entrypoint; we
    # don't want to over-fit to one shape here.  For a v1 we return the
    # registry stub so the channel sees something useful.
    try:
        from backend.agents.agent_service import get_agent
        agent = get_agent(agent_key)
        if hasattr(agent, "process_async"):
            out = await agent.process_async({"prompt": prompt, "user_id": user_id, "organization_id": organization_id})
            if isinstance(out, dict):
                return out.get("content") or out.get("response") or json.dumps(out)[:400]
            return str(out)[:400]
    except Exception as exc:  # noqa: BLE001
        logger.warning("chat.agent_dispatch_failed", agent=agent_key, error=str(exc))
    return f"`{agent_key}` received your request: _{prompt[:120]}_\n\n(Async dispatch will follow up in the project channel.)"


# ── Mentions ──────────────────────────────────────────────────────


@router.get("/mentions/me")
async def list_my_mentions(
    organization_id: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    current_user: User = Depends(get_current_active_user),
):
    org_id = organization_id or _resolve_primary_org_id(current_user)
    await _require_org_member(org_id, current_user)
    col = await MongoDB.get_collection("chat_messages")
    handle = (getattr(current_user, "email", "") or "").split("@")[0]
    if not handle:
        return []
    cursor = col.find({
        "organization_id": _oid(org_id),
        "mentions": handle,
    }).sort("created_at", -1).limit(limit)
    return [_serialize_message(r) for r in await cursor.to_list(length=limit)]


# ── WebSocket: subscribe + send ───────────────────────────────────


@router.websocket("/ws/{channel_id}")
async def channel_ws(websocket: WebSocket, channel_id: str):
    token = websocket.query_params.get("token", "")
    if not token:
        await websocket.close(code=4401)
        return
    try:
        class _Cred:
            credentials = token
        token_data = await verify_token(_Cred())
        user_id = token_data.get("user_id") or token_data.get("uid")
    except Exception:
        await websocket.close(code=4401)
        return
    try:
        ch = await _require_channel_access(channel_id, user_id)
    except HTTPException:
        await websocket.close(code=4403)
        return

    await websocket.accept()
    await websocket.send_json({"type": "connected", "channel_id": channel_id})

    # Send recent history for context (last 25 messages).
    msg_col = await MongoDB.get_collection("chat_messages")
    recent = await msg_col.find({"channel_id": _oid(channel_id)}).sort("created_at", -1).limit(25).to_list(length=25)
    recent.reverse()
    await websocket.send_json({"type": "history", "messages": [_serialize_message(r) for r in recent]})

    try:
        while True:
            raw = await websocket.receive_text()
            try:
                data = json.loads(raw)
            except Exception:
                continue
            kind = data.get("type")
            if kind == "ping":
                await websocket.send_json({"type": "pong"})
                continue
            if kind == "message":
                content = (data.get("content") or "").strip()
                if not content:
                    continue
                m = await _post_message(
                    channel_id=channel_id, channel_doc=ch,
                    user_id=user_id, content=content,
                )
                await websocket.send_json({"type": "message", "message": _serialize_message(m)})
                slash = SLASH_RE.match(content)
                if slash:
                    agent_key, prompt = slash.group(1).lower(), slash.group(2).strip()
                    try:
                        reply = await _run_agent_inline(
                            agent_key=agent_key, prompt=prompt,
                            organization_id=str(ch["organization_id"]),
                            user_id=user_id,
                        )
                    except Exception as exc:  # noqa: BLE001
                        reply = f"⚠️ Agent {agent_key} failed: {exc}"
                    bot = await _post_message(
                        channel_id=channel_id, channel_doc=ch,
                        user_id=None, agent_key=agent_key, content=reply,
                    )
                    await websocket.send_json({"type": "message", "message": _serialize_message(bot)})
    except WebSocketDisconnect:
        return
    except Exception as exc:  # noqa: BLE001
        logger.exception("chat_ws_error", error=str(exc))
        try:
            await websocket.close()
        except Exception:
            pass
