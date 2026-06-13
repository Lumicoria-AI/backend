"""
Lumicoria AI — Realtime presence + typing service.

Tracks who is online per organization and which room (project / team /
chat channel / task) each socket is currently subscribed to.  Broadcasts:

  - presence.update         { org_id, user_id, online: true/false, last_seen }
  - typing.start            { room, user_id, started_at }
  - typing.stop             { room, user_id }

This is a lightweight in-process broker.  For multi-worker deployments
the same shape can be lifted onto Redis pub/sub later — the public API
(`mark_online`, `mark_offline`, `subscribe`, `unsubscribe`,
`broadcast_typing`) stays the same.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple
import asyncio
import structlog

logger = structlog.get_logger(__name__)


class PresenceBroker:
    def __init__(self) -> None:
        # user_id -> list[(websocket, org_id)]
        self._sockets: Dict[str, List[Tuple[Any, str]]] = {}
        # org_id -> set[user_id]
        self._org_online: Dict[str, Set[str]] = {}
        # room -> set[(user_id, websocket)]
        self._room_subs: Dict[str, Set[Tuple[str, Any]]] = {}
        # user_id -> last_seen_iso
        self._last_seen: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    # ── Lifecycle ──────────────────────────────────────────────────

    async def mark_online(self, websocket: Any, *, user_id: str, organization_id: str) -> None:
        async with self._lock:
            self._sockets.setdefault(user_id, []).append((websocket, organization_id))
            self._org_online.setdefault(organization_id, set()).add(user_id)
            self._last_seen[user_id] = datetime.now(timezone.utc).isoformat()
        await self._broadcast_to_org(
            organization_id,
            {"type": "presence.update", "data": {"user_id": user_id, "online": True}},
            exclude_user=user_id,
        )

    async def mark_offline(self, websocket: Any, *, user_id: str) -> None:
        org_id: Optional[str] = None
        async with self._lock:
            socks = self._sockets.get(user_id, [])
            self._sockets[user_id] = [(ws, oid) for (ws, oid) in socks if ws is not websocket]
            if not self._sockets[user_id]:
                # User has no remaining sockets in any org — drop fully.
                self._sockets.pop(user_id, None)
                # Drop from every org bucket they were in.
                for oid, members in list(self._org_online.items()):
                    if user_id in members:
                        members.discard(user_id)
                        if not org_id:
                            org_id = oid
            # Tear down all room subs tied to this socket.
            for room, subs in list(self._room_subs.items()):
                self._room_subs[room] = {(uid, ws) for (uid, ws) in subs if ws is not websocket}
                if not self._room_subs[room]:
                    del self._room_subs[room]
            self._last_seen[user_id] = datetime.now(timezone.utc).isoformat()

        if org_id:
            await self._broadcast_to_org(
                org_id,
                {
                    "type": "presence.update",
                    "data": {
                        "user_id": user_id,
                        "online": False,
                        "last_seen": self._last_seen.get(user_id),
                    },
                },
                exclude_user=user_id,
            )

    # ── Rooms ──────────────────────────────────────────────────────

    async def subscribe(self, websocket: Any, *, user_id: str, room: str) -> None:
        async with self._lock:
            self._room_subs.setdefault(room, set()).add((user_id, websocket))

    async def unsubscribe(self, websocket: Any, *, user_id: str, room: str) -> None:
        async with self._lock:
            subs = self._room_subs.get(room)
            if not subs:
                return
            subs.discard((user_id, websocket))
            if not subs:
                self._room_subs.pop(room, None)

    # ── Typing ─────────────────────────────────────────────────────

    async def broadcast_typing(self, *, room: str, user_id: str, typing: bool) -> None:
        payload = {
            "type": "typing.start" if typing else "typing.stop",
            "data": {
                "room": room,
                "user_id": user_id,
                "at": datetime.now(timezone.utc).isoformat(),
            },
        }
        await self._broadcast_to_room(room, payload, exclude_user=user_id)

    # ── Snapshots ──────────────────────────────────────────────────

    def online_users_for_org(self, organization_id: str) -> List[str]:
        return sorted(list(self._org_online.get(organization_id, set())))

    def last_seen(self, user_id: str) -> Optional[str]:
        return self._last_seen.get(user_id)

    # ── Internals ──────────────────────────────────────────────────

    async def _broadcast_to_org(self, org_id: str, message: dict, *, exclude_user: Optional[str] = None) -> None:
        targets: List[Any] = []
        for uid in list(self._org_online.get(org_id, set())):
            if uid == exclude_user:
                continue
            for ws, oid in self._sockets.get(uid, []):
                if oid == org_id:
                    targets.append(ws)
        for ws in targets:
            try:
                await ws.send_json(message)
            except Exception:
                pass

    async def _broadcast_to_room(self, room: str, message: dict, *, exclude_user: Optional[str] = None) -> None:
        subs = list(self._room_subs.get(room, set()))
        for uid, ws in subs:
            if uid == exclude_user:
                continue
            try:
                await ws.send_json(message)
            except Exception:
                pass


presence_broker = PresenceBroker()
