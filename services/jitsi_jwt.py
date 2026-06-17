"""
Lumicoria Huddle — Jitsi JWT signer.

Self-hosted Jitsi (meet.lumicoria.ai) authenticates room access with a
per-user JWT signed by JITSI_APP_SECRET. Public Jitsi (`meet.jit.si`)
returns None — JitsiEmbed accepts an optional jwt prop, so unauthenticated
public rooms keep working unchanged.

Claim shape (Jitsi-standard):
  iss  = JITSI_APP_ID
  aud  = "jitsi"
  sub  = JITSI_DOMAIN
  room = "*" (allow any room) OR a specific room_name
  iat  = now
  exp  = now + JITSI_JWT_TTL_SECONDS
  context = {
      "user": {
          "name":  display_name,
          "email": email,
          "avatar": avatar_url,
          "id":    user_id,
      },
      "features": {
          "recording":     bool,
          "livestreaming": bool,
          "transcription": bool,
          "outbound-call": bool,
      },
      "moderator": bool
  }
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import jwt as pyjwt
import structlog

from backend.core.config import settings

logger = structlog.get_logger(__name__)


def is_self_hosted() -> bool:
    """True when the deployment uses our own Jitsi (meet.lumicoria.ai)
    with JWT auth, False for public meet.jit.si."""
    return bool(settings.JITSI_APP_ID and settings.JITSI_APP_SECRET and settings.JITSI_DOMAIN != "meet.jit.si")


def sign_room_jwt(
    *,
    room: str = "*",
    user_id: Optional[str] = None,
    display_name: Optional[str] = None,
    email: Optional[str] = None,
    avatar_url: Optional[str] = None,
    moderator: bool = False,
    allow_recording: bool = False,
    allow_livestreaming: bool = False,
    ttl_seconds: Optional[int] = None,
) -> Optional[str]:
    """Return a signed Jitsi JWT, or None if self-hosting isn't configured."""
    if not is_self_hosted():
        return None

    now = int(time.time())
    exp = now + (ttl_seconds or settings.JITSI_JWT_TTL_SECONDS)

    payload: Dict[str, Any] = {
        "iss": settings.JITSI_APP_ID,
        "aud": "jitsi",
        "sub": settings.JITSI_DOMAIN,
        "room": room,
        "iat": now,
        "nbf": now - 5,
        "exp": exp,
        "context": {
            "user": {
                "id": user_id or "anonymous",
                "name": display_name or "Guest",
                "email": email or "",
                "avatar": avatar_url or "",
            },
            "features": {
                "recording": allow_recording,
                "livestreaming": allow_livestreaming,
                "transcription": True,
                "outbound-call": False,
            },
            "moderator": moderator,
        },
    }

    try:
        token = pyjwt.encode(payload, settings.JITSI_APP_SECRET, algorithm="HS256")
        # PyJWT < 2 returns bytes
        if isinstance(token, bytes):
            token = token.decode("utf-8")
        return token
    except Exception as e:
        logger.warning("jitsi_jwt_sign_failed", error=str(e))
        return None


def domain() -> str:
    """The Jitsi domain the frontend should embed."""
    return settings.JITSI_DOMAIN
