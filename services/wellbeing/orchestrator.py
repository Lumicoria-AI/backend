"""Service-layer orchestrator for the Well-being Coach.

Responsibilities:
  - Lock the LLM provider to Gemini (per platform decision).
  - Build agent config + strict JSON prompts for each action:
       generate_recommendations, generate_break, chat_response,
       weekly_reflection.
  - Run the agent.
  - Persist recommendations / chat history through the wellbeing
    repository when relevant.

The agent itself (`backend/agents/wellbeing_agent.py`) does the LLM
call.  This wrapper owns tenant-scoping, persistence, and result
normalisation.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Dict, List, Optional

import structlog

from .sanitize import clean_text, clean_parameters, coerce_jsonable

logger = structlog.get_logger(__name__)


# Per-organization async locks so two simultaneous requests from the
# same workspace can't collide on the in-process agent.
_org_locks: Dict[str, asyncio.Lock] = {}


def _lock_for(org_id: str) -> asyncio.Lock:
    if org_id not in _org_locks:
        _org_locks[org_id] = asyncio.Lock()
    return _org_locks[org_id]


# ── Provider locked to Gemini ──────────────────────────────────────


_GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"


def _build_agent_config(model_override: Optional[str] = None) -> Dict[str, Any]:
    model = model_override or _GEMINI_DEFAULT_MODEL
    return {
        "provider": "gemini",
        "model": model,
        # BaseAgent reads `agent_model_config`, not `model_config`.
        "agent_model_config": {
            "model": model,
            "temperature": 0.4,
            # The Coach's structured output is small (8 recs, a few
            # paragraphs).  4k is enough; we leave headroom for the
            # weekly reflection which can run longer.
            "max_tokens": 8192,
        },
    }


def _new_agent(model_override: Optional[str] = None):
    """Instantiate a fresh wellbeing agent.  We don't reuse a global
    singleton because per-call config can differ (e.g., a power-user
    on Pro could override model)."""
    from ...agents.wellbeing_agent import WellbeingAgent

    return WellbeingAgent(_build_agent_config(model_override))


# ── Action runners ─────────────────────────────────────────────────


async def generate_recommendations(
    *,
    organization_id: str,
    user_id: str,
    user_data: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Ask the agent for up to ~8 personalised recommendations.

    `user_data` should carry whatever signals the caller has: latest
    metrics, recent activities, productivity stats, etc.  The agent
    decides what to surface.
    """
    if not organization_id:
        raise ValueError("organization_id is required")

    async with _lock_for(organization_id):
        agent = _new_agent(model_override)
        request = {
            "action": "recommendations",
            "data": coerce_jsonable({"user_data": user_data}),
            "context": clean_parameters(context),
            "parameters": {},
        }
        try:
            result = await agent.process_async(request)
        except Exception as e:  # noqa: BLE001
            logger.error("wellbeing_recommendations_failed", error=str(e))
            return []

    items = (result or {}).get("results") or []
    if isinstance(items, dict):
        # Some prompts come back wrapped — unwrap the common keys.
        items = (
            items.get("recommendations")
            or items.get("items")
            or []
        )
    if not isinstance(items, list):
        return []
    # Persist via the existing repository so the dashboard can show
    # them later even if the agent times out.
    try:
        from ...db.mongodb.repositories.wellbeing_repository import (
            wellbeing_repository,
        )

        for rec in items[:8]:
            try:
                await wellbeing_repository.save_recommendation(
                    user_id=user_id,
                    organization_id=organization_id,
                    recommendation=rec,
                )
            except Exception:  # noqa: BLE001
                continue
    except Exception as e:  # noqa: BLE001
        logger.warning("wellbeing_save_recs_failed", error=str(e))

    return items[:8]


async def generate_break(
    *,
    organization_id: str,
    user_id: str,
    user_data: Dict[str, Any],
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Return `{break_type, duration_minutes, reason, suggested_activities}`."""
    if not organization_id:
        raise ValueError("organization_id is required")

    async with _lock_for(organization_id):
        agent = _new_agent(model_override)
        request = {
            "action": "break_recommendation",
            "data": coerce_jsonable({"user_data": user_data}),
            "context": {},
            "parameters": {"request_type": "break_recommendation"},
        }
        try:
            result = await agent.process_async(request)
        except Exception as e:  # noqa: BLE001
            logger.error("wellbeing_break_failed", error=str(e))
            return _fallback_break()

    inner = (result or {}).get("results") or {}
    if isinstance(inner, list) and inner:
        inner = inner[0]
    if not isinstance(inner, dict) or not inner:
        return _fallback_break()
    return {
        "break_type": str(inner.get("break_type") or "micro_break"),
        "duration_minutes": int(inner.get("duration_minutes") or 5),
        "reason": str(inner.get("reason") or "Take a moment to reset."),
        "suggested_activities": [
            str(a) for a in (inner.get("suggested_activities") or [])
            if a
        ][:5],
    }


def _fallback_break() -> Dict[str, Any]:
    return {
        "break_type": "micro_break",
        "duration_minutes": 5,
        "reason": "You've been working steadily — a quick reset will help.",
        "suggested_activities": [
            "Stand up and stretch",
            "Look at something 20 feet away for 20 seconds",
            "Take a few deep breaths",
        ],
    }


async def chat_with_coach(
    *,
    organization_id: str,
    user_id: str,
    message: str,
    history: Optional[List[Dict[str, str]]] = None,
    user_data: Optional[Dict[str, Any]] = None,
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Conversational coach turn.  Returns
    `{response, follow_up_suggestions[], action_buttons[]}`.
    """
    cleaned_message = clean_text(message, max_len=4_000)
    if not cleaned_message:
        return {
            "response": "Tell me what's on your mind — I'm here.",
            "follow_up_suggestions": [],
            "action_buttons": [],
        }

    async with _lock_for(organization_id):
        agent = _new_agent(model_override)
        request = {
            "action": "chat",
            "data": coerce_jsonable({
                "message": cleaned_message,
                "history": (history or [])[-10:],
                "user_data": user_data or {},
            }),
            "context": {},
            "parameters": {"request_type": "chat"},
        }
        try:
            result = await agent.process_async(request)
        except Exception as e:  # noqa: BLE001
            logger.error("wellbeing_chat_failed", error=str(e))
            return {
                "response": "I'm having trouble responding right now. Try again in a moment.",
                "follow_up_suggestions": [],
                "action_buttons": [],
            }

    inner = (result or {}).get("results") or {}
    if not isinstance(inner, dict):
        inner = {"response": str(inner)}
    return {
        "response": str(inner.get("response") or "").strip()
        or "I'm here whenever you'd like to talk.",
        "follow_up_suggestions": [
            str(s) for s in (inner.get("follow_up_suggestions") or []) if s
        ][:4],
        "action_buttons": [
            {"label": str(a.get("label") or ""), "action": str(a.get("action") or "")}
            for a in (inner.get("action_buttons") or [])
            if isinstance(a, dict) and a.get("label")
        ][:3],
    }


async def weekly_reflection(
    *,
    organization_id: str,
    user_id: str,
    user_data: Dict[str, Any],
    model_override: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate the weekly summary that powers the email digest and
    the Coach's weekly panel.  Returns
    `{summary, highlights[], concerns[], focus_for_next_week[], encouragement}`.
    """
    async with _lock_for(organization_id):
        agent = _new_agent(model_override)
        request = {
            "action": "weekly_reflection",
            "data": coerce_jsonable({"user_data": user_data}),
            "context": {},
            "parameters": {"request_type": "weekly_reflection"},
        }
        try:
            result = await agent.process_async(request)
        except Exception as e:  # noqa: BLE001
            logger.error("wellbeing_reflection_failed", error=str(e))
            return _fallback_reflection()

    inner = (result or {}).get("results") or {}
    if not isinstance(inner, dict) or not inner:
        return _fallback_reflection()
    return {
        "summary": str(inner.get("summary") or "").strip(),
        "highlights": [str(x) for x in (inner.get("highlights") or []) if x][:5],
        "concerns": [str(x) for x in (inner.get("concerns") or []) if x][:5],
        "focus_for_next_week": [
            str(x) for x in (inner.get("focus_for_next_week") or []) if x
        ][:5],
        "encouragement": str(inner.get("encouragement") or "").strip(),
    }


def _fallback_reflection() -> Dict[str, Any]:
    return {
        "summary": "Here's a fresh week. Pick one habit you'd like to build and start small.",
        "highlights": [],
        "concerns": [],
        "focus_for_next_week": [
            "Aim for one short break every hour of focused work.",
            "Log your mood once a day.",
        ],
        "encouragement": "Small steady habits compound. You've got this.",
    }
