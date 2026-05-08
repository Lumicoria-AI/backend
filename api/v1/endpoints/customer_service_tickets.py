"""Operator-facing ticket endpoints.

All require auth + AGENT/customer_service/EXECUTE permission and are
tenant-scoped via the caller's `organization_id`.  The legacy mock-only
endpoints in `customer_service.py` are unaffected — those stay for the
LLM dispatch surface (`/process`, `/analyze-feedback`, …).

Security posture:
    - SQLAlchemy ORM throughout (no string-formatted SQL).
    - Pydantic validators with strict length caps.
    - HTML sanitization in the service layer (single source of truth).
    - permission_repository.check_permission gate.
    - Tenant isolation on every query.
    - Activity logged on every mutation.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, EmailStr, Field, field_validator

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.permission_repository import permission_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.customer_service import (
    analytics as analytics_svc,
    branding as branding_svc,
    draft as draft_svc,
    templates as templates_svc,
    tickets as tickets_svc,
)

router = APIRouter()
logger = structlog.get_logger(__name__)


# ── Permission helper ───────────────────────────────────────────────────


async def _require_cs_permission(current_user: User) -> str:
    """Return the tenant scope id, or raise 403.

    Permission is checked against the user's actual organization_id
    (which may be None for personal accounts — `check_permission`
    treats no-org as permitted).  Tenant-scoping for SQL queries falls
    back to user_id so personal accounts get a private namespace.
    """
    user_id = str(current_user.id)
    permission_org = getattr(current_user, "organization_id", None)
    has = await permission_repository.check_permission(
        user_id=user_id,
        organization_id=permission_org,
        resource_type="AGENT",
        resource_id="customer_service",
        permission_type="EXECUTE",
    )
    if not has:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to use the customer service agent",
        )
    return permission_org or user_id


# ── Pydantic schemas ────────────────────────────────────────────────────


VALID_PRIORITIES = {"High", "Medium", "Low"}
VALID_STATUSES = {"Open", "In Progress", "Resolved", "Closed", "Cancelled"}


class TicketCreateRequest(BaseModel):
    customer_email: EmailStr
    customer_name: Optional[str] = Field(None, max_length=200)
    subject: str = Field(..., min_length=1, max_length=500)
    body: str = Field(..., min_length=1, max_length=10_000)
    priority: Optional[str] = Field("Medium")
    category: Optional[str] = Field(None, max_length=64)
    channel: Optional[str] = Field("manual", max_length=32)
    meta: Optional[Dict[str, Any]] = None

    @field_validator("priority")
    @classmethod
    def _check_priority(cls, v: Optional[str]) -> Optional[str]:
        if v and v not in VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {sorted(VALID_PRIORITIES)}")
        return v


class TicketUpdateRequest(BaseModel):
    status: Optional[str] = None
    priority: Optional[str] = None
    category: Optional[str] = Field(None, max_length=64)
    assigned_user_id: Optional[str] = Field(None, max_length=64)

    @field_validator("status")
    @classmethod
    def _check_status(cls, v: Optional[str]) -> Optional[str]:
        if v and v not in VALID_STATUSES:
            raise ValueError(f"status must be one of {sorted(VALID_STATUSES)}")
        return v

    @field_validator("priority")
    @classmethod
    def _check_priority(cls, v: Optional[str]) -> Optional[str]:
        if v and v not in VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {sorted(VALID_PRIORITIES)}")
        return v


class TicketReplyRequest(BaseModel):
    body: str = Field(..., min_length=1, max_length=20_000)
    template_id: Optional[str] = Field(None, max_length=64)
    transition_status: Optional[str] = Field(None)

    @field_validator("transition_status")
    @classmethod
    def _check_status(cls, v: Optional[str]) -> Optional[str]:
        if v and v not in VALID_STATUSES:
            raise ValueError(f"transition_status must be one of {sorted(VALID_STATUSES)}")
        return v


# ── Endpoints ───────────────────────────────────────────────────────────


@router.get("")
async def list_tickets(
    status: Optional[str] = Query(None),
    priority: Optional[str] = Query(None),
    category: Optional[str] = Query(None, max_length=64),
    channel: Optional[str] = Query(None, max_length=32),
    assigned_to_me: bool = Query(False),
    search: Optional[str] = Query(None, max_length=200),
    time_range: Optional[str] = Query(None, pattern="^(1d|7d|30d|90d|1y)$"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
):
    """Tenant-scoped paginated list."""
    org_id = await _require_cs_permission(current_user)
    user_id = str(current_user.id)
    return await tickets_svc.list_tickets(
        org_id,
        status=status,
        priority=priority,
        category=category,
        channel=channel,
        assigned_user_id=user_id if assigned_to_me else None,
        search=search,
        time_range=time_range,
        limit=limit,
        offset=offset,
    )


@router.get("/{ticket_id}")
async def get_ticket(
    ticket_id: str,
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_cs_permission(current_user)
    ticket = await tickets_svc.get_ticket(org_id, ticket_id, include_replies=True)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")
    return ticket


@router.post("", status_code=status.HTTP_201_CREATED)
async def create_ticket(
    payload: TicketCreateRequest,
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_cs_permission(current_user)
    user_id = str(current_user.id)

    ticket = await tickets_svc.create_ticket(
        organization_id=org_id,
        customer_email=payload.customer_email,
        customer_name=payload.customer_name,
        subject=payload.subject,
        body=payload.body,
        priority=payload.priority,
        category=payload.category,
        channel=payload.channel or "manual",
        submitter_user_id=user_id,
        meta=payload.meta or {},
    )

    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="customer_service.ticket_created",
        details={"ticket_id": ticket["id"], "channel": ticket["channel"]},
        related_resource_type="TICKET",
        related_resource_id=ticket["id"],
        agent_name="Customer Service Agent",
    )
    return ticket


@router.patch("/{ticket_id}")
async def update_ticket(
    ticket_id: str,
    payload: TicketUpdateRequest,
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_cs_permission(current_user)
    user_id = str(current_user.id)

    fields = payload.model_dump(exclude_unset=True)
    updated = await tickets_svc.update_ticket(org_id, ticket_id, fields)
    if not updated:
        raise HTTPException(status_code=404, detail="Ticket not found")

    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="customer_service.ticket_updated",
        details={"ticket_id": ticket_id, "fields": list(fields.keys())},
        related_resource_type="TICKET",
        related_resource_id=ticket_id,
        agent_name="Customer Service Agent",
    )
    return updated


@router.delete("/{ticket_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_ticket(
    ticket_id: str,
    current_user: User = Depends(get_current_active_user),
):
    org_id = await _require_cs_permission(current_user)
    user_id = str(current_user.id)

    ok = await tickets_svc.soft_delete_ticket(org_id, ticket_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Ticket not found")

    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="customer_service.ticket_deleted",
        details={"ticket_id": ticket_id},
        related_resource_type="TICKET",
        related_resource_id=ticket_id,
        agent_name="Customer Service Agent",
    )


@router.post("/{ticket_id}/reply")
async def reply_to_ticket(
    ticket_id: str,
    payload: TicketReplyRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Operator reply.  Persists the message, transitions the ticket to
    `In Progress` (or whatever the operator chose), bumps template
    usage_count, and logs activity."""
    org_id = await _require_cs_permission(current_user)
    user_id = str(current_user.id)

    transition = payload.transition_status or "In Progress"
    reply = await tickets_svc.add_reply(
        organization_id=org_id,
        ticket_id=ticket_id,
        body=payload.body,
        author_type="operator",
        author_user_id=user_id,
        author_display_name=getattr(current_user, "full_name", None),
        template_id=payload.template_id,
        transition_status=transition,
    )
    if not reply:
        raise HTTPException(status_code=404, detail="Ticket not found")

    if payload.template_id:
        await templates_svc.increment_usage(org_id, payload.template_id)

    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="customer_service.ticket_replied",
        details={"ticket_id": ticket_id, "template_id": payload.template_id},
        related_resource_type="TICKET",
        related_resource_id=ticket_id,
        agent_name="Customer Service Agent",
    )

    # Fan out the two notifications: email the customer + in-app
    # delivery confirmation for the operator who just sent it.  We
    # need the full ticket dict (with customer_email) and the org's
    # branding (for slug / display_name / sla / support_email).
    try:
        from backend.services.customer_service import notify as cs_notify
        from backend.services.customer_service import branding as branding_svc
        full_ticket = await tickets_svc.get_ticket(org_id, ticket_id, include_replies=False)
        org_branding = await branding_svc.get_for_org(org_id)
        if full_ticket:
            await cs_notify.on_operator_reply_sent(
                ticket=full_ticket,
                reply=reply,
                branding=org_branding,
                operator_user_id=user_id,
            )
    except Exception as e:  # noqa: BLE001
        logger.warning("reply_notify_failed", error=str(e), ticket_id=ticket_id)

    return reply


@router.post("/{ticket_id}/ai-draft")
async def ai_draft_for_ticket(
    ticket_id: str,
    current_user: User = Depends(get_current_active_user),
):
    """Generate a RAG-grounded draft reply for an existing ticket.

    Returns the draft text + the citations and prior tickets used so
    the operator can verify before sending.
    """
    org_id = await _require_cs_permission(current_user)
    user_id = str(current_user.id)

    ticket = await tickets_svc.get_ticket(org_id, ticket_id, include_replies=False)
    if not ticket:
        raise HTTPException(status_code=404, detail="Ticket not found")

    draft_context = await draft_svc.build_draft_context(
        ticket=ticket,
        organization_id=org_id,
        user_id=user_id,
    )
    grounded_prompt = draft_svc.build_grounded_prompt_body(draft_context)

    # Reuse the existing CustomerServiceAgent dispatcher.  Honour the
    # tenant's configured default provider + model so AI Draft uses
    # whatever the rest of the platform is using (e.g. Gemini per .env).
    #
    # IMPORTANT: BaseAgent reads from `agent_model_config`, not
    # `model_config` — passing the wrong key silently falls through to
    # the agent's default Perplexity model.
    from backend.agents.customer_service_agent import CustomerServiceAgent
    from backend.core.config import settings

    provider = (settings.DEFAULT_LLM_PROVIDER or "gemini").lower()
    model_map = {
        "gemini": getattr(settings, "GEMINI_MODEL", None) or "gemini-2.5-flash",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
        "mistral": "mistral-small-latest",
        "perplexity": "sonar",
    }
    model_name = model_map.get(provider, "sonar")

    agent = CustomerServiceAgent({
        "provider": provider,
        "model": model_name,
        "agent_model_config": {
            "model": model_name,
            "temperature": 0.5,
            "max_tokens": 1500,
        },
    })
    try:
        result = await agent.process_async({
            "content": grounded_prompt,
            "request_type": "generate_response",
            "context": {"ticket_id": ticket_id, "organization_id": org_id},
        })
    except Exception as e:  # noqa: BLE001
        logger.error("ai_draft_failed", error=str(e), ticket_id=ticket_id)
        raise HTTPException(status_code=502, detail=f"AI draft failed: {e}")

    # Pull the draft text out of the agent's flexible response shape.
    response_blob = result.get("response") if isinstance(result, dict) else None
    draft_text = ""
    if isinstance(response_blob, dict):
        draft_text = (
            response_blob.get("response")
            or response_blob.get("draft")
            or response_blob.get("message")
            or response_blob.get("text")
            or ""
        )
    if not draft_text:
        draft_text = result.get("raw_response") or "" if isinstance(result, dict) else ""

    citations = [
        {
            "title": c.get("title"),
            "document_id": c.get("document_id"),
            "page_number": c.get("page_number"),
            "source": c.get("source"),
        }
        for c in (draft_context.get("rag_chunks") or [])
    ]

    return {
        "ticket_id": ticket_id,
        "draft": draft_text,
        "model_used": result.get("model_used") if isinstance(result, dict) else None,
        "citations": citations,
        "prior_tickets_used": [
            {"ticket_id": p.get("ticket_id"), "subject": p.get("subject")}
            for p in (draft_context.get("prior_tickets") or [])
        ],
        "matching_template_id": (draft_context.get("matching_template") or {}).get("id"),
    }
