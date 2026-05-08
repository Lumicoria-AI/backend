"""Public portal endpoints — anonymous, rate-limited.

These are the only customer-service routes that DO NOT require auth.
They power the hosted support portal at `/portal/{slug}` and (later)
the embeddable widget JS.

Threat model:
    - High-volume abuse:        rate_limit decorator (Redis-backed).
    - Slug enumeration:         404 returned for unknown slugs.
    - Captcha bypass:           hCaptcha verification when org enables it.
    - Status-page enumeration:  privacy-gated by submitter email match;
                                 returns 404 (not 401) on mismatch.
    - XSS in stored content:    bleach sanitization at the service layer.
    - SQL injection:            parameterized queries via SQLAlchemy.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import httpx
import structlog
from fastapi import APIRouter, HTTPException, Path, Query, Request, status
from pydantic import BaseModel, EmailStr, Field, field_validator

from backend.core.security import rate_limit
from backend.services.customer_service import (
    branding as branding_svc,
    tickets as tickets_svc,
)

router = APIRouter()
logger = structlog.get_logger(__name__)


_HCAPTCHA_VERIFY_URL = "https://hcaptcha.com/siteverify"
_VALID_PRIORITIES = {"High", "Medium", "Low"}


# ── Pydantic request models ─────────────────────────────────────────────


class PublicTicketCreateRequest(BaseModel):
    """Schema for the anonymous ticket-submission form on the portal."""
    customer_email: EmailStr
    customer_name: Optional[str] = Field(None, max_length=200)
    subject: str = Field(..., min_length=1, max_length=500)
    body: str = Field(..., min_length=1, max_length=10_000)
    priority: Optional[str] = Field("Medium")
    category: Optional[str] = Field(None, max_length=64)
    captcha_token: Optional[str] = Field(None, max_length=2048)

    @field_validator("priority")
    @classmethod
    def _check_priority(cls, v: Optional[str]) -> Optional[str]:
        if v and v not in _VALID_PRIORITIES:
            raise ValueError(f"priority must be one of {sorted(_VALID_PRIORITIES)}")
        return v


# ── Helpers ─────────────────────────────────────────────────────────────


async def _resolve_slug_or_404(slug: str) -> Dict[str, Any]:
    """Lookup the org_branding row by slug; raise 404 if missing.

    Side effect: returns the branding dict so callers can use the org id
    + captcha flag without a second query.
    """
    if not branding_svc.is_valid_slug(slug):
        raise HTTPException(status_code=404, detail="Portal not found")
    branding = await branding_svc.get_by_slug(slug)
    if not branding:
        raise HTTPException(status_code=404, detail="Portal not found")
    return branding


async def _verify_captcha_if_enabled(
    branding: Dict[str, Any], token: Optional[str], remote_ip: Optional[str]
) -> None:
    """Best-effort hCaptcha verification.  When `captcha_enabled=True`
    on the org and a `HCAPTCHA_SECRET` env var is configured, we POST
    to hCaptcha; otherwise the gate is permissive (logged warning)."""
    if not branding.get("captcha_enabled"):
        return
    secret = os.getenv("HCAPTCHA_SECRET", "").strip()
    if not secret:
        logger.warning(
            "captcha_enabled_without_secret",
            org_slug=branding.get("slug"),
            hint="Set HCAPTCHA_SECRET env var to enforce.",
        )
        return
    if not token:
        raise HTTPException(status_code=400, detail="Captcha token required")
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                _HCAPTCHA_VERIFY_URL,
                data={"secret": secret, "response": token, "remoteip": remote_ip or ""},
            )
            if resp.status_code != 200:
                raise HTTPException(status_code=400, detail="Captcha verification failed")
            j = resp.json()
            if not j.get("success"):
                raise HTTPException(status_code=400, detail="Captcha rejected")
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        logger.error("captcha_verify_error", error=str(e))
        raise HTTPException(status_code=502, detail="Captcha service error")


def _sanitize_branding_for_public(branding: Dict[str, Any]) -> Dict[str, Any]:
    """Strip operator-only fields before sending the branding dict to
    anonymous portal visitors."""
    return {
        "slug": branding.get("slug"),
        "display_name": branding.get("display_name"),
        "logo_url": branding.get("logo_url"),
        "primary_color": branding.get("primary_color"),
        "accent_color": branding.get("accent_color"),
        "hero_copy": branding.get("hero_copy"),
        "support_email": branding.get("support_email"),
        "sla_response_minutes": branding.get("sla_response_minutes"),
        "captcha_enabled": branding.get("captcha_enabled"),
        "public_categories": branding.get("public_categories", []),
    }


# ── Endpoints ───────────────────────────────────────────────────────────


@router.get("/portal/{slug}/branding")
@rate_limit(limit=60, window=60)
async def get_public_branding(
    request: Request,
    slug: str = Path(..., min_length=2, max_length=64),
):
    """Public read of an organization's portal branding."""
    branding = await _resolve_slug_or_404(slug)
    return _sanitize_branding_for_public(branding)


@router.post("/portal/{slug}/tickets", status_code=status.HTTP_201_CREATED)
@rate_limit(limit=5, window=900)
async def create_public_ticket(
    request: Request,
    payload: PublicTicketCreateRequest,
    slug: str = Path(..., min_length=2, max_length=64),
):
    """Anonymous ticket submission via the hosted portal.

    Heavy rate limit (5 per 15 minutes per IP) plus optional captcha.
    """
    branding = await _resolve_slug_or_404(slug)
    org_id = branding["organization_id"]

    remote_ip = request.client.host if request.client else None
    await _verify_captcha_if_enabled(branding, payload.captcha_token, remote_ip)

    ticket = await tickets_svc.create_ticket(
        organization_id=org_id,
        customer_email=str(payload.customer_email),
        customer_name=payload.customer_name,
        subject=payload.subject,
        body=payload.body,
        priority=payload.priority or "Medium",
        category=payload.category,
        channel="portal",
        submitter_user_id=None,
        meta={
            "remote_ip": remote_ip,
            "user_agent": request.headers.get("user-agent", "")[:500],
            "referer": request.headers.get("referer", "")[:500],
            "portal_slug": slug,
        },
    )

    # Activity log fire-and-forget; don't fail the user's submission if
    # MongoDB hiccups.  Attribute to the org owner (ObjectId) and put the
    # ticket id inside `details` rather than `related_resource_id` —
    # activity_repository validates related_resource_id as an ObjectId
    # too, and our ticket ids are short slugs (TK-xxxxxxxx).
    try:
        from backend.services.activity_logger import log_activity
        await log_activity(
            user_id=str(org_id),
            organization_id=org_id,
            activity_type="customer_service.ticket_submitted_public",
            details={
                "ticket_id": ticket["id"],
                "slug": slug,
                "submitted_by": "anonymous_portal",
                "customer_email": ticket.get("customer_email"),
            },
            agent_name="Customer Service Agent",
        )
    except Exception:  # noqa: BLE001
        pass

    # Fan out: confirmation email to the customer + in-app/push to the
    # org's operators.  Best-effort — never fail the public submission
    # because of a downstream notification hiccup.
    try:
        from backend.services.customer_service import notify as cs_notify
        await cs_notify.on_ticket_created(
            ticket=ticket,
            branding=branding,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("ticket_create_notify_failed", error=str(e), ticket_id=ticket.get("id"))

    return {
        "ticket_id": ticket["id"],
        "status": ticket["status"],
        "status_url": f"/portal/{slug}/status/{ticket['id']}",
        "created_at": ticket["created_at"],
    }


@router.get("/portal/{slug}/tickets/{ticket_id}/status")
@rate_limit(limit=30, window=900)
async def get_public_ticket_status(
    request: Request,
    slug: str = Path(..., min_length=2, max_length=64),
    ticket_id: str = Path(..., min_length=4, max_length=64),
    email: str = Query(..., min_length=3, max_length=320),
):
    """Privacy-gated status check.

    Requires the submitter's email as a query param; if it doesn't match,
    we return 404 (not 401) so this endpoint can't be used to enumerate
    valid ticket IDs.
    """
    branding = await _resolve_slug_or_404(slug)
    org_id = branding["organization_id"]

    ticket = await tickets_svc.get_ticket_for_public_status(
        org_slug_to_id=org_id,
        ticket_id=ticket_id,
        customer_email=email,
    )
    if not ticket:
        raise HTTPException(status_code=404, detail="Not found")

    # Public response: redact internal-only fields before returning.
    safe = {
        "id": ticket["id"],
        "subject": ticket["subject"],
        "status": ticket["status"],
        "priority": ticket["priority"],
        "category": ticket["category"],
        "created_at": ticket["created_at"],
        "updated_at": ticket["updated_at"],
        "resolved_at": ticket["resolved_at"],
        "replies": [
            {
                "author_type": r["author_type"],
                "author_display_name": r["author_display_name"],
                "body": r["body"],
                "created_at": r["created_at"],
                "is_ai": r["author_type"] == "agent_ai",
            }
            for r in (ticket.get("replies") or [])
            if r["author_type"] in ("operator", "agent_ai")
        ],
    }
    return safe


class PublicCustomerReplyRequest(BaseModel):
    customer_email: EmailStr
    body: str = Field(..., min_length=1, max_length=10_000)


@router.get("/portal/{slug}/help")
@rate_limit(limit=120, window=60)
async def list_public_help_articles(
    request: Request,
    slug: str = Path(..., min_length=2, max_length=64),
    category: Optional[str] = None,
    limit: int = 50,
):
    """Public list of published help-center articles for an org's portal."""
    branding = await _resolve_slug_or_404(slug)
    from backend.services.customer_service import articles as articles_svc
    items = await articles_svc.list_articles_public(
        branding["organization_id"], category=category, limit=limit,
    )
    return {
        "branding": _sanitize_branding_for_public(branding),
        "articles": items,
    }


@router.get("/portal/{slug}/help/{article_slug}")
@rate_limit(limit=120, window=60)
async def get_public_help_article(
    request: Request,
    slug: str = Path(..., min_length=2, max_length=64),
    article_slug: str = Path(..., min_length=1, max_length=120),
):
    """Public view of a single help-center article."""
    branding = await _resolve_slug_or_404(slug)
    from backend.services.customer_service import articles as articles_svc
    article = await articles_svc.get_article_public(
        branding["organization_id"], article_slug,
    )
    if not article:
        raise HTTPException(status_code=404, detail="Article not found")
    return {
        "branding": _sanitize_branding_for_public(branding),
        "article": article,
    }


@router.post("/portal/{slug}/help/{article_slug}/vote", status_code=status.HTTP_204_NO_CONTENT)
@rate_limit(limit=20, window=900)
async def vote_public_help_article(
    request: Request,
    slug: str = Path(..., min_length=2, max_length=64),
    article_slug: str = Path(..., min_length=1, max_length=120),
    helpful: bool = True,
):
    """Thumbs-up / thumbs-down on a help-center article."""
    branding = await _resolve_slug_or_404(slug)
    from backend.services.customer_service import articles as articles_svc
    ok = await articles_svc.vote_article_public(
        branding["organization_id"], article_slug, helpful=helpful,
    )
    if not ok:
        raise HTTPException(status_code=404, detail="Article not found")


@router.post("/portal/{slug}/tickets/{ticket_id}/replies", status_code=status.HTTP_201_CREATED)
@rate_limit(limit=10, window=900)
async def post_public_customer_reply(
    request: Request,
    payload: PublicCustomerReplyRequest,
    slug: str = Path(..., min_length=2, max_length=64),
    ticket_id: str = Path(..., min_length=4, max_length=64),
):
    """Allow the original submitter to add a follow-up reply via the
    public status page.  Email must match the original submitter."""
    branding = await _resolve_slug_or_404(slug)
    org_id = branding["organization_id"]

    # Re-use the privacy-gated lookup as the auth check.
    confirm = await tickets_svc.get_ticket_for_public_status(
        org_slug_to_id=org_id,
        ticket_id=ticket_id,
        customer_email=str(payload.customer_email),
    )
    if not confirm:
        raise HTTPException(status_code=404, detail="Not found")

    reply = await tickets_svc.add_reply(
        organization_id=org_id,
        ticket_id=ticket_id,
        body=payload.body,
        author_type="customer",
        author_user_id=None,
        author_display_name=confirm.get("customer_name") or confirm.get("customer_email"),
        # Don't auto-transition status; operator decides next action.
        transition_status=None,
    )
    if not reply:
        raise HTTPException(status_code=404, detail="Not found")
    return {"id": reply["id"], "created_at": reply["created_at"]}
