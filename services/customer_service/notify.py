"""Customer-service notification dispatcher.

Single entry point for the two ticket events that fan out to multiple
channels:

    on_ticket_created()
        ─→ email customer       (confirmation with ticket id + status url)
        ─→ in-app + push for operators

    on_operator_reply_sent()
        ─→ email customer       (reply body + status url)
        ─→ in-app for operator  (delivery confirmation)

Every channel is best-effort.  A failure on one (e.g. SendGrid down,
push subscription expired) never breaks the others or the user-facing
HTTP request — failures are logged via structlog and swallowed.

Operators are NEVER emailed about new tickets.  They get in-app +
push only — high-volume agents can receive 500+ tickets/day, email
would be unbearable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)


# ── Helpers ─────────────────────────────────────────────────────────────


def _truncate(text: str, n: int = 240) -> str:
    if not text:
        return ""
    text = text.replace("\r", "").strip()
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"


def _format_date(iso: Optional[str]) -> str:
    if not iso:
        return ""
    try:
        d = datetime.fromisoformat(iso.replace("Z", "+00:00"))
    except Exception:
        return iso
    return d.strftime("%B %-d, %Y at %-I:%M %p UTC")


def _portal_base_url() -> str:
    """Public-facing base URL for status / portal links in customer emails.

    Resolution order:
      1. settings.PUBLIC_BASE_URL  (the explicit production URL)
      2. settings.FRONTEND_URL     (the same FE URL OAuth redirects use)
      3. localhost:8080            (Docker nginx default for this project)
    """
    try:
        from backend.core.config import settings  # type: ignore
        url = (
            getattr(settings, "PUBLIC_BASE_URL", None)
            or getattr(settings, "FRONTEND_URL", None)
            or "http://localhost:8080"
        )
    except Exception:
        url = "http://localhost:8080"
    return url.rstrip("/")


def _status_url(slug: Optional[str], ticket_id: str, email: str) -> Optional[str]:
    """Compose the customer-facing status URL.  Returns None if we
    don't have a slug — better to omit the link than send a broken one."""
    if not slug:
        return None
    from urllib.parse import quote
    return f"{_portal_base_url()}/portal/{slug}/status/{ticket_id}?email={quote(email)}"


# ── Customer-side: email on ticket creation ─────────────────────────────


async def email_customer_ticket_received(
    *,
    ticket: Dict[str, Any],
    branding: Optional[Dict[str, Any]] = None,
) -> None:
    """Send the 'we got your message' email.  branding may be the public
    branding dict (slug, display_name, support_email, sla_response_minutes)."""
    customer_email = (ticket.get("customer_email") or "").strip()
    if not customer_email:
        logger.warning("ticket_email_skipped_no_address", ticket_id=ticket.get("id"))
        return
    try:
        from backend.services.email_service import get_email_service  # type: ignore
        email_service = await get_email_service()
    except Exception as e:  # noqa: BLE001
        logger.warning("email_service_unavailable", error=str(e))
        return

    org_name = (branding or {}).get("display_name") or "Support"
    slug = (branding or {}).get("slug")
    template_data = {
        "customer_name": ticket.get("customer_name") or "there",
        "ticket_id": ticket.get("id"),
        "ticket_subject": ticket.get("subject"),
        "ticket_body": _truncate(ticket.get("body") or "", 1000),
        "submitted_at": _format_date(ticket.get("created_at")),
        "organization_name": org_name,
        "support_email": (branding or {}).get("support_email"),
        "sla_minutes": (branding or {}).get("sla_response_minutes"),
        "status_url": _status_url(slug, ticket.get("id"), customer_email),
        "subject_line": f"We've received your request — {ticket.get('id')}",
    }

    try:
        # NOTE: omitting `metadata=` here — SendGrid's `add_custom_arg`
        # raises 'str object has no attribute get' on certain SDK
        # versions when passed a plain dict.  Tags survive (categories)
        # for filtering inside SendGrid's UI without that footgun.
        result = await email_service.send(
            to=customer_email,
            subject=f"[{ticket.get('id')}] We've received your request",
            template_name="ticket_received",
            template_data=template_data,
            from_name=org_name,
            reply_to=(branding or {}).get("support_email"),
            tags=["customer_service", "ticket_received"],
        )
        logger.info(
            "customer_email_ticket_received_sent",
            ticket_id=ticket.get("id"),
            success=getattr(result, "success", None),
            provider=getattr(result, "provider_used", None),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("customer_email_ticket_received_failed", error=str(e), ticket_id=ticket.get("id"))


# ── Customer-side: email on operator reply ──────────────────────────────


async def email_customer_reply(
    *,
    ticket: Dict[str, Any],
    reply: Dict[str, Any],
    branding: Optional[Dict[str, Any]] = None,
) -> None:
    customer_email = (ticket.get("customer_email") or "").strip()
    if not customer_email:
        return
    try:
        from backend.services.email_service import get_email_service  # type: ignore
        email_service = await get_email_service()
    except Exception as e:  # noqa: BLE001
        logger.warning("email_service_unavailable", error=str(e))
        return

    org_name = (branding or {}).get("display_name") or "Support"
    slug = (branding or {}).get("slug")
    operator_name = (
        reply.get("author_display_name")
        or org_name
        or "Support"
    )

    template_data = {
        "customer_name": ticket.get("customer_name") or "there",
        "ticket_id": ticket.get("id"),
        "ticket_subject": ticket.get("subject"),
        "reply_body": reply.get("body") or "",
        "operator_name": operator_name,
        "organization_name": org_name,
        "support_email": (branding or {}).get("support_email"),
        "status_url": _status_url(slug, ticket.get("id"), customer_email),
        "original_subject": ticket.get("subject"),
        "original_excerpt": _truncate(ticket.get("body") or "", 320),
        "original_date": _format_date(ticket.get("created_at")),
        "subject_line": f"Re: {ticket.get('subject') or 'Your support request'} — {ticket.get('id')}",
    }

    try:
        result = await email_service.send(
            to=customer_email,
            subject=f"Re: {ticket.get('subject') or 'Your support request'} [{ticket.get('id')}]",
            template_name="ticket_reply",
            template_data=template_data,
            from_name=org_name,
            reply_to=(branding or {}).get("support_email"),
            tags=["customer_service", "ticket_reply"],
            metadata={
                "ticket_id": ticket.get("id"),
                "reply_id": reply.get("id"),
                "organization_id": ticket.get("organization_id"),
            },
        )
        logger.info(
            "customer_email_reply_sent",
            ticket_id=ticket.get("id"),
            reply_id=reply.get("id"),
            success=getattr(result, "success", None),
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("customer_email_reply_failed", error=str(e), ticket_id=ticket.get("id"))


# ── Operator-side: in-app + push notifications ──────────────────────────


async def _resolve_operator_user_ids(organization_id: str) -> List[str]:
    """Best-effort lookup of operator user_ids for an org.

    For personal accounts we use `organization_id == user_id`, so the
    org id IS the operator's user id.  Real multi-user orgs would
    enumerate via a member repository; we fall back to using the org id
    itself which works for the personal-account path.
    """
    if not organization_id:
        return []
    return [organization_id]


async def notify_operators_new_ticket(
    *,
    ticket: Dict[str, Any],
    branding: Optional[Dict[str, Any]] = None,
) -> None:
    """In-app notification + web push for every operator in the org.
    Never fires email — operators get inbox volume that way."""
    try:
        # Use the singleton notification_service (a NotificationService instance).
        from backend.services import notification_service as _ns_module  # type: ignore
        notification_service = getattr(_ns_module, "notification_service", None)
        if notification_service is None:
            # Fallback: instantiate.  The class is named NotificationService.
            notification_service = _ns_module.NotificationService()
        # IMPORTANT: the live Notification model lives in
        # `db/mongodb/models/notification.py` and uses a different enum
        # value set than the legacy `models/mongodb_models.py` enums.
        from backend.db.mongodb.models.notification import (  # type: ignore
            NotificationType, NotificationPriority,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("operator_notify_unavailable", error=str(e))
        return

    org_id = ticket.get("organization_id")
    operator_ids = await _resolve_operator_user_ids(org_id)
    if not operator_ids:
        return

    org_name = (branding or {}).get("display_name") or "Support"
    title = f"New ticket — {ticket.get('priority', 'Medium')} priority"
    content = (
        f"{ticket.get('customer_name') or ticket.get('customer_email')} on {org_name}: "
        f"{_truncate(ticket.get('subject') or '', 120)}"
    )
    metadata = {
        "ticket_id": ticket.get("id"),
        "organization_id": org_id,
        "priority": ticket.get("priority"),
        "category": ticket.get("category"),
        "channel": ticket.get("channel"),
        "deeplink": f"/agents/customer-service?ticket={ticket.get('id')}",
    }
    # Map ticket priority to the live NotificationPriority enum
    # (LOW | NORMAL | HIGH | URGENT — note: no MEDIUM).
    priority_map = {
        "High": NotificationPriority.HIGH,
        "Medium": NotificationPriority.NORMAL,
        "Low": NotificationPriority.LOW,
    }
    chosen_priority = priority_map.get(
        ticket.get("priority", "Medium"), NotificationPriority.NORMAL,
    )

    for op_id in operator_ids:
        try:
            await notification_service.create_in_app_notification(
                user_id=op_id,
                title=title,
                content=content,
                # IN_APP is the correct value for the operator inbox bell.
                notification_type=NotificationType.IN_APP,
                priority=chosen_priority,
                metadata=metadata,
                send_realtime=True,
                send_push=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("operator_notify_new_ticket_failed", error=str(e), user_id=op_id)


async def notify_operator_reply_delivered(
    *,
    ticket: Dict[str, Any],
    reply: Dict[str, Any],
    operator_user_id: str,
) -> None:
    """In-app receipt for the operator who just sent a reply: confirms
    the email reached the customer (or that it was queued for delivery).
    No push — the operator just initiated the action, an in-app toast
    is enough."""
    if not operator_user_id:
        return
    try:
        from backend.services import notification_service as _ns_module  # type: ignore
        notification_service = getattr(_ns_module, "notification_service", None)
        if notification_service is None:
            notification_service = _ns_module.NotificationService()
        from backend.db.mongodb.models.notification import (  # type: ignore
            NotificationType, NotificationPriority,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("operator_notify_unavailable", error=str(e))
        return

    customer = ticket.get("customer_email") or "the customer"
    title = "Reply delivered"
    content = f"Your reply on {ticket.get('id')} was delivered to {customer}."
    try:
        await notification_service.create_in_app_notification(
            user_id=operator_user_id,
            title=title,
            content=content,
            notification_type=NotificationType.IN_APP,
            priority=NotificationPriority.LOW,
            metadata={
                "ticket_id": ticket.get("id"),
                "reply_id": reply.get("id"),
                "customer_email": ticket.get("customer_email"),
                "organization_id": ticket.get("organization_id"),
                "deeplink": f"/agents/customer-service?ticket={ticket.get('id')}",
            },
            send_realtime=True,
            send_push=False,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("operator_notify_reply_delivered_failed", error=str(e), user_id=operator_user_id)


# ── Composite handlers — called by endpoints ────────────────────────────


async def on_ticket_created(
    *,
    ticket: Dict[str, Any],
    branding: Optional[Dict[str, Any]] = None,
) -> None:
    """Fan out the two notifications for a new ticket.  Each side is
    independently best-effort — a failed customer email does not skip
    the operator notification and vice versa."""
    await email_customer_ticket_received(ticket=ticket, branding=branding)
    await notify_operators_new_ticket(ticket=ticket, branding=branding)


async def on_operator_reply_sent(
    *,
    ticket: Dict[str, Any],
    reply: Dict[str, Any],
    branding: Optional[Dict[str, Any]] = None,
    operator_user_id: Optional[str] = None,
) -> None:
    """Fan out the two notifications for an operator reply."""
    await email_customer_reply(ticket=ticket, reply=reply, branding=branding)
    if operator_user_id:
        await notify_operator_reply_delivered(
            ticket=ticket, reply=reply, operator_user_id=operator_user_id,
        )
