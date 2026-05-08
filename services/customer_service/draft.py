"""RAG-grounded AI draft context builder.

Called from `/customer-service/tickets/{id}/ai-draft`.  Pulls four
sources of context, all org-scoped:
    1. Top-N RAG chunks via `context_service.get_context_for_query`.
    2. Top-N prior resolved tickets in the same category.
    3. Best-matching response template by category.
    4. Org branding tone hints (sla_response_minutes, support_email).

Output is a plain dict the existing CustomerServiceAgent prompt-builder
extends without breaking when fields are absent.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog

from . import templates as templates_svc
from . import tickets as tickets_svc

logger = structlog.get_logger(__name__)


async def _safe_get_rag_chunks(
    *,
    query: str,
    user_id: str,
    organization_id: str,
    k: int = 5,
) -> List[Dict[str, Any]]:
    """Best-effort RAG retrieval; returns [] if vector store is down or
    embeddings fail.  Never raises into the caller."""
    if not query or not user_id:
        return []
    try:
        from ...services.context_service import context_service  # type: ignore
    except Exception:
        return []
    try:
        result = await context_service.get_context_for_query(
            query=query,
            user_id=user_id,
            organization_id=organization_id,
            k=k,
            include_sources=["upload", "web", "manual_entry"],
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("draft_rag_search_failed", error=str(e))
        return []
    chunks = result.get("context") if isinstance(result, dict) else None
    if not chunks:
        return []
    # Normalize to a small, prompt-friendly shape.
    out: List[Dict[str, Any]] = []
    for c in chunks[:k]:
        if not isinstance(c, dict):
            continue
        meta = c.get("metadata") or {}
        out.append({
            "content": (c.get("content") or "")[:1500],
            "title": meta.get("title") or meta.get("filename"),
            "document_id": meta.get("document_id"),
            "page_number": meta.get("page_number"),
            "source": meta.get("source"),
            "score": c.get("score"),
        })
    return out


async def _safe_get_branding_hints(organization_id: str) -> Dict[str, Any]:
    """Pull tone hints from org_branding.  Defaults are fine if absent."""
    try:
        from ...db.postgres import get_async_sessionmaker
        from ...db.postgres_models import OrgBrandingSQL
        from sqlalchemy import select
    except Exception:
        return {}
    SessionLocal = get_async_sessionmaker()
    try:
        async with SessionLocal() as session:
            row = (await session.execute(
                select(OrgBrandingSQL).where(OrgBrandingSQL.organization_id == organization_id)
            )).scalar_one_or_none()
            if not row:
                return {}
            return {
                "display_name": row.display_name,
                "support_email": row.support_email,
                "sla_response_minutes": row.sla_response_minutes,
                "hero_copy": row.hero_copy,
            }
    except Exception as e:  # noqa: BLE001
        logger.warning("draft_branding_fetch_failed", error=str(e))
        return {}


async def build_draft_context(
    *,
    ticket: Dict[str, Any],
    organization_id: str,
    user_id: str,
    k_rag: int = 5,
    k_prior: int = 3,
) -> Dict[str, Any]:
    """Assemble the enriched context dict that's passed as
    `context` to `CustomerServiceAgent.process_async(...)` when
    `request_type='generate_response'`.

    The agent's prompt builder reads `context['rag_chunks']`,
    `context['prior_tickets']`, and `context['matching_template']`
    when present.
    """
    subject = (ticket.get("subject") or "").strip()
    body = (ticket.get("body") or "").strip()
    category = ticket.get("category")
    query_text = f"{subject}\n\n{body}".strip() or subject

    rag_chunks = await _safe_get_rag_chunks(
        query=query_text,
        user_id=user_id,
        organization_id=organization_id,
        k=k_rag,
    )

    prior_tickets = []
    try:
        prior_tickets = await tickets_svc.search_resolved_for_context(
            organization_id,
            category=category,
            subject_hint=subject,
            limit=k_prior,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("draft_prior_tickets_failed", error=str(e))

    matching_template = None
    try:
        matching_template = await templates_svc.find_best_match_for_category(
            organization_id, category
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("draft_template_match_failed", error=str(e))

    branding = await _safe_get_branding_hints(organization_id)

    return {
        "ticket_id": ticket.get("id"),
        "ticket_subject": subject,
        "ticket_body": body,
        "ticket_category": category,
        "ticket_priority": ticket.get("priority"),
        "customer_name": ticket.get("customer_name"),
        "customer_email": ticket.get("customer_email"),
        "rag_chunks": rag_chunks,
        "prior_tickets": prior_tickets,
        "matching_template": matching_template,
        "branding": branding,
    }


def build_grounded_prompt_body(context: Dict[str, Any]) -> str:
    """Compose the human-facing prompt body that the agent's `content`
    field receives.  This is what the LLM actually sees as the user
    message — it embeds the RAG citations and prior cases inline so the
    agent doesn't have to re-fetch them.

    The agent's existing prompt template prepends a system message about
    tone; we only need to provide the rich content here.
    """
    parts: List[str] = []
    parts.append(
        f"You are drafting a customer-support reply for ticket "
        f"{context.get('ticket_id', '?')} in {context.get('branding', {}).get('display_name', 'this organization')}."
    )
    if context.get("branding", {}).get("sla_response_minutes"):
        parts.append(
            f"Our SLA target is to respond within "
            f"{context['branding']['sla_response_minutes']} minutes."
        )

    parts.append("\n--- CUSTOMER MESSAGE ---")
    parts.append(f"From: {context.get('customer_name') or 'Customer'} "
                 f"<{context.get('customer_email') or ''}>")
    parts.append(f"Subject: {context.get('ticket_subject', '')}")
    parts.append(f"Priority: {context.get('ticket_priority', 'Medium')}")
    if context.get("ticket_category"):
        parts.append(f"Category: {context['ticket_category']}")
    parts.append("")
    parts.append(context.get("ticket_body", ""))

    rag = context.get("rag_chunks") or []
    if rag:
        parts.append("\n--- RELEVANT KNOWLEDGE BASE EXCERPTS ---")
        for i, chunk in enumerate(rag, 1):
            title = chunk.get("title") or "(untitled)"
            page = chunk.get("page_number")
            head = f"[{i}] {title}" + (f", page {page}" if page else "")
            parts.append(head)
            parts.append((chunk.get("content") or "").strip())
            parts.append("")

    prior = context.get("prior_tickets") or []
    if prior:
        parts.append("\n--- HOW SIMILAR PAST TICKETS WERE RESOLVED ---")
        for i, p in enumerate(prior, 1):
            parts.append(f"[Past ticket {i}] {p.get('subject', '')}")
            if p.get("resolution"):
                parts.append(f"Resolution: {p['resolution']}")
            parts.append("")

    template = context.get("matching_template")
    if template:
        parts.append("\n--- SUGGESTED TEMPLATE STRUCTURE ---")
        parts.append(f"Template: {template.get('name')}")
        parts.append(template.get("body", ""))

    parts.append("\n--- INSTRUCTIONS ---")
    parts.append(
        "Draft a clear, empathetic, on-brand reply addressed to the customer. "
        "Use information from the knowledge base excerpts and resolved-ticket "
        "patterns above to ground your answer. If you cite specifics, reference "
        "[1], [2], etc. inline so the operator can verify. Keep it concise — "
        "operators will edit before sending. Sign off as 'Support'."
    )
    return "\n".join(parts)
