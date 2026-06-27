"""Org branding service — single source of truth for portal/widget config.

The slug is the public URL segment.  We enforce a strict regex at the
write layer; database has a unique index across all orgs so we never
end up with two tenants competing for `/portal/agripro`.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from sqlalchemy import select, update

from ...db.postgres import get_async_sessionmaker
from ...db.postgres_models import OrgBrandingSQL
from .sanitize import clean_plain_text, clean_rich_text, sanitize_string_list

logger = structlog.get_logger(__name__)


_SLUG_RE = re.compile(r"^[a-z0-9](?:[a-z0-9-]{1,62}[a-z0-9])?$")
_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def is_valid_slug(slug: str) -> bool:
    return bool(slug and _SLUG_RE.match(slug))


def to_dict(row: OrgBrandingSQL) -> Dict[str, Any]:
    return {
        "organization_id": row.organization_id,
        "slug": row.slug,
        "display_name": row.display_name,
        "logo_url": row.logo_url,
        "primary_color": row.primary_color,
        "accent_color": row.accent_color,
        "hero_copy": row.hero_copy,
        "support_email": row.support_email,
        "sla_response_minutes": row.sla_response_minutes,
        "captcha_enabled": row.captcha_enabled,
        "public_categories": list(row.public_categories or []),
        # Meeting (Jitsi) branding — see OrgBrandingSQL for fallback rules.
        "meeting_app_name": getattr(row, "meeting_app_name", None),
        "meeting_logo_url": getattr(row, "meeting_logo_url", None),
        "meeting_favicon_url": getattr(row, "meeting_favicon_url", None),
        "meeting_watermark_link": getattr(row, "meeting_watermark_link", None),
        "meeting_welcome_message": getattr(row, "meeting_welcome_message", None),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _default_payload(organization_id: str) -> Dict[str, Any]:
    """Sensible empty defaults so the operator UI never shows a 404 on
    fresh orgs that haven't configured branding yet."""
    return {
        "organization_id": organization_id,
        "slug": "",
        "display_name": "",
        "logo_url": None,
        "primary_color": "#4f46e5",
        "accent_color": "#6366f1",
        "hero_copy": None,
        "support_email": None,
        "sla_response_minutes": 60,
        "captcha_enabled": False,
        "public_categories": [
            "technical_support",
            "billing_issue",
            "feature_request",
            "general_inquiry",
        ],
        "created_at": None,
        "updated_at": None,
    }


async def get_for_org(organization_id: str) -> Dict[str, Any]:
    """Fetch the branding row for an org; return defaults if missing."""
    if not organization_id:
        return _default_payload("")
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        row = (await session.execute(
            select(OrgBrandingSQL).where(OrgBrandingSQL.organization_id == organization_id)
        )).scalar_one_or_none()
        if not row:
            return _default_payload(organization_id)
        return to_dict(row)


async def get_by_slug(slug: str) -> Optional[Dict[str, Any]]:
    """Public-portal lookup.  Returns None if no org owns the slug."""
    if not is_valid_slug(slug):
        return None
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        row = (await session.execute(
            select(OrgBrandingSQL).where(OrgBrandingSQL.slug == slug)
        )).scalar_one_or_none()
        return to_dict(row) if row else None


def _validate_and_clean(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Pure function: validate + sanitize a branding payload.  Caller is
    responsible for rejecting on invalid slug (we return the cleaned slug
    only when it passes the regex)."""
    out: Dict[str, Any] = {}

    if "slug" in payload:
        slug = (payload["slug"] or "").strip().lower()
        if not is_valid_slug(slug):
            raise ValueError(f"invalid slug: {slug!r}")
        out["slug"] = slug
    if "display_name" in payload:
        out["display_name"] = clean_plain_text(payload["display_name"], max_len=200)
    if "logo_url" in payload:
        url = (payload["logo_url"] or "").strip()
        out["logo_url"] = (url[:1000] if url else None)
    if "primary_color" in payload:
        c = (payload["primary_color"] or "").strip()
        if not _HEX_COLOR_RE.match(c):
            raise ValueError(f"invalid primary_color: {c!r}")
        out["primary_color"] = c
    if "accent_color" in payload:
        c = (payload["accent_color"] or "").strip()
        if not _HEX_COLOR_RE.match(c):
            raise ValueError(f"invalid accent_color: {c!r}")
        out["accent_color"] = c
    if "hero_copy" in payload:
        out["hero_copy"] = clean_rich_text(payload["hero_copy"] or "", max_len=2000) or None
    if "support_email" in payload:
        em = (payload["support_email"] or "").strip().lower()
        out["support_email"] = (em[:320] if em else None)
    if "sla_response_minutes" in payload:
        try:
            v = int(payload["sla_response_minutes"])
        except (TypeError, ValueError):
            v = 60
        out["sla_response_minutes"] = max(1, min(v, 60 * 24 * 7))
    if "captcha_enabled" in payload:
        out["captcha_enabled"] = bool(payload["captcha_enabled"])
    if "public_categories" in payload:
        out["public_categories"] = sanitize_string_list(
            payload["public_categories"] or [], max_each=64
        )

    # ── Meeting (Jitsi) branding overrides ─────────────────────────────
    if "meeting_app_name" in payload:
        v = (payload["meeting_app_name"] or "").strip()
        out["meeting_app_name"] = (v[:120] if v else None)
    if "meeting_logo_url" in payload:
        v = (payload["meeting_logo_url"] or "").strip()
        out["meeting_logo_url"] = (v[:1000] if v else None)
    if "meeting_favicon_url" in payload:
        v = (payload["meeting_favicon_url"] or "").strip()
        out["meeting_favicon_url"] = (v[:1000] if v else None)
    if "meeting_watermark_link" in payload:
        v = (payload["meeting_watermark_link"] or "").strip()
        out["meeting_watermark_link"] = (v[:512] if v else None)
    if "meeting_welcome_message" in payload:
        out["meeting_welcome_message"] = (
            clean_rich_text(payload["meeting_welcome_message"] or "", max_len=2000) or None
        )

    return out


async def upsert(
    organization_id: str,
    payload: Dict[str, Any],
) -> Dict[str, Any]:
    """Create or update the branding row for an org.

    Slug is required on first create.  The (slug) unique index will
    raise IntegrityError if another org already claims it — we surface
    that as ValueError so the endpoint can return a 409.
    """
    cleaned = _validate_and_clean(payload)

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        row = (await session.execute(
            select(OrgBrandingSQL).where(OrgBrandingSQL.organization_id == organization_id)
        )).scalar_one_or_none()

        if row is None:
            if "slug" not in cleaned or "display_name" not in cleaned:
                raise ValueError("slug and display_name are required on first create")
            row = OrgBrandingSQL(
                organization_id=organization_id,
                slug=cleaned["slug"],
                display_name=cleaned["display_name"],
                logo_url=cleaned.get("logo_url"),
                primary_color=cleaned.get("primary_color", "#4f46e5"),
                accent_color=cleaned.get("accent_color", "#6366f1"),
                hero_copy=cleaned.get("hero_copy"),
                support_email=cleaned.get("support_email"),
                sla_response_minutes=cleaned.get("sla_response_minutes", 60),
                captcha_enabled=cleaned.get("captcha_enabled", False),
                public_categories=cleaned.get("public_categories", []),
            )
            session.add(row)
        else:
            for k, v in cleaned.items():
                setattr(row, k, v)
            row.updated_at = datetime.utcnow()

        try:
            await session.commit()
        except Exception as e:  # noqa: BLE001
            # Most likely the slug uniqueness violation; surface as ValueError.
            raise ValueError(f"branding upsert failed: {e}") from e
        await session.refresh(row)
        return to_dict(row)
