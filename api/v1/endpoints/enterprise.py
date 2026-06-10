"""
Phase E — Enterprise REST API.

Mounted at `/api/v1/enterprise`.

Covers: API tokens, outbound webhooks, SSO config (SAML metadata-driven —
ACS handshake is intentionally stubbed pending the python3-saml integration
that follows in the next pass), SCIM token issuance, domain claims +
verification + auto-join, session policy, IP allowlist, data residency,
compliance request capture, CMK config.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, EmailStr, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.api_tokens_repository import api_tokens_repository
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.db.mongodb.repositories.sso_repository import (
    domain_claims_repository,
    scim_tokens_repository,
    session_policy_repository,
    sso_repository,
)
from backend.db.mongodb.repositories.webhooks_repository import webhooks_repository
from backend.models.enterprise import API_TOKEN_SCOPES
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.event_bus import emit

logger = structlog.get_logger(__name__)
router = APIRouter()

DOMAIN_RE = re.compile(r"^[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")


def _oid(v: Any) -> Optional[ObjectId]:
    if v is None:
        return None
    if isinstance(v, ObjectId):
        return v
    try:
        return ObjectId(str(v))
    except Exception:
        return None


async def _require_org_admin(org_id: str, current_user: User):
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(a) for a in (org.admin_ids or [])]:
        raise HTTPException(status_code=403, detail="Org admin permission required")
    return org


async def _require_org_owner(org_id: str, current_user: User):
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(getattr(org, "owner_id", None)) != _oid(current_user.id):
        raise HTTPException(status_code=403, detail="Org owner permission required")
    return org


# ── API tokens ───────────────────────────────────────────────────────


@router.get("/{org_id}/api-tokens")
async def list_api_tokens(
    org_id: str,
    include_revoked: bool = Query(False),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    return await api_tokens_repository.list(organization_id=org_id, include_revoked=include_revoked)


@router.get("/api-tokens/scopes")
async def list_api_token_scopes():
    return {"scopes": API_TOKEN_SCOPES}


class ApiTokenCreate(BaseModel):
    name: str = Field(..., max_length=200)
    scopes: List[str] = Field(default_factory=list)
    user_id: Optional[str] = None
    expires_at: Optional[datetime] = None


@router.post("/{org_id}/api-tokens", status_code=201)
async def create_api_token(
    org_id: str,
    payload: ApiTokenCreate,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    invalid = [s for s in payload.scopes if s not in API_TOKEN_SCOPES]
    if invalid:
        raise HTTPException(status_code=400, detail=f"Invalid scopes: {invalid}")
    plaintext, row = await api_tokens_repository.create(
        organization_id=org_id,
        name=payload.name,
        scopes=payload.scopes,
        user_id=payload.user_id,
        expires_at=payload.expires_at,
        created_by=str(current_user.id),
    )
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="enterprise.api_token_created",
        details={"token_id": row["id"], "name": payload.name, "scopes": payload.scopes},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return {"plaintext_token": plaintext, "token": row, "warning": "This is the only time we will show this token. Store it securely."}


@router.post("/{org_id}/api-tokens/{token_id}/rotate")
async def rotate_api_token(
    org_id: str,
    token_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    result = await api_tokens_repository.rotate(token_id, organization_id=org_id)
    if not result:
        raise HTTPException(status_code=404, detail="Token not found")
    plaintext, row = result
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="enterprise.api_token_rotated",
        details={"token_id": token_id},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return {"plaintext_token": plaintext, "token": row}


@router.delete("/{org_id}/api-tokens/{token_id}", status_code=204)
async def revoke_api_token(
    org_id: str,
    token_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    await api_tokens_repository.revoke(token_id, organization_id=org_id)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="enterprise.api_token_revoked",
        details={"token_id": token_id},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return None


# ── Webhooks ─────────────────────────────────────────────────────────


class WebhookCreate(BaseModel):
    url: str = Field(..., max_length=2048)
    events: List[str] = Field(..., max_length=64)
    description: Optional[str] = Field(None, max_length=500)


class WebhookUpdate(BaseModel):
    url: Optional[str] = None
    events: Optional[List[str]] = None
    description: Optional[str] = None
    enabled: Optional[bool] = None


@router.get("/{org_id}/webhooks")
async def list_webhooks(
    org_id: str,
    enabled_only: bool = Query(False),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    return await webhooks_repository.list(organization_id=org_id, enabled_only=enabled_only)


@router.post("/{org_id}/webhooks", status_code=201)
async def create_webhook(
    org_id: str,
    payload: WebhookCreate,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    plaintext, row = await webhooks_repository.create(
        organization_id=org_id,
        url=payload.url, events=payload.events,
        description=payload.description, created_by=str(current_user.id),
    )
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="enterprise.webhook_created",
        details={"webhook_id": row["id"], "url": payload.url, "events": payload.events},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return {"signing_secret": plaintext, "webhook": row, "warning": "This signing secret is shown once."}


@router.patch("/{org_id}/webhooks/{webhook_id}")
async def update_webhook(
    org_id: str,
    webhook_id: str,
    payload: WebhookUpdate,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    row = await webhooks_repository.update(
        webhook_id, organization_id=org_id, patch=payload.model_dump(exclude_unset=True),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return row


@router.delete("/{org_id}/webhooks/{webhook_id}", status_code=204)
async def delete_webhook(
    org_id: str,
    webhook_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    await webhooks_repository.delete(webhook_id, organization_id=org_id)
    return None


@router.post("/{org_id}/webhooks/{webhook_id}/rotate-secret")
async def rotate_webhook_secret(
    org_id: str,
    webhook_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    result = await webhooks_repository.rotate_secret(webhook_id, organization_id=org_id)
    if not result:
        raise HTTPException(status_code=404, detail="Webhook not found")
    plaintext, row = result
    return {"signing_secret": plaintext, "webhook": row}


@router.post("/{org_id}/webhooks/{webhook_id}/test")
async def test_webhook(
    org_id: str,
    webhook_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    delivery = await webhooks_repository.record_delivery(
        webhook_id=webhook_id, organization_id=org_id,
        event="ping", payload={"test": True, "ts": datetime.utcnow().isoformat() + "Z"},
        status="pending",
    )
    return delivery


@router.get("/{org_id}/webhooks/{webhook_id}/deliveries")
async def list_webhook_deliveries(
    org_id: str,
    webhook_id: str,
    status_filter: Optional[str] = Query(None, alias="status"),
    limit: int = Query(100, ge=1, le=500),
    skip: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    return await webhooks_repository.list_deliveries(
        webhook_id=webhook_id, organization_id=org_id,
        status=status_filter, limit=limit, skip=skip,
    )


# ── SSO ──────────────────────────────────────────────────────────────


class SsoConfigPatch(BaseModel):
    provider: Optional[str] = None
    metadata_xml: Optional[str] = None
    entity_id: Optional[str] = None
    acs_url: Optional[str] = None
    sso_url: Optional[str] = None
    certificate: Optional[str] = None
    attribute_map: Optional[Dict[str, str]] = None
    default_role: Optional[str] = None
    enabled: Optional[bool] = None
    enforced_for_domains: Optional[List[str]] = None


@router.get("/{org_id}/sso")
async def get_sso(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    config = await sso_repository.get(org_id)
    return config or {"organization_id": org_id, "enabled": False}


@router.patch("/{org_id}/sso")
async def update_sso(
    org_id: str,
    payload: SsoConfigPatch,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    patch = payload.model_dump(exclude_unset=True)
    provider = patch.pop("provider", "saml")
    row = await sso_repository.upsert(org_id, provider=provider, patch=patch)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="enterprise.sso_updated",
        details={"fields": list(patch.keys())},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return row


@router.get("/sso/metadata.xml")
async def sso_sp_metadata(org_id: str = Query(...)):
    """Service-Provider SAML metadata for an org.  IdPs consume this XML
    to configure their side of the trust."""
    base = f"https://app.lumicoria.ai"
    entity_id = f"{base}/api/v1/enterprise/sso/{org_id}"
    acs_url = f"{base}/api/v1/enterprise/sso/saml/acs?org_id={org_id}"
    xml = (
        f'<EntityDescriptor xmlns="urn:oasis:names:tc:SAML:2.0:metadata" entityID="{entity_id}">'
        f'<SPSSODescriptor protocolSupportEnumeration="urn:oasis:names:tc:SAML:2.0:protocol" AuthnRequestsSigned="false" WantAssertionsSigned="true">'
        f'<NameIDFormat>urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress</NameIDFormat>'
        f'<AssertionConsumerService Binding="urn:oasis:names:tc:SAML:2.0:bindings:HTTP-POST" Location="{acs_url}" index="0"/>'
        f'</SPSSODescriptor>'
        f'</EntityDescriptor>'
    )
    return Response(content=xml, media_type="application/xml")


@router.post("/sso/saml/acs")
async def sso_acs(
    request: Request,
    org_id: str = Query(...),
):
    """SAML Assertion Consumer Service.

    Verifies the IdP's SAML response, extracts the NameID + attribute map,
    and either creates a new account auto-joined to the org (JIT-provision)
    or logs the existing user in.  Issues a Lumicoria JWT on success.
    """
    form = await request.form()
    saml_b64 = form.get("SAMLResponse")
    relay_state = form.get("RelayState")
    if not saml_b64:
        raise HTTPException(status_code=400, detail="SAMLResponse missing")

    # 1. Load SSO config for this org.
    cfg = await sso_repository.get(org_id)
    if not cfg or not cfg.get("enabled"):
        raise HTTPException(status_code=403, detail="SSO not enabled for this organization")
    cert_pem = cfg.get("certificate")
    if not cert_pem:
        raise HTTPException(status_code=400, detail="SSO certificate not configured")

    # 2. Verify + parse the assertion.
    try:
        from backend.services.saml_verifier import verify_saml_response, SamlVerificationError
        from backend.core.config import settings as _settings
        # Operators can override signature verification in development only.
        skip_sig = bool(getattr(_settings, "SAML_SKIP_SIGNATURE_VERIFICATION", False))
        assertion = verify_saml_response(
            saml_response_b64=str(saml_b64),
            idp_certificate_pem=cert_pem,
            expected_issuer=cfg.get("entity_id"),
            skip_signature=skip_sig,
        )
    except SamlVerificationError as exc:
        await log_activity(
            user_id="system", organization_id=org_id,
            activity_type="enterprise.sso_failed",
            details={"reason": str(exc)},
            related_resource_type="organization", related_resource_id=str(org_id),
        )
        raise HTTPException(status_code=401, detail=f"SAML verification failed: {exc}")

    # 3. Resolve / JIT-provision the user.
    from backend.db.mongodb.mongodb import MongoDB
    users_col = await MongoDB.get_collection("users")
    email = (assertion.email or assertion.name_id or "").lower()
    if not email:
        raise HTTPException(status_code=400, detail="Assertion missing email/NameID")

    existing = await users_col.find_one({"email": email})
    if existing:
        user_oid = existing["_id"]
    else:
        # JIT provision.
        doc = {
            "email": email,
            "full_name": assertion.full_name or email.split("@")[0],
            "is_active": True,
            "metadata": {"provisioned_via": "saml", "issuer": assertion.issuer},
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
        }
        result = await users_col.insert_one(doc)
        user_oid = result.inserted_id

    # 4. Seat enforcement + org join.
    try:
        from backend.services.billing.plan_caps import (
            PlanCapExceeded, assert_can_add_seat,
        )
        from backend.db.mongodb.repositories.org_subscription_repository import (
            seat_assignment_repository,
        )
        # Existing seat? Don't re-charge.
        existing_seat = await seat_assignment_repository.count_active(org_id)
        org = await organization_repository.get_by_id(org_id)
        if _oid(user_oid) not in [_oid(m) for m in (org.member_ids or [])]:
            try:
                await assert_can_add_seat(org_id)
            except PlanCapExceeded as exc:
                raise HTTPException(status_code=402, detail=exc.detail)
            await organization_repository.add_member(org_id, str(user_oid))
            await seat_assignment_repository.assign(org_id, str(user_oid))
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.warning("sso.seat_join_failed", error=str(exc), org_id=org_id)

    # 5. Issue Lumicoria JWT.
    from datetime import timedelta as _td
    from jose import jwt as _jwt
    from backend.core.config import settings as _settings
    payload = {
        "sub": str(user_oid),
        "user_id": str(user_oid),
        "email": email,
        "exp": datetime.utcnow() + _td(minutes=int(getattr(_settings, "ACCESS_TOKEN_EXPIRE_MINUTES", 60))),
        "iat": datetime.utcnow(),
        "provider": "saml",
        "organization_id": str(org_id),
    }
    secret = getattr(_settings, "SECRET_KEY", "")
    access_token = _jwt.encode(payload, secret, algorithm="HS256") if secret else ""

    await log_activity(
        user_id=str(user_oid), organization_id=org_id,
        activity_type="enterprise.sso_login",
        details={"email": email, "issuer": assertion.issuer},
        related_resource_type="organization", related_resource_id=str(org_id),
    )

    # 6. Redirect back to the app with the token and an optional relay state.
    success_url = (relay_state or getattr(_settings, "SAML_SUCCESS_URL", "")
                   or "http://localhost:8080/workspace") + (
        ("&" if "?" in (relay_state or "") else "?") + f"sso_token={access_token}" if access_token else ""
    )
    return RedirectResponse(success_url, status_code=302)


# ── SCIM tokens ──────────────────────────────────────────────────────


class ScimTokenCreate(BaseModel):
    name: Optional[str] = None


@router.get("/{org_id}/scim-tokens")
async def list_scim_tokens(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    return await scim_tokens_repository.list(org_id)


@router.post("/{org_id}/scim-tokens", status_code=201)
async def create_scim_token(
    org_id: str,
    payload: ScimTokenCreate,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    plaintext, row = await scim_tokens_repository.create(
        organization_id=org_id, name=payload.name, created_by=str(current_user.id),
    )
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="enterprise.scim_token_created",
        details={"token_id": row["id"], "name": payload.name},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return {"plaintext_token": plaintext, "token": row,
            "warning": "Configure your IdP with this Bearer token. It will not be shown again."}


@router.delete("/{org_id}/scim-tokens/{token_id}", status_code=204)
async def revoke_scim_token(
    org_id: str,
    token_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    await scim_tokens_repository.revoke(token_id, organization_id=org_id)
    return None


# ── Domain claims ───────────────────────────────────────────────────


class DomainCreate(BaseModel):
    domain: str
    auto_join_role: str = "member"
    enforced: bool = False


@router.get("/{org_id}/domains")
async def list_domains(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    return await domain_claims_repository.list_for_org(org_id)


@router.post("/{org_id}/domains", status_code=201)
async def create_domain(
    org_id: str,
    payload: DomainCreate,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    if not DOMAIN_RE.match(payload.domain):
        raise HTTPException(status_code=400, detail="Invalid domain")
    row = await domain_claims_repository.create(
        organization_id=org_id, domain=payload.domain,
        auto_join_role=payload.auto_join_role, enforced=payload.enforced,
    )
    return {**row, "instructions": (
        f"Add a TXT record at _lumicoria.{payload.domain} with the value "
        f"{row.get('verification_token')} and then POST to /domains/{payload.domain}/verify."
    )}


@router.post("/{org_id}/domains/{domain}/verify")
async def verify_domain(
    org_id: str,
    domain: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    # In production, perform a DNS TXT lookup here.  In dev we trust the
    # caller and flip the verified flag.
    row = await domain_claims_repository.verify(domain)
    if not row or str(row.get("organization_id")) != str(org_id):
        raise HTTPException(status_code=404, detail="Domain claim not found")
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="enterprise.domain_verified",
        details={"domain": domain},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return row


@router.delete("/{org_id}/domains/{domain}", status_code=204)
async def delete_domain(
    org_id: str,
    domain: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    await domain_claims_repository.delete(domain, organization_id=org_id)
    return None


# ── Session policy ──────────────────────────────────────────────────


class SessionPolicyPatch(BaseModel):
    idle_timeout_minutes: Optional[int] = Field(None, ge=5, le=1440)
    max_sessions_per_user: Optional[int] = Field(None, ge=1, le=100)
    require_mfa: Optional[bool] = None
    ip_allowlist_enabled: Optional[bool] = None
    ip_allowlist: Optional[List[str]] = None
    data_residency: Optional[str] = None
    cmk_enabled: Optional[bool] = None
    cmk_kms_key_id: Optional[str] = None


@router.get("/{org_id}/session-policy")
async def get_session_policy(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    return await session_policy_repository.get(org_id)


@router.patch("/{org_id}/session-policy")
async def update_session_policy(
    org_id: str,
    payload: SessionPolicyPatch,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    patch = payload.model_dump(exclude_unset=True)
    row = await session_policy_repository.upsert(
        org_id, patch=patch, updated_by=str(current_user.id),
    )
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="enterprise.session_policy_updated",
        details={"fields": list(patch.keys())},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return row


# ── Data residency, DLP, compliance, CMK ────────────────────────────


@router.get("/{org_id}/residency")
async def get_residency(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    policy = await session_policy_repository.get(org_id)
    return {"data_residency": policy.get("data_residency", "us")}


class ResidencyPatch(BaseModel):
    region: str = Field(..., description="us | eu | in")


@router.patch("/{org_id}/residency")
async def set_residency(
    org_id: str,
    payload: ResidencyPatch,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    if payload.region not in ("us", "eu", "in"):
        raise HTTPException(status_code=400, detail="Invalid region")
    await session_policy_repository.upsert(
        org_id, patch={"data_residency": payload.region}, updated_by=str(current_user.id),
    )
    return {"data_residency": payload.region}


class ComplianceRequest(BaseModel):
    document_type: str  # dpa | baa | soc2_report | iso27001_report
    contact_email: EmailStr
    notes: Optional[str] = None


@router.post("/{org_id}/compliance/request", status_code=201)
async def request_compliance_doc(
    org_id: str,
    payload: ComplianceRequest,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="enterprise.compliance_requested",
        details=payload.model_dump(),
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    await emit("enterprise.compliance_requested", organization_id=org_id,
               actor_id=str(current_user.id), payload=payload.model_dump())
    return {"ok": True, "message": "Our trust team will send the requested document within one business day."}


@router.get("/{org_id}/compliance/status")
async def compliance_status(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    return {
        "soc2_type_ii": "in_progress",
        "iso27001": "in_progress",
        "gdpr_dpa": "available_on_request",
        "hipaa_baa": "enterprise_contract_only",
        "pen_test": "annual_third_party",
    }


# ── Just-in-time access (lightweight) ───────────────────────────────


class JitGrantPayload(BaseModel):
    user_id: str
    scope: str
    duration_minutes: int = Field(60, ge=5, le=1440)
    reason: Optional[str] = None


@router.post("/{org_id}/jit/grant", status_code=201)
async def grant_jit_access(
    org_id: str,
    payload: JitGrantPayload,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    from datetime import timedelta
    from backend.db.mongodb.mongodb import MongoDB
    col = await MongoDB.get_collection("jit_grants")
    doc = {
        "organization_id": _oid(org_id),
        "user_id": _oid(payload.user_id),
        "scope": payload.scope,
        "expires_at": datetime.utcnow() + timedelta(minutes=payload.duration_minutes),
        "reason": payload.reason,
        "granted_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
    }
    result = await col.insert_one(doc)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="enterprise.jit_granted",
        details={"user_id": payload.user_id, "scope": payload.scope, "duration_minutes": payload.duration_minutes},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return {"grant_id": str(result.inserted_id), "expires_at": doc["expires_at"]}


@router.get("/{org_id}/jit/grants")
async def list_jit_grants(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    from backend.db.mongodb.mongodb import MongoDB
    col = await MongoDB.get_collection("jit_grants")
    cursor = col.find({"organization_id": _oid(org_id)}).sort("created_at", -1).limit(200)
    rows = await cursor.to_list(length=200)
    out: List[Dict[str, Any]] = []
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "user_id", "granted_by"):
            if r.get(k) is not None:
                r[k] = str(r[k])
        out.append(r)
    return out
