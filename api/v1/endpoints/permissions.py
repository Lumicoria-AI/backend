"""
Lumicoria AI — User capabilities probe.

GET /api/v1/permissions/me?organization_id=<org_id>

Returns the signed-in user's effective capability surface inside the
given organization (or their primary org if omitted).  Powers the
frontend PermissionsContext, which drives sidebar / button gating.

Keep this endpoint side-effect-free and fast — the frontend may poll it
on every nav change.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.models.user import User

router = APIRouter()


def _oid(value: Any) -> Optional[ObjectId]:
    try:
        return ObjectId(str(value))
    except Exception:
        return None


def _primary_org_id(user: User) -> Optional[str]:
    """Pick the first organization the user belongs to, with sensible
    fallbacks across the few different field names different codepaths
    use to attach the org."""
    for key in ("organization_id", "primary_organization_id", "default_organization_id"):
        v = getattr(user, key, None)
        if v:
            return str(v)
    orgs = getattr(user, "organization_ids", None) or []
    if orgs:
        return str(orgs[0])
    return None


@router.get("/me")
async def get_my_permissions(
    organization_id: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    org_id = organization_id or _primary_org_id(current_user)
    if not org_id:
        # Unauthenticated / no-org users see the baseline capabilities
        # so the frontend can still render a sane "no workspace" state
        # without erroring.
        return {
            "user_id": str(current_user.id),
            "organization_id": None,
            "plan": "free",
            "is_org_owner": False,
            "is_org_admin": False,
            "role": "guest",
            "can": _capability_matrix(plan="free", is_admin=False, is_owner=False),
        }

    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    user_oid = _oid(current_user.id)
    admin_ids = [_oid(a) for a in (org.admin_ids or [])]
    is_admin = user_oid in admin_ids
    is_owner = user_oid == _oid(getattr(org, "owner_id", None))
    plan = (getattr(org, "plan", None) or "free").lower()
    role = "owner" if is_owner else "admin" if is_admin else "member"

    return {
        "user_id": str(current_user.id),
        "organization_id": org_id,
        "plan": plan,
        "is_org_owner": bool(is_owner),
        "is_org_admin": bool(is_admin or is_owner),
        "role": role,
        "can": _capability_matrix(plan=plan, is_admin=bool(is_admin or is_owner), is_owner=bool(is_owner)),
    }


def _capability_matrix(*, plan: str, is_admin: bool, is_owner: bool) -> Dict[str, bool]:
    """Capability flags consumed by the frontend PermissionsContext.

    Plans are gated: SSO/SCIM = Business+/Enterprise, audit export =
    Team+, custom domains = Business+/Enterprise.  Admin/Owner overrides
    are mixed in here so the frontend only needs to read a single bool
    per surface.
    """
    paid = plan in {"team", "business", "enterprise"}
    business_plus = plan in {"business", "enterprise"}
    return {
        "view_workspace": True,
        "create_project": True,
        "create_team": is_admin,
        "invite_members": is_admin,
        "manage_members": is_admin,
        "manage_billing": is_owner,
        "manage_branding": is_admin,
        "manage_settings": is_admin,
        "manage_automations": is_admin,
        "manage_webhooks": is_admin,
        "manage_api_tokens": is_admin,
        "manage_integrations": is_admin,
        "manage_sso": is_admin and business_plus,
        "manage_scim": is_admin and business_plus,
        "manage_custom_domains": is_admin and business_plus,
        "view_audit": is_admin,
        "export_audit": is_admin and paid,
        "manage_seats": is_admin and paid,
        "manage_enterprise_features": is_admin and business_plus,
    }
