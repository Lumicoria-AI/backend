"""
Phase E — SCIM 2.0 endpoints.

Mounted at `/api/v1/scim/v2`.

Bearer-token authenticated (the token is issued via
`POST /api/v1/enterprise/{org_id}/scim-tokens` and stored in
`scim_tokens.token_hash`).

Supports the standard SCIM 2.0 User and Group resources for Okta /
Azure AD / OneLogin auto-provisioning.

Filter parsing is intentionally narrow — we support the operators Okta &
Azure AD actually use against us in production (`eq`, `sw`, `co`).
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import structlog
from bson import ObjectId
from fastapi import APIRouter, Body, Header, HTTPException, Query, Request, status

from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.db.mongodb.repositories.org_subscription_repository import seat_assignment_repository
from backend.db.mongodb.repositories.sso_repository import scim_tokens_repository
from backend.db.mongodb.repositories.user_repository import get_user_repository
from backend.services.activity_logger import log_activity
from backend.services.event_bus import emit

logger = structlog.get_logger(__name__)
router = APIRouter()


async def _authenticate(authorization: Optional[str]) -> Dict[str, Any]:
    """Validate the bearer token, return the associated SCIM token row."""
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = authorization.split(" ", 1)[1].strip()
    row = await scim_tokens_repository.lookup_by_plaintext(token)
    if not row:
        raise HTTPException(status_code=401, detail="Invalid SCIM token")
    return row


# ── Discovery endpoints (unauthenticated by spec) ───────────────────


@router.get("/ServiceProviderConfig")
async def service_provider_config():
    return {
        "schemas": ["urn:ietf:params:scim:schemas:core:2.0:ServiceProviderConfig"],
        "documentationUri": "https://lumicoria.ai/docs/scim",
        "patch": {"supported": True},
        "bulk": {"supported": False},
        "filter": {"supported": True, "maxResults": 200},
        "changePassword": {"supported": False},
        "sort": {"supported": True},
        "etag": {"supported": False},
        "authenticationSchemes": [
            {
                "type": "oauthbearertoken",
                "name": "OAuth Bearer Token",
                "description": "Per-org SCIM token issued from the Enterprise admin panel.",
            }
        ],
    }


@router.get("/ResourceTypes")
async def resource_types():
    base = [
        {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:ResourceType"],
            "id": "User",
            "name": "User",
            "endpoint": "/Users",
            "schema": "urn:ietf:params:scim:schemas:core:2.0:User",
        },
        {
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:ResourceType"],
            "id": "Group",
            "name": "Group",
            "endpoint": "/Groups",
            "schema": "urn:ietf:params:scim:schemas:core:2.0:Group",
        },
    ]
    return {"totalResults": 2, "Resources": base}


@router.get("/Schemas")
async def schemas():
    return {
        "totalResults": 2,
        "Resources": [
            {"id": "urn:ietf:params:scim:schemas:core:2.0:User", "name": "User"},
            {"id": "urn:ietf:params:scim:schemas:core:2.0:Group", "name": "Group"},
        ],
    }


# ── Filter parsing ──────────────────────────────────────────────────


FILTER_RE = re.compile(
    r'^\s*([a-zA-Z][a-zA-Z0-9_.]*)\s+(eq|sw|co)\s+"(.*)"\s*$',
)


def _parse_filter(f: Optional[str]) -> Dict[str, Any]:
    if not f:
        return {}
    m = FILTER_RE.match(f)
    if not m:
        return {}
    attr, op, value = m.groups()
    if attr in ("userName", "userName.value"):
        attr = "email"
    if attr in ("emails", "emails.value"):
        attr = "email"
    if attr == "externalId":
        attr = "external_id"
    if op == "eq":
        return {attr: value}
    if op == "sw":
        return {attr: {"$regex": f"^{re.escape(value)}", "$options": "i"}}
    if op == "co":
        return {attr: {"$regex": re.escape(value), "$options": "i"}}
    return {}


# ── Users ───────────────────────────────────────────────────────────


def _user_to_scim(user: Dict[str, Any], org_id: str) -> Dict[str, Any]:
    d = dict(user)
    uid = str(d.get("_id") or d.get("id"))
    email = d.get("email") or ""
    name = d.get("full_name") or d.get("name") or ""
    parts = name.split(" ", 1)
    return {
        "schemas": ["urn:ietf:params:scim:schemas:core:2.0:User"],
        "id": uid,
        "externalId": d.get("external_id"),
        "userName": email,
        "name": {
            "givenName": parts[0] if parts else "",
            "familyName": parts[1] if len(parts) > 1 else "",
            "formatted": name,
        },
        "emails": [{"value": email, "primary": True}],
        "active": bool(d.get("is_active", True)),
        "meta": {
            "resourceType": "User",
            "created": (d.get("created_at") or datetime.utcnow()).isoformat() + "Z" if isinstance(d.get("created_at"), datetime) else d.get("created_at"),
            "lastModified": (d.get("updated_at") or datetime.utcnow()).isoformat() + "Z" if isinstance(d.get("updated_at"), datetime) else d.get("updated_at"),
            "location": f"/api/v1/scim/v2/Users/{uid}",
        },
    }


@router.get("/Users")
async def list_users(
    request: Request,
    filter: Optional[str] = Query(None),
    startIndex: int = Query(1, ge=1),
    count: int = Query(100, ge=1, le=200),
    authorization: Optional[str] = Header(None),
):
    scim = await _authenticate(authorization)
    org_id = scim["organization_id"]
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    extra = _parse_filter(filter)

    users_col = await MongoDB.get_collection("users")
    member_oids = [m for m in (org.member_ids or [])]
    query: Dict[str, Any] = {"_id": {"$in": member_oids}}
    query.update(extra)

    skip = startIndex - 1
    total = await users_col.count_documents(query)
    cursor = users_col.find(query).skip(skip).limit(count)
    rows = await cursor.to_list(length=count)
    return {
        "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
        "totalResults": total,
        "startIndex": startIndex,
        "itemsPerPage": len(rows),
        "Resources": [_user_to_scim(r, org_id) for r in rows],
    }


@router.post("/Users", status_code=201)
async def create_user(
    body: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
):
    scim = await _authenticate(authorization)
    org_id = scim["organization_id"]
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")

    email = body.get("userName")
    if not email:
        raise HTTPException(status_code=400, detail="userName is required")
    name_obj = body.get("name") or {}
    full_name = (
        name_obj.get("formatted")
        or " ".join(filter(None, [name_obj.get("givenName"), name_obj.get("familyName")])).strip()
        or email.split("@")[0]
    )

    users_col = await MongoDB.get_collection("users")
    existing = await users_col.find_one({"email": email})
    if existing:
        uid = existing["_id"]
        await organization_repository.add_member(str(org_id), str(uid))
    else:
        doc = {
            "email": email,
            "full_name": full_name,
            "is_active": bool(body.get("active", True)),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "external_id": body.get("externalId"),
            "metadata": {"provisioned_via": "scim"},
        }
        result = await users_col.insert_one(doc)
        uid = result.inserted_id
        existing = await users_col.find_one({"_id": uid})
        await organization_repository.add_member(str(org_id), str(uid))

    # Reserve a seat
    try:
        await seat_assignment_repository.assign(str(org_id), str(uid), assigned_by=scim["id"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("scim.seat_assign_failed", error=str(exc))

    await log_activity(
        user_id=str(uid), organization_id=str(org_id),
        activity_type="scim.user_provisioned",
        details={"email": email},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    await emit("scim.user_provisioned", organization_id=str(org_id),
               resource_type="user", resource_id=str(uid),
               payload={"email": email})
    return _user_to_scim(existing, str(org_id))


@router.get("/Users/{user_id}")
async def get_user(user_id: str, authorization: Optional[str] = Header(None)):
    scim = await _authenticate(authorization)
    users_col = await MongoDB.get_collection("users")
    row = await users_col.find_one({"_id": ObjectId(user_id)})
    if not row:
        raise HTTPException(status_code=404, detail="User not found")
    return _user_to_scim(row, scim["organization_id"])


@router.put("/Users/{user_id}")
async def replace_user(
    user_id: str,
    body: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
):
    scim = await _authenticate(authorization)
    users_col = await MongoDB.get_collection("users")
    name_obj = body.get("name") or {}
    full_name = name_obj.get("formatted") or " ".join(
        filter(None, [name_obj.get("givenName"), name_obj.get("familyName")])
    )
    patch: Dict[str, Any] = {
        "email": body.get("userName"),
        "full_name": full_name,
        "is_active": bool(body.get("active", True)),
        "external_id": body.get("externalId"),
        "updated_at": datetime.utcnow(),
    }
    patch = {k: v for k, v in patch.items() if v is not None}
    await users_col.update_one({"_id": ObjectId(user_id)}, {"$set": patch})
    if patch.get("is_active") is False:
        await seat_assignment_repository.remove(scim["organization_id"], user_id)
    row = await users_col.find_one({"_id": ObjectId(user_id)})
    return _user_to_scim(row, scim["organization_id"])


@router.patch("/Users/{user_id}")
async def patch_user(
    user_id: str,
    body: Dict[str, Any] = Body(...),
    authorization: Optional[str] = Header(None),
):
    scim = await _authenticate(authorization)
    users_col = await MongoDB.get_collection("users")
    set_patch: Dict[str, Any] = {"updated_at": datetime.utcnow()}
    for op in body.get("Operations", []) or []:
        path = (op.get("path") or "").lower()
        value = op.get("value")
        if path == "active" or (isinstance(value, dict) and "active" in value):
            active = value if path == "active" else value.get("active")
            set_patch["is_active"] = bool(active)
        if path == "username" or (isinstance(value, dict) and "userName" in value):
            email = value if path == "username" else value.get("userName")
            if email:
                set_patch["email"] = email
        if path.startswith("name.") and value:
            set_patch.setdefault("full_name", value)
    await users_col.update_one({"_id": ObjectId(user_id)}, {"$set": set_patch})
    if set_patch.get("is_active") is False:
        await seat_assignment_repository.remove(scim["organization_id"], user_id)
        await organization_repository.remove_member(scim["organization_id"], user_id)
    row = await users_col.find_one({"_id": ObjectId(user_id)})
    return _user_to_scim(row, scim["organization_id"])


@router.delete("/Users/{user_id}", status_code=204)
async def delete_user(user_id: str, authorization: Optional[str] = Header(None)):
    scim = await _authenticate(authorization)
    await organization_repository.remove_member(scim["organization_id"], user_id)
    await seat_assignment_repository.remove(scim["organization_id"], user_id)
    await log_activity(
        user_id=str(user_id), organization_id=scim["organization_id"],
        activity_type="scim.user_deprovisioned",
        details={}, related_resource_type="user", related_resource_id=str(user_id),
    )
    return None


# ── Groups (teams) ──────────────────────────────────────────────────


@router.get("/Groups")
async def list_groups(
    filter: Optional[str] = Query(None),
    startIndex: int = Query(1, ge=1),
    count: int = Query(100, ge=1, le=200),
    authorization: Optional[str] = Header(None),
):
    scim = await _authenticate(authorization)
    col = await MongoDB.get_collection("teams")
    query: Dict[str, Any] = {"organization_id": ObjectId(scim["organization_id"])}
    if filter:
        m = FILTER_RE.match(filter)
        if m and m.group(1) == "displayName":
            query["name"] = m.group(3)
    total = await col.count_documents(query)
    cursor = col.find(query).skip(startIndex - 1).limit(count)
    rows = await cursor.to_list(length=count)
    groups = []
    for r in rows:
        groups.append({
            "schemas": ["urn:ietf:params:scim:schemas:core:2.0:Group"],
            "id": str(r["_id"]),
            "displayName": r.get("name"),
            "members": [{"value": str(m), "type": "User"} for m in (r.get("member_ids") or [])],
            "meta": {"resourceType": "Group", "location": f"/api/v1/scim/v2/Groups/{r['_id']}"},
        })
    return {
        "schemas": ["urn:ietf:params:scim:api:messages:2.0:ListResponse"],
        "totalResults": total,
        "startIndex": startIndex,
        "itemsPerPage": len(groups),
        "Resources": groups,
    }
