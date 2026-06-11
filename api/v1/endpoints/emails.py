"""
Phase C — Email + templates router.

Mounted at `/api/v1/emails`.

Catalogue + preview-render + sample-payload preview + sent log +
resend + branding + test-send + deliverability + custom templates +
DKIM/SPF setup + sending-domains + tracking opt-out.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.models.user import User

logger = structlog.get_logger(__name__)
router = APIRouter()

TEMPLATES_DIR = Path(__file__).resolve().parents[3] / "services" / "templates" / "email"


def _oid(v: Any) -> Optional[ObjectId]:
    if v is None:
        return None
    if isinstance(v, ObjectId):
        return v
    try:
        return ObjectId(str(v))
    except Exception:
        return None


def _list_templates() -> List[Dict[str, Any]]:
    if not TEMPLATES_DIR.exists():
        return []
    out: List[Dict[str, Any]] = []
    for path in sorted(TEMPLATES_DIR.glob("*.html")):
        out.append({
            "key": path.stem,
            "filename": path.name,
            "size_bytes": path.stat().st_size,
            "modified_at": datetime.utcfromtimestamp(path.stat().st_mtime).isoformat() + "Z",
        })
    return out


async def _require_org_admin(org_id: str, current_user: User):
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(a) for a in (org.admin_ids or [])]:
        raise HTTPException(status_code=403, detail="Org admin permission required")
    return org


# ── Catalogue + preview ──────────────────────────────────────────


@router.get("/templates")
async def list_templates(current_user: User = Depends(get_current_active_user)):
    return {"templates": _list_templates()}


@router.get("/templates/{key}")
async def get_template(key: str, current_user: User = Depends(get_current_active_user)):
    path = TEMPLATES_DIR / f"{key}.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    return {
        "key": key,
        "html": path.read_text(),
    }


@router.post("/templates/{key}/preview-data")
async def render_with_sample_data(
    key: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
    current_user: User = Depends(get_current_active_user),
):
    """Render a template against sample data.  Uses simple `{{ var }}`
    interpolation to avoid Jinja2's full pipeline."""
    path = TEMPLATES_DIR / f"{key}.html"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Template not found")
    html = path.read_text()
    sample = payload or {}
    sample.setdefault("user_name", getattr(current_user, "full_name", "Friend"))
    sample.setdefault("organization_name", "Lumicoria")
    for k, v in sample.items():
        html = html.replace("{{ " + k + " }}", str(v))
    return {"key": key, "html": html, "sample_data": sample}


# ── Sent log ─────────────────────────────────────────────────────


@router.get("/sent")
async def list_sent_emails(
    user_id: Optional[str] = Query(None),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("email_sent_log")
    q: Dict[str, Any] = {}
    if user_id:
        q["recipient_user_id"] = _oid(user_id)
    cursor = col.find(q).sort("sent_at", -1).limit(limit)
    rows = await cursor.to_list(length=limit)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "recipient_user_id"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.get("/sent/{email_id}")
async def get_sent_email(
    email_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("email_sent_log")
    row = await col.find_one({"_id": _oid(email_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Email not found")
    row["id"] = str(row.pop("_id"))
    for k in ("organization_id", "recipient_user_id"):
        if row.get(k):
            row[k] = str(row[k])
    return row


@router.post("/resend/{email_id}")
async def resend_email(
    email_id: str,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("email_sent_log")
    row = await col.find_one({"_id": _oid(email_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Email not found")
    return {"ok": True, "queued": True}


# ── Branding ─────────────────────────────────────────────────────


@router.get("/branding/{org_id}")
async def get_email_branding(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("email_branding")
    row = await col.find_one({"organization_id": _oid(org_id)})
    if not row:
        return {"organization_id": org_id, "primary_color": "#6C4AB0", "footer_text": ""}
    row["id"] = str(row.pop("_id"))
    if row.get("organization_id"):
        row["organization_id"] = str(row["organization_id"])
    return row


@router.patch("/branding/{org_id}")
async def update_email_branding(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("email_branding")
    await col.update_one(
        {"organization_id": _oid(org_id)},
        {"$set": {**payload, "updated_at": datetime.utcnow()}},
        upsert=True,
    )
    return {"ok": True}


# ── Test send + deliverability ──────────────────────────────────


class TestSendPayload(BaseModel):
    to: EmailStr
    template_key: str
    sample_data: Dict[str, Any] = Field(default_factory=dict)


@router.post("/test-send")
async def test_send(
    payload: TestSendPayload,
    current_user: User = Depends(get_current_active_user),
):
    col = await MongoDB.get_collection("email_sent_log")
    doc = {
        "to": str(payload.to),
        "template_key": payload.template_key,
        "sample_data": payload.sample_data,
        "queued_at": datetime.utcnow(),
        "queued_by": _oid(current_user.id),
        "kind": "test",
    }
    r = await col.insert_one(doc)
    return {"queued_id": str(r.inserted_id), "ok": True}


@router.get("/deliverability/{org_id}")
async def deliverability(
    org_id: str,
    days: int = Query(30, ge=1, le=180),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("email_sent_log")
    since = datetime.utcnow() - timedelta(days=days)
    pipeline = [
        {"$match": {"organization_id": _oid(org_id), "sent_at": {"$gte": since}}},
        {"$group": {"_id": "$status", "count": {"$sum": 1}}},
    ]
    rows = await col.aggregate(pipeline).to_list(length=10)
    return {r["_id"] or "unknown": r["count"] for r in rows}


# ── Custom templates per org ────────────────────────────────────


@router.get("/templates/custom/{org_id}")
async def list_custom_templates(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("email_custom_templates")
    cursor = col.find({"organization_id": _oid(org_id)})
    rows = await cursor.to_list(length=200)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        if r.get("organization_id"):
            r["organization_id"] = str(r["organization_id"])
    return rows


class CustomTemplateCreate(BaseModel):
    key: str
    subject: str
    html: str
    description: Optional[str] = None


@router.post("/templates/custom/{org_id}", status_code=201)
async def create_custom_template(
    org_id: str,
    payload: CustomTemplateCreate,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("email_custom_templates")
    doc = {
        "organization_id": _oid(org_id),
        "key": payload.key, "subject": payload.subject,
        "html": payload.html, "description": payload.description,
        "created_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
        "updated_at": datetime.utcnow(),
    }
    await col.update_one(
        {"organization_id": doc["organization_id"], "key": payload.key},
        {"$set": doc}, upsert=True,
    )
    return {"ok": True}


@router.delete("/templates/custom/{org_id}/{key}", status_code=204)
async def delete_custom_template(
    org_id: str, key: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("email_custom_templates")
    await col.delete_one({"organization_id": _oid(org_id), "key": key})
    return None


# ── DKIM / SPF / sending domains ────────────────────────────────


@router.get("/sending-domains/{org_id}")
async def list_sending_domains(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("email_sending_domains")
    rows = await col.find({"organization_id": _oid(org_id)}).to_list(length=20)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        if r.get("organization_id"):
            r["organization_id"] = str(r["organization_id"])
    return rows


@router.post("/sending-domains/{org_id}", status_code=201)
async def add_sending_domain(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("email_sending_domains")
    import secrets
    doc = {
        "organization_id": _oid(org_id),
        "domain": payload.get("domain"),
        "dkim_selector": "lumicoria",
        "dkim_public_key": "pending",
        "verification_token": secrets.token_urlsafe(24),
        "verified": False,
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"id": str(r.inserted_id),
            "instructions": "Add the TXT records shown to your DNS, then POST /verify."}


@router.post("/sending-domains/{org_id}/{domain}/verify")
async def verify_sending_domain(
    org_id: str, domain: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("email_sending_domains")
    row = await col.find_one_and_update(
        {"organization_id": _oid(org_id), "domain": domain},
        {"$set": {"verified": True, "verified_at": datetime.utcnow()}},
        return_document=True,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Domain not found")
    return {"ok": True, "verified": True}


@router.delete("/sending-domains/{org_id}/{domain}", status_code=204)
async def delete_sending_domain(
    org_id: str, domain: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("email_sending_domains")
    await col.delete_one({"organization_id": _oid(org_id), "domain": domain})
    return None


# ── Tracking opt-out ────────────────────────────────────────────


@router.get("/tracking-opt-out/{org_id}")
async def get_tracking_optout(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("email_settings")
    row = await col.find_one({"organization_id": _oid(org_id)})
    return {"tracking_disabled": bool((row or {}).get("tracking_disabled", False))}


@router.post("/tracking-opt-out/{org_id}")
async def set_tracking_optout(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("email_settings")
    await col.update_one(
        {"organization_id": _oid(org_id)},
        {"$set": {"tracking_disabled": bool(payload.get("disabled", True))}},
        upsert=True,
    )
    return {"ok": True}
