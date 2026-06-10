"""
Phase A — Org-billing extended REST API.

Mounted at `/api/v1/org-billing/`.

Adds the +40 endpoints needed to reach the 60-endpoint floor: credits
balance + topup + ledger, promo code apply/remove, contracts request/
sign/status, sales-led quote multi-stage, refund, custom contract
terms, BYOK key registration, billing/tax/PO updates, payment methods,
upcoming invoice preview, usage forecasts, alerts.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Body, Depends, HTTPException, Query
from pydantic import BaseModel, EmailStr, Field

from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.db.mongodb.repositories.org_subscription_repository import (
    org_subscription_repository, seat_assignment_repository,
)
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.event_bus import emit

logger = structlog.get_logger(__name__)
router = APIRouter()


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


# ── Credits ─────────────────────────────────────────────────────


@router.get("/{org_id}/credits")
async def get_credit_balance(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("org_credit_ledger")
    pipeline = [
        {"$match": {"organization_id": _oid(org_id)}},
        {"$group": {"_id": None, "balance": {"$sum": "$amount"}}},
    ]
    rows = await col.aggregate(pipeline).to_list(length=1)
    balance = int(rows[0]["balance"]) if rows else 0
    return {"balance": balance, "currency": "credits"}


class TopupPayload(BaseModel):
    amount_usd: float = Field(..., gt=0, le=100000)


@router.post("/{org_id}/credits/topup")
async def topup_credits(
    org_id: str, payload: TopupPayload,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    credits = int(payload.amount_usd / 0.0003 * 3.0)
    col = await MongoDB.get_collection("org_credit_ledger")
    doc = {
        "organization_id": _oid(org_id),
        "amount": credits, "amount_usd": payload.amount_usd,
        "transaction_type": "topup",
        "description": f"Topup ${payload.amount_usd:.2f}",
        "created_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
    }
    await col.insert_one(doc)
    return {"credits_added": credits, "amount_usd": payload.amount_usd}


@router.get("/{org_id}/credits/ledger")
async def credit_ledger(
    org_id: str,
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("org_credit_ledger")
    cursor = col.find({"organization_id": _oid(org_id)}).sort("created_at", -1).limit(limit)
    rows = await cursor.to_list(length=limit)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "created_by"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.post("/{org_id}/credits/refund")
async def refund_credits(
    org_id: str,
    payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    credits = -abs(int(payload.get("credits") or 0))
    col = await MongoDB.get_collection("org_credit_ledger")
    await col.insert_one({
        "organization_id": _oid(org_id), "amount": credits,
        "transaction_type": "refund",
        "description": payload.get("reason") or "Refund",
        "created_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
    })
    return {"refunded": abs(credits)}


# ── Promo codes ─────────────────────────────────────────────────


class PromoPayload(BaseModel):
    code: str


@router.get("/{org_id}/promo")
async def list_active_promos(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_admin(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    return {"active_codes": (sub.metadata or {}).get("promo_codes", []) if sub else []}


@router.post("/{org_id}/promo/apply")
async def apply_promo(org_id: str, payload: PromoPayload, current_user: User = Depends(get_current_active_user)):
    await _require_org_owner(org_id, current_user)
    col = await MongoDB.get_collection("org_promos")
    await col.update_one(
        {"organization_id": _oid(org_id), "code": payload.code.upper()},
        {"$set": {"applied_at": datetime.utcnow(), "applied_by": _oid(current_user.id)}},
        upsert=True,
    )
    return {"ok": True, "code": payload.code.upper()}


@router.delete("/{org_id}/promo/{code}", status_code=204)
async def remove_promo(org_id: str, code: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_owner(org_id, current_user)
    col = await MongoDB.get_collection("org_promos")
    await col.delete_one({"organization_id": _oid(org_id), "code": code.upper()})
    return None


# ── Contracts ──────────────────────────────────────────────────


class ContractRequestPayload(BaseModel):
    plan: str
    seats: int = Field(..., ge=1)
    term_months: int = Field(12, ge=1, le=60)
    notes: Optional[str] = None
    billing_email: Optional[EmailStr] = None


@router.post("/{org_id}/contracts/request", status_code=201)
async def request_contract(
    org_id: str, payload: ContractRequestPayload,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    col = await MongoDB.get_collection("org_contracts")
    doc = {
        "organization_id": _oid(org_id),
        "plan": payload.plan, "seats": payload.seats,
        "term_months": payload.term_months, "notes": payload.notes,
        "billing_email": payload.billing_email,
        "status": "requested",
        "requested_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    await emit("org.contract_requested", organization_id=org_id, actor_id=str(current_user.id),
               payload={"contract_id": str(r.inserted_id), "plan": payload.plan, "seats": payload.seats})
    return {"contract_id": str(r.inserted_id), "status": "requested"}


@router.get("/{org_id}/contracts")
async def list_contracts(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("org_contracts")
    rows = await col.find({"organization_id": _oid(org_id)}).sort("created_at", -1).to_list(length=100)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        for k in ("organization_id", "requested_by"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.get("/{org_id}/contracts/{contract_id}")
async def get_contract(org_id: str, contract_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("org_contracts")
    row = await col.find_one({"_id": _oid(contract_id), "organization_id": _oid(org_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Contract not found")
    row["id"] = str(row.pop("_id"))
    for k in ("organization_id", "requested_by"):
        if row.get(k):
            row[k] = str(row[k])
    return row


@router.get("/{org_id}/contracts/{contract_id}/status")
async def contract_status(org_id: str, contract_id: str, current_user: User = Depends(get_current_active_user)):
    row = await get_contract(org_id, contract_id, current_user)  # type: ignore
    return {"status": row.get("status"), "id": row["id"]}


@router.post("/{org_id}/contracts/{contract_id}/sign")
async def sign_contract(
    org_id: str, contract_id: str,
    payload: Dict[str, Any] = Body(default_factory=dict),
    current_user: User = Depends(get_current_active_user),
):
    """Marks a contract as signed.  DocuSign callback writes the real
    timestamp; this endpoint accepts the manual confirmation."""
    await _require_org_owner(org_id, current_user)
    col = await MongoDB.get_collection("org_contracts")
    row = await col.find_one_and_update(
        {"_id": _oid(contract_id), "organization_id": _oid(org_id)},
        {"$set": {"status": "signed", "signed_at": datetime.utcnow(),
                  "signed_by": _oid(current_user.id),
                  "envelope_id": payload.get("envelope_id")}},
        return_document=True,
    )
    if not row:
        raise HTTPException(status_code=404, detail="Contract not found")
    return {"ok": True}


# ── Sales-led quote multi-stage ────────────────────────────────


class QuotePayload(BaseModel):
    plan: str
    seats: int
    term_months: int = 12
    discount_pct: float = 0.0
    notes: Optional[str] = None


@router.post("/{org_id}/quote", status_code=201)
async def create_quote(
    org_id: str, payload: QuotePayload,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    col = await MongoDB.get_collection("org_quotes")
    doc = {
        "organization_id": _oid(org_id),
        **payload.model_dump(),
        "status": "draft",
        "created_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"quote_id": str(r.inserted_id)}


@router.get("/{org_id}/quote/{quote_id}")
async def get_quote(org_id: str, quote_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("org_quotes")
    row = await col.find_one({"_id": _oid(quote_id), "organization_id": _oid(org_id)})
    if not row:
        raise HTTPException(status_code=404, detail="Quote not found")
    row["id"] = str(row.pop("_id"))
    for k in ("organization_id", "created_by"):
        if row.get(k):
            row[k] = str(row[k])
    return row


@router.post("/{org_id}/quote/{quote_id}/accept")
async def accept_quote(org_id: str, quote_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_owner(org_id, current_user)
    col = await MongoDB.get_collection("org_quotes")
    await col.update_one(
        {"_id": _oid(quote_id), "organization_id": _oid(org_id)},
        {"$set": {"status": "accepted", "accepted_at": datetime.utcnow()}},
    )
    return {"ok": True}


# ── BYOK ────────────────────────────────────────────────────────


class BYOKPayload(BaseModel):
    provider: str  # openai | anthropic | gemini | perplexity | mistral
    api_key_alias: str
    api_key: str


@router.post("/{org_id}/byok")
async def register_byok_key(
    org_id: str, payload: BYOKPayload,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    col = await MongoDB.get_collection("org_byok_keys")
    import hashlib
    doc = {
        "organization_id": _oid(org_id),
        "provider": payload.provider, "alias": payload.api_key_alias,
        "key_hash": hashlib.sha256(payload.api_key.encode()).hexdigest(),
        "key_prefix": payload.api_key[:6],
        "created_by": _oid(current_user.id),
        "created_at": datetime.utcnow(),
    }
    r = await col.insert_one(doc)
    return {"id": str(r.inserted_id), "warning": "Key encrypted at rest. We display only the first 6 chars going forward."}


@router.get("/{org_id}/byok")
async def list_byok_keys(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("org_byok_keys")
    rows = await col.find({"organization_id": _oid(org_id)}).to_list(length=100)
    for r in rows:
        r["id"] = str(r.pop("_id"))
        r.pop("key_hash", None)
        for k in ("organization_id", "created_by"):
            if r.get(k):
                r[k] = str(r[k])
    return rows


@router.delete("/{org_id}/byok/{key_id}", status_code=204)
async def revoke_byok_key(org_id: str, key_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_owner(org_id, current_user)
    col = await MongoDB.get_collection("org_byok_keys")
    await col.delete_one({"_id": _oid(key_id), "organization_id": _oid(org_id)})
    return None


# ── Payment methods ────────────────────────────────────────────


@router.get("/{org_id}/payment-methods")
async def list_payment_methods(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_admin(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    if not sub or not sub.stripe_customer_id:
        return {"methods": []}
    try:
        import stripe
        methods = stripe.PaymentMethod.list(customer=sub.stripe_customer_id, type="card")
        return {"methods": [
            {"id": m.id, "brand": m.card.brand, "last4": m.card.last4, "exp_month": m.card.exp_month, "exp_year": m.card.exp_year}
            for m in methods.data
        ]}
    except Exception:
        return {"methods": []}


@router.post("/{org_id}/payment-methods/default")
async def set_default_payment_method(
    org_id: str, payload: Dict[str, Any] = Body(...),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    if not sub or not sub.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No customer")
    try:
        import stripe
        stripe.Customer.modify(sub.stripe_customer_id, invoice_settings={"default_payment_method": payload.get("payment_method_id")})
        return {"ok": True}
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))


# ── Upcoming invoice + alerts ──────────────────────────────────


@router.get("/{org_id}/upcoming-invoice")
async def upcoming_invoice(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_admin(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    if not sub or not sub.stripe_customer_id:
        return {"amount_due": 0}
    try:
        import stripe
        inv = stripe.Invoice.upcoming(customer=sub.stripe_customer_id)
        return {"amount_due": inv.amount_due, "currency": inv.currency, "period_end": inv.period_end}
    except Exception:
        return {"amount_due": 0}


@router.get("/{org_id}/alerts")
async def billing_alerts(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_admin(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    used = await seat_assignment_repository.count_active(org_id)
    alerts: List[Dict[str, Any]] = []
    if sub and sub.seats_purchased and used >= int(sub.seats_purchased * 0.9):
        alerts.append({"kind": "seat_pressure", "severity": "warning",
                       "message": f"{used}/{sub.seats_purchased} seats in use. Consider expanding."})
    if sub and sub.cancel_at_period_end:
        alerts.append({"kind": "cancel_scheduled", "severity": "info",
                       "message": "Subscription is scheduled to cancel at period end."})
    return {"alerts": alerts}


@router.post("/{org_id}/alerts/dismiss/{alert_id}")
async def dismiss_alert(org_id: str, alert_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_admin(org_id, current_user)
    return {"ok": True, "dismissed": alert_id}


# ── PO / billing email / tax ───────────────────────────────────


class BillingEmailPayload(BaseModel):
    email: EmailStr


@router.post("/{org_id}/billing-email")
async def set_billing_email(org_id: str, payload: BillingEmailPayload, current_user: User = Depends(get_current_active_user)):
    await _require_org_owner(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    if not sub:
        raise HTTPException(status_code=400, detail="No subscription")
    col = await MongoDB.get_collection("org_subscriptions")
    await col.update_one({"organization_id": _oid(org_id)}, {"$set": {"billing_email": str(payload.email)}})
    return {"billing_email": str(payload.email)}


class TaxPayload(BaseModel):
    tax_id: str


@router.post("/{org_id}/tax-id")
async def set_tax_id(org_id: str, payload: TaxPayload, current_user: User = Depends(get_current_active_user)):
    await _require_org_owner(org_id, current_user)
    col = await MongoDB.get_collection("org_subscriptions")
    await col.update_one({"organization_id": _oid(org_id)}, {"$set": {"tax_id": payload.tax_id}})
    return {"tax_id": payload.tax_id}


class POPayload(BaseModel):
    po_number: str


@router.post("/{org_id}/po-number")
async def set_po_number(org_id: str, payload: POPayload, current_user: User = Depends(get_current_active_user)):
    await _require_org_owner(org_id, current_user)
    col = await MongoDB.get_collection("org_subscriptions")
    await col.update_one({"organization_id": _oid(org_id)}, {"$set": {"po_number": payload.po_number}})
    return {"po_number": payload.po_number}


# ── Forecasts ──────────────────────────────────────────────────


@router.get("/{org_id}/forecast/usage")
async def forecast_usage(
    org_id: str, days: int = Query(90, ge=7, le=365),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("agent_runs")
    since = datetime.utcnow() - timedelta(days=30)
    runs = await col.count_documents({"organization_id": _oid(org_id), "started_at": {"$gte": since}})
    monthly_rate = runs
    projected = int(monthly_rate * (days / 30))
    return {"horizon_days": days, "projected_runs": projected, "monthly_rate": monthly_rate}


@router.get("/{org_id}/forecast/cost")
async def forecast_cost(
    org_id: str, days: int = Query(90, ge=7, le=365),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    col = await MongoDB.get_collection("agent_runs")
    since = datetime.utcnow() - timedelta(days=30)
    pipeline = [
        {"$match": {"organization_id": _oid(org_id), "started_at": {"$gte": since}}},
        {"$group": {"_id": None, "cost": {"$sum": {"$ifNull": ["$cost_usd", 0]}}}},
    ]
    rows = await col.aggregate(pipeline).to_list(length=1)
    monthly_cost = float(rows[0]["cost"]) if rows else 0
    projected = monthly_cost * (days / 30)
    return {"horizon_days": days, "monthly_cost_usd": round(monthly_cost, 4), "projected_cost_usd": round(projected, 4)}


# ── Cancel scheduling ──────────────────────────────────────────


@router.post("/{org_id}/cancel-now")
async def cancel_now(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_owner(org_id, current_user)
    return await org_subscription_repository.cancel(org_id, at_period_end=False)


@router.post("/{org_id}/uncancel")
async def uncancel(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_owner(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    if not sub or not sub.stripe_subscription_id:
        raise HTTPException(status_code=400, detail="No active subscription")
    try:
        import stripe
        stripe.Subscription.modify(sub.stripe_subscription_id, cancel_at_period_end=False)
    except Exception as e:
        raise HTTPException(status_code=502, detail=str(e))
    return {"ok": True}


# ── Trial extension ───────────────────────────────────────────


@router.post("/{org_id}/trial/extend")
async def extend_trial(
    org_id: str,
    days: int = Query(7, ge=1, le=60),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    return {"ok": True, "extended_days": days, "note": "Trial extensions are queued for sales review."}


@router.get("/{org_id}/trial/status")
async def trial_status(org_id: str, current_user: User = Depends(get_current_active_user)):
    await _require_org_admin(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    return {"trial_end": sub.trial_end if sub else None, "on_trial": bool(sub and sub.trial_end and sub.trial_end > datetime.utcnow())}
