"""
Phase A3 — Org-scoped billing REST API.

Mounted under `/api/v1/org-billing`.

Powers Team / Business / Enterprise plans (per-seat, billed to the org).
Complements the existing user-scoped `/billing` router which keeps powering
individual plans (Free / Starter / Professional).

Stripe wiring reuses `backend/services/billing_service.py` primitives where
possible; webhook receiver is separate from the user-scoped one because
Stripe events for org subscriptions can be distinguished by the customer's
metadata (we tag `metadata.organization_id` at customer creation).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import stripe
import structlog
from bson import ObjectId
from fastapi import APIRouter, Body, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, EmailStr, Field

from backend.api.deps import get_current_active_user
from backend.core.config import settings
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.db.mongodb.repositories.org_subscription_repository import (
    org_subscription_repository,
    seat_assignment_repository,
)
from backend.db.mongodb.repositories.billing_repository import payment_event_repository
from backend.models.billing import (
    OrgSubscriptionInDB,
    PLAN_LIMITS,
    SubscriptionPlan,
    SubscriptionStatus,
    get_plan_limits,
)
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.billing.plan_caps import PlanCapExceeded, assert_can_add_seat
from backend.services.event_bus import emit

logger = structlog.get_logger(__name__)
router = APIRouter()


# ── Stripe price resolution ──────────────────────────────────────────


def _price_id_for(plan: SubscriptionPlan, cadence: str) -> str:
    """Map (plan, cadence) → Stripe Price ID from environment config."""
    table: Dict[str, str] = {
        f"{SubscriptionPlan.TEAM.value}:monthly": settings.STRIPE_PRICE_TEAM_MONTHLY,
        f"{SubscriptionPlan.TEAM.value}:annual": settings.STRIPE_PRICE_TEAM_YEARLY,
        f"{SubscriptionPlan.BUSINESS.value}:monthly": settings.STRIPE_PRICE_BUSINESS_MONTHLY,
        f"{SubscriptionPlan.BUSINESS.value}:annual": settings.STRIPE_PRICE_BUSINESS_YEARLY,
        f"{SubscriptionPlan.ENTERPRISE.value}:monthly": settings.STRIPE_PRICE_ENTERPRISE_MONTHLY,
        f"{SubscriptionPlan.ENTERPRISE.value}:annual": settings.STRIPE_PRICE_ENTERPRISE_YEARLY,
    }
    cadence = (cadence or "monthly").lower()
    if cadence not in ("monthly", "annual"):
        cadence = "monthly"
    price = table.get(f"{plan.value if isinstance(plan, SubscriptionPlan) else str(plan)}:{cadence}", "")
    if not price:
        raise HTTPException(
            status_code=503,
            detail=(
                f"Stripe price not configured for plan={plan} cadence={cadence}. "
                "Set the matching STRIPE_PRICE_* env var."
            ),
        )
    return price


# ── Helpers ──────────────────────────────────────────────────────────


def _oid(value: Any) -> Optional[ObjectId]:
    if value is None:
        return None
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value))
    except Exception:
        return None


async def _require_org_admin(org_id: str, current_user: User):
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    uid = _oid(current_user.id)
    admin_ids = [_oid(a) for a in (org.admin_ids or [])]
    if uid not in admin_ids:
        raise HTTPException(status_code=403, detail="Org admin permission required")
    return org


async def _require_org_owner(org_id: str, current_user: User):
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(getattr(org, "owner_id", None)) != _oid(current_user.id):
        raise HTTPException(status_code=403, detail="Org owner permission required")
    return org


def _serialize_subscription(sub: Optional[OrgSubscriptionInDB]) -> Dict[str, Any]:
    if not sub:
        return {
            "plan": SubscriptionPlan.FREE.value,
            "status": "inactive",
            "seats_purchased": 0,
            "seats_used": 0,
            "current_period_end": None,
        }
    return {
        "plan": sub.plan.value if hasattr(sub.plan, "value") else str(sub.plan),
        "status": sub.status.value if hasattr(sub.status, "value") else str(sub.status),
        "cadence": sub.cadence,
        "seats_purchased": sub.seats_purchased,
        "seats_used": sub.seats_used,
        "billing_email": sub.billing_email,
        "tax_id": sub.tax_id,
        "po_number": sub.po_number,
        "cancel_at_period_end": sub.cancel_at_period_end,
        "current_period_start": sub.current_period_start,
        "current_period_end": sub.current_period_end,
        "trial_end": sub.trial_end,
    }


def _per_seat_price(plan: SubscriptionPlan, cadence: str) -> Dict[str, Any]:
    limits = get_plan_limits(plan.value if isinstance(plan, SubscriptionPlan) else str(plan))
    monthly = limits.get("price_monthly_per_seat") or 0
    discount = limits.get("annual_discount_pct") or 0
    if cadence == "annual" and discount:
        annual_per_seat = round(monthly * (1 - discount / 100), 2)
        return {"per_seat_monthly_equiv": annual_per_seat, "monthly": monthly, "discount_pct": discount}
    return {"per_seat_monthly_equiv": monthly, "monthly": monthly, "discount_pct": 0}


# ── Public pricing preview (no auth) ─────────────────────────────────


class QuoteRequest(BaseModel):
    plan: SubscriptionPlan
    cadence: str = Field("monthly", description="monthly or annual")
    seats: int = Field(..., ge=1, le=10_000)


@router.post("/quote")
async def org_billing_quote(payload: QuoteRequest):
    """Public pricing preview — used by the marketing site and the in-app
    Pricing page.  No authentication required."""
    plan = payload.plan
    cadence = payload.cadence.lower()
    if cadence not in ("monthly", "annual"):
        raise HTTPException(status_code=400, detail="cadence must be monthly or annual")
    limits = get_plan_limits(plan.value)
    monthly_per_seat = limits.get("price_monthly_per_seat") or 0
    discount_pct = limits.get("annual_discount_pct") or 0

    if cadence == "annual":
        effective_per_seat = round(monthly_per_seat * (1 - discount_pct / 100), 2)
        monthly_total = round(effective_per_seat * payload.seats, 2)
        annual_total = round(monthly_total * 12, 2)
    else:
        effective_per_seat = monthly_per_seat
        monthly_total = round(monthly_per_seat * payload.seats, 2)
        annual_total = round(monthly_total * 12, 2)

    # Enterprise floor: minimum $1,500/mo
    enterprise_floor = 1500.0 if plan == SubscriptionPlan.ENTERPRISE else 0.0
    if enterprise_floor and monthly_total < enterprise_floor:
        monthly_total = enterprise_floor
        annual_total = round(monthly_total * 12, 2)
        effective_per_seat = round(monthly_total / payload.seats, 2)

    return {
        "plan": plan.value,
        "cadence": cadence,
        "seats": payload.seats,
        "per_seat_monthly_effective": effective_per_seat,
        "per_seat_monthly_list": monthly_per_seat,
        "discount_pct": discount_pct,
        "monthly_total": monthly_total,
        "annual_total": annual_total,
        "currency": "USD",
        "enterprise_floor_applied": bool(enterprise_floor and monthly_total == enterprise_floor),
    }


@router.get("/plans")
async def org_billing_plans():
    """Catalogue of org-scoped plans + limits + per-seat pricing."""
    out: List[Dict[str, Any]] = []
    for plan in (SubscriptionPlan.TEAM, SubscriptionPlan.BUSINESS, SubscriptionPlan.ENTERPRISE):
        limits = get_plan_limits(plan.value)
        out.append({
            "plan": plan.value,
            "display_name": limits.get("display_name") or plan.value.title(),
            "per_seat_monthly": limits.get("price_monthly_per_seat"),
            "annual_discount_pct": limits.get("annual_discount_pct") or 0,
            "limits": {k: v for k, v in limits.items() if k.startswith("max_")},
            "capabilities": {
                "teams_enabled": limits.get("teams_enabled", False),
                "sso_enabled": limits.get("sso_enabled", False),
                "scim_enabled": limits.get("scim_enabled", False),
                "strict_mode_enabled": limits.get("strict_mode_enabled", False),
                "data_residency": limits.get("data_residency", False),
                "audit_export_enabled": limits.get("audit_export_enabled", False),
                "advanced_features": limits.get("advanced_features", False),
                "priority_support": limits.get("priority_support", False),
                "api_access": limits.get("api_access", False),
                "custom_agent_templates": limits.get("custom_agent_templates", False),
            },
        })
    return {"plans": out}


# ── Subscription read ────────────────────────────────────────────────


@router.get("/{org_id}/subscription")
async def get_org_subscription(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    return _serialize_subscription(sub)


@router.get("/{org_id}/usage")
async def get_org_usage(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    plan = (sub.plan.value if (sub and hasattr(sub.plan, "value")) else (sub.plan if sub else SubscriptionPlan.FREE.value))
    limits = get_plan_limits(plan)
    active_seats = await seat_assignment_repository.count_active(org_id)
    return {
        "plan": plan,
        "seats_purchased": sub.seats_purchased if sub else 0,
        "seats_used": active_seats,
        "seats_remaining": (sub.seats_purchased - active_seats) if sub else 0,
        "limits": {k: v for k, v in limits.items() if k.startswith("max_")},
    }


# ── Stripe Checkout + Customer Portal ────────────────────────────────


class CheckoutRequest(BaseModel):
    plan: SubscriptionPlan
    cadence: str = "monthly"
    seats: int = Field(..., ge=1, le=10_000)
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None
    billing_email: Optional[EmailStr] = None


@router.post("/{org_id}/checkout")
async def create_org_checkout(
    org_id: str,
    payload: CheckoutRequest,
    current_user: User = Depends(get_current_active_user),
):
    """Create a Stripe Checkout session for an org-scoped per-seat subscription."""
    org = await _require_org_admin(org_id, current_user)
    price_id = _price_id_for(payload.plan, payload.cadence)

    # Find or create the Stripe customer for the org.
    sub = await org_subscription_repository.get_for_org(org_id)
    customer_id = sub.stripe_customer_id if sub and sub.stripe_customer_id else None
    if not customer_id:
        try:
            customer = stripe.Customer.create(
                email=payload.billing_email or getattr(current_user, "email", None),
                name=org.name,
                metadata={"organization_id": str(org_id), "type": "org"},
            )
            customer_id = customer["id"]
        except Exception as exc:  # noqa: BLE001
            logger.exception("org_billing.customer_create_failed", error=str(exc))
            raise HTTPException(status_code=502, detail="Could not create Stripe customer")

    try:
        session = stripe.checkout.Session.create(
            customer=customer_id,
            mode="subscription",
            line_items=[{"price": price_id, "quantity": int(payload.seats)}],
            allow_promotion_codes=True,
            success_url=payload.success_url or settings.STRIPE_SUCCESS_URL,
            cancel_url=payload.cancel_url or settings.STRIPE_CANCEL_URL,
            metadata={
                "organization_id": str(org_id),
                "plan": payload.plan.value,
                "cadence": payload.cadence,
                "seats": str(payload.seats),
                "created_by": str(current_user.id),
            },
            subscription_data={
                "metadata": {
                    "organization_id": str(org_id),
                    "plan": payload.plan.value,
                    "cadence": payload.cadence,
                },
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("org_billing.checkout_failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Could not create Checkout session")

    # Persist a tentative subscription row so we can find it via webhook.
    await org_subscription_repository.upsert(
        org_id,
        plan=payload.plan,
        status=SubscriptionStatus.INCOMPLETE,
        seats_purchased=payload.seats,
        cadence=payload.cadence,
        stripe_customer_id=customer_id,
        stripe_price_id=price_id,
        billing_email=str(payload.billing_email) if payload.billing_email else None,
    )

    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="org_billing.checkout_started",
        details={"plan": payload.plan.value, "seats": payload.seats, "cadence": payload.cadence},
        related_resource_type="organization", related_resource_id=str(org_id),
    )

    return {"checkout_url": session.url, "session_id": session.id}


class PortalRequest(BaseModel):
    return_url: Optional[str] = None


@router.post("/{org_id}/portal")
async def create_org_portal_session(
    org_id: str,
    payload: PortalRequest,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    if not sub or not sub.stripe_customer_id:
        raise HTTPException(status_code=400, detail="No active subscription to manage")
    try:
        portal = stripe.billing_portal.Session.create(
            customer=sub.stripe_customer_id,
            return_url=payload.return_url or settings.STRIPE_SUCCESS_URL,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("org_billing.portal_failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Could not open Customer Portal")
    return {"portal_url": portal.url}


# ── Seats ────────────────────────────────────────────────────────────


@router.get("/{org_id}/seats")
async def list_seats(
    org_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    rows = await seat_assignment_repository.list_active(org_id)
    sub = await org_subscription_repository.get_for_org(org_id)
    return {
        "purchased": sub.seats_purchased if sub else 0,
        "used": len(rows),
        "assignments": [
            {
                "user_id": r.user_id,
                "assigned_at": r.assigned_at,
                "assigned_by": r.assigned_by,
                "metadata": r.metadata,
            }
            for r in rows
        ],
    }


class SeatAssignPayload(BaseModel):
    user_id: str


@router.post("/{org_id}/seats/assign")
async def assign_seat(
    org_id: str,
    payload: SeatAssignPayload,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    try:
        await assert_can_add_seat(org_id)
    except PlanCapExceeded as exc:
        raise HTTPException(status_code=402, detail=exc.detail)
    row = await seat_assignment_repository.assign(
        org_id, payload.user_id, assigned_by=str(current_user.id),
    )
    used = await seat_assignment_repository.count_active(org_id)
    await org_subscription_repository.update_seats(org_id, used=used)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="org_billing.seat_assigned",
        details={"user_id": payload.user_id},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    await emit("org.seat_assigned", organization_id=org_id, actor_id=str(current_user.id),
               resource_type="organization", resource_id=org_id,
               payload={"user_id": payload.user_id, "seats_used": used})
    return {"ok": True, "seats_used": used}


@router.delete("/{org_id}/seats/{user_id}", status_code=204)
async def remove_seat(
    org_id: str,
    user_id: str,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    removed = await seat_assignment_repository.remove(org_id, user_id)
    if not removed:
        raise HTTPException(status_code=404, detail="Seat not found")
    used = await seat_assignment_repository.count_active(org_id)
    await org_subscription_repository.update_seats(org_id, used=used)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="org_billing.seat_removed",
        details={"user_id": user_id},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return None


class SeatBuyPayload(BaseModel):
    quantity: int = Field(..., ge=1, le=10_000)


@router.post("/{org_id}/seats/buy")
async def buy_seats(
    org_id: str,
    payload: SeatBuyPayload,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    if not sub or not sub.stripe_subscription_id:
        raise HTTPException(status_code=400, detail="No active subscription")
    new_qty = sub.seats_purchased + payload.quantity
    try:
        sub_obj = stripe.Subscription.retrieve(sub.stripe_subscription_id)
        item_id = sub_obj["items"]["data"][0]["id"]
        stripe.SubscriptionItem.modify(
            item_id,
            quantity=new_qty,
            proration_behavior="create_prorations",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("org_billing.buy_seats_failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Stripe seat increase failed")
    await org_subscription_repository.update_seats(org_id, purchased=new_qty)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="org_billing.seats_bought",
        details={"quantity": payload.quantity, "new_total": new_qty},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return {"ok": True, "purchased": new_qty}


class SeatRemoveQuantityPayload(BaseModel):
    quantity: int = Field(..., ge=1, le=10_000)


@router.post("/{org_id}/seats/return")
async def return_seats(
    org_id: str,
    payload: SeatRemoveQuantityPayload,
    current_user: User = Depends(get_current_active_user),
):
    """Reduce purchased seats.  Requires usage to fit the new count."""
    await _require_org_owner(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    if not sub or not sub.stripe_subscription_id:
        raise HTTPException(status_code=400, detail="No active subscription")
    new_qty = max(1, sub.seats_purchased - payload.quantity)
    used = await seat_assignment_repository.count_active(org_id)
    if new_qty < used:
        raise HTTPException(
            status_code=400,
            detail=f"Cannot reduce seats below current usage ({used}). Remove members first.",
        )
    try:
        sub_obj = stripe.Subscription.retrieve(sub.stripe_subscription_id)
        item_id = sub_obj["items"]["data"][0]["id"]
        stripe.SubscriptionItem.modify(
            item_id,
            quantity=new_qty,
            proration_behavior="create_prorations",
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("org_billing.return_seats_failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Stripe seat reduction failed")
    await org_subscription_repository.update_seats(org_id, purchased=new_qty)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="org_billing.seats_returned",
        details={"quantity": payload.quantity, "new_total": new_qty},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return {"ok": True, "purchased": new_qty}


@router.get("/{org_id}/seats/forecast")
async def seat_forecast(
    org_id: str,
    horizon_days: int = Query(90, ge=7, le=365),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    used = await seat_assignment_repository.count_active(org_id)
    # Simple linear projection from current util — replaced by analytics
    # v2 in Phase D with cohort growth modelling.
    utilisation = (used / sub.seats_purchased) if (sub and sub.seats_purchased) else 0.0
    projected_used = round(used * (1 + (utilisation * horizon_days / 30) * 0.05), 0)
    return {
        "purchased": sub.seats_purchased if sub else 0,
        "used_today": used,
        "projected_used": int(projected_used),
        "horizon_days": horizon_days,
    }


# ── Plan + cadence changes ───────────────────────────────────────────


class PlanChangePayload(BaseModel):
    plan: SubscriptionPlan
    cadence: Optional[str] = None  # if None, keep current


@router.post("/{org_id}/plan/change")
async def change_plan(
    org_id: str,
    payload: PlanChangePayload,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    if not sub or not sub.stripe_subscription_id:
        raise HTTPException(status_code=400, detail="No active subscription")
    cadence = payload.cadence or sub.cadence
    new_price = _price_id_for(payload.plan, cadence)
    try:
        sub_obj = stripe.Subscription.retrieve(sub.stripe_subscription_id)
        item_id = sub_obj["items"]["data"][0]["id"]
        stripe.SubscriptionItem.modify(item_id, price=new_price, proration_behavior="create_prorations")
    except Exception as exc:  # noqa: BLE001
        logger.exception("org_billing.plan_change_failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Plan change failed")
    await org_subscription_repository.upsert(
        org_id, plan=payload.plan, status=sub.status, seats_purchased=sub.seats_purchased,
        cadence=cadence, stripe_customer_id=sub.stripe_customer_id,
        stripe_subscription_id=sub.stripe_subscription_id, stripe_price_id=new_price,
    )
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="org_billing.plan_changed",
        details={"plan": payload.plan.value, "cadence": cadence},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return {"ok": True, "plan": payload.plan.value, "cadence": cadence}


class CadenceChangePayload(BaseModel):
    cadence: str  # monthly | annual


@router.post("/{org_id}/cadence")
async def change_cadence(
    org_id: str,
    payload: CadenceChangePayload,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    if not sub:
        raise HTTPException(status_code=400, detail="No active subscription")
    plan = sub.plan if isinstance(sub.plan, SubscriptionPlan) else SubscriptionPlan(sub.plan)
    return await change_plan(  # type: ignore[misc]
        org_id, PlanChangePayload(plan=plan, cadence=payload.cadence), current_user
    )


# ── Cancel / restart ─────────────────────────────────────────────────


@router.post("/{org_id}/cancel")
async def cancel_subscription(
    org_id: str,
    at_period_end: bool = Query(True),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    if not sub or not sub.stripe_subscription_id:
        raise HTTPException(status_code=400, detail="No active subscription")
    try:
        stripe.Subscription.modify(sub.stripe_subscription_id, cancel_at_period_end=bool(at_period_end))
    except Exception as exc:  # noqa: BLE001
        logger.exception("org_billing.cancel_failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Cancel failed")
    updated = await org_subscription_repository.cancel(org_id, at_period_end=at_period_end)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="org_billing.subscription_cancelled",
        details={"at_period_end": at_period_end},
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    return _serialize_subscription(updated)


# ── Invoices ─────────────────────────────────────────────────────────


@router.get("/{org_id}/invoices")
async def list_org_invoices(
    org_id: str,
    limit: int = Query(20, ge=1, le=100),
    starting_after: Optional[str] = Query(None),
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    if not sub or not sub.stripe_customer_id:
        return {"invoices": []}
    try:
        kwargs: Dict[str, Any] = {"customer": sub.stripe_customer_id, "limit": limit}
        if starting_after:
            kwargs["starting_after"] = starting_after
        result = stripe.Invoice.list(**kwargs)
    except Exception as exc:  # noqa: BLE001
        logger.exception("org_billing.invoices_failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Could not load invoices")
    invoices = [
        {
            "id": inv["id"],
            "number": inv.get("number"),
            "status": inv.get("status"),
            "amount_due": inv.get("amount_due"),
            "amount_paid": inv.get("amount_paid"),
            "currency": inv.get("currency"),
            "hosted_invoice_url": inv.get("hosted_invoice_url"),
            "invoice_pdf": inv.get("invoice_pdf"),
            "created": inv.get("created"),
        }
        for inv in result["data"]
    ]
    return {"invoices": invoices, "has_more": result.get("has_more")}


# ── Billing details (tax / PO / email) ───────────────────────────────


class BillingDetailsPatch(BaseModel):
    billing_email: Optional[EmailStr] = None
    tax_id: Optional[str] = None
    po_number: Optional[str] = None


@router.patch("/{org_id}/details")
async def update_billing_details(
    org_id: str,
    payload: BillingDetailsPatch,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    if not sub:
        raise HTTPException(status_code=400, detail="No active subscription")
    patch = payload.model_dump(exclude_unset=True)
    if not patch:
        return _serialize_subscription(sub)
    # Best-effort: update Stripe customer fields too
    if sub.stripe_customer_id and patch.get("billing_email"):
        try:
            stripe.Customer.modify(sub.stripe_customer_id, email=patch["billing_email"])
        except Exception:  # noqa: BLE001
            pass
    updated = await org_subscription_repository.upsert(
        org_id,
        plan=sub.plan if isinstance(sub.plan, SubscriptionPlan) else SubscriptionPlan(sub.plan),
        status=sub.status if isinstance(sub.status, SubscriptionStatus) else SubscriptionStatus(sub.status),
        seats_purchased=sub.seats_purchased,
        cadence=sub.cadence,
        stripe_customer_id=sub.stripe_customer_id,
        stripe_subscription_id=sub.stripe_subscription_id,
        stripe_price_id=sub.stripe_price_id,
        billing_email=patch.get("billing_email") or sub.billing_email,
    )
    return _serialize_subscription(updated)


# ── Sales-led flow ───────────────────────────────────────────────────


class TalkToSalesPayload(BaseModel):
    company_name: Optional[str] = None
    seats: Optional[int] = None
    use_case: Optional[str] = None
    contact_email: Optional[EmailStr] = None
    notes: Optional[str] = Field(None, max_length=4000)


@router.post("/{org_id}/talk-to-sales", status_code=201)
async def talk_to_sales(
    org_id: str,
    payload: TalkToSalesPayload,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_admin(org_id, current_user)
    await log_activity(
        user_id=str(current_user.id), organization_id=org_id,
        activity_type="org_billing.sales_request",
        details=payload.model_dump(exclude_none=True),
        related_resource_type="organization", related_resource_id=str(org_id),
    )
    await emit("org.sales_request", organization_id=org_id, actor_id=str(current_user.id),
               resource_type="organization", resource_id=org_id,
               payload=payload.model_dump(exclude_none=True))
    return {"ok": True, "message": "Our sales team will reach out within one business day."}


class PromoCodePayload(BaseModel):
    code: str


@router.post("/{org_id}/promo-code")
async def apply_promo_code(
    org_id: str,
    payload: PromoCodePayload,
    current_user: User = Depends(get_current_active_user),
):
    await _require_org_owner(org_id, current_user)
    sub = await org_subscription_repository.get_for_org(org_id)
    if not sub or not sub.stripe_subscription_id:
        raise HTTPException(status_code=400, detail="No active subscription")
    try:
        codes = stripe.PromotionCode.list(code=payload.code, active=True, limit=1)
        if not codes["data"]:
            raise HTTPException(status_code=404, detail="Invalid or expired promo code")
        promo = codes["data"][0]
        stripe.Subscription.modify(sub.stripe_subscription_id, promotion_code=promo["id"])
    except HTTPException:
        raise
    except Exception as exc:  # noqa: BLE001
        logger.exception("org_billing.promo_failed", error=str(exc))
        raise HTTPException(status_code=502, detail="Promo code could not be applied")
    return {"ok": True}


# ── Webhook ──────────────────────────────────────────────────────────


@router.post("/webhook", status_code=200)
async def org_billing_webhook(request: Request):
    """Stripe webhook receiver for org-scoped events.  Idempotent via
    `payment_event_repository.has_processed()`."""
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature", "")
    try:
        event = stripe.Webhook.construct_event(
            payload, sig_header, settings.STRIPE_WEBHOOK_SECRET,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("org_billing.webhook_invalid_signature", error=str(exc))
        return {"ok": False}

    # Idempotency via the shared payment_events collection.
    try:
        if await payment_event_repository.is_event_processed(event["id"]):
            return {"ok": True, "duplicate": True}
    except Exception:  # noqa: BLE001 — best-effort
        pass

    org_id = None
    obj = event.get("data", {}).get("object", {}) or {}
    meta = obj.get("metadata") or {}
    if meta.get("organization_id"):
        org_id = meta["organization_id"]
    elif obj.get("customer"):
        try:
            customer = stripe.Customer.retrieve(obj["customer"])
            org_id = (customer.get("metadata") or {}).get("organization_id")
        except Exception:  # noqa: BLE001
            org_id = None

    if not org_id:
        return {"ok": True, "skipped": "no_org"}

    etype = event["type"]
    try:
        if etype in ("checkout.session.completed", "customer.subscription.created", "customer.subscription.updated"):
            sub_id = obj.get("subscription") or obj.get("id")
            if isinstance(sub_id, str) and sub_id.startswith("sub_"):
                sub_obj = stripe.Subscription.retrieve(sub_id)
                plan_name = (sub_obj.get("metadata") or {}).get("plan") or meta.get("plan")
                plan = SubscriptionPlan(plan_name) if plan_name else SubscriptionPlan.TEAM
                cadence = (sub_obj.get("metadata") or {}).get("cadence") or meta.get("cadence", "monthly")
                items = sub_obj["items"]["data"]
                qty = int(items[0]["quantity"]) if items else 1
                price_id = items[0]["price"]["id"] if items else None
                status_str = sub_obj.get("status", "active")
                await org_subscription_repository.upsert(
                    org_id,
                    plan=plan,
                    status=SubscriptionStatus(status_str) if status_str in SubscriptionStatus._value2member_map_ else SubscriptionStatus.ACTIVE,
                    seats_purchased=qty,
                    cadence=cadence,
                    stripe_customer_id=sub_obj["customer"],
                    stripe_subscription_id=sub_obj["id"],
                    stripe_price_id=price_id,
                )
        elif etype == "customer.subscription.deleted":
            await org_subscription_repository.cancel(org_id, at_period_end=False)
        # Persist the event for idempotency.
        try:
            from backend.models.billing import PaymentEventInDB
            sub_id_candidate = obj.get("subscription")
            if not sub_id_candidate and isinstance(obj.get("id"), str) and obj["id"].startswith("sub_"):
                sub_id_candidate = obj["id"]
            inv_id_candidate = obj.get("invoice")
            if not inv_id_candidate and isinstance(obj.get("id"), str) and obj["id"].startswith("in_"):
                inv_id_candidate = obj["id"]
            await payment_event_repository.record_event(PaymentEventInDB(
                stripe_event_id=event["id"],
                event_type=etype,
                user_id=None,
                stripe_customer_id=obj.get("customer"),
                stripe_subscription_id=sub_id_candidate,
                stripe_invoice_id=inv_id_candidate,
                amount=obj.get("amount_paid") or obj.get("amount_due"),
                currency=obj.get("currency"),
                raw_event=dict(event) if not isinstance(event, dict) else event,
            ))
        except Exception:  # noqa: BLE001 — non-critical
            pass
    except Exception as exc:  # noqa: BLE001
        logger.exception("org_billing.webhook_process_failed", error=str(exc), type=etype)

    return {"ok": True}
