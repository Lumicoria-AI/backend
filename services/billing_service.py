"""
Lumicoria AI — Stripe Billing Service

Production-grade Stripe integration with:
  • Hosted Checkout sessions
  • Customer creation & management
  • Webhook signature verification
  • Idempotent event processing
  • Subscription lifecycle management
  • Customer Portal integration
  • Plan-based feature gating
  • Usage tracking & limit enforcement
  • Admin override support
  • Grace period handling

SECURITY:
  - All Stripe keys loaded from environment (never hardcoded)
  - Webhook signature verified with stripe.Webhook.construct_event()
  - No subscription state trusted from client
  - All billing validation is server-side
  - Sensitive data never logged
"""

import stripe
import asyncio
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, Tuple
from backend.core.config import settings
from backend.models.billing import (
    SubscriptionInDB,
    UsageTrackingInDB,
    PaymentEventInDB,
    SubscriptionPlan,
    SubscriptionStatus,
    PaymentStatus,
    PLAN_LIMITS,
    get_plan_limits,
    CreateCheckoutResponse,
    CustomerPortalResponse,
    SubscriptionResponse,
    UsageResponse,
)
from backend.db.mongodb.repositories.billing_repository import (
    subscription_repository,
    usage_tracking_repository,
    payment_event_repository,
)
from backend.db.mongodb.repositories.credits_repository import CreditLedgerRepository
from backend.db.mongodb.repositories.invoice_repository import InvoiceRepository
from backend.services.notification_service import notification_service
import structlog

logger = structlog.get_logger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Initialize Stripe SDK
# ─────────────────────────────────────────────────────────────────────────────

stripe.api_key = settings.STRIPE_SECRET_KEY
stripe.api_version = "2024-12-18.acacia"  # Pin API version for stability

# Map Stripe Price IDs → internal plan names
# Loaded from environment — NEVER hardcoded
PRICE_TO_PLAN: Dict[str, SubscriptionPlan] = {}


def _build_price_map():
    """Build the price-to-plan mapping from environment config."""
    global PRICE_TO_PLAN
    mappings = [
        (settings.STRIPE_PRICE_STARTER_MONTHLY, SubscriptionPlan.STARTER),
        (settings.STRIPE_PRICE_STARTER_YEARLY, SubscriptionPlan.STARTER),
        (settings.STRIPE_PRICE_PRO_MONTHLY, SubscriptionPlan.PROFESSIONAL),
        (settings.STRIPE_PRICE_PRO_YEARLY, SubscriptionPlan.PROFESSIONAL),
        (settings.STRIPE_PRICE_ENTERPRISE, SubscriptionPlan.ENTERPRISE),
    ]
    for price_id, plan in mappings:
        if price_id:
            PRICE_TO_PLAN[price_id] = plan
    logger.info("Stripe price-to-plan mapping built", count=len(PRICE_TO_PLAN))


# Build map at module load (settings are already validated)
_build_price_map()


def _resolve_plan_from_price(price_id: str) -> SubscriptionPlan:
    """Resolve a Stripe Price ID to an internal plan. Defaults to FREE."""
    return PRICE_TO_PLAN.get(price_id, SubscriptionPlan.FREE)


# ─────────────────────────────────────────────────────────────────────────────
# Customer Management
# ─────────────────────────────────────────────────────────────────────────────

async def get_or_create_stripe_customer(
    user_id: str,
    email: str,
    name: Optional[str] = None,
) -> str:
    """
    Get existing Stripe customer or create a new one.
    Links the Stripe customer to our internal user ID via metadata.
    Returns the stripe_customer_id.
    """
    # Check if user already has a subscription record
    sub = await subscription_repository.get_by_user_id(user_id)
    if sub and sub.stripe_customer_id:
        return sub.stripe_customer_id

    # Create Stripe customer
    customer = stripe.Customer.create(
        email=email,
        name=name,
        metadata={
            "lumicoria_user_id": user_id,
            "environment": settings.ENVIRONMENT,
        },
    )

    # Create subscription document (starts as FREE)
    subscription = SubscriptionInDB(
        user_id=user_id,
        stripe_customer_id=customer.id,
        plan=SubscriptionPlan.FREE,
        status=SubscriptionStatus.ACTIVE,
    )
    await subscription_repository.create(subscription)

    logger.info(
        "Stripe customer created",
        user_id=user_id,
        stripe_customer_id=customer.id,
    )
    return customer.id


# ─────────────────────────────────────────────────────────────────────────────
# Checkout
# ─────────────────────────────────────────────────────────────────────────────

async def create_checkout_session(
    user_id: str,
    email: str,
    name: Optional[str],
    price_id: str,
    success_url: Optional[str] = None,
    cancel_url: Optional[str] = None,
) -> CreateCheckoutResponse:
    """
    Create a Stripe Checkout Session for subscription purchase.
    Uses Stripe Hosted Checkout (PCI compliant — no card data touches our servers).
    """
    # Get or create customer
    stripe_customer_id = await get_or_create_stripe_customer(user_id, email, name)

    # Validate price_id exists in our mapping
    if price_id not in PRICE_TO_PLAN:
        raise ValueError(f"Invalid price_id: {price_id}")

    session = stripe.checkout.Session.create(
        customer=stripe_customer_id,
        payment_method_types=["card"],
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=success_url or settings.STRIPE_SUCCESS_URL,
        cancel_url=cancel_url or settings.STRIPE_CANCEL_URL,
        metadata={
            "lumicoria_user_id": user_id,
            "environment": settings.ENVIRONMENT,
        },
        subscription_data={
            "metadata": {
                "lumicoria_user_id": user_id,
            },
            "trial_period_days": 14 if _resolve_plan_from_price(price_id) != SubscriptionPlan.ENTERPRISE else None,
        },
        allow_promotion_codes=True,
        billing_address_collection="required",
        customer_update={
            "address": "auto",
            "name": "auto",
        },
    )

    logger.info(
        "Checkout session created",
        user_id=user_id,
        price_id=price_id,
        session_id=session.id,
    )

    return CreateCheckoutResponse(
        checkout_url=session.url,
        session_id=session.id,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Customer Portal
# ─────────────────────────────────────────────────────────────────────────────

async def create_customer_portal_session(
    user_id: str,
    return_url: Optional[str] = None,
) -> CustomerPortalResponse:
    """Create a Stripe Customer Portal session for self-service billing management."""
    sub = await subscription_repository.get_by_user_id(user_id)
    if not sub or not sub.stripe_customer_id:
        raise ValueError("No billing account found. Please subscribe first.")

    session = stripe.billing_portal.Session.create(
        customer=sub.stripe_customer_id,
        return_url=return_url or settings.STRIPE_SUCCESS_URL,
    )

    return CustomerPortalResponse(portal_url=session.url)


# ─────────────────────────────────────────────────────────────────────────────
# Webhook Processing
# ─────────────────────────────────────────────────────────────────────────────

# Events we actually handle — ignore everything else
HANDLED_EVENTS = {
    "checkout.session.completed",
    "customer.subscription.created",
    "customer.subscription.updated",
    "customer.subscription.deleted",
    "invoice.payment_succeeded",
    "invoice.payment_failed",
}


def verify_webhook_signature(payload: bytes, sig_header: str) -> stripe.Event:
    """
    Verify Stripe webhook signature.
    SECURITY: This MUST be called with the raw request body (bytes).
    Raises stripe.error.SignatureVerificationError if invalid.
    """
    return stripe.Webhook.construct_event(
        payload,
        sig_header,
        settings.STRIPE_WEBHOOK_SECRET,
    )


async def process_webhook_event(event: stripe.Event) -> Dict[str, Any]:
    """
    Process a verified Stripe webhook event.
    
    IDEMPOTENCY: Each event is processed exactly once.
    The payment_events collection has a unique index on stripe_event_id.
    """
    event_id = event.id
    event_type = event.type

    # Filter unhandled events
    if event_type not in HANDLED_EVENTS:
        logger.debug("Ignoring unhandled webhook event", event_type=event_type)
        return {"status": "ignored", "event_type": event_type}

    # Idempotency check
    if await payment_event_repository.is_event_processed(event_id):
        logger.info("Webhook event already processed (idempotent skip)", event_id=event_id)
        return {"status": "already_processed", "event_id": event_id}

    # Route to handler
    result = {"status": "processed", "event_id": event_id, "event_type": event_type}

    try:
        if event_type == "checkout.session.completed":
            await _handle_checkout_completed(event)
        elif event_type == "customer.subscription.created":
            await _handle_subscription_created(event)
        elif event_type == "customer.subscription.updated":
            await _handle_subscription_updated(event)
        elif event_type == "customer.subscription.deleted":
            await _handle_subscription_deleted(event)
        elif event_type == "invoice.payment_succeeded":
            await _handle_payment_succeeded(event)
        elif event_type == "invoice.payment_failed":
            await _handle_payment_failed(event)

        # Record the event for idempotency
        payment_event = PaymentEventInDB(
            stripe_event_id=event_id,
            event_type=event_type,
            stripe_customer_id=_safe_get_customer_id(event),
            status=PaymentStatus.SUCCEEDED,
        )
        await payment_event_repository.record_event(payment_event)

    except Exception as e:
        logger.error(
            "Webhook processing error",
            event_id=event_id,
            event_type=event_type,
            error=str(e),
        )
        # Record failed event for debugging
        payment_event = PaymentEventInDB(
            stripe_event_id=event_id,
            event_type=event_type,
            stripe_customer_id=_safe_get_customer_id(event),
            status=PaymentStatus.FAILED,
        )
        await payment_event_repository.record_event(payment_event)
        raise

    return result


def _safe_get_customer_id(event: stripe.Event) -> Optional[str]:
    """Safely extract customer ID from any Stripe event."""
    obj = event.data.object
    if hasattr(obj, "customer"):
        cid = obj.customer
        return cid if isinstance(cid, str) else getattr(cid, "id", None)
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Notification Helper
# ─────────────────────────────────────────────────────────────────────────────

async def _notify_billing_event(
    stripe_customer_id: str,
    event: str,
    details: dict = None,
) -> None:
    """Resolve user from Stripe customer ID and fire billing notification."""
    try:
        sub = await subscription_repository.get_by_stripe_customer_id(stripe_customer_id)
        if not sub:
            logger.warning("Cannot notify — no subscription for customer", customer_id=stripe_customer_id)
            return
        # Get user email from Stripe customer object
        customer = stripe.Customer.retrieve(stripe_customer_id)
        email = customer.get("email", "")
        name = customer.get("name", "there")
        await notification_service.send_billing_notification(
            user_id=sub.user_id,
            email=email,
            event=event,
            details={"name": name, **(details or {})},
        )
    except Exception as e:
        logger.error("Billing notification failed", customer_id=stripe_customer_id, error=str(e))


# ─────────────────────────────────────────────────────────────────────────────
# Individual Webhook Handlers
# ─────────────────────────────────────────────────────────────────────────────

async def _handle_checkout_completed(event: stripe.Event):
    """Handle checkout.session.completed — user completed payment."""
    session = event.data.object
    customer_id = session.customer
    subscription_id = session.subscription
    user_id = session.metadata.get("lumicoria_user_id")

    if not subscription_id:
        logger.info("Checkout completed without subscription (one-time?)", session_id=session.id)
        return

    # Fetch the subscription from Stripe to get plan details
    stripe_sub = stripe.Subscription.retrieve(subscription_id)
    price_id = stripe_sub["items"]["data"][0]["price"]["id"] if stripe_sub["items"]["data"] else None
    plan = _resolve_plan_from_price(price_id) if price_id else SubscriptionPlan.FREE

    # Update our database
    await subscription_repository.update_from_stripe(
        stripe_customer_id=customer_id,
        update_data={
            "stripe_subscription_id": subscription_id,
            "stripe_price_id": price_id,
            "plan": plan,
            "status": SubscriptionStatus(stripe_sub.status),
            "current_period_start": datetime.utcfromtimestamp(stripe_sub.current_period_start),
            "current_period_end": datetime.utcfromtimestamp(stripe_sub.current_period_end),
            "trial_end": (
                datetime.utcfromtimestamp(stripe_sub.trial_end) if stripe_sub.trial_end else None
            ),
        },
    )
    logger.info(
        "Checkout completed — subscription activated",
        user_id=user_id,
        plan=plan,
        subscription_id=subscription_id,
    )
    # Fire-and-forget notification
    asyncio.ensure_future(_notify_billing_event(
        customer_id, "checkout_completed", {"plan": plan.value}
    ))


async def _handle_subscription_created(event: stripe.Event):
    """Handle customer.subscription.created."""
    subscription = event.data.object
    customer_id = subscription.customer
    price_id = subscription["items"]["data"][0]["price"]["id"] if subscription["items"]["data"] else None
    plan = _resolve_plan_from_price(price_id) if price_id else SubscriptionPlan.FREE

    await subscription_repository.update_from_stripe(
        stripe_customer_id=customer_id,
        update_data={
            "stripe_subscription_id": subscription.id,
            "stripe_price_id": price_id,
            "plan": plan,
            "status": SubscriptionStatus(subscription.status),
            "current_period_start": datetime.utcfromtimestamp(subscription.current_period_start),
            "current_period_end": datetime.utcfromtimestamp(subscription.current_period_end),
            "cancel_at_period_end": subscription.cancel_at_period_end,
            "trial_start": (
                datetime.utcfromtimestamp(subscription.trial_start) if subscription.trial_start else None
            ),
            "trial_end": (
                datetime.utcfromtimestamp(subscription.trial_end) if subscription.trial_end else None
            ),
        },
    )
    logger.info("Subscription created", customer_id=customer_id, plan=plan)


async def _handle_subscription_updated(event: stripe.Event):
    """Handle customer.subscription.updated — upgrade, downgrade, cancel, resume."""
    subscription = event.data.object
    customer_id = subscription.customer
    price_id = subscription["items"]["data"][0]["price"]["id"] if subscription["items"]["data"] else None
    plan = _resolve_plan_from_price(price_id) if price_id else SubscriptionPlan.FREE

    update_data = {
        "stripe_price_id": price_id,
        "plan": plan,
        "status": SubscriptionStatus(subscription.status),
        "cancel_at_period_end": subscription.cancel_at_period_end,
        "current_period_start": datetime.utcfromtimestamp(subscription.current_period_start),
        "current_period_end": datetime.utcfromtimestamp(subscription.current_period_end),
    }

    if subscription.canceled_at:
        update_data["canceled_at"] = datetime.utcfromtimestamp(subscription.canceled_at)
    if subscription.ended_at:
        update_data["ended_at"] = datetime.utcfromtimestamp(subscription.ended_at)
    if subscription.trial_end:
        update_data["trial_end"] = datetime.utcfromtimestamp(subscription.trial_end)

    await subscription_repository.update_from_stripe(
        stripe_customer_id=customer_id,
        update_data=update_data,
    )
    logger.info(
        "Subscription updated",
        customer_id=customer_id,
        plan=plan,
        status=subscription.status,
        cancel_at_period_end=subscription.cancel_at_period_end,
    )
    # Fire-and-forget notification
    asyncio.ensure_future(_notify_billing_event(
        customer_id, "subscription_updated", {"plan": plan.value}
    ))


async def _handle_subscription_deleted(event: stripe.Event):
    """Handle customer.subscription.deleted — subscription ended."""
    subscription = event.data.object
    customer_id = subscription.customer

    await subscription_repository.update_from_stripe(
        stripe_customer_id=customer_id,
        update_data={
            "plan": SubscriptionPlan.FREE,
            "status": SubscriptionStatus.CANCELED,
            "stripe_subscription_id": None,
            "stripe_price_id": None,
            "ended_at": datetime.utcnow(),
        },
    )
    logger.info("Subscription deleted — reverted to FREE", customer_id=customer_id)
    # Fire-and-forget notification
    asyncio.ensure_future(_notify_billing_event(
        customer_id, "subscription_deleted"
    ))


async def _handle_payment_succeeded(event: stripe.Event):
    """Handle invoice.payment_succeeded — record successful payment, save invoice, grant credits."""
    invoice = event.data.object
    customer_id = invoice.customer
    
    # Get user_id from subscription or customer metadata
    user_id = None
    sub = await subscription_repository.get_by_stripe_customer_id(customer_id)
    if sub:
        user_id = sub.user_id

    # Save invoice to database
    if user_id:
        from backend.db.mongodb.repositories.billing_repository import get_invoice_repository
        invoice_repo = await get_invoice_repository()
        await invoice_repo.create_or_update_from_stripe(user_id, invoice)
        
        # Grant credits for payment
        from backend.db.mongodb.repositories.billing_repository import get_credits_repository
        credits_repo = await get_credits_repository()
        await grant_credits_for_payment(user_id, credits_repo, invoice.amount_paid, invoice.id)

    # Update subscription period if this is a subscription invoice
    if invoice.subscription:
        stripe_sub = stripe.Subscription.retrieve(invoice.subscription)
        await subscription_repository.update_from_stripe(
            stripe_customer_id=customer_id,
            update_data={
                "status": SubscriptionStatus.ACTIVE,
                "current_period_start": datetime.utcfromtimestamp(stripe_sub.current_period_start),
                "current_period_end": datetime.utcfromtimestamp(stripe_sub.current_period_end),
            },
        )
    logger.info(
        "Payment succeeded",
        customer_id=customer_id,
        user_id=user_id,
        amount=invoice.amount_paid,
        currency=invoice.currency,
    )
    # Fire-and-forget notification
    amount_str = f"${invoice.amount_paid / 100:.2f} {invoice.currency.upper()}" if invoice.amount_paid else "your invoice"
    asyncio.ensure_future(_notify_billing_event(
        customer_id, "payment_succeeded", {"amount": amount_str}
    ))


async def _handle_payment_failed(event: stripe.Event):
    """Handle invoice.payment_failed — flag subscription as past_due."""
    invoice = event.data.object
    customer_id = invoice.customer

    if invoice.subscription:
        await subscription_repository.update_from_stripe(
            stripe_customer_id=customer_id,
            update_data={
                "status": SubscriptionStatus.PAST_DUE,
            },
        )
    logger.warning(
        "Payment failed",
        customer_id=customer_id,
        amount=invoice.amount_due,
    )
    # Fire-and-forget notification (urgent — payment failed)
    asyncio.ensure_future(_notify_billing_event(
        customer_id, "payment_failed", {"amount": f"${invoice.amount_due / 100:.2f}" if invoice.amount_due else "N/A"}
    ))


# ─────────────────────────────────────────────────────────────────────────────
# Subscription & Usage Queries
# ─────────────────────────────────────────────────────────────────────────────

async def get_user_subscription(user_id: str) -> SubscriptionResponse:
    """Get the user's current subscription status."""
    sub = await subscription_repository.get_by_user_id(user_id)

    if not sub:
        return SubscriptionResponse(
            plan=SubscriptionPlan.FREE,
            status=SubscriptionStatus.ACTIVE,
            is_active=True,
        )

    # Check admin override
    effective_plan = sub.plan
    if sub.is_admin_override and sub.admin_override_plan:
        if sub.admin_override_expires is None or sub.admin_override_expires > datetime.utcnow():
            effective_plan = sub.admin_override_plan

    is_active = sub.status in (
        SubscriptionStatus.ACTIVE,
        SubscriptionStatus.TRIALING,
    )

    # Grace period: allow 3 days past_due before cutting off
    if sub.status == SubscriptionStatus.PAST_DUE:
        if sub.current_period_end and (datetime.utcnow() - sub.current_period_end) < timedelta(days=3):
            is_active = True

    return SubscriptionResponse(
        plan=effective_plan,
        status=sub.status,
        cancel_at_period_end=sub.cancel_at_period_end,
        current_period_start=sub.current_period_start,
        current_period_end=sub.current_period_end,
        trial_end=sub.trial_end,
        is_active=is_active,
    )


async def get_user_effective_plan(user_id: str) -> SubscriptionPlan:
    """Get the user's effective plan (considering admin overrides)."""
    sub_response = await get_user_subscription(user_id)
    return sub_response.plan


async def get_user_usage(user_id: str) -> UsageResponse:
    """Get the user's current usage against their plan limits."""
    sub = await get_user_subscription(user_id)
    limits = get_plan_limits(sub.plan)
    usage = await usage_tracking_repository.get_current_usage(user_id)

    now = datetime.utcnow()
    agent_runs = usage.agent_runs if usage else 0
    docs = usage.documents_processed if usage else 0
    kb_queries = usage.knowledge_base_queries if usage else 0

    max_runs = limits["max_agent_runs_per_month"]
    usage_pct = (agent_runs / max_runs * 100) if max_runs > 0 else 0.0

    return UsageResponse(
        plan=sub.plan,
        month=now.month,
        year=now.year,
        agent_runs=agent_runs,
        agent_runs_limit=max_runs,
        documents_processed=docs,
        documents_limit=limits["max_documents_per_month"],
        knowledge_base_queries=kb_queries,
        knowledge_base_queries_limit=limits["max_knowledge_base_queries"],
        usage_percent=min(usage_pct, 100.0),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Feature Gating — called BEFORE agent execution
# ─────────────────────────────────────────────────────────────────────────────

async def require_active_subscription(user_id: str) -> SubscriptionResponse:
    """
    Verify the user has an active subscription (including FREE).
    Raises ValueError if subscription is inactive.
    """
    sub = await get_user_subscription(user_id)
    if not sub.is_active:
        raise ValueError(
            f"Subscription is {sub.status}. Please update your billing information."
        )
    return sub


async def require_plan(
    user_id: str,
    minimum_plan: SubscriptionPlan,
) -> SubscriptionResponse:
    """
    Verify the user's plan meets the minimum required.
    Plan hierarchy: FREE < STARTER < PROFESSIONAL < ENTERPRISE
    """
    plan_order = {
        SubscriptionPlan.FREE: 0,
        SubscriptionPlan.STARTER: 1,
        SubscriptionPlan.PROFESSIONAL: 2,
        SubscriptionPlan.ENTERPRISE: 3,
    }
    sub = await require_active_subscription(user_id)
    if plan_order.get(sub.plan, 0) < plan_order.get(minimum_plan, 0):
        raise ValueError(
            f"This feature requires the {minimum_plan.value} plan or higher. "
            f"You are currently on the {sub.plan.value} plan."
        )
    return sub


async def check_usage_limits(
    user_id: str,
    usage_type: str = "agent_runs",
) -> Tuple[bool, int, int]:
    """
    Check if user is within their plan's usage limits.
    
    Returns:
        (allowed, current_usage, limit)
    """
    sub = await get_user_subscription(user_id)
    limits = get_plan_limits(sub.plan)
    usage = await usage_tracking_repository.get_current_usage(user_id)

    limit_map = {
        "agent_runs": ("agent_runs", "max_agent_runs_per_month"),
        "documents": ("documents_processed", "max_documents_per_month"),
        "kb_queries": ("knowledge_base_queries", "max_knowledge_base_queries"),
    }

    usage_field, limit_field = limit_map.get(usage_type, ("agent_runs", "max_agent_runs_per_month"))
    current = getattr(usage, usage_field, 0) if usage else 0
    limit = limits.get(limit_field, 0)

    # -1 means unlimited (Enterprise)
    if limit == -1:
        return (True, current, -1)

    return (current < limit, current, limit)


async def enforce_agent_run_limit(user_id: str, agent_type: str = "unknown"):
    """
    Enforce agent run limit BEFORE execution.
    Raises ValueError if limit exceeded.
    """
    allowed, current, limit = await check_usage_limits(user_id, "agent_runs")
    if not allowed:
        raise ValueError(
            f"Monthly agent run limit reached ({current}/{limit}). "
            f"Upgrade your plan for more runs."
        )

    # Track the usage (atomic increment)
    await usage_tracking_repository.increment_agent_runs(user_id, agent_type)


async def enforce_document_limit(user_id: str):
    """Enforce document processing limit."""
    allowed, current, limit = await check_usage_limits(user_id, "documents")
    if not allowed:
        raise ValueError(
            f"Monthly document limit reached ({current}/{limit}). "
            f"Upgrade your plan to process more documents."
        )
    await usage_tracking_repository.increment_documents(user_id)


async def enforce_kb_query_limit(user_id: str):
    """Enforce knowledge base query limit."""
    allowed, current, limit = await check_usage_limits(user_id, "kb_queries")
    if not allowed:
        raise ValueError(
            f"Monthly knowledge base query limit reached ({current}/{limit}). "
            f"Upgrade your plan for more queries."
        )
    await usage_tracking_repository.increment_kb_queries(user_id)


async def check_model_access(user_id: str, model_provider: str) -> bool:
    """Check if user's plan allows access to a specific model provider."""
    sub = await get_user_subscription(user_id)
    limits = get_plan_limits(sub.plan)
    allowed_models = limits.get("allowed_models", ["default"])
    return model_provider in allowed_models or "default" in allowed_models


# ─────────────────────────────────────────────────────────────────────────────
# Invoice & Receipt Export
# ─────────────────────────────────────────────────────────────────────────────

async def get_user_invoices(
    user_id: str,
    invoice_repo: InvoiceRepository,
    limit: int = 50,
    skip: int = 0,
    status: Optional[str] = None,
):
    """
    Get user's invoices with pagination.
    
    Args:
        user_id: User ID
        invoice_repo: Invoice repository instance
        limit: Max invoices to return
        skip: Skip count for pagination
        status: Filter by status (paid, open, etc.)
        
    Returns:
        InvoiceListResponse with paginated invoices
    """
    return await invoice_repo.get_user_invoices(user_id, limit, skip, status)


async def get_invoice_pdf(
    user_id: str,
    invoice_id: str,
    invoice_repo: InvoiceRepository,
) -> Optional[str]:
    """
    Get Stripe-hosted PDF URL for an invoice.
    
    Args:
        user_id: User ID (for authorization check)
        invoice_id: Stripe invoice ID
        invoice_repo: Invoice repository instance
        
    Returns:
        PDF URL if invoice exists and belongs to user, None otherwise
    """
    invoice = await invoice_repo.get_by_stripe_invoice_id(invoice_id)
    
    if not invoice or invoice.user_id != user_id:
        logger.warning("Invoice access denied", user_id=user_id, invoice_id=invoice_id)
        return None
    
    # If we have cached PDF URL, return it
    if invoice.invoice_pdf_url:
        return invoice.invoice_pdf_url
    
    # Otherwise, fetch from Stripe
    try:
        stripe_invoice = stripe.Invoice.retrieve(invoice_id)
        pdf_url = stripe_invoice.get("invoice_pdf")
        
        # Update our record
        if pdf_url:
            await invoice_repo.collection.update_one(
                {"stripe_invoice_id": invoice_id},
                {"$set": {"invoice_pdf_url": pdf_url, "updated_at": datetime.utcnow()}}
            )
        
        return pdf_url
    except stripe.error.StripeError as e:
        logger.error("Failed to fetch invoice PDF from Stripe", error=str(e))
        return None


async def export_invoice(
    user_id: str,
    invoice_id: str,
    invoice_repo: InvoiceRepository,
) -> Optional[Dict[str, Any]]:
    """
    Export invoice data for download (JSON format).
    
    Args:
        user_id: User ID
        invoice_id: Stripe invoice ID
        invoice_repo: Invoice repository instance
        
    Returns:
        Invoice data dict or None if not found/authorized
    """
    invoice = await invoice_repo.get_by_stripe_invoice_id(invoice_id)
    
    if not invoice or invoice.user_id != user_id:
        logger.warning("Invoice export denied", user_id=user_id, invoice_id=invoice_id)
        return None
    
    return {
        "invoice_id": invoice.stripe_invoice_id,
        "invoice_number": invoice.invoice_number,
        "customer_id": invoice.stripe_customer_id,
        "amount_due": invoice.amount_due / 100,  # Convert cents to dollars
        "amount_paid": invoice.amount_paid / 100,
        "currency": invoice.currency.upper(),
        "status": invoice.status,
        "invoice_date": invoice.invoice_date.isoformat(),
        "paid_at": invoice.paid_at.isoformat() if invoice.paid_at else None,
        "line_items": invoice.line_items,
        "pdf_url": invoice.invoice_pdf_url,
        "hosted_url": invoice.hosted_invoice_url,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Credits Ledger
# ─────────────────────────────────────────────────────────────────────────────

async def get_credit_balance(user_id: str, credits_repo: CreditLedgerRepository) -> int:
    """Get user's current credit balance."""
    return await credits_repo.get_balance(user_id)


async def get_credit_ledger(
    user_id: str,
    credits_repo: CreditLedgerRepository,
    limit: int = 50,
    skip: int = 0,
):
    """Get user's credit transaction history."""
    return await credits_repo.get_ledger(user_id, limit, skip)


async def grant_credits_for_payment(
    user_id: str,
    credits_repo: CreditLedgerRepository,
    amount_paid: int,
    stripe_invoice_id: str,
):
    """
    Grant credits when user makes a payment.
    Conversion rate: $1 = 100 credits
    """
    from backend.models.billing import TransactionType
    
    credits_amount = amount_paid  # amount_paid is already in cents = credits
    
    await credits_repo.grant_credits(
        user_id=user_id,
        amount=credits_amount,
        description=f"Payment received (Invoice: {stripe_invoice_id})",
        transaction_type=TransactionType.CREDIT,
        metadata={"stripe_invoice_id": stripe_invoice_id},
    )
    
    logger.info(
        "Credits granted for payment",
        user_id=user_id,
        credits=credits_amount,
        invoice_id=stripe_invoice_id,
    )

