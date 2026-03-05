"""
Lumicoria AI — Billing API Endpoints

Production-grade REST endpoints for Stripe billing:
  • POST /billing/checkout          — Create Checkout session
  • POST /billing/portal            — Create Customer Portal session
  • POST /billing/webhook           — Stripe webhook receiver
  • GET  /billing/subscription      — Get current subscription
  • GET  /billing/usage             — Get current usage & limits
  • GET  /billing/plans             — Get available plans (public)
  • POST /billing/admin/override    — Admin plan override

SECURITY:
  - All endpoints (except webhook + plans) require authentication
  - Webhook uses raw body + Stripe signature verification
  - Rate limiting on checkout endpoint
  - No subscription state trusted from client
"""

from fastapi import APIRouter, Request, HTTPException, Depends, status
from fastapi.responses import JSONResponse
from backend.core.security import verify_token, rate_limit
from backend.models.billing import (
    CreateCheckoutRequest,
    CreateCheckoutResponse,
    CustomerPortalRequest,
    CustomerPortalResponse,
    SubscriptionResponse,
    UsageResponse,
    PlanInfoResponse,
    AdminOverrideRequest,
    SubscriptionPlan,
    PLAN_LIMITS,
    get_plan_limits,
    InvoiceListResponse,
    InvoiceResponse,
    CreditBalanceResponse,
    CreditLedgerResponse,
)
from backend.services import billing_service
import stripe
import structlog

logger = structlog.get_logger(__name__)

router = APIRouter()


# ─────────────────────────────────────────────────────────────────────────────
# Create Checkout Session
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/checkout",
    response_model=CreateCheckoutResponse,
    summary="Create Stripe Checkout session",
)
async def create_checkout(
    request: CreateCheckoutRequest,
    token_data: dict = Depends(verify_token),
):
    """
    Create a Stripe Checkout session for subscription purchase.
    Redirects the user to Stripe's hosted checkout page.
    """
    user_id = token_data.get("user_id") or token_data.get("uid")
    email = token_data.get("email")

    if not user_id or not email:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        result = await billing_service.create_checkout_session(
            user_id=user_id,
            email=email,
            name=None,
            price_id=request.price_id,
            success_url=request.success_url,
            cancel_url=request.cancel_url,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except stripe.error.StripeError as e:
        logger.error("Stripe error creating checkout", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Payment service unavailable. Please try again.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Customer Portal
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/portal",
    response_model=CustomerPortalResponse,
    summary="Create Stripe Customer Portal session",
)
async def create_portal(
    request: CustomerPortalRequest,
    token_data: dict = Depends(verify_token),
):
    """
    Create a Stripe Customer Portal session for self-service billing.
    Users can update payment methods, view invoices, and cancel subscriptions.
    """
    user_id = token_data.get("user_id") or token_data.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        result = await billing_service.create_customer_portal_session(
            user_id=user_id,
            return_url=request.return_url,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))
    except stripe.error.StripeError as e:
        logger.error("Stripe error creating portal", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Payment service unavailable.",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Stripe Webhook
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/webhook",
    summary="Stripe webhook receiver",
    include_in_schema=False,  # Hide from OpenAPI docs
)
async def stripe_webhook(request: Request):
    """
    Receive and process Stripe webhook events.
    
    SECURITY:
      - Raw body used for signature verification (not parsed JSON)
      - Stripe-Signature header verified against STRIPE_WEBHOOK_SECRET
      - Each event processed exactly once (idempotent)
      - Returns 200 even on processing errors to prevent Stripe retries
        for events we've already acknowledged
    """
    # Get raw body for signature verification
    payload = await request.body()
    sig_header = request.headers.get("Stripe-Signature")

    if not sig_header:
        logger.warning("Webhook request missing Stripe-Signature header")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Missing Stripe-Signature header",
        )

    # Verify signature
    try:
        event = billing_service.verify_webhook_signature(payload, sig_header)
    except stripe.error.SignatureVerificationError as e:
        logger.warning("Webhook signature verification failed", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid signature",
        )

    # Process the event
    try:
        result = await billing_service.process_webhook_event(event)
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        logger.error(
            "Webhook processing error",
            event_id=event.id,
            event_type=event.type,
            error=str(e),
        )
        # Return 200 to acknowledge receipt — Stripe will not retry
        # The error is logged and the event is marked as failed in our DB
        return JSONResponse(
            status_code=200,
            content={"status": "error", "message": "Processing failed"},
        )


# ─────────────────────────────────────────────────────────────────────────────
# Subscription Status
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/subscription",
    response_model=SubscriptionResponse,
    summary="Get current subscription",
)
async def get_subscription(
    token_data: dict = Depends(verify_token),
):
    """Get the authenticated user's current subscription status."""
    user_id = token_data.get("user_id") or token_data.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    return await billing_service.get_user_subscription(user_id)


# ─────────────────────────────────────────────────────────────────────────────
# Usage Dashboard
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/usage",
    response_model=UsageResponse,
    summary="Get current usage & limits",
)
async def get_usage(
    token_data: dict = Depends(verify_token),
):
    """Get the authenticated user's current usage against their plan limits."""
    user_id = token_data.get("user_id") or token_data.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    return await billing_service.get_user_usage(user_id)


# ─────────────────────────────────────────────────────────────────────────────
# Public Plans
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/plans",
    response_model=list[PlanInfoResponse],
    summary="Get available plans",
)
async def get_plans():
    """Get all available subscription plans (public — no auth required)."""
    plans = []
    for plan_key, limits in PLAN_LIMITS.items():
        plans.append(PlanInfoResponse(
            plan=plan_key,
            display_name=limits["display_name"],
            price_monthly=limits["price_monthly"],
            max_agents=limits["max_agents"],
            max_agent_runs_per_month=limits["max_agent_runs_per_month"],
            max_documents_per_month=limits["max_documents_per_month"],
            max_file_upload_mb=limits["max_file_upload_mb"],
            max_knowledge_base_queries=limits["max_knowledge_base_queries"],
            advanced_features=limits["advanced_features"],
            priority_support=limits["priority_support"],
            api_access=limits["api_access"],
            custom_agent_templates=limits["custom_agent_templates"],
        ))
    return plans


# ─────────────────────────────────────────────────────────────────────────────
# Admin Override
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/admin/override",
    response_model=SubscriptionResponse,
    summary="Admin: override user plan",
)
async def admin_override_plan(
    request: AdminOverrideRequest,
    token_data: dict = Depends(verify_token),
):
    """
    Admin endpoint to override a user's subscription plan.
    Requires is_superuser or admin role.
    """
    # Check admin permissions
    is_admin = token_data.get("is_admin") or token_data.get("is_superuser")
    if not is_admin:
        # Check from DB if needed
        from backend.db.mongodb.repositories.user_repository import user_repository
        admin_user_id = token_data.get("user_id") or token_data.get("uid")
        admin_user = await user_repository.get_user_by_id(admin_user_id)
        if not admin_user or not getattr(admin_user, "is_superuser", False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Admin access required",
            )

    from backend.db.mongodb.repositories.billing_repository import subscription_repository
    await subscription_repository.set_admin_override(
        user_id=request.user_id,
        plan=request.plan,
        expires_at=request.expires_at,
    )

    return await billing_service.get_user_subscription(request.user_id)


# ─────────────────────────────────────────────────────────────────────────────
# Invoice & Receipt Export
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/invoices",
    response_model=InvoiceListResponse,
    summary="Get user invoices",
)
async def get_invoices(
    token_data: dict = Depends(verify_token),
    limit: int = 50,
    skip: int = 0,
    status: str = None,
):
    """
    Get user's invoice history with pagination.
    Invoices are synced from Stripe webhooks.
    """
    user_id = token_data.get("user_id") or token_data.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        from backend.db.mongodb.repositories.billing_repository import get_invoice_repository
        invoice_repo = await get_invoice_repository()
        return await billing_service.get_user_invoices(
            user_id=user_id,
            invoice_repo=invoice_repo,
            limit=limit,
            skip=skip,
            status=status,
        )
    except Exception as e:
        logger.error("Failed to fetch invoices", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch invoices",
        )


@router.get(
    "/invoices/{invoice_id}/pdf",
    summary="Get invoice PDF URL",
)
async def get_invoice_pdf(
    invoice_id: str,
    token_data: dict = Depends(verify_token),
):
    """
    Get Stripe-hosted PDF URL for an invoice.
    Returns redirect URL or direct PDF link.
    """
    user_id = token_data.get("user_id") or token_data.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        from backend.db.mongodb.repositories.billing_repository import get_invoice_repository
        invoice_repo = await get_invoice_repository()
        pdf_url = await billing_service.get_invoice_pdf(user_id, invoice_id, invoice_repo)
        
        if not pdf_url:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invoice not found or access denied",
            )
        
        return {"pdf_url": pdf_url}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to fetch invoice PDF", invoice_id=invoice_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch invoice PDF",
        )


@router.get(
    "/invoices/{invoice_id}/export",
    summary="Export invoice data",
)
async def export_invoice(
    invoice_id: str,
    token_data: dict = Depends(verify_token),
):
    """
    Export invoice data as JSON for download.
    Useful for accounting integrations.
    """
    user_id = token_data.get("user_id") or token_data.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        from backend.db.mongodb.repositories.billing_repository import get_invoice_repository
        invoice_repo = await get_invoice_repository()
        invoice_data = await billing_service.export_invoice(user_id, invoice_id, invoice_repo)
        
        if not invoice_data:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Invoice not found or access denied",
            )
        
        return invoice_data
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Failed to export invoice", invoice_id=invoice_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to export invoice",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Credits Ledger
# ─────────────────────────────────────────────────────────────────────────────

@router.get(
    "/credits/balance",
    response_model=CreditBalanceResponse,
    summary="Get credit balance",
)
async def get_credit_balance(
    token_data: dict = Depends(verify_token),
):
    """
    Get user's current credit balance.
    Credits are earned through payments and spent on usage.
    """
    user_id = token_data.get("user_id") or token_data.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        from backend.db.mongodb.repositories.billing_repository import get_credits_repository
        credits_repo = await get_credits_repository()
        balance = await billing_service.get_credit_balance(user_id, credits_repo)
        
        return CreditBalanceResponse(
            user_id=user_id,
            balance=balance,
            currency="credits",
        )
    except Exception as e:
        logger.error("Failed to fetch credit balance", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch credit balance",
        )


@router.get(
    "/credits/ledger",
    response_model=CreditLedgerResponse,
    summary="Get credit transaction history",
)
async def get_credit_ledger(
    token_data: dict = Depends(verify_token),
    limit: int = 50,
    skip: int = 0,
):
    """
    Get user's credit transaction history with pagination.
    Shows all credits earned and spent over time.
    """
    user_id = token_data.get("user_id") or token_data.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        from backend.db.mongodb.repositories.billing_repository import get_credits_repository
        credits_repo = await get_credits_repository()
        return await billing_service.get_credit_ledger(
            user_id=user_id,
            credits_repo=credits_repo,
            limit=limit,
            skip=skip,
        )
    except Exception as e:
        logger.error("Failed to fetch credit ledger", user_id=user_id, error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to fetch credit ledger",
        )
