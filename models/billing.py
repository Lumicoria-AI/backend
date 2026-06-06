"""
Lumicoria AI — Billing & Subscription Models

Production-grade Pydantic models for Stripe billing integration.
All subscription state is SERVER-AUTHORITATIVE — never trust the client.
"""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, ConfigDict
from enum import Enum


# ─────────────────────────────────────────────────────────────────────────────
# Enums
# ─────────────────────────────────────────────────────────────────────────────

class SubscriptionPlan(str, Enum):
    """Available subscription plans — mapped 1:1 to Stripe Price IDs.

    Individual plans (user-scoped): FREE, STARTER, PROFESSIONAL.
    Team plans (org-scoped):        TEAM, BUSINESS, ENTERPRISE.
    """
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
    TEAM = "team"
    BUSINESS = "business"
    ENTERPRISE = "enterprise"


class SubscriptionStatus(str, Enum):
    """Stripe subscription lifecycle states."""
    ACTIVE = "active"
    PAST_DUE = "past_due"
    CANCELED = "canceled"
    INCOMPLETE = "incomplete"
    INCOMPLETE_EXPIRED = "incomplete_expired"
    TRIALING = "trialing"
    UNPAID = "unpaid"
    PAUSED = "paused"


class PaymentStatus(str, Enum):
    """Invoice payment states."""
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    PENDING = "pending"
    REFUNDED = "refunded"


# ─────────────────────────────────────────────────────────────────────────────
# Plan Limits — the SINGLE SOURCE OF TRUTH for feature gating
# ─────────────────────────────────────────────────────────────────────────────

PLAN_LIMITS: Dict[str, Dict[str, Any]] = {
    SubscriptionPlan.FREE: {
        "display_name": "Free",
        "price_monthly": 0,
        "max_agents": 2,
        "max_agent_runs_per_month": 50,
        "max_documents_per_month": 10,
        "max_file_upload_mb": 5,
        "max_knowledge_base_queries": 20,
        "allowed_models": ["default"],
        "advanced_features": False,
        "priority_support": False,
        "api_access": False,
        "custom_agent_templates": False,
    },
    SubscriptionPlan.STARTER: {
        "display_name": "Starter",
        "price_monthly": 29,
        "max_agents": 5,
        "max_agent_runs_per_month": 500,
        "max_documents_per_month": 100,
        "max_file_upload_mb": 10,
        "max_knowledge_base_queries": 200,
        "allowed_models": ["default", "perplexity", "gemini", "openai"],
        "advanced_features": False,
        "priority_support": False,
        "api_access": False,
        "custom_agent_templates": False,
    },
    SubscriptionPlan.PROFESSIONAL: {
        "display_name": "Professional",
        "price_monthly": 79,
        "max_agents": 15,
        "max_agent_runs_per_month": 5000,
        "max_documents_per_month": 1000,
        "max_file_upload_mb": 50,
        "max_knowledge_base_queries": 2000,
        "allowed_models": ["default", "perplexity", "gemini", "openai", "anthropic", "mistral"],
        "advanced_features": True,
        "priority_support": True,
        "api_access": True,
        "custom_agent_templates": True,
    },
    SubscriptionPlan.ENTERPRISE: {
        "display_name": "Enterprise",
        "price_monthly": None,  # Custom pricing
        "max_agents": -1,  # Unlimited
        "max_agent_runs_per_month": -1,
        "max_documents_per_month": -1,
        "max_file_upload_mb": 100,
        "max_knowledge_base_queries": -1,
        "allowed_models": ["default", "perplexity", "gemini", "openai", "anthropic", "mistral"],
        "advanced_features": True,
        "priority_support": True,
        "api_access": True,
        "custom_agent_templates": True,
        "teams_enabled": True,
        "sso_enabled": True,
        "scim_enabled": True,
        "strict_mode_enabled": True,
        "data_residency": True,
        "audit_export_enabled": True,
    },
    # ── Team plans (org-scoped, per-seat) ──
    SubscriptionPlan.TEAM: {
        "display_name": "Team",
        "price_monthly_per_seat": 39,
        "annual_discount_pct": 15,
        "max_agents": 25,
        "max_agent_runs_per_month": 15_000,
        "max_documents_per_month": 3_000,
        "max_file_upload_mb": 50,
        "max_knowledge_base_queries": 5_000,
        "allowed_models": ["default", "perplexity", "gemini", "openai", "anthropic", "mistral"],
        "advanced_features": True,
        "priority_support": False,
        "api_access": True,
        "custom_agent_templates": True,
        "teams_enabled": True,
        "sso_enabled": False,
        "scim_enabled": False,
        "strict_mode_enabled": False,
        "data_residency": False,
        "audit_export_enabled": False,
    },
    SubscriptionPlan.BUSINESS: {
        "display_name": "Business",
        "price_monthly_per_seat": 79,
        "annual_discount_pct": 15,
        "max_agents": 100,
        "max_agent_runs_per_month": 75_000,
        "max_documents_per_month": 15_000,
        "max_file_upload_mb": 100,
        "max_knowledge_base_queries": 25_000,
        "allowed_models": ["default", "perplexity", "gemini", "openai", "anthropic", "mistral"],
        "advanced_features": True,
        "priority_support": True,
        "api_access": True,
        "custom_agent_templates": True,
        "teams_enabled": True,
        "sso_enabled": True,
        "scim_enabled": False,
        "strict_mode_enabled": True,
        "data_residency": False,
        "audit_export_enabled": True,
    },
}


def get_plan_limits(plan: str) -> Dict[str, Any]:
    """Get the limits for a given plan. Defaults to FREE if plan unknown."""
    return PLAN_LIMITS.get(plan, PLAN_LIMITS[SubscriptionPlan.FREE])


# ─────────────────────────────────────────────────────────────────────────────
# Database Document Models (MongoDB)
# ─────────────────────────────────────────────────────────────────────────────

class SubscriptionInDB(BaseModel):
    """MongoDB document schema for subscriptions."""
    user_id: str
    stripe_customer_id: str
    stripe_subscription_id: Optional[str] = None
    stripe_price_id: Optional[str] = None
    plan: SubscriptionPlan = SubscriptionPlan.FREE
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    cancel_at_period_end: bool = False
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    trial_start: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    # Admin override — allows bypassing plan limits
    is_admin_override: bool = False
    admin_override_plan: Optional[SubscriptionPlan] = None
    admin_override_expires: Optional[datetime] = None
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class OrgSubscriptionInDB(BaseModel):
    """Org-scoped subscription (parallel to user-scoped `SubscriptionInDB`).

    Used for Team / Business / Enterprise plans where billing follows the
    organisation rather than a single user.  The user-scoped subscriptions
    remain in place for individual plans.
    """
    organization_id: str
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    stripe_price_id: Optional[str] = None
    plan: SubscriptionPlan = SubscriptionPlan.TEAM
    status: SubscriptionStatus = SubscriptionStatus.ACTIVE
    cadence: str = "monthly"             # "monthly" | "annual"
    seats_purchased: int = 1
    seats_used: int = 1
    billing_email: Optional[str] = None
    tax_id: Optional[str] = None
    po_number: Optional[str] = None
    cancel_at_period_end: bool = False
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    trial_start: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    canceled_at: Optional[datetime] = None
    ended_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class SeatAssignmentInDB(BaseModel):
    """One active seat assignment for a user inside an organisation."""
    organization_id: str
    user_id: str
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    assigned_by: Optional[str] = None
    removed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(from_attributes=True)


class UsageTrackingInDB(BaseModel):
    """MongoDB document schema for monthly usage tracking."""
    user_id: str
    month: int  # 1-12
    year: int
    agent_runs: int = 0
    documents_processed: int = 0
    knowledge_base_queries: int = 0
    file_uploads: int = 0
    # Per-agent breakdown
    agent_usage_breakdown: Dict[str, int] = Field(default_factory=dict)
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


class PaymentEventInDB(BaseModel):
    """MongoDB document for idempotent webhook event tracking."""
    stripe_event_id: str  # Unique — used for idempotency
    event_type: str
    user_id: Optional[str] = None
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    stripe_invoice_id: Optional[str] = None
    amount: Optional[int] = None  # cents
    currency: Optional[str] = None
    status: PaymentStatus = PaymentStatus.PENDING
    raw_event: Optional[Dict[str, Any]] = None  # Stripped of secrets
    processed_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(from_attributes=True)


class TransactionType(str, Enum):
    """Credit transaction types."""
    CREDIT = "credit"  # Adding credits
    DEBIT = "debit"    # Spending credits
    REFUND = "refund"  # Refunding credits
    ADJUSTMENT = "adjustment"  # Admin adjustment
    BONUS = "bonus"    # Promotional credits


class CreditLedgerInDB(BaseModel):
    """MongoDB document for credits transaction ledger."""
    user_id: str
    transaction_type: TransactionType
    amount: int = Field(..., description="Amount in credits (positive for credit, negative for debit)")
    balance_after: int = Field(..., description="User's credit balance after this transaction")
    description: str = Field(..., description="Human-readable description of the transaction")
    # Associated entities
    stripe_invoice_id: Optional[str] = None
    stripe_payment_intent_id: Optional[str] = None
    agent_run_id: Optional[str] = None
    document_id: Optional[str] = None
    # Metadata
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)
    created_by: Optional[str] = Field(None, description="User ID who created this transaction (for admin adjustments)")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    model_config = ConfigDict(from_attributes=True)


class InvoiceInDB(BaseModel):
    """MongoDB document for invoice/receipt storage."""
    user_id: str
    stripe_invoice_id: str
    stripe_customer_id: str
    stripe_subscription_id: Optional[str] = None
    # Invoice details
    invoice_number: Optional[str] = None
    invoice_pdf_url: Optional[str] = None  # Stripe-hosted PDF
    hosted_invoice_url: Optional[str] = None  # Stripe-hosted invoice page
    amount_due: int = Field(..., description="Amount in cents")
    amount_paid: int = Field(..., description="Amount paid in cents")
    currency: str = "usd"
    status: str = Field(..., description="Invoice status (draft, open, paid, void, uncollectible)")
    # Dates
    invoice_date: datetime
    due_date: Optional[datetime] = None
    paid_at: Optional[datetime] = None
    # Line items (simplified)
    line_items: List[Dict[str, Any]] = Field(default_factory=list)
    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: Optional[datetime] = None

    model_config = ConfigDict(from_attributes=True)


# ─────────────────────────────────────────────────────────────────────────────
# API Request / Response Models
# ─────────────────────────────────────────────────────────────────────────────

class CreateCheckoutRequest(BaseModel):
    """Request body for creating a Stripe Checkout session."""
    price_id: str = Field(..., description="Stripe Price ID for the chosen plan")
    success_url: Optional[str] = None
    cancel_url: Optional[str] = None


class CreateCheckoutResponse(BaseModel):
    """Response with the Stripe Checkout session URL."""
    checkout_url: str
    session_id: str


class CustomerPortalRequest(BaseModel):
    """Request to open Stripe Customer Portal."""
    return_url: Optional[str] = None


class CustomerPortalResponse(BaseModel):
    """Response with Stripe Customer Portal URL."""
    portal_url: str


class SubscriptionResponse(BaseModel):
    """User-facing subscription data."""
    plan: SubscriptionPlan
    status: SubscriptionStatus
    cancel_at_period_end: bool = False
    current_period_start: Optional[datetime] = None
    current_period_end: Optional[datetime] = None
    trial_end: Optional[datetime] = None
    is_active: bool = False


class UsageResponse(BaseModel):
    """User-facing usage data."""
    plan: SubscriptionPlan
    month: int
    year: int
    agent_runs: int = 0
    agent_runs_limit: int = 0
    documents_processed: int = 0
    documents_limit: int = 0
    knowledge_base_queries: int = 0
    knowledge_base_queries_limit: int = 0
    usage_percent: float = 0.0


class PlanInfoResponse(BaseModel):
    """Public pricing tier information."""
    plan: SubscriptionPlan
    display_name: str
    price_monthly: Optional[int]
    max_agents: int
    max_agent_runs_per_month: int
    max_documents_per_month: int
    max_file_upload_mb: int
    max_knowledge_base_queries: int
    advanced_features: bool
    priority_support: bool
    api_access: bool
    custom_agent_templates: bool


class AdminOverrideRequest(BaseModel):
    """Admin request to override a user's plan."""
    user_id: str
    plan: SubscriptionPlan
    expires_at: Optional[datetime] = None
    reason: Optional[str] = None


class CreditBalanceResponse(BaseModel):
    """User's current credit balance."""
    user_id: str
    balance: int
    currency: str = "credits"


class CreditTransactionResponse(BaseModel):
    """Single credit transaction entry."""
    transaction_type: TransactionType
    amount: int
    balance_after: int
    description: str
    created_at: datetime
    metadata: Optional[Dict[str, Any]] = None


class CreditLedgerResponse(BaseModel):
    """Paginated credit transaction history."""
    balance: int
    transactions: List[CreditTransactionResponse]
    total_count: int
    page: int
    page_size: int


class InvoiceResponse(BaseModel):
    """Invoice/receipt data."""
    invoice_id: str
    invoice_number: Optional[str]
    amount_due: int
    amount_paid: int
    currency: str
    status: str
    invoice_date: datetime
    paid_at: Optional[datetime]
    invoice_pdf_url: Optional[str]
    hosted_invoice_url: Optional[str]
    line_items: List[Dict[str, Any]]


class InvoiceListResponse(BaseModel):
    """List of invoices."""
    invoices: List[InvoiceResponse]
    total_count: int

