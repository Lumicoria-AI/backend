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
    """Available subscription plans — mapped 1:1 to Stripe Price IDs."""
    FREE = "free"
    STARTER = "starter"
    PROFESSIONAL = "professional"
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
