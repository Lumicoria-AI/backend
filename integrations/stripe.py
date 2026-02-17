"""
Lumicoria AI — Stripe Integration Module

This module re-exports the billing service for use via the integrations package.
The actual implementation lives in backend/services/billing_service.py.

Usage:
    from backend.integrations.stripe import billing_service
    
    await billing_service.create_checkout_session(...)
"""

# Re-export the billing service
from backend.services.billing_service import (
    create_checkout_session,
    create_customer_portal_session,
    get_or_create_stripe_customer,
    get_user_subscription,
    get_user_usage,
    require_active_subscription,
    require_plan,
    check_usage_limits,
    enforce_agent_run_limit,
    enforce_document_limit,
    enforce_kb_query_limit,
    check_model_access,
    verify_webhook_signature,
    process_webhook_event,
)

__all__ = [
    "create_checkout_session",
    "create_customer_portal_session",
    "get_or_create_stripe_customer",
    "get_user_subscription",
    "get_user_usage",
    "require_active_subscription",
    "require_plan",
    "check_usage_limits",
    "enforce_agent_run_limit",
    "enforce_document_limit",
    "enforce_kb_query_limit",
    "check_model_access",
    "verify_webhook_signature",
    "process_webhook_event",
]
