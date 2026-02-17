"""
Lumicoria AI — Billing Middleware & Dependencies

FastAPI dependencies for subscription enforcement.
Import these into any endpoint that needs billing checks.

Usage:
    @router.post("/agents/execute")
    async def execute_agent(
        ...,
        billing: BillingCheck = Depends(require_subscription),
    ):
        # billing.plan, billing.user_id available
        ...
"""

from dataclasses import dataclass
from typing import Optional
from fastapi import Depends, HTTPException, status
from backend.core.security import verify_token
from backend.models.billing import SubscriptionPlan, SubscriptionResponse
from backend.services import billing_service
import structlog

logger = structlog.get_logger(__name__)


@dataclass
class BillingCheck:
    """Result of a billing check — injected into endpoints."""
    user_id: str
    plan: SubscriptionPlan
    is_active: bool
    subscription: SubscriptionResponse


async def require_subscription(
    token_data: dict = Depends(verify_token),
) -> BillingCheck:
    """
    FastAPI dependency: require an active subscription (including FREE).
    Raises 402 Payment Required if subscription is inactive.
    """
    user_id = token_data.get("user_id") or token_data.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        sub = await billing_service.require_active_subscription(user_id)
        return BillingCheck(
            user_id=user_id,
            plan=sub.plan,
            is_active=sub.is_active,
            subscription=sub,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=str(e),
        )


async def require_starter_plan(
    token_data: dict = Depends(verify_token),
) -> BillingCheck:
    """Require at least Starter plan."""
    user_id = token_data.get("user_id") or token_data.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        sub = await billing_service.require_plan(user_id, SubscriptionPlan.STARTER)
        return BillingCheck(
            user_id=user_id,
            plan=sub.plan,
            is_active=sub.is_active,
            subscription=sub,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=str(e),
        )


async def require_pro_plan(
    token_data: dict = Depends(verify_token),
) -> BillingCheck:
    """Require at least Professional plan."""
    user_id = token_data.get("user_id") or token_data.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        sub = await billing_service.require_plan(user_id, SubscriptionPlan.PROFESSIONAL)
        return BillingCheck(
            user_id=user_id,
            plan=sub.plan,
            is_active=sub.is_active,
            subscription=sub,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=str(e),
        )


async def require_enterprise_plan(
    token_data: dict = Depends(verify_token),
) -> BillingCheck:
    """Require Enterprise plan."""
    user_id = token_data.get("user_id") or token_data.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        sub = await billing_service.require_plan(user_id, SubscriptionPlan.ENTERPRISE)
        return BillingCheck(
            user_id=user_id,
            plan=sub.plan,
            is_active=sub.is_active,
            subscription=sub,
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=str(e),
        )


async def enforce_agent_limit(
    token_data: dict = Depends(verify_token),
) -> BillingCheck:
    """
    Enforce agent run limit. Use as a dependency on agent execution endpoints.
    Increments usage counter atomically after validation.
    """
    user_id = token_data.get("user_id") or token_data.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        sub = await billing_service.require_active_subscription(user_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=str(e),
        )

    try:
        await billing_service.enforce_agent_run_limit(user_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )

    return BillingCheck(
        user_id=user_id,
        plan=sub.plan,
        is_active=sub.is_active,
        subscription=sub,
    )


async def enforce_document_limit(
    token_data: dict = Depends(verify_token),
) -> BillingCheck:
    """Enforce document processing limit."""
    user_id = token_data.get("user_id") or token_data.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        sub = await billing_service.require_active_subscription(user_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=str(e),
        )

    try:
        await billing_service.enforce_document_limit(user_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )

    return BillingCheck(
        user_id=user_id,
        plan=sub.plan,
        is_active=sub.is_active,
        subscription=sub,
    )


async def enforce_kb_query_limit(
    token_data: dict = Depends(verify_token),
) -> BillingCheck:
    """Enforce knowledge base query limit."""
    user_id = token_data.get("user_id") or token_data.get("uid")
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    try:
        sub = await billing_service.require_active_subscription(user_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_402_PAYMENT_REQUIRED,
            detail=str(e),
        )

    try:
        await billing_service.enforce_kb_query_limit(user_id)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=str(e),
        )

    return BillingCheck(
        user_id=user_id,
        plan=sub.plan,
        is_active=sub.is_active,
        subscription=sub,
    )
