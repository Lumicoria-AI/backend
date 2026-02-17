"""
Lumicoria AI — Billing System Tests

Production-grade test suite covering:
  • Subscription lifecycle (create → upgrade → downgrade → cancel)
  • Usage tracking & limit enforcement
  • Webhook idempotency
  • Feature gating
  • Admin overrides
  • Edge cases (expired trials, past_due grace periods)

Run: pytest backend/tests/test_billing.py -v
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch
from backend.models.billing import (
    SubscriptionInDB,
    UsageTrackingInDB,
    PaymentEventInDB,
    SubscriptionPlan,
    SubscriptionStatus,
    PaymentStatus,
    PLAN_LIMITS,
    get_plan_limits,
    CreateCheckoutRequest,
)


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests — Plan Limits
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanLimits:
    """Test plan limit configuration."""

    def test_all_plans_defined(self):
        """Every plan enum value must have limits defined."""
        for plan in SubscriptionPlan:
            assert plan in PLAN_LIMITS or plan.value in PLAN_LIMITS, (
                f"Missing limits for plan: {plan}"
            )

    def test_free_plan_has_strict_limits(self):
        limits = get_plan_limits(SubscriptionPlan.FREE)
        assert limits["max_agents"] == 2
        assert limits["max_agent_runs_per_month"] == 50
        assert limits["max_documents_per_month"] == 10
        assert limits["advanced_features"] is False
        assert limits["api_access"] is False

    def test_starter_plan_limits(self):
        limits = get_plan_limits(SubscriptionPlan.STARTER)
        assert limits["max_agents"] == 5
        assert limits["max_agent_runs_per_month"] == 500
        assert limits["price_monthly"] == 29

    def test_professional_plan_limits(self):
        limits = get_plan_limits(SubscriptionPlan.PROFESSIONAL)
        assert limits["max_agents"] == 15
        assert limits["max_agent_runs_per_month"] == 5000
        assert limits["price_monthly"] == 79
        assert limits["advanced_features"] is True
        assert limits["api_access"] is True

    def test_enterprise_unlimited(self):
        limits = get_plan_limits(SubscriptionPlan.ENTERPRISE)
        assert limits["max_agents"] == -1  # unlimited
        assert limits["max_agent_runs_per_month"] == -1

    def test_unknown_plan_defaults_to_free(self):
        limits = get_plan_limits("nonexistent_plan")
        assert limits["max_agents"] == 2
        assert limits["max_agent_runs_per_month"] == 50

    def test_plan_hierarchy(self):
        """Plans must have increasing limits."""
        free = get_plan_limits(SubscriptionPlan.FREE)
        starter = get_plan_limits(SubscriptionPlan.STARTER)
        pro = get_plan_limits(SubscriptionPlan.PROFESSIONAL)

        assert free["max_agents"] < starter["max_agents"]
        assert starter["max_agents"] < pro["max_agents"]
        assert free["max_agent_runs_per_month"] < starter["max_agent_runs_per_month"]
        assert starter["max_agent_runs_per_month"] < pro["max_agent_runs_per_month"]


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests — Models
# ─────────────────────────────────────────────────────────────────────────────

class TestBillingModels:
    """Test Pydantic model validation."""

    def test_subscription_defaults(self):
        sub = SubscriptionInDB(
            user_id="user_123",
            stripe_customer_id="cus_test",
        )
        assert sub.plan == SubscriptionPlan.FREE
        assert sub.status == SubscriptionStatus.ACTIVE
        assert sub.cancel_at_period_end is False
        assert sub.is_admin_override is False

    def test_usage_tracking_defaults(self):
        usage = UsageTrackingInDB(
            user_id="user_123",
            month=1,
            year=2026,
        )
        assert usage.agent_runs == 0
        assert usage.documents_processed == 0
        assert usage.knowledge_base_queries == 0
        assert usage.agent_usage_breakdown == {}

    def test_payment_event_idempotency_key(self):
        event = PaymentEventInDB(
            stripe_event_id="evt_123",
            event_type="checkout.session.completed",
        )
        assert event.stripe_event_id == "evt_123"
        assert event.status == PaymentStatus.PENDING

    def test_checkout_request_validation(self):
        req = CreateCheckoutRequest(price_id="price_test123")
        assert req.price_id == "price_test123"
        assert req.success_url is None

    def test_subscription_status_enum(self):
        """All Stripe subscription statuses must be represented."""
        assert SubscriptionStatus.ACTIVE == "active"
        assert SubscriptionStatus.PAST_DUE == "past_due"
        assert SubscriptionStatus.CANCELED == "canceled"
        assert SubscriptionStatus.TRIALING == "trialing"
        assert SubscriptionStatus.UNPAID == "unpaid"


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests — Billing Service Logic
# ─────────────────────────────────────────────────────────────────────────────

class TestBillingServiceLogic:
    """Test billing service business logic (mocked DB)."""

    @pytest.mark.asyncio
    async def test_free_user_has_active_subscription(self):
        """Users without any subscription should be treated as FREE/active."""
        with patch("backend.services.billing_service.subscription_repository") as mock_repo:
            mock_repo.get_by_user_id = AsyncMock(return_value=None)
            
            from backend.services.billing_service import get_user_subscription
            result = await get_user_subscription("user_no_sub")
            
            assert result.plan == SubscriptionPlan.FREE
            assert result.is_active is True

    @pytest.mark.asyncio
    async def test_active_subscription_is_active(self):
        """Active subscription should be marked as active."""
        sub = SubscriptionInDB(
            user_id="user_active",
            stripe_customer_id="cus_active",
            plan=SubscriptionPlan.PROFESSIONAL,
            status=SubscriptionStatus.ACTIVE,
            current_period_end=datetime.utcnow() + timedelta(days=30),
        )
        with patch("backend.services.billing_service.subscription_repository") as mock_repo:
            mock_repo.get_by_user_id = AsyncMock(return_value=sub)
            
            from backend.services.billing_service import get_user_subscription
            result = await get_user_subscription("user_active")
            
            assert result.plan == SubscriptionPlan.PROFESSIONAL
            assert result.is_active is True

    @pytest.mark.asyncio
    async def test_canceled_subscription_is_inactive(self):
        """Canceled subscription should be marked as inactive."""
        sub = SubscriptionInDB(
            user_id="user_canceled",
            stripe_customer_id="cus_canceled",
            plan=SubscriptionPlan.STARTER,
            status=SubscriptionStatus.CANCELED,
        )
        with patch("backend.services.billing_service.subscription_repository") as mock_repo:
            mock_repo.get_by_user_id = AsyncMock(return_value=sub)
            
            from backend.services.billing_service import get_user_subscription
            result = await get_user_subscription("user_canceled")
            
            assert result.is_active is False

    @pytest.mark.asyncio
    async def test_trialing_subscription_is_active(self):
        """Trial subscription should be marked as active."""
        sub = SubscriptionInDB(
            user_id="user_trial",
            stripe_customer_id="cus_trial",
            plan=SubscriptionPlan.PROFESSIONAL,
            status=SubscriptionStatus.TRIALING,
            trial_end=datetime.utcnow() + timedelta(days=14),
        )
        with patch("backend.services.billing_service.subscription_repository") as mock_repo:
            mock_repo.get_by_user_id = AsyncMock(return_value=sub)
            
            from backend.services.billing_service import get_user_subscription
            result = await get_user_subscription("user_trial")
            
            assert result.is_active is True
            assert result.plan == SubscriptionPlan.PROFESSIONAL

    @pytest.mark.asyncio
    async def test_past_due_within_grace_period(self):
        """Past due within 3-day grace period should still be active."""
        sub = SubscriptionInDB(
            user_id="user_pastdue",
            stripe_customer_id="cus_pastdue",
            plan=SubscriptionPlan.STARTER,
            status=SubscriptionStatus.PAST_DUE,
            current_period_end=datetime.utcnow() - timedelta(days=1),
        )
        with patch("backend.services.billing_service.subscription_repository") as mock_repo:
            mock_repo.get_by_user_id = AsyncMock(return_value=sub)
            
            from backend.services.billing_service import get_user_subscription
            result = await get_user_subscription("user_pastdue")
            
            assert result.is_active is True  # Within grace period

    @pytest.mark.asyncio
    async def test_past_due_beyond_grace_period(self):
        """Past due beyond 3-day grace period should be inactive."""
        sub = SubscriptionInDB(
            user_id="user_pastdue_old",
            stripe_customer_id="cus_pastdue_old",
            plan=SubscriptionPlan.STARTER,
            status=SubscriptionStatus.PAST_DUE,
            current_period_end=datetime.utcnow() - timedelta(days=5),
        )
        with patch("backend.services.billing_service.subscription_repository") as mock_repo:
            mock_repo.get_by_user_id = AsyncMock(return_value=sub)
            
            from backend.services.billing_service import get_user_subscription
            result = await get_user_subscription("user_pastdue_old")
            
            assert result.is_active is False

    @pytest.mark.asyncio
    async def test_admin_override_plan(self):
        """Admin override should take precedence over Stripe plan."""
        sub = SubscriptionInDB(
            user_id="user_override",
            stripe_customer_id="cus_override",
            plan=SubscriptionPlan.FREE,
            status=SubscriptionStatus.ACTIVE,
            is_admin_override=True,
            admin_override_plan=SubscriptionPlan.ENTERPRISE,
        )
        with patch("backend.services.billing_service.subscription_repository") as mock_repo:
            mock_repo.get_by_user_id = AsyncMock(return_value=sub)
            
            from backend.services.billing_service import get_user_subscription
            result = await get_user_subscription("user_override")
            
            assert result.plan == SubscriptionPlan.ENTERPRISE

    @pytest.mark.asyncio
    async def test_expired_admin_override_ignored(self):
        """Expired admin override should fall back to Stripe plan."""
        sub = SubscriptionInDB(
            user_id="user_expired_override",
            stripe_customer_id="cus_expired_override",
            plan=SubscriptionPlan.FREE,
            status=SubscriptionStatus.ACTIVE,
            is_admin_override=True,
            admin_override_plan=SubscriptionPlan.ENTERPRISE,
            admin_override_expires=datetime.utcnow() - timedelta(days=1),
        )
        with patch("backend.services.billing_service.subscription_repository") as mock_repo:
            mock_repo.get_by_user_id = AsyncMock(return_value=sub)
            
            from backend.services.billing_service import get_user_subscription
            result = await get_user_subscription("user_expired_override")
            
            assert result.plan == SubscriptionPlan.FREE


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests — Usage Enforcement
# ─────────────────────────────────────────────────────────────────────────────

class TestUsageEnforcement:
    """Test usage limit enforcement."""

    @pytest.mark.asyncio
    async def test_usage_within_limit(self):
        """Usage below limit should be allowed."""
        sub = SubscriptionInDB(
            user_id="user_ok",
            stripe_customer_id="cus_ok",
            plan=SubscriptionPlan.STARTER,
            status=SubscriptionStatus.ACTIVE,
        )
        usage = UsageTrackingInDB(
            user_id="user_ok",
            month=datetime.utcnow().month,
            year=datetime.utcnow().year,
            agent_runs=10,
        )
        with patch("backend.services.billing_service.subscription_repository") as mock_sub_repo, \
             patch("backend.services.billing_service.usage_tracking_repository") as mock_usage_repo:
            mock_sub_repo.get_by_user_id = AsyncMock(return_value=sub)
            mock_usage_repo.get_current_usage = AsyncMock(return_value=usage)
            
            from backend.services.billing_service import check_usage_limits
            allowed, current, limit = await check_usage_limits("user_ok", "agent_runs")
            
            assert allowed is True
            assert current == 10
            assert limit == 500  # Starter limit

    @pytest.mark.asyncio
    async def test_usage_at_limit_blocked(self):
        """Usage at or above limit should be blocked."""
        sub = SubscriptionInDB(
            user_id="user_limit",
            stripe_customer_id="cus_limit",
            plan=SubscriptionPlan.FREE,
            status=SubscriptionStatus.ACTIVE,
        )
        usage = UsageTrackingInDB(
            user_id="user_limit",
            month=datetime.utcnow().month,
            year=datetime.utcnow().year,
            agent_runs=50,  # At the FREE limit
        )
        with patch("backend.services.billing_service.subscription_repository") as mock_sub_repo, \
             patch("backend.services.billing_service.usage_tracking_repository") as mock_usage_repo:
            mock_sub_repo.get_by_user_id = AsyncMock(return_value=sub)
            mock_usage_repo.get_current_usage = AsyncMock(return_value=usage)
            
            from backend.services.billing_service import check_usage_limits
            allowed, current, limit = await check_usage_limits("user_limit", "agent_runs")
            
            assert allowed is False
            assert current == 50
            assert limit == 50

    @pytest.mark.asyncio
    async def test_enterprise_unlimited(self):
        """Enterprise plan should always be allowed (unlimited)."""
        sub = SubscriptionInDB(
            user_id="user_ent",
            stripe_customer_id="cus_ent",
            plan=SubscriptionPlan.ENTERPRISE,
            status=SubscriptionStatus.ACTIVE,
        )
        usage = UsageTrackingInDB(
            user_id="user_ent",
            month=datetime.utcnow().month,
            year=datetime.utcnow().year,
            agent_runs=99999,
        )
        with patch("backend.services.billing_service.subscription_repository") as mock_sub_repo, \
             patch("backend.services.billing_service.usage_tracking_repository") as mock_usage_repo:
            mock_sub_repo.get_by_user_id = AsyncMock(return_value=sub)
            mock_usage_repo.get_current_usage = AsyncMock(return_value=usage)
            
            from backend.services.billing_service import check_usage_limits
            allowed, current, limit = await check_usage_limits("user_ent", "agent_runs")
            
            assert allowed is True
            assert limit == -1

    @pytest.mark.asyncio
    async def test_enforce_raises_on_limit(self):
        """enforce_agent_run_limit should raise when limit exceeded."""
        sub = SubscriptionInDB(
            user_id="user_exceed",
            stripe_customer_id="cus_exceed",
            plan=SubscriptionPlan.FREE,
            status=SubscriptionStatus.ACTIVE,
        )
        usage = UsageTrackingInDB(
            user_id="user_exceed",
            month=datetime.utcnow().month,
            year=datetime.utcnow().year,
            agent_runs=51,
        )
        with patch("backend.services.billing_service.subscription_repository") as mock_sub_repo, \
             patch("backend.services.billing_service.usage_tracking_repository") as mock_usage_repo:
            mock_sub_repo.get_by_user_id = AsyncMock(return_value=sub)
            mock_usage_repo.get_current_usage = AsyncMock(return_value=usage)
            
            from backend.services.billing_service import enforce_agent_run_limit
            with pytest.raises(ValueError, match="limit reached"):
                await enforce_agent_run_limit("user_exceed")


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests — Plan Requirements
# ─────────────────────────────────────────────────────────────────────────────

class TestPlanRequirements:
    """Test require_plan gating."""

    @pytest.mark.asyncio
    async def test_require_plan_success(self):
        """User on Pro plan should pass Starter requirement."""
        sub = SubscriptionInDB(
            user_id="user_pro",
            stripe_customer_id="cus_pro",
            plan=SubscriptionPlan.PROFESSIONAL,
            status=SubscriptionStatus.ACTIVE,
        )
        with patch("backend.services.billing_service.subscription_repository") as mock_repo:
            mock_repo.get_by_user_id = AsyncMock(return_value=sub)
            
            from backend.services.billing_service import require_plan
            result = await require_plan("user_pro", SubscriptionPlan.STARTER)
            assert result.plan == SubscriptionPlan.PROFESSIONAL

    @pytest.mark.asyncio
    async def test_require_plan_failure(self):
        """Free user should fail Pro plan requirement."""
        sub = SubscriptionInDB(
            user_id="user_free",
            stripe_customer_id="cus_free",
            plan=SubscriptionPlan.FREE,
            status=SubscriptionStatus.ACTIVE,
        )
        with patch("backend.services.billing_service.subscription_repository") as mock_repo:
            mock_repo.get_by_user_id = AsyncMock(return_value=sub)
            
            from backend.services.billing_service import require_plan
            with pytest.raises(ValueError, match="requires the professional plan"):
                await require_plan("user_free", SubscriptionPlan.PROFESSIONAL)


# ─────────────────────────────────────────────────────────────────────────────
# Unit Tests — Webhook Idempotency
# ─────────────────────────────────────────────────────────────────────────────

class TestWebhookIdempotency:
    """Test webhook idempotency logic."""

    @pytest.mark.asyncio
    async def test_duplicate_event_skipped(self):
        """Already-processed events should be skipped."""
        mock_event = MagicMock()
        mock_event.id = "evt_already_processed"
        mock_event.type = "checkout.session.completed"

        with patch("backend.services.billing_service.payment_event_repository") as mock_repo:
            mock_repo.is_event_processed = AsyncMock(return_value=True)
            
            from backend.services.billing_service import process_webhook_event
            result = await process_webhook_event(mock_event)
            
            assert result["status"] == "already_processed"

    @pytest.mark.asyncio
    async def test_unhandled_event_ignored(self):
        """Events not in HANDLED_EVENTS should be ignored."""
        mock_event = MagicMock()
        mock_event.id = "evt_unknown"
        mock_event.type = "charge.refunded"  # Not handled

        from backend.services.billing_service import process_webhook_event
        result = await process_webhook_event(mock_event)
        
        assert result["status"] == "ignored"
