"""
Phase A verification — Plan-cap enforcement.

Direct unit tests against `backend/services/billing/plan_caps.py`.  A full
E2E pass would require a live Mongo + Stripe; here we exercise the cap
logic with a mocked seat repository so the assertions stay deterministic.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from backend.services.billing import plan_caps
from backend.services.billing.plan_caps import (
    PlanCapDetail,
    PlanCapExceeded,
    _suggest_upgrade,
    assert_can_add_seat,
    assert_can_create_custom_agent,
    assert_can_run_agent,
    assert_can_upload_document,
)


# ── Internals ──────────────────────────────────────────────────────


def test_upgrade_ladder_progression():
    assert _suggest_upgrade("team") == "business"
    assert _suggest_upgrade("business") == "enterprise"
    # Top of ladder — no upgrade available
    assert _suggest_upgrade("enterprise") is None
    # Unknown plan rolls back to the first paid tier
    assert _suggest_upgrade("free") == "team"


def test_plan_cap_detail_payload_shape():
    detail = PlanCapDetail(
        cap="seats", current=3, limit=3, plan="team",
        upgrade_suggested="business", message="No seats left.",
    )
    exc = PlanCapExceeded(detail)
    assert exc.detail["cap"] == "seats"
    assert exc.detail["plan"] == "team"
    assert exc.detail["upgrade_suggested"] == "business"
    assert "No seats left." in str(exc)


# ── Seat assertion ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_assert_can_add_seat_passes_below_purchased():
    with patch.object(plan_caps, "_get_org_plan",
                      AsyncMock(return_value={"plan": "team", "seats_purchased": 5})), \
         patch.object(plan_caps, "_count_active_seats", AsyncMock(return_value=3)):
        await assert_can_add_seat("org1")


@pytest.mark.asyncio
async def test_assert_can_add_seat_blocks_at_purchased():
    """3 used + 1 new = 4 > purchased 3 → should raise."""
    with patch.object(plan_caps, "_get_org_plan",
                      AsyncMock(return_value={"plan": "team", "seats_purchased": 3})), \
         patch.object(plan_caps, "_count_active_seats", AsyncMock(return_value=3)):
        with pytest.raises(PlanCapExceeded) as exc:
            await assert_can_add_seat("org1")
        assert exc.value.detail["cap"] == "seats"
        assert exc.value.detail["current"] == 3
        assert exc.value.detail["limit"] == 3
        assert exc.value.detail["upgrade_suggested"] == "business"


@pytest.mark.asyncio
async def test_assert_can_add_seat_blocks_when_full():
    with patch.object(plan_caps, "_get_org_plan",
                      AsyncMock(return_value={"plan": "business", "seats_purchased": 10})), \
         patch.object(plan_caps, "_count_active_seats", AsyncMock(return_value=10)):
        with pytest.raises(PlanCapExceeded):
            await assert_can_add_seat("org1")


# ── Agent run quota ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_run_quota_passes_under_cap():
    with patch.object(plan_caps, "_get_org_plan",
                      AsyncMock(return_value={"plan": "team", "seats_purchased": 1})), \
         patch.object(plan_caps, "_current_month_agent_runs", AsyncMock(return_value=42)):
        await assert_can_run_agent("org1")


@pytest.mark.asyncio
async def test_run_quota_blocks_at_cap():
    """Free plan has max_agent_runs_per_month=50."""
    with patch.object(plan_caps, "_get_org_plan",
                      AsyncMock(return_value={"plan": "free", "seats_purchased": 1})), \
         patch.object(plan_caps, "_current_month_agent_runs", AsyncMock(return_value=50)):
        with pytest.raises(PlanCapExceeded) as exc:
            await assert_can_run_agent("org1")
        assert exc.value.detail["cap"] == "agent_runs_per_month"


@pytest.mark.asyncio
async def test_run_quota_unlimited_on_enterprise():
    with patch.object(plan_caps, "_get_org_plan",
                      AsyncMock(return_value={"plan": "enterprise", "seats_purchased": 100})), \
         patch.object(plan_caps, "_current_month_agent_runs", AsyncMock(return_value=10_000)):
        # No raise even at huge usage — enterprise is -1 (unlimited)
        await assert_can_run_agent("org1")


# ── Custom agent ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_custom_agent_blocked_on_free_plan():
    """Free plan has custom_agent_templates=False."""
    with patch.object(plan_caps, "_get_org_plan",
                      AsyncMock(return_value={"plan": "free", "seats_purchased": 1})):
        with pytest.raises(PlanCapExceeded) as exc:
            await assert_can_create_custom_agent("org1")
        assert exc.value.detail["cap"] == "custom_agent_templates"


@pytest.mark.asyncio
async def test_custom_agent_blocked_when_count_at_cap():
    """Team plan ships max_agents=25; an org sitting at 25 should be blocked."""
    with patch.object(plan_caps, "_get_org_plan",
                      AsyncMock(return_value={"plan": "team", "seats_purchased": 1})), \
         patch.object(plan_caps, "_custom_agent_count", AsyncMock(return_value=25)):
        with pytest.raises(PlanCapExceeded) as exc:
            await assert_can_create_custom_agent("org1")
        assert exc.value.detail["cap"] == "max_agents"
        assert exc.value.detail["upgrade_suggested"] == "business"


# ── Document upload size ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_upload_blocked_above_plan_size():
    """Free plan caps at 5 MB; a 6 MB upload must be blocked."""
    with patch.object(plan_caps, "_get_org_plan",
                      AsyncMock(return_value={"plan": "free", "seats_purchased": 1})):
        with pytest.raises(PlanCapExceeded) as exc:
            await assert_can_upload_document("org1", 6 * 1024 * 1024)
        assert exc.value.detail["cap"] == "max_file_upload_mb"


@pytest.mark.asyncio
async def test_upload_passes_within_plan_size():
    with patch.object(plan_caps, "_get_org_plan",
                      AsyncMock(return_value={"plan": "team", "seats_purchased": 1})):
        # Team plan caps at 50 MB; 10 MB sails through
        await assert_can_upload_document("org1", 10 * 1024 * 1024)
