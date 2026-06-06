"""
Lumicoria AI — Plan-cap enforcement

Pre-flight guards that turn a numeric / boolean limit into a typed Python
exception or a structured 402-style response.

Usage from a router:

    from backend.services.billing.plan_caps import (
        assert_can_add_seat,
        PlanCapExceeded,
    )

    try:
        await assert_can_add_seat(organization_id)
    except PlanCapExceeded as exc:
        raise HTTPException(status_code=402, detail=exc.detail)

Caps come from `backend/models/billing.py:PLAN_LIMITS`.  Org-scoped plans
(team/business/enterprise) are looked up from the `org_subscriptions`
collection added in Phase A3; until that lands the helpers fall back to the
free tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import structlog
from bson import ObjectId

from backend.db.mongodb.mongodb import MongoDB
from backend.models.billing import PLAN_LIMITS, SubscriptionPlan, get_plan_limits

logger = structlog.get_logger(__name__)


@dataclass
class PlanCapDetail:
    cap: str
    current: int
    limit: int
    plan: str
    upgrade_suggested: Optional[str] = None
    message: str = "Plan limit reached."


class PlanCapExceeded(Exception):
    """Raised when a quota-sensitive action would exceed the active plan."""

    def __init__(self, detail: PlanCapDetail) -> None:
        super().__init__(detail.message)
        self.detail: Dict[str, Any] = {
            "cap": detail.cap,
            "current": detail.current,
            "limit": detail.limit,
            "plan": detail.plan,
            "upgrade_suggested": detail.upgrade_suggested,
            "message": detail.message,
        }


# Org-scoped plans we care about for seat enforcement.
ORG_PAID_PLANS = {"team", "business", "enterprise"}

# Default per-seat allowances when the org has no subscription row yet.
DEFAULT_SEAT_FLOOR = 1

# Map team-plan internal keys to upgrade ladder
PLAN_LADDER = ["team", "business", "enterprise"]


# --------------------------------------------------------------------- helpers

async def _get_org_plan(organization_id: str) -> Dict[str, Any]:
    """Resolve the active plan + caps for an org.  Falls back to FREE."""
    if not organization_id:
        return {"plan": SubscriptionPlan.FREE.value, "seats_purchased": DEFAULT_SEAT_FLOOR}
    try:
        col = await MongoDB.get_collection("org_subscriptions")
        row = await col.find_one({"organization_id": ObjectId(organization_id)})
    except Exception as exc:  # noqa: BLE001
        logger.warning("plan_caps.org_subscription_lookup_failed", error=str(exc))
        row = None
    if not row:
        return {"plan": SubscriptionPlan.FREE.value, "seats_purchased": DEFAULT_SEAT_FLOOR}
    return {
        "plan": row.get("plan", SubscriptionPlan.FREE.value),
        "seats_purchased": int(row.get("seats_purchased") or DEFAULT_SEAT_FLOOR),
        "status": row.get("status"),
    }


async def _count_active_seats(organization_id: str) -> int:
    """Count active seat_assignments for this org."""
    try:
        col = await MongoDB.get_collection("seat_assignments")
        return await col.count_documents({
            "organization_id": ObjectId(organization_id),
            "removed_at": {"$in": [None]},
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("plan_caps.seat_count_failed", error=str(exc))
        return 0


def _suggest_upgrade(current_plan: str) -> Optional[str]:
    try:
        idx = PLAN_LADDER.index(current_plan)
    except ValueError:
        return PLAN_LADDER[0]
    if idx + 1 < len(PLAN_LADDER):
        return PLAN_LADDER[idx + 1]
    return None


# --------------------------------------------------------------------- public guards


async def assert_can_add_seat(organization_id: str) -> None:
    """Block when seats_purchased would be exceeded.  Free orgs get 1 seat."""
    plan = await _get_org_plan(organization_id)
    purchased = plan["seats_purchased"]
    active = await _count_active_seats(organization_id)
    if active + 1 > purchased:
        raise PlanCapExceeded(PlanCapDetail(
            cap="seats",
            current=active,
            limit=purchased,
            plan=plan["plan"],
            upgrade_suggested=_suggest_upgrade(plan["plan"]),
            message=(
                "All seats are filled. Buy more seats from your billing settings, "
                "or upgrade to a higher plan."
            ),
        ))


async def assert_can_run_agent(organization_id: str, plan: Optional[str] = None) -> None:
    """Check monthly agent-run cap against current usage."""
    if plan is None:
        plan = (await _get_org_plan(organization_id))["plan"]
    limits = get_plan_limits(plan)
    cap = int(limits.get("max_agent_runs_per_month") or -1)
    if cap < 0:
        return  # unlimited
    used = await _current_month_agent_runs(organization_id)
    if used >= cap:
        raise PlanCapExceeded(PlanCapDetail(
            cap="agent_runs_per_month",
            current=used,
            limit=cap,
            plan=plan,
            upgrade_suggested=_suggest_upgrade(plan),
            message="Monthly agent-run quota reached.",
        ))


async def assert_can_create_custom_agent(organization_id: str, plan: Optional[str] = None) -> None:
    if plan is None:
        plan = (await _get_org_plan(organization_id))["plan"]
    limits = get_plan_limits(plan)
    if not limits.get("custom_agent_templates", False):
        raise PlanCapExceeded(PlanCapDetail(
            cap="custom_agent_templates",
            current=0,
            limit=0,
            plan=plan,
            upgrade_suggested=_suggest_upgrade(plan),
            message="Custom agents require the Professional plan or higher.",
        ))
    # Numeric cap: max_agents
    cap = int(limits.get("max_agents") or -1)
    if cap < 0:
        return
    used = await _custom_agent_count(organization_id)
    if used >= cap:
        raise PlanCapExceeded(PlanCapDetail(
            cap="max_agents",
            current=used,
            limit=cap,
            plan=plan,
            upgrade_suggested=_suggest_upgrade(plan),
            message="Custom agent quota reached.",
        ))


async def assert_can_upload_document(organization_id: str, size_bytes: int, plan: Optional[str] = None) -> None:
    if plan is None:
        plan = (await _get_org_plan(organization_id))["plan"]
    limits = get_plan_limits(plan)
    max_mb = int(limits.get("max_file_upload_mb") or 0)
    if max_mb and size_bytes > max_mb * 1024 * 1024:
        raise PlanCapExceeded(PlanCapDetail(
            cap="max_file_upload_mb",
            current=size_bytes // (1024 * 1024),
            limit=max_mb,
            plan=plan,
            upgrade_suggested=_suggest_upgrade(plan),
            message=f"File exceeds {max_mb} MB on the {plan} plan.",
        ))


# --------------------------------------------------------------------- usage probes


async def _current_month_agent_runs(organization_id: str) -> int:
    try:
        from datetime import datetime
        now = datetime.utcnow()
        start = datetime(now.year, now.month, 1)
        col = await MongoDB.get_collection("agent_runs")
        return await col.count_documents({
            "organization_id": ObjectId(organization_id),
            "started_at": {"$gte": start},
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("plan_caps.agent_run_count_failed", error=str(exc))
        return 0


async def _custom_agent_count(organization_id: str) -> int:
    try:
        col = await MongoDB.get_collection("agents")
        return await col.count_documents({
            "organization_id": ObjectId(organization_id),
        })
    except Exception as exc:  # noqa: BLE001
        logger.warning("plan_caps.custom_agent_count_failed", error=str(exc))
        return 0
