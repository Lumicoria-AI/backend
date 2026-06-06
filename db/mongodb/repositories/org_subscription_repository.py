"""
OrgSubscriptionRepository — org-scoped Stripe subscriptions + seat tracking.

Backs Team / Business / Enterprise plans.  Pairs with the existing
user-scoped `SubscriptionRepository` (which keeps powering individual
plans).  Seat usage is recorded in the `seat_assignments` collection
maintained by `seat_assignment_repository`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog
from bson import ObjectId
from pymongo import ASCENDING

from backend.db.mongodb.mongodb import MongoDB
from backend.models.billing import OrgSubscriptionInDB, SeatAssignmentInDB, SubscriptionPlan, SubscriptionStatus

logger = structlog.get_logger(__name__)


class OrgSubscriptionRepository:
    """Thin async wrapper over the `org_subscriptions` collection."""

    COLLECTION = "org_subscriptions"

    def __init__(self) -> None:
        self._initialised = False

    async def _ensure_indexes(self) -> None:
        if self._initialised:
            return
        col = await MongoDB.get_collection(self.COLLECTION)
        await col.create_index("organization_id", unique=True)
        await col.create_index("stripe_subscription_id", sparse=True)
        await col.create_index("plan")
        await col.create_index("status")
        self._initialised = True

    async def get_for_org(self, organization_id: str) -> Optional[OrgSubscriptionInDB]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(self.COLLECTION)
        row = await col.find_one({"organization_id": ObjectId(organization_id)})
        if not row:
            return None
        return self._coerce(row)

    async def upsert(
        self,
        organization_id: str,
        *,
        plan: SubscriptionPlan,
        status: SubscriptionStatus = SubscriptionStatus.ACTIVE,
        seats_purchased: int = 1,
        cadence: str = "monthly",
        stripe_customer_id: Optional[str] = None,
        stripe_subscription_id: Optional[str] = None,
        stripe_price_id: Optional[str] = None,
        billing_email: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> OrgSubscriptionInDB:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(self.COLLECTION)
        now = datetime.utcnow()
        update: Dict[str, Any] = {
            "$set": {
                "organization_id": ObjectId(organization_id),
                "plan": plan.value if isinstance(plan, SubscriptionPlan) else str(plan),
                "status": status.value if isinstance(status, SubscriptionStatus) else str(status),
                "seats_purchased": int(seats_purchased),
                "cadence": cadence,
                "updated_at": now,
            },
            "$setOnInsert": {"created_at": now},
        }
        for key, value in (
            ("stripe_customer_id", stripe_customer_id),
            ("stripe_subscription_id", stripe_subscription_id),
            ("stripe_price_id", stripe_price_id),
            ("billing_email", billing_email),
        ):
            if value is not None:
                update["$set"][key] = value
        if metadata:
            update["$set"]["metadata"] = metadata
        await col.update_one({"organization_id": ObjectId(organization_id)}, update, upsert=True)
        row = await col.find_one({"organization_id": ObjectId(organization_id)})
        return self._coerce(row)

    async def update_seats(self, organization_id: str, *, purchased: Optional[int] = None,
                           used: Optional[int] = None) -> Optional[OrgSubscriptionInDB]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(self.COLLECTION)
        patch: Dict[str, Any] = {"updated_at": datetime.utcnow()}
        if purchased is not None:
            patch["seats_purchased"] = int(purchased)
        if used is not None:
            patch["seats_used"] = int(used)
        if len(patch) == 1:
            return await self.get_for_org(organization_id)
        await col.update_one({"organization_id": ObjectId(organization_id)}, {"$set": patch})
        return await self.get_for_org(organization_id)

    async def cancel(self, organization_id: str, at_period_end: bool = True) -> Optional[OrgSubscriptionInDB]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(self.COLLECTION)
        patch: Dict[str, Any] = {
            "cancel_at_period_end": bool(at_period_end),
            "updated_at": datetime.utcnow(),
        }
        if not at_period_end:
            patch["status"] = SubscriptionStatus.CANCELED.value
            patch["canceled_at"] = datetime.utcnow()
        await col.update_one({"organization_id": ObjectId(organization_id)}, {"$set": patch})
        return await self.get_for_org(organization_id)

    @staticmethod
    def _coerce(row: Dict[str, Any]) -> OrgSubscriptionInDB:
        row = dict(row)
        row["organization_id"] = str(row["organization_id"])
        return OrgSubscriptionInDB(**row)


class SeatAssignmentRepository:
    COLLECTION = "seat_assignments"

    def __init__(self) -> None:
        self._initialised = False

    async def _ensure_indexes(self) -> None:
        if self._initialised:
            return
        col = await MongoDB.get_collection(self.COLLECTION)
        await col.create_index(
            [("organization_id", ASCENDING), ("user_id", ASCENDING), ("removed_at", ASCENDING)],
        )
        await col.create_index([("organization_id", ASCENDING), ("assigned_at", ASCENDING)])
        self._initialised = True

    async def assign(self, organization_id: str, user_id: str, *, assigned_by: Optional[str] = None,
                     metadata: Optional[Dict[str, Any]] = None) -> SeatAssignmentInDB:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(self.COLLECTION)
        now = datetime.utcnow()
        # Reactivate if a prior removed row exists, otherwise insert.
        existing = await col.find_one({
            "organization_id": ObjectId(organization_id),
            "user_id": ObjectId(user_id),
        })
        if existing:
            await col.update_one(
                {"_id": existing["_id"]},
                {"$set": {
                    "removed_at": None,
                    "assigned_by": ObjectId(assigned_by) if assigned_by else None,
                    "metadata": metadata or {},
                    "assigned_at": existing.get("assigned_at") or now,
                }},
            )
            row = await col.find_one({"_id": existing["_id"]})
        else:
            doc = {
                "organization_id": ObjectId(organization_id),
                "user_id": ObjectId(user_id),
                "assigned_at": now,
                "assigned_by": ObjectId(assigned_by) if assigned_by else None,
                "removed_at": None,
                "metadata": metadata or {},
            }
            await col.insert_one(doc)
            row = await col.find_one(doc)
        return self._coerce(row)

    async def remove(self, organization_id: str, user_id: str) -> bool:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(self.COLLECTION)
        result = await col.update_one(
            {"organization_id": ObjectId(organization_id), "user_id": ObjectId(user_id), "removed_at": None},
            {"$set": {"removed_at": datetime.utcnow()}},
        )
        return result.modified_count > 0

    async def count_active(self, organization_id: str) -> int:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(self.COLLECTION)
        return await col.count_documents({
            "organization_id": ObjectId(organization_id),
            "removed_at": None,
        })

    async def list_active(self, organization_id: str, limit: int = 500) -> List[SeatAssignmentInDB]:
        await self._ensure_indexes()
        col = await MongoDB.get_collection(self.COLLECTION)
        cursor = col.find({
            "organization_id": ObjectId(organization_id),
            "removed_at": None,
        }).sort("assigned_at", ASCENDING).limit(limit)
        rows = await cursor.to_list(length=limit)
        return [self._coerce(r) for r in rows]

    @staticmethod
    def _coerce(row: Dict[str, Any]) -> SeatAssignmentInDB:
        row = dict(row)
        row["organization_id"] = str(row["organization_id"])
        row["user_id"] = str(row["user_id"])
        if row.get("assigned_by") is not None:
            row["assigned_by"] = str(row["assigned_by"])
        return SeatAssignmentInDB(**row)


org_subscription_repository = OrgSubscriptionRepository()
seat_assignment_repository = SeatAssignmentRepository()
