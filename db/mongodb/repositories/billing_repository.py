"""
Lumicoria AI — Billing Repository (MongoDB)

Production-grade data access layer for subscriptions, usage tracking,
and payment events. Built on the existing BaseRepository pattern.

SECURITY:
  - All writes go through atomic MongoDB operations
  - Usage increments use $inc (atomic, no race conditions)
  - Idempotency enforced via unique stripe_event_id index
"""

from typing import Optional, Dict, Any, List
from datetime import datetime
from bson import ObjectId
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import ASCENDING, DESCENDING, IndexModel
from pymongo.errors import DuplicateKeyError

from backend.db.mongodb.base_repository import BaseRepository
from backend.models.billing import (
    SubscriptionInDB,
    UsageTrackingInDB,
    PaymentEventInDB,
    SubscriptionPlan,
    SubscriptionStatus,
)
import structlog

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Subscription Repository
# ─────────────────────────────────────────────────────────────────────────────

class SubscriptionRepository:
    """MongoDB repository for subscription documents."""

    def __init__(self):
        self._collection: Optional[AsyncIOMotorCollection] = None

    async def _get_collection(self) -> AsyncIOMotorCollection:
        if self._collection is None:
            from backend.db.mongodb.mongodb import MongoDB
            self._collection = await MongoDB.get_collection("subscriptions")
            await self._create_indexes()
        return self._collection

    async def _create_indexes(self):
        """Create unique and lookup indexes."""
        await self._collection.create_index("user_id", unique=True)
        await self._collection.create_index("stripe_customer_id", unique=True, sparse=True)
        await self._collection.create_index("stripe_subscription_id", sparse=True)
        await self._collection.create_index("status")
        logger.info("Subscription indexes created")

    async def get_by_user_id(self, user_id: str) -> Optional[SubscriptionInDB]:
        """Get subscription by user ID."""
        collection = await self._get_collection()
        doc = await collection.find_one({"user_id": user_id})
        return SubscriptionInDB(**doc) if doc else None

    async def get_by_stripe_customer_id(self, stripe_customer_id: str) -> Optional[SubscriptionInDB]:
        """Get subscription by Stripe customer ID."""
        collection = await self._get_collection()
        doc = await collection.find_one({"stripe_customer_id": stripe_customer_id})
        return SubscriptionInDB(**doc) if doc else None

    async def get_by_stripe_subscription_id(self, stripe_subscription_id: str) -> Optional[SubscriptionInDB]:
        """Get subscription by Stripe subscription ID."""
        collection = await self._get_collection()
        doc = await collection.find_one({"stripe_subscription_id": stripe_subscription_id})
        return SubscriptionInDB(**doc) if doc else None

    async def create(self, subscription: SubscriptionInDB) -> SubscriptionInDB:
        """Create a new subscription document."""
        collection = await self._get_collection()
        doc = subscription.model_dump()
        doc["created_at"] = datetime.utcnow()
        try:
            result = await collection.insert_one(doc)
            created = await collection.find_one({"_id": result.inserted_id})
            logger.info(
                "Subscription created",
                user_id=subscription.user_id,
                plan=subscription.plan,
            )
            return SubscriptionInDB(**created)
        except DuplicateKeyError:
            # User already has a subscription — return existing
            existing = await self.get_by_user_id(subscription.user_id)
            logger.warning(
                "Subscription already exists for user",
                user_id=subscription.user_id,
            )
            return existing

    async def update_from_stripe(
        self,
        stripe_customer_id: str,
        update_data: Dict[str, Any],
    ) -> Optional[SubscriptionInDB]:
        """
        Update subscription from Stripe webhook data.
        Uses atomic $set to prevent partial writes.
        """
        collection = await self._get_collection()
        update_data["updated_at"] = datetime.utcnow()
        result = await collection.find_one_and_update(
            {"stripe_customer_id": stripe_customer_id},
            {"$set": update_data},
            return_document=True,
        )
        if result:
            logger.info(
                "Subscription updated from Stripe",
                stripe_customer_id=stripe_customer_id,
                fields=list(update_data.keys()),
            )
            return SubscriptionInDB(**result)
        return None

    async def update_by_stripe_subscription_id(
        self,
        stripe_subscription_id: str,
        update_data: Dict[str, Any],
    ) -> Optional[SubscriptionInDB]:
        """Update subscription by Stripe subscription ID."""
        collection = await self._get_collection()
        update_data["updated_at"] = datetime.utcnow()
        result = await collection.find_one_and_update(
            {"stripe_subscription_id": stripe_subscription_id},
            {"$set": update_data},
            return_document=True,
        )
        if result:
            logger.info(
                "Subscription updated",
                stripe_subscription_id=stripe_subscription_id,
                fields=list(update_data.keys()),
            )
            return SubscriptionInDB(**result)
        return None

    async def set_admin_override(
        self,
        user_id: str,
        plan: SubscriptionPlan,
        expires_at: Optional[datetime] = None,
    ) -> Optional[SubscriptionInDB]:
        """Admin override — grant a user a specific plan."""
        collection = await self._get_collection()
        result = await collection.find_one_and_update(
            {"user_id": user_id},
            {
                "$set": {
                    "is_admin_override": True,
                    "admin_override_plan": plan,
                    "admin_override_expires": expires_at,
                    "updated_at": datetime.utcnow(),
                }
            },
            return_document=True,
        )
        if result:
            logger.info(
                "Admin override set",
                user_id=user_id,
                plan=plan,
                expires_at=expires_at,
            )
            return SubscriptionInDB(**result)
        return None

    async def remove_admin_override(self, user_id: str) -> Optional[SubscriptionInDB]:
        """Remove admin override."""
        collection = await self._get_collection()
        result = await collection.find_one_and_update(
            {"user_id": user_id},
            {
                "$set": {
                    "is_admin_override": False,
                    "admin_override_plan": None,
                    "admin_override_expires": None,
                    "updated_at": datetime.utcnow(),
                }
            },
            return_document=True,
        )
        return SubscriptionInDB(**result) if result else None


# ─────────────────────────────────────────────────────────────────────────────
# Usage Tracking Repository
# ─────────────────────────────────────────────────────────────────────────────

class UsageTrackingRepository:
    """MongoDB repository for per-user per-month usage tracking."""

    def __init__(self):
        self._collection: Optional[AsyncIOMotorCollection] = None

    async def _get_collection(self) -> AsyncIOMotorCollection:
        if self._collection is None:
            from backend.db.mongodb.mongodb import MongoDB
            self._collection = await MongoDB.get_collection("usage_tracking")
            await self._create_indexes()
        return self._collection

    async def _create_indexes(self):
        await self._collection.create_index(
            [("user_id", ASCENDING), ("month", ASCENDING), ("year", ASCENDING)],
            unique=True,
        )
        logger.info("Usage tracking indexes created")

    async def get_current_usage(self, user_id: str) -> Optional[UsageTrackingInDB]:
        """Get usage for the current month."""
        now = datetime.utcnow()
        collection = await self._get_collection()
        doc = await collection.find_one({
            "user_id": user_id,
            "month": now.month,
            "year": now.year,
        })
        return UsageTrackingInDB(**doc) if doc else None

    async def get_or_create_current(self, user_id: str) -> UsageTrackingInDB:
        """Get or create usage document for current month (upsert)."""
        now = datetime.utcnow()
        collection = await self._get_collection()
        result = await collection.find_one_and_update(
            {
                "user_id": user_id,
                "month": now.month,
                "year": now.year,
            },
            {
                "$setOnInsert": {
                    "user_id": user_id,
                    "month": now.month,
                    "year": now.year,
                    "agent_runs": 0,
                    "documents_processed": 0,
                    "knowledge_base_queries": 0,
                    "file_uploads": 0,
                    "agent_usage_breakdown": {},
                    "created_at": now,
                },
                "$set": {"updated_at": now},
            },
            upsert=True,
            return_document=True,
        )
        return UsageTrackingInDB(**result)

    async def increment_agent_runs(
        self,
        user_id: str,
        agent_type: str = "unknown",
        count: int = 1,
    ) -> UsageTrackingInDB:
        """
        Atomically increment agent run count.
        Uses MongoDB $inc — race-condition safe.
        """
        now = datetime.utcnow()
        collection = await self._get_collection()
        result = await collection.find_one_and_update(
            {
                "user_id": user_id,
                "month": now.month,
                "year": now.year,
            },
            {
                "$inc": {
                    "agent_runs": count,
                    f"agent_usage_breakdown.{agent_type}": count,
                },
                "$setOnInsert": {
                    "user_id": user_id,
                    "month": now.month,
                    "year": now.year,
                    "documents_processed": 0,
                    "knowledge_base_queries": 0,
                    "file_uploads": 0,
                    "created_at": now,
                },
                "$set": {"updated_at": now},
            },
            upsert=True,
            return_document=True,
        )
        return UsageTrackingInDB(**result)

    async def increment_documents(self, user_id: str, count: int = 1) -> UsageTrackingInDB:
        """Atomically increment document count."""
        now = datetime.utcnow()
        collection = await self._get_collection()
        result = await collection.find_one_and_update(
            {"user_id": user_id, "month": now.month, "year": now.year},
            {
                "$inc": {"documents_processed": count},
                "$setOnInsert": {
                    "user_id": user_id,
                    "month": now.month,
                    "year": now.year,
                    "agent_runs": 0,
                    "knowledge_base_queries": 0,
                    "file_uploads": 0,
                    "agent_usage_breakdown": {},
                    "created_at": now,
                },
                "$set": {"updated_at": now},
            },
            upsert=True,
            return_document=True,
        )
        return UsageTrackingInDB(**result)

    async def increment_kb_queries(self, user_id: str, count: int = 1) -> UsageTrackingInDB:
        """Atomically increment knowledge base query count."""
        now = datetime.utcnow()
        collection = await self._get_collection()
        result = await collection.find_one_and_update(
            {"user_id": user_id, "month": now.month, "year": now.year},
            {
                "$inc": {"knowledge_base_queries": count},
                "$setOnInsert": {
                    "user_id": user_id,
                    "month": now.month,
                    "year": now.year,
                    "agent_runs": 0,
                    "documents_processed": 0,
                    "file_uploads": 0,
                    "agent_usage_breakdown": {},
                    "created_at": now,
                },
                "$set": {"updated_at": now},
            },
            upsert=True,
            return_document=True,
        )
        return UsageTrackingInDB(**result)

    async def get_usage_history(
        self,
        user_id: str,
        months: int = 6,
    ) -> List[UsageTrackingInDB]:
        """Get usage history for the last N months."""
        collection = await self._get_collection()
        cursor = collection.find(
            {"user_id": user_id}
        ).sort([("year", DESCENDING), ("month", DESCENDING)]).limit(months)
        results = []
        async for doc in cursor:
            results.append(UsageTrackingInDB(**doc))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Payment Event Repository (Idempotency)
# ─────────────────────────────────────────────────────────────────────────────

class PaymentEventRepository:
    """
    MongoDB repository for tracking processed Stripe webhook events.
    Ensures idempotency — each Stripe event is processed exactly once.
    """

    def __init__(self):
        self._collection: Optional[AsyncIOMotorCollection] = None

    async def _get_collection(self) -> AsyncIOMotorCollection:
        if self._collection is None:
            from backend.db.mongodb.mongodb import MongoDB
            self._collection = await MongoDB.get_collection("payment_events")
            await self._create_indexes()
        return self._collection

    async def _create_indexes(self):
        await self._collection.create_index("stripe_event_id", unique=True)
        await self._collection.create_index("user_id")
        await self._collection.create_index("event_type")
        await self._collection.create_index("processed_at")
        logger.info("Payment event indexes created")

    async def is_event_processed(self, stripe_event_id: str) -> bool:
        """Check if a Stripe event has already been processed (idempotency)."""
        collection = await self._get_collection()
        doc = await collection.find_one({"stripe_event_id": stripe_event_id})
        return doc is not None

    async def record_event(self, event: PaymentEventInDB) -> bool:
        """
        Record a processed event. Returns False if already exists (idempotent).
        Uses unique index on stripe_event_id to prevent duplicates atomically.
        """
        collection = await self._get_collection()
        try:
            doc = event.model_dump()
            doc["processed_at"] = datetime.utcnow()
            await collection.insert_one(doc)
            return True
        except DuplicateKeyError:
            logger.warning(
                "Duplicate webhook event — already processed",
                stripe_event_id=event.stripe_event_id,
            )
            return False

    async def get_events_for_user(
        self,
        user_id: str,
        limit: int = 50,
    ) -> List[PaymentEventInDB]:
        """Get recent payment events for a user."""
        collection = await self._get_collection()
        cursor = collection.find(
            {"user_id": user_id}
        ).sort("processed_at", DESCENDING).limit(limit)
        results = []
        async for doc in cursor:
            results.append(PaymentEventInDB(**doc))
        return results


# ─────────────────────────────────────────────────────────────────────────────
# Singleton instances
# ─────────────────────────────────────────────────────────────────────────────

subscription_repository = SubscriptionRepository()
usage_tracking_repository = UsageTrackingRepository()
payment_event_repository = PaymentEventRepository()


# Import and initialize credits and invoice repositories
async def get_credits_repository():
    """Get credits repository with lazy initialization."""
    from backend.db.mongodb.mongodb import MongoDB
    from backend.db.mongodb.repositories.credits_repository import CreditLedgerRepository
    collection = await MongoDB.get_collection("credit_ledger")
    repo = CreditLedgerRepository(collection)
    await repo.ensure_indexes()
    return repo


async def get_invoice_repository():
    """Get invoice repository with lazy initialization."""
    from backend.db.mongodb.mongodb import MongoDB
    from backend.db.mongodb.repositories.invoice_repository import InvoiceRepository
    collection = await MongoDB.get_collection("invoices")
    repo = InvoiceRepository(collection)
    await repo.ensure_indexes()
    return repo
