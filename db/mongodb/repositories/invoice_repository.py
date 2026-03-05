"""
Lumicoria AI — Invoice Repository

Handles invoice/receipt storage and retrieval from MongoDB.
Invoices are synced from Stripe webhooks.
"""

from datetime import datetime
from typing import List, Optional
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import DESCENDING, IndexModel
import structlog

from backend.models.billing import InvoiceInDB, InvoiceResponse, InvoiceListResponse
from backend.db.mongodb.base_repository import BaseRepository

logger = structlog.get_logger(__name__)


class InvoiceRepository(BaseRepository[InvoiceInDB]):
    """Repository for invoice storage and retrieval."""

    def __init__(self, collection: AsyncIOMotorCollection):
        super().__init__(collection, InvoiceInDB)

    async def ensure_indexes(self):
        """Create indexes for efficient querying."""
        indexes = [
            IndexModel([("user_id", DESCENDING), ("invoice_date", DESCENDING)]),
            IndexModel([("stripe_invoice_id", DESCENDING)], unique=True),
            IndexModel([("stripe_customer_id", DESCENDING)]),
            IndexModel([("status", DESCENDING)]),
        ]
        await self.collection.create_indexes(indexes)
        logger.info("Invoice repository indexes created")

    async def create_or_update_from_stripe(
        self,
        user_id: str,
        stripe_invoice: dict,
    ) -> InvoiceInDB:
        """
        Create or update invoice from Stripe invoice object.
        
        Args:
            user_id: User ID
            stripe_invoice: Stripe invoice object from webhook
            
        Returns:
            Created or updated invoice record
        """
        invoice_data = InvoiceInDB(
            user_id=user_id,
            stripe_invoice_id=stripe_invoice["id"],
            stripe_customer_id=stripe_invoice["customer"],
            stripe_subscription_id=stripe_invoice.get("subscription"),
            invoice_number=stripe_invoice.get("number"),
            invoice_pdf_url=stripe_invoice.get("invoice_pdf"),
            hosted_invoice_url=stripe_invoice.get("hosted_invoice_url"),
            amount_due=stripe_invoice["amount_due"],
            amount_paid=stripe_invoice["amount_paid"],
            currency=stripe_invoice["currency"],
            status=stripe_invoice["status"],
            invoice_date=datetime.fromtimestamp(stripe_invoice["created"]),
            due_date=datetime.fromtimestamp(stripe_invoice["due_date"]) if stripe_invoice.get("due_date") else None,
            paid_at=datetime.fromtimestamp(stripe_invoice["status_transitions"]["paid_at"]) if stripe_invoice.get("status_transitions", {}).get("paid_at") else None,
            line_items=[
                {
                    "description": item["description"],
                    "amount": item["amount"],
                    "currency": item["currency"],
                    "quantity": item.get("quantity", 1),
                }
                for item in stripe_invoice.get("lines", {}).get("data", [])
            ],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
        )

        # Upsert by stripe_invoice_id
        doc = invoice_data.model_dump(by_alias=True)
        await self.collection.update_one(
            {"stripe_invoice_id": stripe_invoice["id"]},
            {"$set": doc},
            upsert=True
        )

        logger.info(
            "Invoice synced from Stripe",
            user_id=user_id,
            stripe_invoice_id=stripe_invoice["id"],
            status=stripe_invoice["status"],
        )

        return invoice_data

    async def get_by_stripe_invoice_id(self, stripe_invoice_id: str) -> Optional[InvoiceInDB]:
        """Get invoice by Stripe invoice ID."""
        doc = await self.collection.find_one({"stripe_invoice_id": stripe_invoice_id})
        if doc:
            return InvoiceInDB(**doc)
        return None

    async def get_user_invoices(
        self,
        user_id: str,
        limit: int = 50,
        skip: int = 0,
        status: Optional[str] = None,
    ) -> InvoiceListResponse:
        """
        Get user's invoices with pagination.
        
        Args:
            user_id: User ID
            limit: Maximum number of invoices to return
            skip: Number of invoices to skip
            status: Filter by status (optional)
            
        Returns:
            Paginated list of invoices
        """
        query = {"user_id": user_id}
        if status:
            query["status"] = status

        total_count = await self.collection.count_documents(query)
        
        cursor = self.collection.find(query).sort("invoice_date", DESCENDING).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)

        invoices = [
            InvoiceResponse(
                invoice_id=doc["stripe_invoice_id"],
                invoice_number=doc.get("invoice_number"),
                amount_due=doc["amount_due"],
                amount_paid=doc["amount_paid"],
                currency=doc["currency"],
                status=doc["status"],
                invoice_date=doc["invoice_date"],
                paid_at=doc.get("paid_at"),
                invoice_pdf_url=doc.get("invoice_pdf_url"),
                hosted_invoice_url=doc.get("hosted_invoice_url"),
                line_items=doc.get("line_items", []),
            )
            for doc in docs
        ]

        return InvoiceListResponse(
            invoices=invoices,
            total_count=total_count,
        )


def get_invoice_repository(collection: AsyncIOMotorCollection) -> InvoiceRepository:
    """Factory function for dependency injection."""
    return InvoiceRepository(collection)
