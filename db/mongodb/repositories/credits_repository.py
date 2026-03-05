"""
Lumicoria AI — Credits Ledger Repository

Handles credit transactions, balance tracking, and audit trail.
All operations are atomic to prevent race conditions.
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from motor.motor_asyncio import AsyncIOMotorCollection
from pymongo import DESCENDING, IndexModel
import structlog

from backend.models.billing import (
    CreditLedgerInDB,
    TransactionType,
    CreditBalanceResponse,
    CreditTransactionResponse,
    CreditLedgerResponse,
)
from backend.db.mongodb.base_repository import BaseRepository

logger = structlog.get_logger(__name__)


class CreditLedgerRepository(BaseRepository[CreditLedgerInDB]):
    """Repository for credit transaction ledger with atomic balance tracking."""

    def __init__(self, collection: AsyncIOMotorCollection):
        super().__init__(collection, CreditLedgerInDB)
        self._balance_collection_name = "credit_balances"

    async def ensure_indexes(self):
        """Create indexes for efficient querying and uniqueness."""
        indexes = [
            IndexModel([("user_id", DESCENDING), ("created_at", DESCENDING)]),
            IndexModel([("user_id", DESCENDING), ("transaction_type", DESCENDING)]),
            IndexModel([("stripe_invoice_id", DESCENDING)], sparse=True),
            IndexModel([("agent_run_id", DESCENDING)], sparse=True),
        ]
        await self.collection.create_indexes(indexes)
        logger.info("Credit ledger indexes created")

    async def get_balance(self, user_id: str) -> int:
        """
        Get user's current credit balance.
        Uses a separate collection for O(1) balance lookups.
        """
        db = self.collection.database
        balance_collection = db[self._balance_collection_name]
        
        result = await balance_collection.find_one({"user_id": user_id})
        if result:
            return result.get("balance", 0)
        
        # Initialize balance if not found
        await balance_collection.update_one(
            {"user_id": user_id},
            {"$set": {"balance": 0, "updated_at": datetime.utcnow()}},
            upsert=True
        )
        return 0

    async def add_transaction(
        self,
        user_id: str,
        transaction_type: TransactionType,
        amount: int,
        description: str,
        stripe_invoice_id: Optional[str] = None,
        stripe_payment_intent_id: Optional[str] = None,
        agent_run_id: Optional[str] = None,
        document_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        created_by: Optional[str] = None,
    ) -> CreditLedgerInDB:
        """
        Add a credit transaction atomically.
        
        Args:
            user_id: User ID
            transaction_type: Type of transaction (credit, debit, etc.)
            amount: Amount in credits (positive for credit, negative for debit)
            description: Human-readable description
            
        Returns:
            Created transaction record with updated balance
        """
        db = self.collection.database
        balance_collection = db[self._balance_collection_name]
        
        # Atomic balance update
        result = await balance_collection.find_one_and_update(
            {"user_id": user_id},
            {
                "$inc": {"balance": amount},
                "$set": {"updated_at": datetime.utcnow()},
                "$setOnInsert": {"created_at": datetime.utcnow()}
            },
            upsert=True,
            return_document=True  # Return updated document
        )
        
        balance_after = result["balance"]
        
        # Create ledger entry
        transaction = CreditLedgerInDB(
            user_id=user_id,
            transaction_type=transaction_type,
            amount=amount,
            balance_after=balance_after,
            description=description,
            stripe_invoice_id=stripe_invoice_id,
            stripe_payment_intent_id=stripe_payment_intent_id,
            agent_run_id=agent_run_id,
            document_id=document_id,
            metadata=metadata or {},
            created_by=created_by,
            created_at=datetime.utcnow(),
        )
        
        doc = transaction.model_dump(by_alias=True)
        await self.collection.insert_one(doc)
        
        logger.info(
            "Credit transaction created",
            user_id=user_id,
            transaction_type=transaction_type,
            amount=amount,
            balance_after=balance_after,
        )
        
        return transaction

    async def get_ledger(
        self,
        user_id: str,
        limit: int = 50,
        skip: int = 0,
        transaction_type: Optional[TransactionType] = None,
    ) -> CreditLedgerResponse:
        """
        Get user's credit transaction history with pagination.
        
        Args:
            user_id: User ID
            limit: Maximum number of transactions to return
            skip: Number of transactions to skip (for pagination)
            transaction_type: Filter by transaction type (optional)
            
        Returns:
            Paginated ledger response with current balance
        """
        # Build query
        query = {"user_id": user_id}
        if transaction_type:
            query["transaction_type"] = transaction_type
        
        # Get total count
        total_count = await self.collection.count_documents(query)
        
        # Get transactions
        cursor = self.collection.find(query).sort("created_at", DESCENDING).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)
        
        transactions = [
            CreditTransactionResponse(
                transaction_type=doc["transaction_type"],
                amount=doc["amount"],
                balance_after=doc["balance_after"],
                description=doc["description"],
                created_at=doc["created_at"],
                metadata=doc.get("metadata"),
            )
            for doc in docs
        ]
        
        # Get current balance
        balance = await self.get_balance(user_id)
        
        page = (skip // limit) + 1 if limit > 0 else 1
        
        return CreditLedgerResponse(
            balance=balance,
            transactions=transactions,
            total_count=total_count,
            page=page,
            page_size=limit,
        )

    async def deduct_credits(
        self,
        user_id: str,
        amount: int,
        description: str,
        agent_run_id: Optional[str] = None,
        document_id: Optional[str] = None,
    ) -> bool:
        """
        Deduct credits from user's balance.
        Returns False if insufficient balance.
        
        Args:
            user_id: User ID
            amount: Amount to deduct (positive number)
            description: Reason for deduction
            
        Returns:
            True if successful, False if insufficient balance
        """
        current_balance = await self.get_balance(user_id)
        
        if current_balance < amount:
            logger.warning(
                "Insufficient credits",
                user_id=user_id,
                required=amount,
                available=current_balance,
            )
            return False
        
        # Deduct as negative amount
        await self.add_transaction(
            user_id=user_id,
            transaction_type=TransactionType.DEBIT,
            amount=-amount,
            description=description,
            agent_run_id=agent_run_id,
            document_id=document_id,
        )
        
        return True

    async def grant_credits(
        self,
        user_id: str,
        amount: int,
        description: str,
        transaction_type: TransactionType = TransactionType.CREDIT,
        created_by: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> CreditLedgerInDB:
        """
        Grant credits to user (for payments, bonuses, adjustments).
        
        Args:
            user_id: User ID
            amount: Amount to grant (positive number)
            description: Reason for grant
            transaction_type: Type of credit (CREDIT, BONUS, ADJUSTMENT, REFUND)
            created_by: Admin user ID (for adjustments)
            
        Returns:
            Created transaction record
        """
        return await self.add_transaction(
            user_id=user_id,
            transaction_type=transaction_type,
            amount=amount,
            description=description,
            created_by=created_by,
            metadata=metadata,
        )


# Singleton instance factory
def get_credit_ledger_repository(collection: AsyncIOMotorCollection) -> CreditLedgerRepository:
    """Factory function for dependency injection."""
    return CreditLedgerRepository(collection)
