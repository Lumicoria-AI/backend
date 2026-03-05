"""
Integration Tests for Billing Enforcement & Invoice Export

Run with: pytest backend/tests/test_billing_integration.py -v
"""

import pytest
from datetime import datetime
from unittest.mock import Mock, AsyncMock, patch
from backend.models.billing import (
    SubscriptionPlan,
    SubscriptionStatus,
    TransactionType,
    SubscriptionInDB,
    CreditLedgerInDB,
    InvoiceInDB,
)


class TestBillingEnforcement:
    """Test billing enforcement on agent endpoints."""
    
    @pytest.mark.asyncio
    async def test_enforce_agent_limit_blocks_inactive_subscription(self):
        """Test that inactive subscriptions are blocked."""
        from backend.core.billing import enforce_agent_limit
        from fastapi import HTTPException
        
        # Mock token with user_id
        mock_token = {"user_id": "user_123", "email": "test@example.com"}
        
        # Mock subscription service to return inactive subscription
        with patch("backend.services.billing_service.require_active_subscription") as mock_sub:
            mock_sub.side_effect = ValueError("Subscription is not active")
            
            with pytest.raises(HTTPException) as exc_info:
                await enforce_agent_limit(token_data=mock_token)
            
            assert exc_info.value.status_code == 402
            assert "Payment Required" in str(exc_info.value.detail) or "active" in str(exc_info.value.detail).lower()
    
    @pytest.mark.asyncio
    async def test_enforce_agent_limit_blocks_exceeded_quota(self):
        """Test that exceeded monthly quota is blocked."""
        from backend.core.billing import enforce_agent_limit
        from fastapi import HTTPException
        from backend.models.billing import SubscriptionResponse
        
        mock_token = {"user_id": "user_123", "email": "test@example.com"}
        
        # Mock active subscription but exceeded limit
        mock_subscription = SubscriptionResponse(
            plan=SubscriptionPlan.FREE,
            status=SubscriptionStatus.ACTIVE,
            is_active=True
        )
        
        with patch("backend.services.billing_service.require_active_subscription") as mock_sub:
            mock_sub.return_value = mock_subscription
            
            with patch("backend.services.billing_service.enforce_agent_run_limit") as mock_enforce:
                mock_enforce.side_effect = ValueError("Monthly limit reached (50/50)")
                
                with pytest.raises(HTTPException) as exc_info:
                    await enforce_agent_limit(token_data=mock_token)
                
                assert exc_info.value.status_code == 429
                assert "Too Many Requests" in str(exc_info.value.status_code) or "limit" in str(exc_info.value.detail).lower()


class TestCreditsLedger:
    """Test credits ledger system."""
    
    @pytest.mark.asyncio
    async def test_get_balance_returns_zero_for_new_user(self):
        """Test that new users start with zero balance."""
        from backend.db.mongodb.repositories.credits_repository import CreditLedgerRepository
        
        # Mock collection
        mock_collection = Mock()
        mock_db = Mock()
        mock_balance_collection = AsyncMock()
        mock_balance_collection.find_one.return_value = None
        mock_balance_collection.update_one = AsyncMock()
        mock_db.__getitem__.return_value = mock_balance_collection
        mock_collection.database = mock_db
        
        repo = CreditLedgerRepository(mock_collection)
        balance = await repo.get_balance("new_user_123")
        
        assert balance == 0
        mock_balance_collection.update_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_add_transaction_updates_balance_atomically(self):
        """Test that transactions update balance atomically."""
        from backend.db.mongodb.repositories.credits_repository import CreditLedgerRepository
        
        mock_collection = AsyncMock()
        mock_db = Mock()
        mock_balance_collection = AsyncMock()
        
        # Mock find_one_and_update to return updated balance
        mock_balance_collection.find_one_and_update.return_value = {
            "user_id": "user_123",
            "balance": 2900
        }
        
        mock_db.__getitem__.return_value = mock_balance_collection
        mock_collection.database = mock_db
        mock_collection.insert_one = AsyncMock()
        
        repo = CreditLedgerRepository(mock_collection)
        
        transaction = await repo.add_transaction(
            user_id="user_123",
            transaction_type=TransactionType.CREDIT,
            amount=2900,
            description="Payment received",
            stripe_invoice_id="in_test123"
        )
        
        assert transaction.amount == 2900
        assert transaction.balance_after == 2900
        assert transaction.transaction_type == TransactionType.CREDIT
        mock_balance_collection.find_one_and_update.assert_called_once()
        mock_collection.insert_one.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_deduct_credits_fails_on_insufficient_balance(self):
        """Test that credit deduction fails with insufficient balance."""
        from backend.db.mongodb.repositories.credits_repository import CreditLedgerRepository
        
        mock_collection = AsyncMock()
        mock_db = Mock()
        mock_balance_collection = AsyncMock()
        mock_balance_collection.find_one.return_value = {"balance": 50}
        mock_db.__getitem__.return_value = mock_balance_collection
        mock_collection.database = mock_db
        
        repo = CreditLedgerRepository(mock_collection)
        
        # Try to deduct 100 credits when only 50 available
        result = await repo.deduct_credits(
            user_id="user_123",
            amount=100,
            description="Expensive operation"
        )
        
        assert result is False


class TestInvoiceRepository:
    """Test invoice repository."""
    
    @pytest.mark.asyncio
    async def test_create_invoice_from_stripe(self):
        """Test invoice creation from Stripe webhook."""
        from backend.db.mongodb.repositories.invoice_repository import InvoiceRepository
        
        mock_collection = AsyncMock()
        mock_collection.update_one = AsyncMock()
        mock_collection.create_indexes = AsyncMock()
        
        repo = InvoiceRepository(mock_collection)
        
        stripe_invoice = {
            "id": "in_test123",
            "customer": "cus_test123",
            "subscription": "sub_test123",
            "number": "ABC-0001",
            "invoice_pdf": "https://stripe.com/pdf",
            "hosted_invoice_url": "https://stripe.com/invoice",
            "amount_due": 2900,
            "amount_paid": 2900,
            "currency": "usd",
            "status": "paid",
            "created": 1709640000,
            "due_date": None,
            "status_transitions": {"paid_at": 1709640300},
            "lines": {
                "data": [
                    {
                        "description": "Starter Plan",
                        "amount": 2900,
                        "currency": "usd",
                        "quantity": 1
                    }
                ]
            }
        }
        
        invoice = await repo.create_or_update_from_stripe(
            user_id="user_123",
            stripe_invoice=stripe_invoice
        )
        
        assert invoice.stripe_invoice_id == "in_test123"
        assert invoice.amount_paid == 2900
        assert invoice.status == "paid"
        assert len(invoice.line_items) == 1
        mock_collection.update_one.assert_called_once()


class TestBillingService:
    """Test billing service methods."""
    
    @pytest.mark.asyncio
    async def test_grant_credits_for_payment(self):
        """Test credits are granted on payment."""
        from backend.services.billing_service import grant_credits_for_payment
        
        mock_credits_repo = AsyncMock()
        mock_credits_repo.grant_credits = AsyncMock()
        
        await grant_credits_for_payment(
            user_id="user_123",
            credits_repo=mock_credits_repo,
            amount_paid=2900,
            stripe_invoice_id="in_test123"
        )
        
        # Verify grant_credits was called with correct amount
        mock_credits_repo.grant_credits.assert_called_once()
        call_args = mock_credits_repo.grant_credits.call_args
        assert call_args.kwargs["user_id"] == "user_123"
        assert call_args.kwargs["amount"] == 2900
        assert "Invoice: in_test123" in call_args.kwargs["description"]
    
    @pytest.mark.asyncio
    async def test_get_invoice_pdf_requires_authorization(self):
        """Test that invoice PDF access requires user ownership."""
        from backend.services.billing_service import get_invoice_pdf
        
        # Mock invoice repository
        mock_invoice_repo = AsyncMock()
        mock_invoice = Mock()
        mock_invoice.user_id = "user_123"
        mock_invoice.invoice_pdf_url = None
        mock_invoice_repo.get_by_stripe_invoice_id.return_value = mock_invoice
        
        # User tries to access someone else's invoice
        result = await get_invoice_pdf(
            user_id="user_456",
            invoice_id="in_test123",
            invoice_repo=mock_invoice_repo
        )
        
        assert result is None  # Access denied


class TestAPIEndpoints:
    """Test API endpoint responses."""
    
    @pytest.mark.asyncio
    async def test_get_invoices_endpoint_returns_paginated_list(self):
        """Test invoice list endpoint."""
        from backend.api.v1.endpoints.billing import get_invoices
        from backend.models.billing import InvoiceListResponse, InvoiceResponse
        
        mock_token = {"user_id": "user_123"}
        
        with patch("backend.api.v1.endpoints.billing.get_invoice_repository") as mock_get_repo:
            mock_invoice_repo = AsyncMock()
            mock_invoice_repo.get_user_invoices.return_value = InvoiceListResponse(
                invoices=[
                    InvoiceResponse(
                        invoice_id="in_1",
                        invoice_number="ABC-0001",
                        amount_due=2900,
                        amount_paid=2900,
                        currency="usd",
                        status="paid",
                        invoice_date=datetime.utcnow(),
                        paid_at=datetime.utcnow(),
                        invoice_pdf_url="https://stripe.com/pdf",
                        hosted_invoice_url="https://stripe.com/invoice",
                        line_items=[]
                    )
                ],
                total_count=1
            )
            mock_get_repo.return_value = mock_invoice_repo
            
            with patch("backend.services.billing_service.get_user_invoices") as mock_service:
                mock_service.return_value = mock_invoice_repo.get_user_invoices.return_value
                
                result = await get_invoices(
                    token_data=mock_token,
                    limit=50,
                    skip=0,
                    status=None
                )
                
                assert result.total_count == 1
                assert len(result.invoices) == 1
                assert result.invoices[0].invoice_id == "in_1"


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
