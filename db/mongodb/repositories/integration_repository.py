from typing import Optional, List, Dict, Any, Union
from pymongo import ASCENDING, DESCENDING
from bson import ObjectId
from datetime import datetime, timedelta
from ..base_repository import BaseRepository
from backend.models.integration import Integration, IntegrationCreate, IntegrationConfig, IntegrationType
from backend.models.mongodb_models import (
    MongoBaseModel,
    PyObjectId
)
import structlog
import json
from cryptography.fernet import Fernet
import os

logger = structlog.get_logger()

class IntegrationRepository(BaseRepository[Integration]):
    def __init__(self):
        super().__init__("integrations", Integration)
        # Initialize encryption key for sensitive data
        self._encryption_key = os.getenv("INTEGRATION_ENCRYPTION_KEY", Fernet.generate_key())
        self._cipher_suite = Fernet(self._encryption_key)

    async def _create_indexes(self):
        collection = await self.collection
        # Create indexes for common queries
        await collection.create_index("organization_id")
        await collection.create_index("created_by")
        await collection.create_index("config.type")
        await collection.create_index("status")
        await collection.create_index("created_at")
        # Compound indexes for common queries
        await collection.create_index([
            ("organization_id", ASCENDING),
            ("config.type", ASCENDING),
            ("status", ASCENDING)
        ])
        await collection.create_index([
            ("created_by", ASCENDING),
            ("config.type", ASCENDING),
            ("created_at", DESCENDING)
        ])
        # Text search index for name and error logs
        await collection.create_index([
            ("name", "text"),
            ("error_log.message", "text")
        ])

    def _encrypt_credentials(self, credentials: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive credentials before storing."""
        encrypted_creds = {}
        for key, value in credentials.items():
            if isinstance(value, (str, int, float, bool)):
                encrypted_value = self._cipher_suite.encrypt(str(value).encode())
                encrypted_creds[key] = encrypted_value.decode()
            elif isinstance(value, dict):
                encrypted_creds[key] = self._encrypt_credentials(value)
            else:
                encrypted_creds[key] = value
        return encrypted_creds

    def _decrypt_credentials(self, encrypted_creds: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive credentials after retrieval."""
        decrypted_creds = {}
        for key, value in encrypted_creds.items():
            if isinstance(value, str):
                try:
                    decrypted_value = self._cipher_suite.decrypt(value.encode())
                    decrypted_creds[key] = decrypted_value.decode()
                except Exception:
                    decrypted_creds[key] = value
            elif isinstance(value, dict):
                decrypted_creds[key] = self._decrypt_credentials(value)
            else:
                decrypted_creds[key] = value
        return decrypted_creds

    async def create_integration(
        self,
        integration_data: IntegrationCreate,
        organization_id: str,
        created_by: str
    ) -> Integration:
        """Create a new integration with encrypted credentials."""
        entry_dict = integration_data.dict()
        
        # Encrypt sensitive credentials
        if "credentials" in entry_dict:
            entry_dict["credentials"] = self._encrypt_credentials(entry_dict["credentials"])
        
        entry_dict.update({
            "organization_id": ObjectId(organization_id),
            "created_by": ObjectId(created_by),
            "created_at": datetime.utcnow(),
            "updated_at": datetime.utcnow(),
            "status": "active",
            "error_log": []
        })

        try:
            return await self.create(entry_dict)
        except Exception as e:
            logger.error(
                "Failed to create integration",
                error=str(e),
                organization_id=organization_id,
                integration_type=integration_data.type
            )
            raise

    async def update_integration(
        self,
        integration_id: str,
        update_data: Dict[str, Any],
        encrypt_credentials: bool = True
    ) -> Optional[Integration]:
        """Update integration details with optional credential encryption."""
        if "credentials" in update_data and encrypt_credentials:
            update_data["credentials"] = self._encrypt_credentials(update_data["credentials"])
        
        update_data["updated_at"] = datetime.utcnow()
        return await self.update(integration_id, update_data)

    async def get_integration_by_id(
        self,
        integration_id: str,
        decrypt_credentials: bool = True
    ) -> Optional[Integration]:
        """Get integration by ID with optional credential decryption."""
        integration = await self.find_one({"_id": ObjectId(integration_id)})
        if integration and decrypt_credentials and "credentials" in integration:
            integration["credentials"] = self._decrypt_credentials(integration["credentials"])
        return integration

    async def get_organization_integrations(
        self,
        organization_id: str,
        integration_type: Optional[IntegrationType] = None,
        status: Optional[str] = None,
        skip: int = 0,
        limit: int = 100
    ) -> List[Integration]:
        """Get all integrations for an organization with filtering."""
        filters = {"organization_id": ObjectId(organization_id)}
        if integration_type:
            filters["config.type"] = integration_type
        if status:
            filters["status"] = status

        return await self.find_many(
            filters,
            skip=skip,
            limit=limit,
            sort=[("created_at", DESCENDING)]
        )

    async def add_error_log(
        self,
        integration_id: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None
    ) -> Optional[Integration]:
        """Add an error log entry to the integration."""
        error_entry = {
            "timestamp": datetime.utcnow(),
            "message": error_message,
            "details": error_details or {}
        }

        update_data = {
            "$push": {"error_log": error_entry},
            "updated_at": datetime.utcnow()
        }
        return await self.update(integration_id, update_data)

    async def update_sync_status(
        self,
        integration_id: str,
        last_sync: datetime,
        sync_status: str,
        sync_details: Optional[Dict[str, Any]] = None
    ) -> Optional[Integration]:
        """Update the last sync status of an integration."""
        update_data = {
            "config.last_sync": last_sync,
            "sync_status": sync_status,
            "sync_details": sync_details or {},
            "updated_at": datetime.utcnow()
        }
        return await self.update(integration_id, update_data)

    async def get_active_webhooks(
        self,
        integration_id: str
    ) -> List[Dict[str, Any]]:
        """Get all active webhooks for an integration."""
        integration = await self.find_one({"_id": ObjectId(integration_id)})
        if not integration or "webhooks" not in integration:
            return []
        return [webhook for webhook in integration["webhooks"] if webhook.get("is_active", True)]

    async def add_webhook(
        self,
        integration_id: str,
        webhook_data: Dict[str, Any]
    ) -> Optional[Integration]:
        """Add a new webhook to an integration."""
        webhook_data.update({
            "id": str(ObjectId()),
            "created_at": datetime.utcnow(),
            "is_active": True
        })

        update_data = {
            "$push": {"webhooks": webhook_data},
            "updated_at": datetime.utcnow()
        }
        return await self.update(integration_id, update_data)

    async def update_webhook_status(
        self,
        integration_id: str,
        webhook_id: str,
        is_active: bool,
        last_triggered: Optional[datetime] = None
    ) -> Optional[Integration]:
        """Update the status of a webhook."""
        update_data = {
            "$set": {
                "webhooks.$[webhook].is_active": is_active,
                "updated_at": datetime.utcnow()
            }
        }
        if last_triggered:
            update_data["$set"]["webhooks.$[webhook].last_triggered"] = last_triggered

        return await self.update(
            integration_id,
            update_data,
            array_filters=[{"webhook.id": webhook_id}]
        )

    async def get_integration_stats(
        self,
        organization_id: str,
        time_range: Optional[timedelta] = None
    ) -> Dict[str, Any]:
        """Get statistics about integrations in an organization."""
        match = {"organization_id": ObjectId(organization_id)}
        if time_range:
            match["created_at"] = {
                "$gte": datetime.utcnow() - time_range
            }

        pipeline = [
            {"$match": match},
            {"$group": {
                "_id": {
                    "type": "$config.type",
                    "status": "$status"
                },
                "count": {"$sum": 1},
                "active_webhooks": {
                    "$sum": {
                        "$size": {
                            "$filter": {
                                "input": "$webhooks",
                                "as": "webhook",
                                "cond": {"$eq": ["$$webhook.is_active", True]}
                            }
                        }
                    }
                },
                "error_count": {
                    "$sum": {"$size": "$error_log"}
                }
            }},
            {"$group": {
                "_id": "$_id.type",
                "statuses": {
                    "$push": {
                        "status": "$_id.status",
                        "count": "$count",
                        "active_webhooks": "$active_webhooks",
                        "error_count": "$error_count"
                    }
                }
            }}
        ]

        results = await self.aggregate(pipeline)
        return {
            result["_id"]: result["statuses"]
            for result in results
        }

    async def get_integration_health(
        self,
        integration_id: str,
        time_range: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """Get health metrics for an integration."""
        integration = await self.find_one({"_id": ObjectId(integration_id)})
        if not integration:
            return {}

        # Calculate error rate
        recent_errors = [
            error for error in integration.get("error_log", [])
            if error["timestamp"] >= datetime.utcnow() - time_range
        ]
        error_rate = len(recent_errors) / time_range.total_seconds() if time_range.total_seconds() > 0 else 0

        # Get webhook statistics
        webhooks = integration.get("webhooks", [])
        active_webhooks = [w for w in webhooks if w.get("is_active", True)]
        webhook_success_rate = 0
        if webhooks:
            successful_webhooks = sum(
                1 for w in webhooks
                if w.get("last_triggered") and w.get("last_status") == "success"
            )
            webhook_success_rate = successful_webhooks / len(webhooks)

        return {
            "status": integration.get("status"),
            "last_sync": integration.get("config", {}).get("last_sync"),
            "error_rate": error_rate,
            "recent_errors": len(recent_errors),
            "active_webhooks": len(active_webhooks),
            "webhook_success_rate": webhook_success_rate,
            "sync_status": integration.get("sync_status"),
            "last_error": recent_errors[-1] if recent_errors else None
        }

# Create a singleton instance
integration_repository = IntegrationRepository() 