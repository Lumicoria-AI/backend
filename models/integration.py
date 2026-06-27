from enum import Enum, auto
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator
from datetime import datetime

class IntegrationType(str, Enum):
    """Supported integration types."""
    GOOGLE_WORKSPACE = "google_workspace"
    SLACK = "slack"
    SALESFORCE = "salesforce"
    NOTION = "notion"
    STRIPE = "stripe"

class IntegrationStatus(str, Enum):
    """Status of an integration."""
    ACTIVE = "active"
    INACTIVE = "inactive"
    PENDING = "pending"
    ERROR = "error"
    EXPIRED = "expired"
    CONFIGURATION_REQUIRED = "configuration_required"

class IntegrationSyncStatus(BaseModel):
    """Status of the last synchronization."""
    last_sync_time: Optional[datetime] = None
    status: str = "never_synced"  # success, failed, in_progress, never_synced
    error_message: Optional[str] = None
    items_synced: Optional[int] = None
    
class IntegrationConfig(BaseModel):
    """Configuration for an integration."""
    type: IntegrationType
    sync_frequency: Optional[str] = None  # cron expression if automatic sync is enabled
    sync_enabled: bool = False
    webhooks_enabled: bool = False
    webhook_url: Optional[str] = None
    
class IntegrationErrorLog(BaseModel):
    """Error log entry for an integration."""
    timestamp: datetime
    error_message: str
    action: Optional[str] = None
    
class IntegrationUser(BaseModel):
    """User associated with an integration."""
    id: str
    name: Optional[str] = None
    email: Optional[str] = None
    role: Optional[str] = None

class IntegrationCreate(BaseModel):
    """Payload for creating a new integration."""
    name: str
    type: IntegrationType
    credentials: Dict[str, Any]
    config: Optional[IntegrationConfig] = None
    metadata: Optional[Dict[str, Any]] = None

class IntegrationUpdate(BaseModel):
    """Payload for updating an integration."""
    name: Optional[str] = None
    credentials: Optional[Dict[str, Any]] = None
    config: Optional[IntegrationConfig] = None
    metadata: Optional[Dict[str, Any]] = None
    status: Optional[str] = None

class Integration(BaseModel):
    """Integration model."""
    id: str = Field(default="", alias="_id")
    name: str
    type: IntegrationType
    credentials: Dict[str, Any]  # Encrypted in DB, decrypted when needed
    config: Optional[IntegrationConfig] = None
    organization_id: str
    created_by: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    status: str = "active"
    sync_status: Optional[IntegrationSyncStatus] = None
    users: Optional[List[IntegrationUser]] = None
    error_logs: Optional[List[IntegrationErrorLog]] = None
    metadata: Optional[Dict[str, Any]] = None

    class Config:
        populate_by_name = True
        arbitrary_types_allowed = True

    @field_validator("id", mode="before")
    @classmethod
    def _coerce_object_id(cls, v):
        """Mongo round-trips this field as a BSON ObjectId; the rest of the
        app treats it as a string. Coerce defensively so we accept either
        shape from the persistence layer."""
        return str(v) if v is not None else ""

    # ── Dict-style compatibility shims ──────────────────────────────
    # Many existing call sites read integration data as if it were a
    # raw Mongo dict (``integration.get("status")``, ``integration["x"]``).
    # After we started hydrating to the Pydantic model these calls
    # broke. Rather than rewrite every consumer, the model duck-types
    # as a dict for read access. Writes still go through Pydantic so
    # validation isn't bypassed.

    def get(self, key: str, default: Any = None) -> Any:  # noqa: A003
        # Map `_id` → `id` so legacy code reading either key works.
        if key == "_id":
            return self.id or default
        val = getattr(self, key, default)
        # Common chained pattern is ``integration.get("config", {}).get("type")``.
        # If the field exists but is None and the caller supplied a non-None
        # default, return the default so the chain doesn't crash on ``None.get``.
        if val is None and default is not None:
            return default
        return val

    def __getitem__(self, key: str) -> Any:
        if key == "_id":
            return self.id
        if not hasattr(self, key):
            raise KeyError(key)
        return getattr(self, key)

    def __contains__(self, key: str) -> bool:
        if key == "_id":
            return bool(self.id)
        return key in self.model_fields and getattr(self, key) is not None
