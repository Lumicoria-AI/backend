from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime

from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.models.integration import (
    Integration, IntegrationCreate, IntegrationUpdate,
    IntegrationType, IntegrationStatus
)
from backend.db.mongodb.repositories.integration_repository import integration_repository
from backend.services.integration_service import integration_service

router = APIRouter()


# ── Response / Request Models ─────────────────────────────────────────────

class IntegrationResponse(BaseModel):
    id: str
    name: str
    type: IntegrationType
    organization_id: str
    created_by: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    status: str
    config: Optional[Dict[str, Any]] = None
    sync_status: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class IntegrationConnectRequest(BaseModel):
    """Request to connect an integration with credentials."""
    type: IntegrationType
    name: str
    credentials: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class IntegrationActionRequest(BaseModel):
    """Request to execute an action on an integration."""
    action: str
    data: Dict[str, Any] = Field(default_factory=dict)


class IntegrationActionResponse(BaseModel):
    success: bool
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class AvailableIntegration(BaseModel):
    type: str
    name: str
    description: str
    icon: str
    available_actions: List[str]
    is_configured: bool
    status: str
    category: str


class IntegrationHealthResponse(BaseModel):
    status: Optional[str] = None
    last_sync: Optional[datetime] = None
    error_rate: float = 0.0
    recent_errors: int = 0
    active_webhooks: int = 0
    webhook_success_rate: float = 0.0


# ── Catalog (no auth needed — public info) ────────────────────────────────

INTEGRATION_CATALOG = {
    "google_workspace": {
        "name": "Google Workspace",
        "description": "Calendar, Drive, Docs, Sheets, Gmail — your full Google productivity suite.",
        "icon": "/images/integrations/google-workspace.png",
        "category": "productivity",
        "actions": [
            "create_calendar_event", "list_calendars", "get_upcoming_events",
            "create_document", "list_files", "create_project_folder",
            "send_email", "create_project", "create_project_database",
            "add_project_task", "get_project_tasks", "update_project_task",
            "export_meeting_to_google_workspace",
        ],
        "credential_fields": [
            {"key": "credentials_json", "label": "Service Account JSON", "type": "file"},
            {"key": "delegated_email", "label": "Delegated Email (optional)", "type": "email"},
        ],
    },
    "slack": {
        "name": "Slack",
        "description": "Channels, messages, reminders, file sharing — keep your team in sync.",
        "icon": "/images/integrations/slack.png",
        "category": "communication",
        "actions": [
            "create_project_channel", "add_project_task", "export_meeting_notes",
            "create_reminder", "search_project_content", "upload_project_file",
            "get_channel_members", "archive_project_channel",
        ],
        "credential_fields": [
            {"key": "bot_token", "label": "Bot Token (xoxb-...)", "type": "password"},
            {"key": "app_token", "label": "App Token (xapp-...)", "type": "password"},
            {"key": "signing_secret", "label": "Signing Secret", "type": "password"},
        ],
    },
    "notion": {
        "name": "Notion",
        "description": "Pages, databases, knowledge bases — structured docs and project management.",
        "icon": "/images/integrations/notion.png",
        "category": "productivity",
        "actions": [
            "create_project", "create_project_database", "add_project_task",
            "search_projects", "get_project_tasks", "update_task_status",
            "create_meeting_notes", "export_meeting_to_notion", "create_knowledge_base",
        ],
        "credential_fields": [
            {"key": "api_key", "label": "Internal Integration Token", "type": "password"},
            {"key": "workspace_id", "label": "Workspace ID (optional)", "type": "text"},
        ],
    },
    "salesforce": {
        "name": "Salesforce",
        "description": "CRM contacts, leads, opportunities — manage your sales pipeline.",
        "icon": "/images/integrations/salesforce.png",
        "category": "crm",
        "actions": [],
        "credential_fields": [
            {"key": "client_id", "label": "Consumer Key", "type": "text"},
            {"key": "client_secret", "label": "Consumer Secret", "type": "password"},
            {"key": "instance_url", "label": "Instance URL", "type": "url"},
            {"key": "refresh_token", "label": "Refresh Token", "type": "password"},
        ],
    },
    "stripe": {
        "name": "Stripe",
        "description": "Payments, subscriptions, invoices — your billing backbone.",
        "icon": "/images/integrations/stripe.png",
        "category": "payments",
        "actions": [],
        "credential_fields": [
            {"key": "secret_key", "label": "Secret Key (sk_...)", "type": "password"},
            {"key": "webhook_secret", "label": "Webhook Secret (whsec_...)", "type": "password"},
        ],
    },
}


@router.get("/catalog", response_model=List[AvailableIntegration])
async def get_integration_catalog(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Get full catalog of available integrations with their connection status.
    """
    configured = integration_service.get_available_integrations()
    user_integrations = await integration_repository.get_organization_integrations(
        organization_id=str(current_user.id),
    )

    user_integration_map: Dict[str, Dict] = {}
    for integ in user_integrations:
        integ_type = integ.get("config", {}).get("type") or integ.get("type", "")
        user_integration_map[integ_type] = integ

    result = []
    for type_key, info in INTEGRATION_CATALOG.items():
        server_configured = configured.get(type_key, {}).get("available", False)
        user_integ = user_integration_map.get(type_key)
        integ_status = "not_connected"
        if user_integ:
            integ_status = user_integ.get("status", "active")
        elif server_configured:
            integ_status = "available"

        result.append(AvailableIntegration(
            type=type_key,
            name=info["name"],
            description=info["description"],
            icon=info["icon"],
            available_actions=info["actions"],
            is_configured=server_configured or bool(user_integ),
            status=integ_status,
            category=info["category"],
        ))
    return result


@router.get("/catalog/{integration_type}")
async def get_integration_catalog_detail(
    integration_type: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get detailed catalog info for a specific integration type, including credential fields."""
    info = INTEGRATION_CATALOG.get(integration_type)
    if not info:
        raise HTTPException(status_code=404, detail=f"Unknown integration type: {integration_type}")
    return {
        "type": integration_type,
        **info,
    }


# ── Connect / Disconnect ──────────────────────────────────────────────────

@router.post("/connect", response_model=IntegrationResponse)
async def connect_integration(
    request: IntegrationConnectRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Connect a new integration by saving encrypted credentials.
    """
    integration_data = IntegrationCreate(
        name=request.name,
        type=request.type,
        credentials=request.credentials,
        config=None,
        metadata=request.metadata,
    )
    try:
        integration = await integration_repository.create_integration(
            integration_data=integration_data,
            organization_id=str(current_user.id),
            created_by=str(current_user.id),
        )
        return integration
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/{integration_id}/disconnect")
async def disconnect_integration(
    integration_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Disconnect (deactivate) an integration — keeps the record but clears credentials."""
    integration = await integration_repository.get_integration_by_id(integration_id)
    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")

    await integration_repository.update_integration(
        integration_id=integration_id,
        update_data={"status": "inactive", "credentials": {}},
        encrypt_credentials=False,
    )
    return {"success": True, "message": "Integration disconnected"}


@router.post("/{integration_id}/reconnect", response_model=IntegrationResponse)
async def reconnect_integration(
    integration_id: str,
    credentials: Dict[str, Any],
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Re-connect a previously disconnected integration with new credentials."""
    integration = await integration_repository.get_integration_by_id(integration_id)
    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")

    updated = await integration_repository.update_integration(
        integration_id=integration_id,
        update_data={"status": "active", "credentials": credentials},
    )
    return updated


# ── Execute Actions ────────────────────────────────────────────────────────

@router.post("/{integration_id}/execute", response_model=IntegrationActionResponse)
async def execute_integration_action(
    integration_id: str,
    request: IntegrationActionRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Execute an action on a connected integration."""
    integration = await integration_repository.get_integration_by_id(integration_id)
    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")

    integ_type = integration.get("config", {}).get("type") or integration.get("type", "")

    try:
        result = await integration_service.execute_integration_action(
            integration_type=integ_type,
            action=request.action,
            data=request.data,
        )
        return IntegrationActionResponse(success=True, result=result)
    except Exception as e:
        await integration_repository.add_error_log(
            integration_id=integration_id,
            error_message=str(e),
            error_details={"action": request.action, "data": request.data},
        )
        return IntegrationActionResponse(success=False, error=str(e))


@router.get("/{integration_id}/actions")
async def get_integration_actions(
    integration_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get available actions for a connected integration."""
    integration = await integration_repository.get_integration_by_id(integration_id)
    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")

    integ_type = integration.get("config", {}).get("type") or integration.get("type", "")
    catalog_info = INTEGRATION_CATALOG.get(integ_type, {})
    return {
        "integration_id": integration_id,
        "type": integ_type,
        "actions": catalog_info.get("actions", []),
    }


# ── Health & Sync ──────────────────────────────────────────────────────────

@router.get("/{integration_id}/health", response_model=IntegrationHealthResponse)
async def get_integration_health(
    integration_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get health metrics for an integration."""
    health = await integration_repository.get_integration_health(integration_id)
    if not health:
        raise HTTPException(status_code=404, detail="Integration not found")
    return health


@router.post("/{integration_id}/sync")
async def trigger_sync(
    integration_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Trigger a manual sync for an integration."""
    integration = await integration_repository.get_integration_by_id(integration_id)
    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")

    await integration_repository.update_sync_status(
        integration_id=integration_id,
        last_sync=datetime.utcnow(),
        sync_status="success",
    )
    return {"success": True, "message": "Sync triggered"}


# ── Standard CRUD ──────────────────────────────────────────────────────────

@router.post("/", response_model=IntegrationResponse)
async def create_integration(
    integration_in: IntegrationCreate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Create a new integration configuration."""
    try:
        integration = await integration_repository.create_integration(
            integration_data=integration_in,
            organization_id=str(current_user.id),
            created_by=str(current_user.id),
        )
        return integration
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{integration_id}", response_model=IntegrationResponse)
async def get_integration(
    integration_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get integration details by ID."""
    integration = await integration_repository.get_integration_by_id(integration_id)
    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")
    return integration


@router.put("/{integration_id}", response_model=IntegrationResponse)
async def update_integration(
    integration_id: str,
    integration_in: IntegrationUpdate,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Update an integration configuration."""
    updated = await integration_repository.update_integration(
        integration_id=integration_id,
        update_data=integration_in.dict(exclude_unset=True),
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Integration not found")
    return updated


@router.delete("/{integration_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_integration(
    integration_id: str,
    current_user: User = Depends(get_current_active_user),
) -> None:
    """Delete an integration configuration."""
    success = await integration_repository.delete(integration_id)
    if not success:
        raise HTTPException(status_code=404, detail="Integration not found")


@router.get("/", response_model=List[IntegrationResponse])
async def list_integrations(
    integration_type: Optional[IntegrationType] = Query(None, alias="type"),
    integration_status: Optional[str] = Query(None, alias="status"),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """List integrations for the current organization."""
    integrations = await integration_repository.get_organization_integrations(
        organization_id=str(current_user.id),
        integration_type=integration_type,
        status=integration_status,
    )
    return integrations


# ── Stats ──────────────────────────────────────────────────────────────────

@router.get("/stats/overview")
async def get_integration_stats(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get integration statistics for the organization."""
    stats = await integration_repository.get_integration_stats(
        organization_id=str(current_user.id),
    )
    return stats
