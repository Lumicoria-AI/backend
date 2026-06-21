from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
import json
import secrets
import urllib.parse
import aiohttp
import base64
import time

from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.models.integration import (
    Integration, IntegrationCreate, IntegrationUpdate,
    IntegrationType, IntegrationStatus
)
from backend.db.mongodb.repositories.integration_repository import integration_repository
from backend.services.integration_service import integration_service
from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger("lumicoria.api.integrations")

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
    type: IntegrationType
    name: str
    credentials: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None
    metadata: Optional[Dict[str, Any]] = None


class IntegrationActionRequest(BaseModel):
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
    auth_method: str = "oauth"


class IntegrationHealthResponse(BaseModel):
    status: Optional[str] = None
    last_sync: Optional[datetime] = None
    error_rate: float = 0.0
    recent_errors: int = 0
    active_webhooks: int = 0
    webhook_success_rate: float = 0.0


class OAuthStartResponse(BaseModel):
    auth_url: str
    state: str


class OAuthCallbackRequest(BaseModel):
    code: str
    state: str
    provider: str


# ── OAuth Provider Configuration ──────────────────────────────────────────

OAUTH_PROVIDERS = {
    "google_workspace": {
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "revoke_url": "https://oauth2.googleapis.com/revoke",
        "scopes": [
            "https://www.googleapis.com/auth/calendar",
            "https://www.googleapis.com/auth/drive",
            "https://www.googleapis.com/auth/drive.readonly",
            "https://www.googleapis.com/auth/documents",
            "https://www.googleapis.com/auth/spreadsheets",
            # Gmail read + send (read powers the autonomous morning brain).
            "https://www.googleapis.com/auth/gmail.readonly",
            "https://www.googleapis.com/auth/gmail.modify",
            "https://www.googleapis.com/auth/gmail.send",
        ],
    },
    "slack": {
        "auth_url": "https://slack.com/oauth/v2/authorize",
        "token_url": "https://slack.com/api/oauth.v2.access",
        "revoke_url": "https://slack.com/api/auth.revoke",
        "scopes": [
            "channels:read", "channels:manage", "chat:write", "users:read",
            "files:write", "search:read", "reminders:read", "reminders:write",
            "reactions:write",
        ],
    },
    "notion": {
        "auth_url": "https://api.notion.com/v1/oauth/authorize",
        "token_url": "https://api.notion.com/v1/oauth/token",
        "revoke_url": None,
        "scopes": [],
    },
    "salesforce": {
        "auth_url": "https://login.salesforce.com/services/oauth2/authorize",
        "token_url": "https://login.salesforce.com/services/oauth2/token",
        "revoke_url": "https://login.salesforce.com/services/oauth2/revoke",
        "scopes": ["full", "refresh_token", "api"],
    },
}

STATE_TTL_SECONDS = 600  # 10 minutes


class _OAuthStateStore:
    """Redis-backed OAuth state store.

    The state token guards against CSRF and replay. We persist it in Redis
    (not in process memory) so the authorize and the callback can land on
    different uvicorn workers, replicas, or VMs. TTL is server-side so
    expired states evict automatically.

    Key format: ``oauth:state:{token}`` → JSON ``{user_id, provider, created_at}``
    """

    _PREFIX = "oauth:state:"

    @classmethod
    def _key(cls, state: str) -> str:
        return f"{cls._PREFIX}{state}"

    @classmethod
    async def put(cls, state: str, *, user_id: str, provider: str) -> None:
        from backend.db.redis.redis import RedisClient
        payload = json.dumps({
            "user_id": user_id,
            "provider": provider,
            "created_at": time.time(),
        })
        await RedisClient.set(cls._key(state), payload, expire=STATE_TTL_SECONDS)

    @classmethod
    async def take(cls, state: str) -> Optional[Dict[str, Any]]:
        """Atomically read + delete (single-use)."""
        from backend.db.redis.redis import RedisClient
        client = await RedisClient.get_client()
        # GETDEL is atomic in Redis ≥ 6.2; falls back to GET + DELETE for safety.
        try:
            raw = await client.getdel(cls._key(state))
        except Exception:
            raw = await client.get(cls._key(state))
            if raw:
                await client.delete(cls._key(state))
        if not raw:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None


# Backward-compatible no-op kept so any in-process callers (tests, scripts)
# don't break. Redis handles expiry server-side.
def _cleanup_expired_states() -> None:
    return None


def _get_oauth_client_credentials(provider: str) -> tuple[str, str]:
    """Get client_id and client_secret for an OAuth provider from settings."""
    if provider == "google_workspace":
        client_id = settings.GOOGLE_OAUTH_CLIENT_ID
        client_secret = settings.GOOGLE_OAUTH_CLIENT_SECRET
    elif provider == "slack":
        client_id = settings.SLACK_CLIENT_ID
        client_secret = settings.SLACK_CLIENT_SECRET
    elif provider == "notion":
        client_id = settings.NOTION_OAUTH_CLIENT_ID
        client_secret = settings.NOTION_OAUTH_CLIENT_SECRET
    elif provider == "salesforce":
        client_id = settings.SALESFORCE_CLIENT_ID
        client_secret = settings.SALESFORCE_CLIENT_SECRET
    else:
        raise ValueError(f"Unknown provider: {provider}")

    if not client_id or not client_secret:
        raise ValueError(
            f"OAuth not configured for {provider}. "
            f"Set the client_id and client_secret in your environment variables."
        )
    return client_id, client_secret


# ── Integration Catalog ───────────────────────────────────────────────────

INTEGRATION_CATALOG = {
    "google_workspace": {
        "name": "Google Workspace",
        "description": "Calendar, Drive, Docs, Sheets, Gmail — your full Google productivity suite.",
        "icon": "/images/integrations/google-workspace.png",
        "category": "productivity",
        "auth_method": "oauth",
        "actions": [
            "create_calendar_event", "list_calendars", "get_upcoming_events",
            "create_document", "list_files", "create_project_folder",
            "send_email", "create_project", "create_project_database",
            "add_project_task", "get_project_tasks", "update_project_task",
            "export_meeting_to_google_workspace",
        ],
        "credential_fields": [],
    },
    "slack": {
        "name": "Slack",
        "description": "Channels, messages, reminders, file sharing — keep your team in sync.",
        "icon": "/images/integrations/slack.png",
        "category": "communication",
        "auth_method": "oauth",
        "actions": [
            "create_project_channel", "add_project_task", "export_meeting_notes",
            "create_reminder", "search_project_content", "upload_project_file",
            "get_channel_members", "archive_project_channel",
        ],
        "credential_fields": [],
    },
    "notion": {
        "name": "Notion",
        "description": "Pages, databases, knowledge bases — structured docs and project management.",
        "icon": "/images/integrations/notion.png",
        "category": "productivity",
        "auth_method": "oauth",
        "actions": [
            "create_project", "create_project_database", "add_project_task",
            "search_projects", "get_project_tasks", "update_task_status",
            "create_meeting_notes", "export_meeting_to_notion", "create_knowledge_base",
        ],
        "credential_fields": [],
    },
    "salesforce": {
        "name": "Salesforce",
        "description": "CRM contacts, leads, opportunities — manage your sales pipeline.",
        "icon": "/images/integrations/salesforce.png",
        "category": "crm",
        "auth_method": "oauth",
        "actions": [
            "get_contacts", "create_contact", "get_leads", "create_lead",
            "get_opportunities", "create_opportunity", "search_records",
        ],
        "credential_fields": [],
    },
    "stripe": {
        "name": "Stripe",
        "description": "Payments, subscriptions, invoices — your billing backbone.",
        "icon": "/images/integrations/stripe.png",
        "category": "payments",
        "auth_method": "credentials",
        "actions": [],
        "credential_fields": [
            {"key": "secret_key", "label": "Secret Key (sk_...)", "type": "password"},
            {"key": "webhook_secret", "label": "Webhook Secret (whsec_...)", "type": "password"},
        ],
    },
}


# ── Catalog Endpoints ─────────────────────────────────────────────────────

@router.get("/catalog", response_model=List[AvailableIntegration])
async def get_integration_catalog(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get full catalog of available integrations with their connection status."""
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
            auth_method=info.get("auth_method", "oauth"),
        ))
    return result


@router.get("/catalog/{integration_type}")
async def get_integration_catalog_detail(
    integration_type: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get detailed catalog info for a specific integration type."""
    info = INTEGRATION_CATALOG.get(integration_type)
    if not info:
        raise HTTPException(status_code=404, detail=f"Unknown integration type: {integration_type}")
    result = {"type": integration_type, **info}
    # Normalize field name for frontend compatibility
    if "actions" in result and "available_actions" not in result:
        result["available_actions"] = result.pop("actions")
    return result


# ── OAuth 2.0 Endpoints ──────────────────────────────────────────────────

@router.get("/oauth/{provider}/authorize", response_model=OAuthStartResponse)
async def oauth_authorize(
    provider: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Start an OAuth flow. Returns the provider's authorization URL for the frontend
    to open in a popup. The user authenticates there and gets redirected back.
    """
    if provider not in OAUTH_PROVIDERS:
        raise HTTPException(status_code=400, detail=f"OAuth not supported for: {provider}")

    try:
        client_id, _ = _get_oauth_client_credentials(provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    state = secrets.token_urlsafe(32)
    await _OAuthStateStore.put(state, user_id=str(current_user.id), provider=provider)

    provider_config = OAUTH_PROVIDERS[provider]
    redirect_uri = f"{settings.FRONTEND_URL}/integrations/oauth/callback"

    params: Dict[str, str] = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "state": state,
    }

    if provider == "google_workspace":
        params["scope"] = " ".join(provider_config["scopes"])
        params["access_type"] = "offline"
        params["prompt"] = "consent"
    elif provider == "slack":
        params["scope"] = ",".join(provider_config["scopes"])
    elif provider == "notion":
        params["owner"] = "user"
    elif provider == "salesforce":
        params["scope"] = " ".join(provider_config["scopes"])
        params["prompt"] = "login consent"

    auth_url = f"{provider_config['auth_url']}?{urllib.parse.urlencode(params)}"

    return OAuthStartResponse(auth_url=auth_url, state=state)


@router.post("/oauth/callback", response_model=IntegrationResponse)
async def oauth_callback(
    request: OAuthCallbackRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Exchange an OAuth authorization code for tokens and store them encrypted.
    Called by the frontend after the popup redirect.
    """
    pending = await _OAuthStateStore.take(request.state)
    if not pending:
        raise HTTPException(status_code=400, detail="Invalid or expired OAuth state. Please try again.")
    if pending["user_id"] != str(current_user.id):
        raise HTTPException(status_code=403, detail="OAuth state does not match current user.")
    if pending["provider"] != request.provider:
        raise HTTPException(status_code=400, detail="Provider mismatch.")

    try:
        client_id, client_secret = _get_oauth_client_credentials(request.provider)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    provider_config = OAUTH_PROVIDERS[request.provider]
    redirect_uri = f"{settings.FRONTEND_URL}/integrations/oauth/callback"

    # Exchange the authorization code for tokens
    try:
        credentials = await _exchange_code_for_tokens(
            provider=request.provider,
            code=request.code,
            client_id=client_id,
            client_secret=client_secret,
            redirect_uri=redirect_uri,
            token_url=provider_config["token_url"],
        )
    except Exception as e:
        logger.error(f"OAuth token exchange failed for {request.provider}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to connect: {str(e)}")

    catalog_info = INTEGRATION_CATALOG.get(request.provider, {})

    # Check if user already has this integration → update instead of create
    existing = await integration_repository.get_organization_integrations(
        organization_id=str(current_user.id),
        integration_type=request.provider,
    )

    if existing:
        updated = await integration_repository.update_integration(
            integration_id=str(existing[0].get("_id", existing[0].get("id"))),
            update_data={"credentials": credentials, "status": "active"},
        )
        return updated
    else:
        integration_data = IntegrationCreate(
            name=catalog_info.get("name", request.provider),
            type=request.provider,
            credentials=credentials,
        )
        integration = await integration_repository.create_integration(
            integration_data=integration_data,
            organization_id=str(current_user.id),
            created_by=str(current_user.id),
        )
        return integration


async def _exchange_code_for_tokens(
    provider: str,
    code: str,
    client_id: str,
    client_secret: str,
    redirect_uri: str,
    token_url: str,
) -> Dict[str, Any]:
    """Exchange an OAuth authorization code for access/refresh tokens."""

    async with aiohttp.ClientSession() as session:
        if provider == "google_workspace":
            async with session.post(token_url, data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            }) as resp:
                data = await resp.json()
                if "error" in data:
                    raise ValueError(f"Google OAuth error: {data.get('error_description', data['error'])}")
                return {
                    "access_token": data["access_token"],
                    "refresh_token": data.get("refresh_token"),
                    "client_id": client_id,
                    "client_secret": client_secret,
                    "token_uri": "https://oauth2.googleapis.com/token",
                    "expires_at": str(time.time() + data.get("expires_in", 3600)),
                    "scope": data.get("scope", ""),
                }

        elif provider == "slack":
            async with session.post(token_url, data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
            }) as resp:
                data = await resp.json()
                if not data.get("ok"):
                    raise ValueError(f"Slack OAuth error: {data.get('error', 'unknown')}")
                return {
                    "bot_token": data["access_token"],
                    "team_id": data.get("team", {}).get("id"),
                    "team_name": data.get("team", {}).get("name"),
                    "bot_user_id": data.get("bot_user_id"),
                    "authed_user_token": data.get("authed_user", {}).get("access_token"),
                }

        elif provider == "notion":
            auth_header = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
            async with session.post(
                token_url,
                json={
                    "grant_type": "authorization_code",
                    "code": code,
                    "redirect_uri": redirect_uri,
                },
                headers={
                    "Authorization": f"Basic {auth_header}",
                    "Content-Type": "application/json",
                },
            ) as resp:
                data = await resp.json()
                if "error" in data:
                    raise ValueError(f"Notion OAuth error: {data.get('error', 'unknown')}")
                return {
                    "api_key": data["access_token"],
                    "workspace_id": data.get("workspace_id"),
                    "workspace_name": data.get("workspace_name"),
                    "bot_id": data.get("bot_id"),
                }

        elif provider == "salesforce":
            async with session.post(token_url, data={
                "code": code,
                "client_id": client_id,
                "client_secret": client_secret,
                "redirect_uri": redirect_uri,
                "grant_type": "authorization_code",
            }) as resp:
                data = await resp.json()
                if "error" in data:
                    raise ValueError(f"Salesforce OAuth error: {data.get('error_description', data['error'])}")
                return {
                    "access_token": data["access_token"],
                    "refresh_token": data.get("refresh_token"),
                    "instance_url": data.get("instance_url"),
                    "token_type": data.get("token_type", "Bearer"),
                    "issued_at": data.get("issued_at"),
                    "id_url": data.get("id"),
                }

        else:
            raise ValueError(f"Unsupported OAuth provider: {provider}")


# ── Connect / Disconnect ──────────────────────────────────────────────────

@router.post("/connect", response_model=IntegrationResponse)
async def connect_integration(
    request: IntegrationConnectRequest,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Connect a new integration with manual credentials (for non-OAuth providers)."""
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
    """Disconnect an integration — revokes tokens with the provider and clears stored credentials."""
    integration = await integration_repository.get_integration_by_id(integration_id)
    if not integration:
        raise HTTPException(status_code=404, detail="Integration not found")

    integ_type = integration.get("config", {}).get("type") or integration.get("type", "")
    credentials = integration.get("credentials", {})

    # Attempt to revoke the token with the provider
    await _revoke_provider_token(integ_type, credentials)

    await integration_repository.update_integration(
        integration_id=integration_id,
        update_data={"status": "inactive", "credentials": {}},
        encrypt_credentials=False,
    )
    return {"success": True, "message": "Integration disconnected"}


async def _revoke_provider_token(provider: str, credentials: Dict[str, Any]):
    """Best-effort revocation of OAuth tokens with the provider."""
    try:
        async with aiohttp.ClientSession() as session:
            if provider == "google_workspace" and credentials.get("access_token"):
                await session.post(
                    "https://oauth2.googleapis.com/revoke",
                    params={"token": credentials["access_token"]},
                )
            elif provider == "slack" and credentials.get("bot_token"):
                await session.post(
                    "https://slack.com/api/auth.revoke",
                    headers={"Authorization": f"Bearer {credentials['bot_token']}"},
                )
            elif provider == "salesforce" and credentials.get("access_token"):
                await session.post(
                    "https://login.salesforce.com/services/oauth2/revoke",
                    params={"token": credentials["access_token"]},
                )
    except Exception as e:
        logger.warning(f"Token revocation failed for {provider}: {e}")


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
            integration_id=integration_id,
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
