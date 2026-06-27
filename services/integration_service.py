from typing import Any, Dict, List, Optional
import structlog
import json
import httpx
from datetime import datetime, timedelta
from pathlib import Path

from backend.db.mongodb.repositories.integration_repository import integration_repository
from backend.models.integration import Integration, IntegrationCreate
from backend.integrations.notion import NotionIntegration
from backend.integrations.google_workspace import GoogleWorkspaceIntegration
from backend.integrations.slack import SlackIntegration
from ..core.config import settings
from backend.core.logging import get_logger

logger = get_logger("lumicoria.services.integration")


class IntegrationService:
    """Service for managing external integrations.

    Initializes server-level integrations from env vars at startup.
    Can also create per-user integration instances from stored credentials.
    """

    def __init__(self):
        self.integrations: Dict[str, Any] = {}
        self._initialize_integrations()

    def _initialize_integrations(self) -> None:
        """Initialize available integrations from environment variables."""

        # Notion
        if settings.NOTION_API_KEY:
            try:
                self.integrations["notion"] = NotionIntegration(
                    api_token=settings.NOTION_API_KEY
                )
                logger.info("Notion integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Notion integration: {e}")

        # Google Workspace
        if settings.GOOGLE_CREDENTIALS_FILE:
            try:
                creds_path = Path(settings.GOOGLE_CREDENTIALS_FILE)
                if creds_path.exists():
                    credentials_info = json.loads(creds_path.read_text())
                    self.integrations["google_workspace"] = GoogleWorkspaceIntegration(
                        credentials_info=credentials_info
                    )
                    logger.info("Google Workspace integration initialized")
                else:
                    logger.warning(f"Google credentials file not found: {settings.GOOGLE_CREDENTIALS_FILE}")
            except Exception as e:
                logger.error(f"Failed to initialize Google Workspace integration: {e}")

        # Slack
        if settings.SLACK_BOT_TOKEN:
            try:
                self.integrations["slack"] = SlackIntegration()
                logger.info("Slack integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Slack integration: {e}")

    # ── Per-user integration instances ─────────────────────────────────────

    async def get_user_integration(
        self, integration_id: str
    ) -> Optional[Any]:
        """
        Create an integration instance from user-stored credentials.
        Handles both legacy (manual paste) and OAuth credential shapes.
        Falls back to the server-level instance if no user credentials exist.
        """
        record = await integration_repository.get_integration_by_id(
            integration_id, decrypt_credentials=True
        )
        if not record:
            return None

        integ_type = record.get("config", {}).get("type") or record.get("type", "")
        credentials = record.get("credentials", {})

        if not credentials:
            return self.integrations.get(integ_type)

        try:
            # ── Notion ────────────────────────────────────────────
            if integ_type == "notion":
                # OAuth shape: { access_token, ... }
                # Legacy shape: { api_key }
                token = credentials.get("access_token") or credentials.get("api_key")
                if token:
                    return NotionIntegration(api_token=token)

            # ── Google Workspace ──────────────────────────────────
            elif integ_type == "google_workspace":
                # OAuth shape: { access_token, refresh_token, token_type, expires_in, scope }
                if credentials.get("access_token"):
                    credentials = await self._ensure_google_token_fresh(
                        integration_id, credentials
                    )
                    # Build an OAuth2 credentials dict that the Google client can use
                    oauth_creds = {
                        "token": credentials["access_token"],
                        "refresh_token": credentials.get("refresh_token"),
                        "token_uri": "https://oauth2.googleapis.com/token",
                        "client_id": settings.GOOGLE_OAUTH_CLIENT_ID,
                        "client_secret": settings.GOOGLE_OAUTH_CLIENT_SECRET,
                        "scopes": (credentials.get("scope") or "").split(),
                    }
                    return GoogleWorkspaceIntegration(credentials_info=oauth_creds)
                # Legacy shape: { credentials_json }
                elif credentials.get("credentials_json"):
                    creds = credentials["credentials_json"]
                    if isinstance(creds, str):
                        creds = json.loads(creds)
                    return GoogleWorkspaceIntegration(credentials_info=creds)

            # ── Slack ─────────────────────────────────────────────
            elif integ_type == "slack":
                # OAuth shape: { access_token, ... }
                # Legacy shape: { bot_token }
                bot_token = credentials.get("access_token") or credentials.get("bot_token")
                if bot_token:
                    return SlackIntegration(bot_token=bot_token)

            # ── Salesforce ────────────────────────────────────────
            elif integ_type == "salesforce":
                # OAuth shape: { access_token, refresh_token, instance_url }
                # Returns credentials dict — no dedicated integration class yet,
                # but the credentials are stored and available for future use
                if credentials.get("access_token") and credentials.get("instance_url"):
                    return {
                        "type": "salesforce",
                        "access_token": credentials["access_token"],
                        "instance_url": credentials["instance_url"],
                        "refresh_token": credentials.get("refresh_token"),
                    }

        except Exception as e:
            logger.error(f"Failed to create user integration instance: {e}")

        return self.integrations.get(integ_type)

    async def _ensure_google_token_fresh(
        self, integration_id: str, credentials: dict
    ) -> dict:
        """
        Check if a Google OAuth access_token is expired (or about to expire).
        If so, use the refresh_token to obtain a new one and persist it.
        Returns the (possibly updated) credentials dict.
        """
        expires_at = credentials.get("expires_at")
        if expires_at:
            # expires_at may be: an epoch int/float, an ISO string, or
            # a stringified float (the integration encrypt layer coerces
            # numerics to str before encryption, so after decrypt every
            # value is a string).
            expiry = None
            if isinstance(expires_at, (int, float)):
                expiry = datetime.utcfromtimestamp(float(expires_at))
            else:
                s = str(expires_at).strip()
                # Try numeric epoch first — handles "1782289517.909566".
                try:
                    expiry = datetime.utcfromtimestamp(float(s))
                except (TypeError, ValueError):
                    try:
                        expiry = datetime.fromisoformat(s)
                    except ValueError:
                        logger.warning(
                            "Could not parse Google expires_at; refreshing token "
                            "preemptively",
                            value_preview=s[:32],
                        )
            # Refresh if we couldn't parse, or less than 5 minutes remain.
            if expiry is not None and datetime.utcnow() < expiry - timedelta(minutes=5):
                return credentials

        refresh_token = credentials.get("refresh_token")
        if not refresh_token:
            logger.warning("Google token may be expired but no refresh_token available")
            return credentials

        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    "https://oauth2.googleapis.com/token",
                    data={
                        "grant_type": "refresh_token",
                        "refresh_token": refresh_token,
                        "client_id": settings.GOOGLE_OAUTH_CLIENT_ID,
                        "client_secret": settings.GOOGLE_OAUTH_CLIENT_SECRET,
                    },
                )
                resp.raise_for_status()
                token_data = resp.json()

            # Update credentials with new token info
            credentials["access_token"] = token_data["access_token"]
            expires_in = token_data.get("expires_in", 3600)
            credentials["expires_at"] = (
                datetime.utcnow() + timedelta(seconds=expires_in)
            ).isoformat()
            if token_data.get("refresh_token"):
                credentials["refresh_token"] = token_data["refresh_token"]

            # Persist updated credentials back to DB
            await integration_repository.update_integration(
                integration_id,
                {"credentials": credentials},
                encrypt_credentials=True,
            )
            logger.info("Refreshed Google OAuth token", integration_id=integration_id)

        except Exception as e:
            logger.error(f"Failed to refresh Google token: {e}")

        return credentials

    # ── Action execution ───────────────────────────────────────────────────

    async def execute_integration_action(
        self,
        integration_type: str,
        action: str,
        data: Dict[str, Any],
        integration_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute an action on an integration.
        If integration_id is provided, uses per-user credentials; otherwise, server-level.
        """
        try:
            if integration_id:
                integration = await self.get_user_integration(integration_id)
            else:
                integration = self.integrations.get(integration_type)

            if not integration:
                raise ValueError(f"Integration type '{integration_type}' not available")

            method = self._resolve_action(integration, integration_type, action)
            if not method:
                raise ValueError(f"Action '{action}' not supported for '{integration_type}'")

            return await method(**data)

        except Exception as e:
            logger.error(
                f"Error executing integration action: {e}",
                integration_type=integration_type,
                action=action,
            )
            raise

    def _resolve_action(self, integration: Any, integration_type: str, action: str):
        """Resolve an action name to an integration method."""
        # Try direct method lookup first
        method = getattr(integration, action, None)
        if method and callable(method):
            return method

        # Fallback to explicit action maps for backwards compat
        action_maps = {
            "notion": {
                "create_project": "create_project",
                "add_task": "add_project_task",
                "export_meeting_notes": "create_meeting_notes",
            },
            "google_workspace": {
                "create_calendar_event": "create_calendar_event",
                "create_document": "create_document",
                "send_email": "send_email",
            },
            "slack": {
                "create_project_channel": "create_project_channel",
                "add_project_task": "add_project_task",
                "export_meeting_notes": "export_meeting_notes",
                "create_reminder": "create_reminder",
                "search_project_content": "search_project_content",
                "upload_project_file": "upload_project_file",
                "get_channel_members": "get_channel_members",
                "archive_project_channel": "archive_project_channel",
            },
        }

        mapped = action_maps.get(integration_type, {}).get(action)
        if mapped:
            method = getattr(integration, mapped, None)
            if method and callable(method):
                return method

        return None

    # ── Availability ───────────────────────────────────────────────────────

    def get_available_integrations(self) -> Dict[str, Any]:
        """Get list of available server-level integrations."""
        return {
            "notion": {
                "available": "notion" in self.integrations,
                "actions": [
                    "create_project", "create_project_database", "add_project_task",
                    "search_projects", "get_project_tasks", "update_task_status",
                    "create_meeting_notes", "export_meeting_to_notion", "create_knowledge_base",
                ],
            },
            "google_workspace": {
                "available": "google_workspace" in self.integrations,
                "actions": [
                    "create_calendar_event", "list_calendars", "get_upcoming_events",
                    "create_document", "list_files", "create_project_folder",
                    "send_email", "create_project", "create_project_database",
                    "add_project_task", "get_project_tasks", "update_project_task",
                    "export_meeting_to_google_workspace",
                ],
            },
            "slack": {
                "available": "slack" in self.integrations,
                "actions": [
                    "create_project_channel", "add_project_task", "export_meeting_notes",
                    "create_reminder", "search_project_content", "upload_project_file",
                    "get_channel_members", "archive_project_channel",
                ],
            },
        }


# Singleton
integration_service = IntegrationService()
