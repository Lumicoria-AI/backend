from typing import Any, Dict, List, Optional
import structlog
import json
from datetime import datetime
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
            if integ_type == "notion" and credentials.get("api_key"):
                return NotionIntegration(api_token=credentials["api_key"])
            elif integ_type == "google_workspace" and credentials.get("credentials_json"):
                creds = credentials["credentials_json"]
                if isinstance(creds, str):
                    creds = json.loads(creds)
                return GoogleWorkspaceIntegration(credentials_info=creds)
            elif integ_type == "slack" and credentials.get("bot_token"):
                return SlackIntegration()
        except Exception as e:
            logger.error(f"Failed to create user integration instance: {e}")

        return self.integrations.get(integ_type)

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
