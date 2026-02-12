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

# Initialize logger
logger = get_logger("lumicoria.services.integration")


class IntegrationService:
    """Service for managing external integrations."""

    def __init__(self):
        """Initialize integration service."""
        self.integrations: Dict[str, Any] = {}
        self._initialize_integrations()

    def _initialize_integrations(self) -> None:
        """Initialize available integrations (each independently — one failure doesn't block the rest)."""

        # ── Notion ──────────────────────────────────────────────────────
        if settings.NOTION_API_KEY:
            try:
                self.integrations["notion"] = NotionIntegration(
                    api_token=settings.NOTION_API_KEY
                )
                logger.info("Notion integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Notion integration: {e}")

        # ── Google Workspace ────────────────────────────────────────────
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
                    logger.warning(
                        f"Google credentials file not found: {settings.GOOGLE_CREDENTIALS_FILE}"
                    )
            except Exception as e:
                logger.error(f"Failed to initialize Google Workspace integration: {e}")

        # ── Slack ───────────────────────────────────────────────────────
        if settings.SLACK_BOT_TOKEN:
            try:
                self.integrations["slack"] = SlackIntegration()
                logger.info("Slack integration initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Slack integration: {e}")
            
    async def execute_integration_action(self,
                                       integration_type: str,
                                       action: str,
                                       data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an action on an integration.
        
        Args:
            integration_type: Type of integration (notion, google_workspace, slack)
            action: Action to execute
            data: Action parameters
            
        Returns:
            Dict containing action result
        """
        try:
            integration = self.integrations.get(integration_type)
            if not integration:
                raise ValueError(f"Integration type '{integration_type}' not found")
                
            # Map action to integration method
            if integration_type == "notion":
                if action == "create_project":
                    return await integration.create_project(**data)
                elif action == "add_task":
                    return await integration.add_task(**data)
                elif action == "export_meeting_notes":
                    return await integration.export_meeting_notes(**data)
                    
            elif integration_type == "google_workspace":
                if action == "create_calendar_event":
                    return await integration.create_calendar_event(**data)
                elif action == "create_document":
                    return await integration.create_document(**data)
                elif action == "send_email":
                    return await integration.send_email(**data)
                    
            elif integration_type == "slack":
                if action == "create_project_channel":
                    return await integration.create_project_channel(**data)
                elif action == "add_project_task":
                    return await integration.add_project_task(**data)
                elif action == "export_meeting_notes":
                    return await integration.export_meeting_notes(**data)
                elif action == "create_reminder":
                    return await integration.create_reminder(**data)
                elif action == "search_project_content":
                    return await integration.search_project_content(**data)
                elif action == "upload_project_file":
                    return await integration.upload_project_file(**data)
                elif action == "get_channel_members":
                    return await integration.get_channel_members(**data)
                elif action == "archive_project_channel":
                    return await integration.archive_project_channel(**data)
                    
            raise ValueError(f"Action '{action}' not supported for integration type '{integration_type}'")

        except Exception as e:
            logger.error(
                f"Error executing integration action: {str(e)}",
                integration_type=integration_type,
                action=action
            )
            raise
            
    def get_available_integrations(self) -> Dict[str, Any]:
        """
        Get list of available integrations.
        
        Returns:
            Dict containing integration information
        """
        return {
            "notion": {
                "available": "notion" in self.integrations,
                "actions": [
                    "create_project",
                    "add_task",
                    "export_meeting_notes"
                ]
            },
            "google_workspace": {
                "available": "google_workspace" in self.integrations,
                "actions": [
                    "create_calendar_event",
                    "create_document",
                    "send_email"
                ]
            },
            "slack": {
                "available": "slack" in self.integrations,
                "actions": [
                    "create_project_channel",
                    "add_project_task",
                    "export_meeting_notes",
                    "create_reminder",
                    "search_project_content",
                    "upload_project_file",
                    "get_channel_members",
                    "archive_project_channel"
                ]
            }
        }

# Create a singleton instance
integration_service = IntegrationService() 