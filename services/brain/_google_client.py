"""Per-run Google Workspace client resolver.

Every fetch node that touches Google calls ``resolve_google_client(user_id)``.
It:

  1. Looks up the user's google_workspace integration record (encrypted
     credentials live in MongoDB ``integrations`` collection).
  2. Refreshes the OAuth access_token if it expires within 5 min — the
     refresh path in ``integration_service._ensure_google_token_fresh``
     persists the new token + expiry back to the record.
  3. Builds a ``GoogleWorkspaceClient`` and returns it.

Returns ``None`` when:
  - The user has no Google integration at all.
  - The integration record exists but is marked ``status != "active"``
    (revoked, paused, errored).
  - Credentials decrypt fails (key rotation bug or corrupt record).

Returning None lets the gate node set ``skip_reason="no_google_integration"``
and the graph routes straight to audit. The fetch nodes themselves
also defensively check — they no-op + log when the client is missing.
"""

from __future__ import annotations

from typing import Optional

import structlog

logger = structlog.get_logger(__name__)


async def resolve_google_client(user_id: str) -> Optional[object]:
    """Resolve the user's Google Workspace client. None if not available."""
    if not user_id:
        return None

    try:
        from backend.db.mongodb.repositories.integration_repository import (
            integration_repository,
        )
        from backend.services.integration_service import integration_service
        from backend.services.ai_clients.google_workspace_client import (
            GoogleWorkspaceClient,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("brain.google_client_imports_failed", error=str(exc))
        return None

    # The integration store keys by `organization_id`. In personal mode
    # (user has no org), the user's own id is used as the org id —
    # matches what api/v1/endpoints/integrations.py does on connect.
    try:
        integrations = await integration_repository.get_organization_integrations(
            organization_id=user_id,
            integration_type="google_workspace",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "brain.google_integration_lookup_failed",
            user_id=user_id, error=str(exc),
        )
        return None

    if not integrations:
        return None

    active = next(
        (i for i in integrations if (i.get("status") or "").lower() == "active"),
        None,
    )
    if active is None:
        # No active record — frontend will show "Reconnect required."
        return None

    integration_id = str(active.get("_id") or active.get("id") or "")
    if not integration_id:
        return None

    # Decrypt + auto-refresh the token.
    try:
        decrypted = await integration_repository.get_integration_by_id(
            integration_id, decrypt_credentials=True,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "brain.google_decrypt_failed",
            user_id=user_id, error=str(exc),
        )
        return None

    if not decrypted or not decrypted.get("credentials"):
        return None

    credentials = decrypted["credentials"]
    try:
        if credentials.get("access_token"):
            credentials = await integration_service._ensure_google_token_fresh(
                integration_id, credentials,
            )
    except Exception as exc:  # noqa: BLE001
        # Token refresh failed — likely the user revoked Lumicoria at
        # Google. Surface via a None return; the next brain run still
        # logs the skip reason.
        logger.warning(
            "brain.google_token_refresh_failed",
            user_id=user_id, error=str(exc),
        )
        return None

    try:
        from backend.core.config import settings
        oauth_creds = {
            "access_token": credentials.get("access_token"),
            "refresh_token": credentials.get("refresh_token"),
            "token_uri": "https://oauth2.googleapis.com/token",
            "client_id": settings.GOOGLE_OAUTH_CLIENT_ID,
            "client_secret": settings.GOOGLE_OAUTH_CLIENT_SECRET,
        }
        return GoogleWorkspaceClient(credentials_info=oauth_creds)
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "brain.google_client_build_failed",
            user_id=user_id, error=str(exc),
        )
        return None
