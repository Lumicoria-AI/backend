"""
Firebase Cloud Messaging (FCM) Push Notification Service.

This module provides:
- FCM integration for push notifications
- Device token management
- Multi-platform push delivery (iOS, Android, Web)
"""

from typing import Dict, Any, Optional, List
import structlog
from pathlib import Path
import json

logger = structlog.get_logger()

# Firebase Admin SDK initialization
_firebase_app = None


def _initialize_firebase():
    """Initialize Firebase Admin SDK if not already initialized."""
    global _firebase_app
    
    if _firebase_app is not None:
        return _firebase_app
    
    try:
        import firebase_admin
        from firebase_admin import credentials
        
        # Check if already initialized
        try:
            _firebase_app = firebase_admin.get_app()
            return _firebase_app
        except ValueError:
            pass
        
        # Look for credentials file in multiple locations
        cred_paths = [
            Path("firebase-credentials.json"),
            Path("backend/firebase-credentials.json"),
            Path("config/firebase-credentials.json"),
        ]
        
        cred_path = None
        for path in cred_paths:
            if path.exists():
                cred_path = path
                break
        
        if cred_path is None:
            logger.warning("firebase_credentials_not_found")
            return None
        
        cred = credentials.Certificate(str(cred_path))
        _firebase_app = firebase_admin.initialize_app(cred)
        logger.info("firebase_initialized", cred_path=str(cred_path))
        return _firebase_app
        
    except ImportError:
        logger.warning("firebase_admin_not_installed")
        return None
    except Exception as e:
        logger.error("firebase_init_error", error=str(e))
        return None


class PushNotificationService:
    """Service for sending push notifications via Firebase Cloud Messaging."""
    
    def __init__(self):
        self._initialized = False
    
    def _ensure_initialized(self) -> bool:
        """Ensure Firebase is initialized."""
        if not self._initialized:
            app = _initialize_firebase()
            self._initialized = app is not None
        return self._initialized
    
    async def send_to_device(
        self,
        device_token: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
        image_url: Optional[str] = None,
        priority: str = "high",
        actions: Optional[List[Dict[str, str]]] = None,
        click_action: Optional[str] = None,
    ) -> bool:
        """
        Send a push notification to a specific device.

        Args:
            device_token: FCM device registration token
            title: Notification title
            body: Notification body text
            data: Optional data payload (for app handling)
            image_url: Optional image URL for rich notifications
            priority: Message priority ("high" or "normal")
            actions: Optional list of {"action": str, "title": str, "url"?: str}
                dicts.  Each becomes a button on the OS notification (web /
                Android).  The service worker reads `event.action` to route
                clicks; the optional `url` is mirrored into the data payload
                so the SW can navigate or fetch it.
            click_action: Optional URL to open when the notification itself is
                tapped (everywhere outside the action buttons).

        Returns:
            True if sent successfully, False otherwise
        """
        if not self._ensure_initialized():
            logger.warning("push_notification_skipped_no_firebase")
            return False

        try:
            from firebase_admin import messaging

            # Build notification
            notification = messaging.Notification(
                title=title,
                body=body,
                image=image_url
            )

            # Normalise actions for the data payload.  FCM requires string
            # values for `data`, so we JSON-encode the list and the SW
            # deserialises it.
            actions_clean: List[Dict[str, str]] = []
            for a in (actions or []):
                if not isinstance(a, dict):
                    continue
                action_id = str(a.get("action") or "").strip()
                a_title = str(a.get("title") or "").strip()
                if not action_id or not a_title:
                    continue
                entry: Dict[str, str] = {"action": action_id, "title": a_title}
                if a.get("url"):
                    entry["url"] = str(a["url"])
                actions_clean.append(entry)

            # Merge action metadata + click_action into the data payload so the
            # SW has everything it needs without re-fetching.
            merged_data: Dict[str, Any] = dict(data or {})
            if click_action:
                merged_data["click_action"] = click_action
            if actions_clean:
                merged_data["actions"] = json.dumps(actions_clean)

            # Web push needs explicit `actions` on the WebpushNotification,
            # and a `fcm_options.link` to make the notification body itself
            # clickable to a URL.
            webpush_notification_kwargs: Dict[str, Any] = {
                "icon": "/lumicoria-logo-white.png",
                "badge": "/icon-192.png",
            }
            if actions_clean:
                # FCM's WebpushNotification.actions takes a list of dicts
                # with `action` and `title` (and optional `icon`).
                webpush_notification_kwargs["actions"] = [
                    {"action": a["action"], "title": a["title"]}
                    for a in actions_clean
                ]
            webpush_fcm_opts = None
            if click_action:
                try:
                    webpush_fcm_opts = messaging.WebpushFCMOptions(link=click_action)
                except Exception:
                    webpush_fcm_opts = None

            # Build message
            message = messaging.Message(
                notification=notification,
                token=device_token,
                data={k: str(v) for k, v in merged_data.items()},  # FCM requires string values
                android=messaging.AndroidConfig(
                    priority=priority,
                    notification=messaging.AndroidNotification(
                        icon="notification_icon",
                        color="#4A90E2",
                        click_action=click_action or None,
                    )
                ),
                apns=messaging.APNSConfig(
                    payload=messaging.APNSPayload(
                        aps=messaging.Aps(
                            badge=1,
                            sound="default",
                            # iOS surfaces actions via a registered category
                            # set up by the client app; we still pass the
                            # action id in the data payload above.
                            category=("LUMICORIA_PROPOSAL" if actions_clean else None),
                        )
                    )
                ),
                webpush=messaging.WebpushConfig(
                    notification=messaging.WebpushNotification(**webpush_notification_kwargs),
                    fcm_options=webpush_fcm_opts,
                )
            )

            # Send message
            response = messaging.send(message)
            logger.info("push_notification_sent", message_id=response)
            return True

        except Exception as e:
            error_msg = str(e)
            
            # Handle invalid token
            if "Requested entity was not found" in error_msg or "not a valid FCM registration token" in error_msg:
                logger.warning("push_notification_invalid_token", token=device_token[:20] + "...")
                # Could trigger token cleanup here
            else:
                logger.error("push_notification_error", error=error_msg)
            
            return False
    
    async def send_to_user(
        self,
        user_id: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None,
        actions: Optional[List[Dict[str, str]]] = None,
        click_action: Optional[str] = None,
    ) -> bool:
        """
        Send a push notification to all devices for a user.
        
        Args:
            user_id: User ID to send to
            title: Notification title
            body: Notification body text
            data: Optional data payload
        
        Returns:
            True if sent to at least one device, False otherwise
        """
        try:
            # Get user's device tokens from repository
            from backend.db.mongodb.repositories.device_token_repository import (
                get_device_token_repository
            )
            
            repository = await get_device_token_repository()
            tokens = await repository.get_user_tokens(user_id)
            
            if not tokens:
                logger.debug("push_notification_no_tokens", user_id=user_id)
                return False
            
            success_count = 0
            invalid_tokens = []
            
            for token_doc in tokens:
                success = await self.send_to_device(
                    device_token=token_doc.token,
                    title=title,
                    body=body,
                    data=data,
                    actions=actions,
                    click_action=click_action,
                )
                if success:
                    success_count += 1
                else:
                    invalid_tokens.append(token_doc.token)
            
            # Clean up invalid tokens
            if invalid_tokens:
                for token in invalid_tokens:
                    await repository.delete_token(user_id, token)
                logger.info(
                    "push_notification_tokens_cleaned",
                    user_id=user_id,
                    count=len(invalid_tokens)
                )
            
            logger.info(
                "push_notification_sent_to_user",
                user_id=user_id,
                success_count=success_count,
                total_tokens=len(tokens)
            )
            
            return success_count > 0
            
        except ImportError:
            logger.warning("device_token_repository_not_available")
            return False
        except Exception as e:
            logger.error("push_notification_user_error", user_id=user_id, error=str(e))
            return False
    
    async def send_to_topic(
        self,
        topic: str,
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Send a push notification to all devices subscribed to a topic.
        
        Args:
            topic: FCM topic name (e.g., "announcements", "updates")
            title: Notification title
            body: Notification body text
            data: Optional data payload
        
        Returns:
            True if sent successfully, False otherwise
        """
        if not self._ensure_initialized():
            return False
        
        try:
            from firebase_admin import messaging
            
            message = messaging.Message(
                notification=messaging.Notification(
                    title=title,
                    body=body
                ),
                topic=topic,
                data={k: str(v) for k, v in (data or {}).items()}
            )
            
            response = messaging.send(message)
            logger.info("push_notification_topic_sent", topic=topic, message_id=response)
            return True
            
        except Exception as e:
            logger.error("push_notification_topic_error", topic=topic, error=str(e))
            return False
    
    async def send_multicast(
        self,
        tokens: List[str],
        title: str,
        body: str,
        data: Optional[Dict[str, Any]] = None
    ) -> Dict[str, int]:
        """
        Send a push notification to multiple devices at once.
        
        Args:
            tokens: List of FCM device tokens
            title: Notification title
            body: Notification body text
            data: Optional data payload
        
        Returns:
            Dict with success_count and failure_count
        """
        if not self._ensure_initialized():
            return {"success_count": 0, "failure_count": len(tokens)}
        
        if not tokens:
            return {"success_count": 0, "failure_count": 0}
        
        try:
            from firebase_admin import messaging
            
            message = messaging.MulticastMessage(
                notification=messaging.Notification(
                    title=title,
                    body=body
                ),
                tokens=tokens,
                data={k: str(v) for k, v in (data or {}).items()}
            )
            
            response = messaging.send_multicast(message)
            
            logger.info(
                "push_notification_multicast_sent",
                success_count=response.success_count,
                failure_count=response.failure_count
            )
            
            return {
                "success_count": response.success_count,
                "failure_count": response.failure_count
            }
            
        except Exception as e:
            logger.error("push_notification_multicast_error", error=str(e))
            return {"success_count": 0, "failure_count": len(tokens)}


# Singleton instance
push_notification_service = PushNotificationService()
