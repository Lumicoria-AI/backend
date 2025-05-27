from typing import Dict, Any, List, Optional
import structlog
import aiohttp
import json
from datetime import datetime
import asyncio
import functools

logger = structlog.get_logger(__name__)

class SlackClient:
    """Client for interacting with Slack APIs."""
    
    def __init__(self, bot_token: str, app_token: Optional[str] = None):
        """
        Initialize Slack client.
        
        Args:
            bot_token: Bot User OAuth Token
            app_token: App-Level Token (for Socket Mode)
        """
        self.bot_token = bot_token
        self.app_token = app_token
        self.base_url = "https://slack.com/api"
        self.headers = {
            "Authorization": f"Bearer {bot_token}",
            "Content-Type": "application/json"
        }
        
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make a request to the Slack API."""
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with aiohttp.ClientSession() as session:
                if method.upper() == "GET":
                    async with session.get(url, headers=self.headers, params=data) as response:
                        result = await response.json()
                else:
                    async with session.request(method, url, headers=self.headers, json=data) as response:
                        result = await response.json()
                
                if not result.get("ok", False):
                    error = result.get("error", "Unknown error")
                    logger.error(f"Slack API error: {error}", endpoint=endpoint)
                    raise Exception(f"Slack API error: {error}")
                    
                return result
                
        except Exception as e:
            logger.error(f"Error making Slack API request: {str(e)}", endpoint=endpoint)
            raise
    
    # Channel Methods
    async def list_channels(self, exclude_archived: bool = True) -> List[Dict[str, Any]]:
        """List all channels in the workspace."""
        data = {"exclude_archived": exclude_archived}
        result = await self._make_request("GET", "conversations.list", data)
        return result.get("channels", [])
    
    async def create_channel(self, name: str, is_private: bool = False) -> Dict[str, Any]:
        """Create a new channel."""
        data = {
            "name": name,
            "is_private": is_private
        }
        result = await self._make_request("POST", "conversations.create", data)
        return result.get("channel", {})
    
    async def join_channel(self, channel: str) -> Dict[str, Any]:
        """Join a channel."""
        data = {"channel": channel}
        result = await self._make_request("POST", "conversations.join", data)
        return result.get("channel", {})
    
    # Message Methods
    async def send_message(self, 
                         channel: str, 
                         text: str, 
                         blocks: Optional[List[Dict[str, Any]]] = None,
                         thread_ts: Optional[str] = None,
                         reply_broadcast: bool = False) -> Dict[str, Any]:
        """
        Send a message to a channel.
        
        Args:
            channel: Channel ID
            text: Message text
            blocks: Message blocks for rich formatting
            thread_ts: Thread timestamp for replies
            reply_broadcast: Whether to broadcast reply to channel
        """
        data = {
            "channel": channel,
            "text": text
        }
        
        if blocks:
            data["blocks"] = blocks
        if thread_ts:
            data["thread_ts"] = thread_ts
            data["reply_broadcast"] = reply_broadcast
            
        result = await self._make_request("POST", "chat.postMessage", data)
        return result.get("message", {})
    
    async def update_message(self,
                           channel: str,
                           ts: str,
                           text: str,
                           blocks: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Update an existing message."""
        data = {
            "channel": channel,
            "ts": ts,
            "text": text
        }
        
        if blocks:
            data["blocks"] = blocks
            
        result = await self._make_request("POST", "chat.update", data)
        return result.get("message", {})
    
    async def delete_message(self, channel: str, ts: str) -> Dict[str, Any]:
        """Delete a message."""
        data = {
            "channel": channel,
            "ts": ts
        }
        return await self._make_request("POST", "chat.delete", data)
    
    # Thread Methods
    async def get_thread_replies(self, channel: str, thread_ts: str) -> List[Dict[str, Any]]:
        """Get replies in a thread."""
        data = {
            "channel": channel,
            "ts": thread_ts
        }
        result = await self._make_request("GET", "conversations.replies", data)
        return result.get("messages", [])
    
    # User Methods
    async def list_users(self) -> List[Dict[str, Any]]:
        """List all users in the workspace."""
        result = await self._make_request("GET", "users.list")
        return result.get("members", [])
    
    async def get_user_info(self, user: str) -> Dict[str, Any]:
        """Get information about a user."""
        data = {"user": user}
        result = await self._make_request("GET", "users.info", data)
        return result.get("user", {})
    
    # File Methods
    async def upload_file(self,
                        channels: List[str],
                        file_path: str,
                        title: Optional[str] = None,
                        initial_comment: Optional[str] = None,
                        thread_ts: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a file to Slack.
        
        Args:
            channels: List of channel IDs to share the file with
            file_path: Path to the file to upload
            title: Title of the file
            initial_comment: Initial comment to post with the file
            thread_ts: Thread timestamp to post in
        """
        data = {
            "channels": ",".join(channels)
        }
        
        if title:
            data["title"] = title
        if initial_comment:
            data["initial_comment"] = initial_comment
        if thread_ts:
            data["thread_ts"] = thread_ts
            
        try:
            async with aiohttp.ClientSession() as session:
                with open(file_path, 'rb') as f:
                    form = aiohttp.FormData()
                    form.add_field('file',
                                 f,
                                 filename=file_path.split('/')[-1],
                                 content_type='application/octet-stream')
                    
                    for key, value in data.items():
                        form.add_field(key, value)
                        
                    async with session.post(
                        f"{self.base_url}/files.upload",
                        headers={"Authorization": f"Bearer {self.bot_token}"},
                        data=form
                    ) as response:
                        result = await response.json()
                        
                        if not result.get("ok", False):
                            error = result.get("error", "Unknown error")
                            logger.error(f"Slack file upload error: {error}")
                            raise Exception(f"Slack file upload error: {error}")
                            
                        return result.get("file", {})
                        
        except Exception as e:
            logger.error(f"Error uploading file to Slack: {str(e)}")
            raise
    
    # Reaction Methods
    async def add_reaction(self, channel: str, timestamp: str, name: str) -> Dict[str, Any]:
        """Add a reaction to a message."""
        data = {
            "channel": channel,
            "timestamp": timestamp,
            "name": name
        }
        return await self._make_request("POST", "reactions.add", data)
    
    async def remove_reaction(self, channel: str, timestamp: str, name: str) -> Dict[str, Any]:
        """Remove a reaction from a message."""
        data = {
            "channel": channel,
            "timestamp": timestamp,
            "name": name
        }
        return await self._make_request("POST", "reactions.remove", data)
    
    # Search Methods
    async def search_messages(self, query: str, count: int = 20) -> List[Dict[str, Any]]:
        """Search for messages."""
        data = {
            "query": query,
            "count": count
        }
        result = await self._make_request("GET", "search.messages", data)
        return result.get("messages", {}).get("matches", [])
    
    # Team Methods
    async def get_team_info(self) -> Dict[str, Any]:
        """Get information about the workspace."""
        result = await self._make_request("GET", "team.info")
        return result.get("team", {})
    
    # Integration Methods
    async def create_reminder(self,
                            text: str,
                            time: str,
                            user: Optional[str] = None,
                            channel: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a reminder.
        
        Args:
            text: Reminder text
            time: When to send the reminder (Unix timestamp or natural language)
            user: User ID to remind (defaults to the authenticated user)
            channel: Channel ID to post the reminder in
        """
        data = {
            "text": text,
            "time": time
        }
        
        if user:
            data["user"] = user
        if channel:
            data["channel"] = channel
            
        result = await self._make_request("POST", "reminders.add", data)
        return result.get("reminder", {})
    
    async def list_reminders(self) -> List[Dict[str, Any]]:
        """List all reminders for the authenticated user."""
        result = await self._make_request("GET", "reminders.list")
        return result.get("reminders", [])
    
    async def complete_reminder(self, reminder_id: str) -> Dict[str, Any]:
        """Mark a reminder as complete."""
        data = {"reminder": reminder_id}
        return await self._make_request("POST", "reminders.complete", data)
    
    async def delete_reminder(self, reminder_id: str) -> Dict[str, Any]:
        """Delete a reminder."""
        data = {"reminder": reminder_id}
        return await self._make_request("POST", "reminders.delete", data) 