
from typing import Dict, Any, List, Optional
import structlog
from datetime import datetime
from ..services.ai_clients.slack_client import SlackClient
from ..core.config import settings

logger = structlog.get_logger(__name__)

class SlackIntegration:
    """Integration with Slack workspace."""
    
    def __init__(self):
        """Initialize Slack integration."""
        self.client = SlackClient(
            bot_token=settings.SLACK_BOT_TOKEN,
            app_token=settings.SLACK_APP_TOKEN
        )
        self._validate_connection()
        
    def _validate_connection(self) -> None:
        """Validate Slack connection by getting team info."""
        try:
            team_info = self.client.get_team_info()
            logger.info(
                "Slack connection validated",
                team_name=team_info.get("name"),
                team_id=team_info.get("id")
            )
        except Exception as e:
            logger.error(f"Failed to validate Slack connection: {str(e)}")
            raise
    
    async def create_project_channel(self, 
                                   project_name: str, 
                                   description: str,
                                   is_private: bool = False) -> Dict[str, Any]:
        """
        Create a new Slack channel for a project.
        
        Args:
            project_name: Name of the project
            description: Project description
            is_private: Whether the channel should be private
            
        Returns:
            Dict containing channel information
        """
        try:
            # Create channel
            channel = await self.client.create_channel(
                name=project_name.lower().replace(" ", "-"),
                is_private=is_private
            )
            
            # Post initial message with project info
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Project: {project_name}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": description
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": f"Created by Lumicoria.ai on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                        }
                    ]
                }
            ]
            
            await self.client.send_message(
                channel=channel["id"],
                text=f"Project channel created: {project_name}",
                blocks=blocks
            )
            
            return channel
            
        except Exception as e:
            logger.error(f"Error creating project channel: {str(e)}")
            raise
    
    async def add_project_task(self,
                             channel: str,
                             task_name: str,
                             description: str,
                             assignee: Optional[str] = None,
                             due_date: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a task to a project channel.
        
        Args:
            channel: Channel ID
            task_name: Name of the task
            description: Task description
            assignee: User ID to assign the task to
            due_date: Due date for the task
            
        Returns:
            Dict containing message information
        """
        try:
            blocks = [
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*New Task:* {task_name}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": description
                    }
                }
            ]
            
            if assignee or due_date:
                elements = []
                if assignee:
                    elements.append({
                        "type": "mrkdwn",
                        "text": f"*Assigned to:* <@{assignee}>"
                    })
                if due_date:
                    elements.append({
                        "type": "mrkdwn",
                        "text": f"*Due:* {due_date}"
                    })
                    
                blocks.append({
                    "type": "context",
                    "elements": elements
                })
            
            # Add action buttons
            blocks.append({
                "type": "actions",
                "elements": [
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Mark Complete",
                            "emoji": True
                        },
                        "style": "primary",
                        "value": "complete"
                    },
                    {
                        "type": "button",
                        "text": {
                            "type": "plain_text",
                            "text": "Add Comment",
                            "emoji": True
                        },
                        "value": "comment"
                    }
                ]
            })
            
            message = await self.client.send_message(
                channel=channel,
                text=f"New task: {task_name}",
                blocks=blocks
            )
            
            return message
            
        except Exception as e:
            logger.error(f"Error adding project task: {str(e)}")
            raise
    
    async def export_meeting_notes(self,
                                 channel: str,
                                 meeting_title: str,
                                 notes: str,
                                 participants: List[str],
                                 date: str) -> Dict[str, Any]:
        """
        Export meeting notes to a Slack channel.
        
        Args:
            channel: Channel ID
            meeting_title: Title of the meeting
            notes: Meeting notes content
            participants: List of participant user IDs
            date: Meeting date
            
        Returns:
            Dict containing message information
        """
        try:
            # Format participants list
            participants_text = "\n".join([f"• <@{p}>" for p in participants])
            
            blocks = [
                {
                    "type": "header",
                    "text": {
                        "type": "plain_text",
                        "text": f"Meeting Notes: {meeting_title}"
                    }
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": f"*Date:* {date}\n*Participants:*\n{participants_text}"
                    }
                },
                {
                    "type": "divider"
                },
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": notes
                    }
                },
                {
                    "type": "context",
                    "elements": [
                        {
                            "type": "mrkdwn",
                            "text": "Generated by Lumicoria.ai"
                        }
                    ]
                }
            ]
            
            message = await self.client.send_message(
                channel=channel,
                text=f"Meeting Notes: {meeting_title}",
                blocks=blocks
            )
            
            return message
            
        except Exception as e:
            logger.error(f"Error exporting meeting notes: {str(e)}")
            raise
    
    async def create_reminder(self,
                            text: str,
                            time: str,
                            channel: Optional[str] = None,
                            user: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a reminder in Slack.
        
        Args:
            text: Reminder text
            time: When to send the reminder
            channel: Channel ID to post in
            user: User ID to remind
            
        Returns:
            Dict containing reminder information
        """
        try:
            reminder = await self.client.create_reminder(
                text=text,
                time=time,
                channel=channel,
                user=user
            )
            
            return reminder
            
        except Exception as e:
            logger.error(f"Error creating reminder: {str(e)}")
            raise
    
    async def search_project_content(self,
                                   query: str,
                                   channel: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Search for project-related content in Slack.
        
        Args:
            query: Search query
            channel: Optional channel ID to limit search to
            
        Returns:
            List of matching messages
        """
        try:
            search_query = query
            if channel:
                search_query = f"in:{channel} {query}"
                
            results = await self.client.search_messages(query=search_query)
            return results
            
        except Exception as e:
            logger.error(f"Error searching project content: {str(e)}")
            raise
    
    async def upload_project_file(self,
                                channel: str,
                                file_path: str,
                                title: Optional[str] = None,
                                comment: Optional[str] = None) -> Dict[str, Any]:
        """
        Upload a file to a project channel.
        
        Args:
            channel: Channel ID
            file_path: Path to the file
            title: Optional file title
            comment: Optional comment to post with the file
            
        Returns:
            Dict containing file information
        """
        try:
            file = await self.client.upload_file(
                channels=[channel],
                file_path=file_path,
                title=title,
                initial_comment=comment
            )
            
            return file
            
        except Exception as e:
            logger.error(f"Error uploading project file: {str(e)}")
            raise
    
    async def get_channel_members(self, channel: str) -> List[Dict[str, Any]]:
        """
        Get members of a channel.
        
        Args:
            channel: Channel ID
            
        Returns:
            List of channel members
        """
        try:
            # First get channel info to ensure we have access
            channel_info = await self.client._make_request(
                "GET",
                "conversations.info",
                {"channel": channel}
            )
            
            # Then get members
            result = await self.client._make_request(
                "GET",
                "conversations.members",
                {"channel": channel}
            )
            
            # Get user info for each member
            members = []
            for user_id in result.get("members", []):
                user_info = await self.client.get_user_info(user=user_id)
                members.append(user_info)
                
            return members
            
        except Exception as e:
            logger.error(f"Error getting channel members: {str(e)}")
            raise
    
    async def archive_project_channel(self, channel: str) -> Dict[str, Any]:
        """
        Archive a project channel.
        
        Args:
            channel: Channel ID
            
        Returns:
            Dict containing channel information
        """
        try:
            result = await self.client._make_request(
                "POST",
                "conversations.archive",
                {"channel": channel}
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Error archiving project channel: {str(e)}")
            raise 