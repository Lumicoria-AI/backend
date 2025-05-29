from typing import Dict, Any, List, Optional
import structlog
from datetime import datetime
import json
from backend.services.ai_clients.notion_client import NotionClient, NotionPage, NotionBlock
from backend.models.integration import Integration, IntegrationType
from backend.core.config import settings

logger = structlog.get_logger(__name__)

class NotionIntegration:
    """
    Integration with Notion API for project management, knowledge management,
    and collaborative documentation.
    """
    
    def __init__(self, api_token: str):
        self.client = NotionClient(api_token)
    
    async def create_project(self, title: str, description: str, due_date: Optional[datetime] = None,
                           status: str = "Not Started", parent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new project page in Notion.
        """
        return await self.client.create_project(title, description, due_date, status, parent_id)
    
    async def create_project_database(self, title: str, parent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a project management database in Notion.
        """
        return await self.client.create_project_database(title, parent_id)
    
    async def add_project_task(self, database_id: str, task_name: str, 
                             description: str, due_date: Optional[datetime] = None,
                             status: str = "Not Started", priority: str = "Medium", 
                             assigned_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a task to a project database in Notion.
        """
        task_properties = {
            "Description": description,
            "Status": status,
            "Priority": priority
        }
        
        if due_date:
            task_properties["Due Date"] = due_date
            
        if assigned_to:
            task_properties["Assigned To"] = assigned_to
            
        return await self.client.add_project_task(database_id, task_name, task_properties)
    
    async def search_projects(self, query: str) -> List[Dict[str, Any]]:
        """
        Search for project pages or databases in Notion.
        """
        search_results = await self.client.search_pages(query)
        
        if "error" in search_results:
            logger.error("Error searching Notion projects", error=search_results["error"])
            return []
            
        results = search_results.get("results", [])
        return results
    
    async def get_project_tasks(self, database_id: str, 
                              filter_status: Optional[str] = None,
                              filter_priority: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get tasks from a project database with optional filtering.
        """
        filter_obj = {}
        if filter_status:
            filter_obj["property"] = "Status"
            filter_obj["select"] = {"equals": filter_status}
        elif filter_priority:
            filter_obj["property"] = "Priority"
            filter_obj["select"] = {"equals": filter_priority}
            
        query_results = await self.client.query_database(database_id, filter_obj=filter_obj if filter_obj else None)
        
        if "error" in query_results:
            logger.error("Error querying Notion database", error=query_results["error"])
            return []
            
        results = query_results.get("results", [])
        return results
    
    async def update_task_status(self, task_id: str, status: str) -> Dict[str, Any]:
        """
        Update the status of a task in Notion.
        """
        properties = {
            "Status": {"type": "select", "name": status}
        }
        return await self.client.update_page(task_id, properties)
    
    async def create_meeting_notes(self, title: str, attendees: List[str], 
                                notes: str, action_items: List[Dict[str, Any]],
                                parent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create meeting notes as a Notion page with structured content.
        """
        # Format attendees as a bulleted list
        attendees_text = "\n".join([f"- {attendee}" for attendee in attendees])
        
        # Format action items
        action_items_text = ""
        for item in action_items:
            assignee = item.get("assignee", "Unassigned")
            deadline = item.get("deadline", "No deadline")
            task = item.get("task", "")
            action_items_text += f"- [ ] {task} (@{assignee}, Due: {deadline})\n"
        
        blocks = [
            NotionBlock(type="heading_1", content="Meeting Notes"),
            NotionBlock(type="paragraph", content=f"Date: {datetime.now().strftime('%Y-%m-%d')}"),
            NotionBlock(type="heading_2", content="Attendees"),
            NotionBlock(type="paragraph", content=attendees_text),
            NotionBlock(type="heading_2", content="Notes"),
            NotionBlock(type="paragraph", content=notes),
            NotionBlock(type="heading_2", content="Action Items"),
            NotionBlock(type="paragraph", content=action_items_text)
        ]
        
        page = NotionPage(
            title=title,
            blocks=blocks,
            parent_id=parent_id
        )
        
        return await self.client.create_page(page)
    
    async def export_meeting_to_notion(self, meeting_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export meeting data processed by MeetingAgent to Notion.
        """
        title = meeting_data.get("metadata", {}).get("title", f"Meeting Notes - {datetime.now().strftime('%Y-%m-%d')}")
        
        # Get participants
        participants = meeting_data.get("metadata", {}).get("participants", [])
        if not isinstance(participants, list):
            participants = [str(participants)]
            
        # Format summary and key points
        summary = meeting_data.get("summary", "No summary available.")
        key_points = meeting_data.get("key_points", [])
        key_points_text = "\n".join([f"- {point}" for point in key_points])
        
        # Format decisions
        decisions = meeting_data.get("decisions", [])
        decisions_text = "\n".join([f"- {decision}" for decision in decisions])
        
        # Get action items
        action_items = meeting_data.get("action_items", [])
        action_items_text = ""
        for item in action_items:
            if isinstance(item, dict):
                task = item.get("task", "")
                assignee = item.get("assignee", "Unassigned")
                deadline = item.get("deadline", "No deadline")
                action_items_text += f"- [ ] {task} (@{assignee}, Due: {deadline})\n"
            else:
                action_items_text += f"- [ ] {item}\n"
                
        # Build blocks
        blocks = [
            NotionBlock(type="heading_1", content="Meeting Summary"),
            NotionBlock(type="paragraph", content=summary),
            NotionBlock(type="heading_2", content="Participants"),
            NotionBlock(type="paragraph", content="\n".join([f"- {p}" for p in participants])),
            NotionBlock(type="heading_2", content="Key Points"),
            NotionBlock(type="paragraph", content=key_points_text),
            NotionBlock(type="heading_2", content="Decisions"),
            NotionBlock(type="paragraph", content=decisions_text),
            NotionBlock(type="heading_2", content="Action Items"),
            NotionBlock(type="paragraph", content=action_items_text)
        ]
        
        # Add questions and concerns if available
        if meeting_data.get("questions"):
            questions_text = "\n".join([f"- {q}" for q in meeting_data.get("questions", [])])
            blocks.append(NotionBlock(type="heading_2", content="Questions"))
            blocks.append(NotionBlock(type="paragraph", content=questions_text))
            
        if meeting_data.get("concerns"):
            concerns_text = "\n".join([f"- {c}" for c in meeting_data.get("concerns", [])])
            blocks.append(NotionBlock(type="heading_2", content="Concerns"))
            blocks.append(NotionBlock(type="paragraph", content=concerns_text))
        
        # Create the page
        page = NotionPage(
            title=title,
            blocks=blocks
        )
        
        return await self.client.create_page(page)
        
    async def create_knowledge_base(self, title: str, content_sections: Dict[str, str], parent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a structured knowledge base page in Notion.
        """
        blocks = [NotionBlock(type="heading_1", content=title)]
        
        for section_title, section_content in content_sections.items():
            blocks.append(NotionBlock(type="heading_2", content=section_title))
            blocks.append(NotionBlock(type="paragraph", content=section_content))
        
        page = NotionPage(
            title=title,
            blocks=blocks,
            parent_id=parent_id
        )
        
        return await self.client.create_page(page)
