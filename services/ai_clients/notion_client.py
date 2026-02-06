from typing import Dict, Any, List, Optional
import structlog
import aiohttp
import json
from datetime import datetime
from pydantic import BaseModel

logger = structlog.get_logger(__name__)

class NotionBlock(BaseModel):
    """Model for Notion block content."""
    type: str
    content: Any

class NotionPage(BaseModel):
    """Model for Notion page content."""
    title: str
    blocks: List[NotionBlock]
    parent_id: Optional[str] = None
    properties: Optional[Dict[str, Any]] = None

class NotionClient:
    """Client for interacting with Notion API."""
    
    def __init__(self, api_token: str, version: str = "2022-06-28"):
        self.api_token = api_token
        self.base_url = "https://api.notion.com/v1"
        self.version = version
        self.headers = {
            "Authorization": f"Bearer {api_token}",
            "Content-Type": "application/json",
            "Notion-Version": version
        }
    
    async def _make_request(self, method: str, endpoint: str, data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Make an HTTP request to the Notion API."""
        url = f"{self.base_url}{endpoint}"
        
        async with aiohttp.ClientSession() as session:
            try:
                async with session.request(
                    method=method,
                    url=url,
                    headers=self.headers,
                    json=data
                ) as response:
                    response_json = await response.json()
                    
                    if not response.ok:
                        error_message = response_json.get("message", "Unknown error")
                        logger.error(
                            "Notion API request failed",
                            status=response.status,
                            error=error_message,
                            endpoint=endpoint
                        )
                        return {"error": error_message, "status_code": response.status}
                    
                    return response_json
            except Exception as e:
                logger.error("Error making Notion API request", error=str(e), endpoint=endpoint)
                return {"error": str(e)}
    
    async def search_pages(self, query: str) -> Dict[str, Any]:
        """Search for Notion pages."""
        data = {"query": query, "sort": {"direction": "descending", "timestamp": "last_edited_time"}}
        return await self._make_request("POST", "/search", data)
    
    async def get_page(self, page_id: str) -> Dict[str, Any]:
        """Get a Notion page by ID."""
        return await self._make_request("GET", f"/pages/{page_id}")
    
    async def create_page(self, page: NotionPage) -> Dict[str, Any]:
        """Create a new Notion page."""
        # Format parent reference
        if page.parent_id:
            parent = {"page_id": page.parent_id}
        else:
            # Default to workspace root if no parent specified
            parent = {"workspace": True}
        
        # Format properties (including title)
        properties = page.properties or {}
        properties["title"] = {
            "title": [{"text": {"content": page.title}}]
        }
        
        # Format children (blocks)
        children = []
        for block in page.blocks:
            if block.type == "paragraph":
                children.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": block.content}}]
                    }
                })
            elif block.type == "heading_1":
                children.append({
                    "object": "block",
                    "type": "heading_1",
                    "heading_1": {
                        "rich_text": [{"type": "text", "text": {"content": block.content}}]
                    }
                })
            elif block.type == "heading_2":
                children.append({
                    "object": "block",
                    "type": "heading_2",
                    "heading_2": {
                        "rich_text": [{"type": "text", "text": {"content": block.content}}]
                    }
                })
            elif block.type == "heading_3":
                children.append({
                    "object": "block",
                    "type": "heading_3",
                    "heading_3": {
                        "rich_text": [{"type": "text", "text": {"content": block.content}}]
                    }
                })
            elif block.type == "bulleted_list_item":
                children.append({
                    "object": "block",
                    "type": "bulleted_list_item",
                    "bulleted_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": block.content}}]
                    }
                })
            elif block.type == "numbered_list_item":
                children.append({
                    "object": "block",
                    "type": "numbered_list_item",
                    "numbered_list_item": {
                        "rich_text": [{"type": "text", "text": {"content": block.content}}]
                    }
                })
            elif block.type == "to_do":
                children.append({
                    "object": "block",
                    "type": "to_do",
                    "to_do": {
                        "rich_text": [{"type": "text", "text": {"content": block.content.get("text", "")}}],
                        "checked": block.content.get("checked", False)
                    }
                })
            # Add support for other block types as needed
        
        # Create the page
        data = {
            "parent": parent,
            "properties": properties,
            "children": children
        }
        
        return await self._make_request("POST", "/pages", data)
    
    async def update_page(self, page_id: str, properties: Dict[str, Any]) -> Dict[str, Any]:
        """Update a Notion page's properties."""
        formatted_properties = {}
        
        # Format properties for Notion API
        for key, value in properties.items():
            if key == "title":
                formatted_properties["title"] = {
                    "title": [{"text": {"content": value}}]
                }
            elif isinstance(value, str):
                formatted_properties[key] = {
                    "rich_text": [{"text": {"content": value}}]
                }
            elif isinstance(value, bool):
                formatted_properties[key] = {
                    "checkbox": value
                }
            elif isinstance(value, (int, float)):
                formatted_properties[key] = {
                    "number": value
                }
            elif isinstance(value, datetime):
                formatted_properties[key] = {
                    "date": {"start": value.isoformat()}
                }
            elif isinstance(value, dict) and value.get("type") == "select":
                formatted_properties[key] = {
                    "select": {"name": value.get("name")}
                }
            # Add other property types as needed
        
        data = {"properties": formatted_properties}
        return await self._make_request("PATCH", f"/pages/{page_id}", data)
    
    async def add_blocks_to_page(self, page_id: str, blocks: List[NotionBlock]) -> Dict[str, Any]:
        """Add blocks to an existing Notion page."""
        formatted_blocks = []
        
        for block in blocks:
            if block.type == "paragraph":
                formatted_blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {
                        "rich_text": [{"type": "text", "text": {"content": block.content}}]
                    }
                })
            # Add other block types formatting as needed, similar to create_page method
        
        data = {"children": formatted_blocks}
        return await self._make_request("PATCH", f"/blocks/{page_id}/children", data)
    
    async def create_database(self, title: str, properties: Dict[str, Dict[str, Any]], parent_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new Notion database."""
        # Format parent reference
        if parent_id:
            parent = {"page_id": parent_id}
        else:
            # Default to workspace root if no parent specified
            parent = {"workspace": True}
        
        # Format title
        title_content = [{"type": "text", "text": {"content": title}}]
        
        data = {
            "parent": parent,
            "title": title_content,
            "properties": properties
        }
        
        return await self._make_request("POST", "/databases", data)
    
    async def query_database(self, database_id: str, filter_obj: Optional[Dict[str, Any]] = None, 
                           sorts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """Query a Notion database with optional filter and sort."""
        data = {}
        if filter_obj:
            data["filter"] = filter_obj
        if sorts:
            data["sorts"] = sorts
        
        return await self._make_request("POST", f"/databases/{database_id}/query", data)
    
    async def create_project(self, title: str, description: str, due_date: Optional[datetime] = None, 
                          status: str = "Not Started", parent_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a new project page with common project fields."""
        blocks = [
            NotionBlock(type="paragraph", content=description)
        ]
        
        properties = {
            "Status": {"type": "select", "name": status}
        }
        
        if due_date:
            properties["Due Date"] = {"date": {"start": due_date.isoformat()}}
        
        page = NotionPage(
            title=title,
            blocks=blocks,
            parent_id=parent_id,
            properties=properties
        )
        
        return await self.create_page(page)
    
    async def create_project_database(self, title: str, parent_id: Optional[str] = None) -> Dict[str, Any]:
        """Create a standardized project tracking database."""
        properties = {
            "Name": {"title": {}},  # The title property is required
            "Status": {
                "select": {
                    "options": [
                        {"name": "Not Started", "color": "gray"},
                        {"name": "In Progress", "color": "blue"},
                        {"name": "Completed", "color": "green"},
                        {"name": "On Hold", "color": "yellow"},
                        {"name": "Cancelled", "color": "red"}
                    ]
                }
            },
            "Priority": {
                "select": {
                    "options": [
                        {"name": "Low", "color": "gray"},
                        {"name": "Medium", "color": "yellow"},
                        {"name": "High", "color": "red"}
                    ]
                }
            },
            "Due Date": {"date": {}},
            "Assigned To": {"rich_text": {}},
            "Progress": {"number": {}},
            "Description": {"rich_text": {}}
        }
        
        return await self.create_database(title, properties, parent_id)
    
    async def add_project_task(self, database_id: str, task_name: str, task_properties: Dict[str, Any]) -> Dict[str, Any]:
        """Add a task to a project database."""
        # Format properties for Notion API
        formatted_properties = {
            "Name": {"title": [{"text": {"content": task_name}}]}
        }
        
        for key, value in task_properties.items():
            if key == "Status":
                formatted_properties["Status"] = {"select": {"name": value}}
            elif key == "Priority":
                formatted_properties["Priority"] = {"select": {"name": value}}
            elif key == "Due Date" and isinstance(value, datetime):
                formatted_properties["Due Date"] = {"date": {"start": value.isoformat()}}
            elif key == "Assigned To":
                formatted_properties["Assigned To"] = {"rich_text": [{"text": {"content": value}}]}
            elif key == "Progress" and isinstance(value, (int, float)):
                formatted_properties["Progress"] = {"number": value}
            elif key == "Description":
                formatted_properties["Description"] = {"rich_text": [{"text": {"content": value}}]}
            # Add other property types as needed
        
        data = {
            "parent": {"database_id": database_id},
            "properties": formatted_properties
        }
        
        return await self._make_request("POST", "/pages", data)
