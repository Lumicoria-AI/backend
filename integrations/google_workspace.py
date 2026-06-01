from typing import Dict, Any, List, Optional
import structlog
from datetime import datetime
import json
from backend.services.ai_clients.google_workspace_client import GoogleWorkspaceClient

logger = structlog.get_logger(__name__)

class GoogleWorkspaceIntegration:
    """
    Integration with Google Workspace APIs for calendar, drive, email, docs, and sheets functionality.
    """
    
    def __init__(self, credentials_info: Dict[str, Any], user_email: Optional[str] = None):
        self.client = GoogleWorkspaceClient(credentials_info, user_email)
    
    # ----- Calendar Methods -----
    
    async def list_calendars(self) -> List[Dict[str, Any]]:
        """
        List available calendars for the authenticated user.
        """
        try:
            return await self.client.list_calendars()
        except Exception as e:
            logger.error("Error listing calendars", error=str(e))
            return []
    
    async def create_calendar_event(self, 
                                  summary: str, 
                                  description: str, 
                                  start_time: datetime, 
                                  end_time: datetime, 
                                  attendees: List[str] = None, 
                                  calendar_id: str = "primary",
                                  location: str = None,
                                  conference_type: str = None) -> Dict[str, Any]:
        """
        Create a calendar event with optional video conferencing.
        """
        try:
            return await self.client.create_calendar_event(
                summary=summary,
                description=description,
                start_time=start_time,
                end_time=end_time,
                attendees=attendees,
                calendar_id=calendar_id,
                location=location,
                conference_type=conference_type
            )
        except Exception as e:
            logger.error("Error creating calendar event", error=str(e))
            return {"error": str(e)}
    
    async def get_upcoming_events(self,
                               calendar_id: str = "primary",
                               max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Get upcoming events from a calendar.
        """
        try:
            return await self.client.get_events(
                calendar_id=calendar_id,
                max_results=max_results
            )
        except Exception as e:
            logger.error("Error getting upcoming events", error=str(e))
            return []

    async def update_calendar_event(
        self,
        event_id: str,
        *,
        calendar_id: str = "primary",
        summary: Optional[str] = None,
        description: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        location: Optional[str] = None,
        status: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Patch an existing Google Calendar event.

        Phase 3: lets the Lumicoria calendar mirror local edits onto Google
        without recreating the event (which would orphan invitations).
        """
        try:
            return await self.client.update_calendar_event(
                event_id=event_id,
                calendar_id=calendar_id,
                summary=summary,
                description=description,
                start_time=start_time,
                end_time=end_time,
                location=location,
                status=status,
            )
        except Exception as e:
            logger.error("Error updating calendar event", error=str(e), event_id=event_id)
            return {"error": str(e)}

    async def delete_calendar_event(
        self,
        event_id: str,
        calendar_id: str = "primary",
    ) -> bool:
        """Delete a Google Calendar event.  Returns True on success or 404."""
        try:
            return await self.client.delete_calendar_event(
                event_id=event_id, calendar_id=calendar_id
            )
        except Exception as e:
            logger.error("Error deleting calendar event", error=str(e), event_id=event_id)
            return False

    # ----- Drive Methods -----
    
    async def list_files(self, folder_id: str = None, query: str = None, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        List files in Google Drive.
        """
        try:
            return await self.client.list_files(
                folder_id=folder_id,
                query=query,
                max_results=max_results
            )
        except Exception as e:
            logger.error("Error listing Drive files", error=str(e))
            return []
    
    async def create_document(self, title: str, content: str = None, folder_id: str = None) -> Dict[str, Any]:
        """
        Create a Google Document.
        """
        try:
            return await self.client.create_document(
                title=title,
                content=content,
                folder_id=folder_id
            )
        except Exception as e:
            logger.error("Error creating Google Document", error=str(e))
            return {"error": str(e)}
    
    async def create_project_folder(self, title: str, parent_id: str = None) -> Dict[str, Any]:
        """
        Create a folder in Google Drive for project management.
        """
        try:
            drive_service = self.client._get_service(
                "drive", "v3", ["https://www.googleapis.com/auth/drive"]
            )
            
            folder_metadata = {
                'name': title,
                'mimeType': 'application/vnd.google-apps.folder'
            }
            
            if parent_id:
                folder_metadata['parents'] = [parent_id]
            
            def _create_folder():
                return drive_service.files().create(
                    body=folder_metadata,
                    fields='id, name, webViewLink'
                ).execute()
            
            folder = await self.client.run_in_executor(_create_folder)
            return folder
        except Exception as e:
            logger.error("Error creating project folder", error=str(e))
            return {"error": str(e)}
    
    # ----- Email Methods -----
    
    async def send_email(self, 
                       to: List[str], 
                       subject: str, 
                       body: str, 
                       cc: List[str] = None,
                       bcc: List[str] = None,
                       html_content: str = None) -> Dict[str, Any]:
        """
        Send an email using Gmail API.
        """
        try:
            return await self.client.send_email(
                to=to,
                subject=subject,
                body=body,
                cc=cc,
                bcc=bcc,
                html_content=html_content
            )
        except Exception as e:
            logger.error("Error sending email", error=str(e))
            return {"error": str(e)}
    
    # ----- Project Management Methods -----
    
    async def create_project(self, title: str, description: str, due_date: Optional[datetime] = None, 
                          status: str = "Not Started", parent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a project as a folder with documents in Google Drive.
        """
        try:
            # Create a project folder
            folder = await self.create_project_folder(title, parent_id)
            
            if "error" in folder:
                return folder
            
            folder_id = folder.get("id")
            
            # Create a project overview document
            content = f"# {title}\n\n"
            content += f"Status: {status}\n"
            if due_date:
                content += f"Due Date: {due_date.strftime('%Y-%m-%d')}\n"
            content += f"\n## Description\n{description}\n"
            
            project_doc = await self.client.create_document(
                title=f"{title} - Overview",
                content=content,
                folder_id=folder_id
            )
            
            return {
                "id": folder_id,
                "title": title,
                "url": folder.get("webViewLink", ""),
                "document_id": project_doc.get("id", ""),
                "document_url": f"https://docs.google.com/document/d/{project_doc.get('id', '')}/edit"
            }
            
        except Exception as e:
            logger.error("Error creating project", error=str(e))
            return {"error": str(e)}
    
    async def create_project_database(self, title: str, parent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a project management spreadsheet in Google Sheets.
        """
        try:
            # Create initial data for the project management spreadsheet
            data = [
                ["Task", "Description", "Status", "Priority", "Assignee", "Due Date"],
                # Empty rows for tasks to be added
                ["", "", "Not Started", "Medium", "", ""],
                ["", "", "Not Started", "Medium", "", ""],
                ["", "", "Not Started", "Medium", "", ""],
            ]
            
            spreadsheet = await self.client.create_spreadsheet(
                title=title,
                data=data,
                folder_id=parent_id
            )
            
            return {
                "id": spreadsheet.get("spreadsheetId"),
                "title": title,
                "url": spreadsheet.get("url")
            }
            
        except Exception as e:
            logger.error("Error creating project database", error=str(e))
            return {"error": str(e)}
    
    async def add_project_task(self, database_id: str, task_name: str, 
                             description: str, due_date: Optional[datetime] = None,
                             status: str = "Not Started", priority: str = "Medium", 
                             assigned_to: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a task to a project spreadsheet in Google Sheets.
        """
        try:
            sheets_service = self.client._get_service(
                "sheets", "v4", ["https://www.googleapis.com/auth/spreadsheets"]
            )
            
            # First, get the current values to determine next row
            def _get_values():
                return sheets_service.spreadsheets().values().get(
                    spreadsheetId=database_id,
                    range="Sheet1!A:A"
                ).execute()
            
            values_result = await self.client.run_in_executor(_get_values)
            values = values_result.get("values", [])
            next_row = len(values) + 1
            
            # Format due date
            due_date_str = due_date.strftime("%Y-%m-%d") if due_date else ""
            
            # Add the task
            task_data = [[
                task_name, 
                description, 
                status, 
                priority, 
                assigned_to or "", 
                due_date_str
            ]]
            
            def _add_task():
                body = {
                    'values': task_data
                }
                return sheets_service.spreadsheets().values().update(
                    spreadsheetId=database_id,
                    range=f"Sheet1!A{next_row}",
                    valueInputOption='RAW',
                    body=body
                ).execute()
            
            result = await self.client.run_in_executor(_add_task)
            
            return {
                "id": f"{database_id}:{next_row}",
                "task": task_name,
                "row": next_row,
                "spreadsheet_id": database_id
            }
            
        except Exception as e:
            logger.error("Error adding project task", error=str(e))
            return {"error": str(e)}
    
    async def get_project_tasks(self, database_id: str, 
                              filter_status: Optional[str] = None,
                              filter_priority: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get tasks from a project spreadsheet with optional filtering.
        """
        try:
            sheets_service = self.client._get_service(
                "sheets", "v4", ["https://www.googleapis.com/auth/spreadsheets"]
            )
            
            def _get_values():
                return sheets_service.spreadsheets().values().get(
                    spreadsheetId=database_id,
                    range="Sheet1!A:F"
                ).execute()
            
            values_result = await self.client.run_in_executor(_get_values)
            rows = values_result.get("values", [])
            
            # Extract header and data rows
            if len(rows) <= 1:
                return []
            
            header = rows[0]
            data_rows = rows[1:]
            
            tasks = []
            for i, row in enumerate(data_rows):
                # Skip empty rows
                if not row or len(row) == 0 or not row[0]:
                    continue
                
                # Ensure row has enough elements
                while len(row) < 6:
                    row.append("")
                
                task = {
                    "id": f"{database_id}:{i+2}",  # +2 because of 0-indexing and header
                    "task": row[0],
                    "description": row[1],
                    "status": row[2],
                    "priority": row[3],
                    "assignee": row[4],
                    "due_date": row[5]
                }
                
                # Apply filters if specified
                if filter_status and task["status"] != filter_status:
                    continue
                if filter_priority and task["priority"] != filter_priority:
                    continue
                
                tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logger.error("Error getting project tasks", error=str(e))
            return []
    
    async def update_task_status(self, task_id: str, status: str) -> Dict[str, Any]:
        """
        Update the status of a task in a project spreadsheet.
        """
        try:
            # Parse the composite ID (spreadsheetId:row)
            parts = task_id.split(":")
            if len(parts) != 2:
                return {"error": "Invalid task ID format"}
            
            spreadsheet_id = parts[0]
            row = int(parts[1])
            
            sheets_service = self.client._get_service(
                "sheets", "v4", ["https://www.googleapis.com/auth/spreadsheets"]
            )
            
            def _update_status():
                body = {
                    'values': [[status]]
                }
                return sheets_service.spreadsheets().values().update(
                    spreadsheetId=spreadsheet_id,
                    range=f"Sheet1!C{row}",
                    valueInputOption='RAW',
                    body=body
                ).execute()
            
            result = await self.client.run_in_executor(_update_status)
            
            return {
                "id": task_id,
                "status": status,
                "updated": True
            }
            
        except Exception as e:
            logger.error("Error updating task status", error=str(e))
            return {"error": str(e)}
    
    # ----- Meeting Export Methods -----
    
    async def export_meeting_to_google_workspace(self, meeting_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Export meeting data to Google Workspace (document and optional calendar event).
        """
        try:
            # Export to Google Doc
            doc_result = await self.client.export_meeting_to_doc(meeting_data)
            
            # Get action items for potential calendar events
            action_items = meeting_data.get("action_items", [])
            
            # In a real implementation, you might add calendar events for action items with due dates
            
            return {
                "id": doc_result.get("id", ""),
                "title": doc_result.get("title", ""),
                "url": f"https://docs.google.com/document/d/{doc_result.get('id', '')}/edit",
                "type": "google_doc"
            }
            
        except Exception as e:
            logger.error("Error exporting meeting", error=str(e))
            return {"error": str(e)}