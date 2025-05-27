from typing import Dict, Any, List, Optional
import structlog
import aiohttp
import json
from datetime import datetime, timedelta
import google.oauth2.credentials
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import asyncio
import functools
import base64
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

logger = structlog.get_logger(__name__)

class GoogleWorkspaceClient:
    """Client for interacting with Google Workspace APIs."""
    
    def __init__(self, credentials_info: Dict[str, Any], user_email: Optional[str] = None):
        """
        Initialize Google Workspace client.
        
        Args:
            credentials_info: Service account or OAuth credentials
            user_email: User email for impersonation when using service account
        """
        self.credentials_info = credentials_info
        self.user_email = user_email
        self.services = {}
        
    def _get_credentials(self, scopes: List[str]):
        """Get credentials for Google API access."""
        creds = None
        
        # Check if it's service account credentials
        if "type" in self.credentials_info and self.credentials_info.get("type") == "service_account":
            # Service account with optional user impersonation
            try:
                creds = Credentials.from_service_account_info(
                    self.credentials_info,
                    scopes=scopes
                )
                
                if self.user_email:
                    # Impersonate the user
                    creds = creds.with_subject(self.user_email)
                    
            except Exception as e:
                logger.error("Error creating service account credentials", error=str(e))
                raise e
        else:
            # OAuth credentials
            try:
                creds = google.oauth2.credentials.Credentials(
                    token=self.credentials_info.get("access_token"),
                    refresh_token=self.credentials_info.get("refresh_token"),
                    client_id=self.credentials_info.get("client_id"),
                    client_secret=self.credentials_info.get("client_secret"),
                    token_uri="https://oauth2.googleapis.com/token",
                    scopes=scopes
                )
            except Exception as e:
                logger.error("Error creating OAuth credentials", error=str(e))
                raise e
                
        return creds
    
    def _get_service(self, api_name: str, api_version: str, scopes: List[str]):
        """Get or create Google API service."""
        service_key = f"{api_name}_{api_version}"
        
        if service_key not in self.services:
            credentials = self._get_credentials(scopes)
            self.services[service_key] = build(api_name, api_version, credentials=credentials)
            
        return self.services[service_key]
    
    async def run_in_executor(self, func, *args, **kwargs):
        """Run synchronous Google API calls in an executor."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, functools.partial(func, *args, **kwargs)
        )
    
    # Calendar API methods
    async def list_calendars(self) -> List[Dict[str, Any]]:
        """List user's calendars."""
        try:
            calendar_service = self._get_service(
                "calendar", "v3", ["https://www.googleapis.com/auth/calendar"]
            )
            
            def _list_calendars():
                result = calendar_service.calendarList().list().execute()
                return result.get('items', [])
                
            calendars = await self.run_in_executor(_list_calendars)
            return calendars
        except HttpError as e:
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
        Create a calendar event.
        
        Args:
            summary: Event title
            description: Event description
            start_time: Event start time
            end_time: Event end time
            attendees: List of attendee emails
            calendar_id: Calendar ID (default is "primary")
            location: Event location
            conference_type: Video conference type ("hangoutsMeet" or "hangouts")
            
        Returns:
            Created event data
        """
        try:
            calendar_service = self._get_service(
                "calendar", "v3", ["https://www.googleapis.com/auth/calendar"]
            )
            
            event = {
                'summary': summary,
                'description': description,
                'start': {
                    'dateTime': start_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'end': {
                    'dateTime': end_time.isoformat(),
                    'timeZone': 'UTC',
                },
                'reminders': {
                    'useDefault': True
                },
            }
            
            if attendees:
                event['attendees'] = [{'email': email} for email in attendees]
            
            if location:
                event['location'] = location
                
            # Add video conferencing if requested
            if conference_type:
                if conference_type == "hangoutsMeet":
                    event['conferenceData'] = {
                        'createRequest': {
                            'requestId': f"meet-{datetime.now().timestamp():.0f}",
                            'conferenceSolutionKey': {
                                'type': 'hangoutsMeet'
                            }
                        }
                    }
                elif conference_type == "hangouts":
                    event['conferenceData'] = {
                        'createRequest': {
                            'requestId': f"hangouts-{datetime.now().timestamp():.0f}",
                            'conferenceSolutionKey': {
                                'type': 'eventHangout'
                            }
                        }
                    }
            
            def _create_event():
                params = {'conferenceDataVersion': 1} if conference_type else {}
                return calendar_service.events().insert(
                    calendarId=calendar_id,
                    body=event,
                    sendUpdates='all',
                    **params
                ).execute()
                
            created_event = await self.run_in_executor(_create_event)
            return created_event
        except Exception as e:
            logger.error("Error creating calendar event", error=str(e))
            raise e
            
    async def get_events(self, 
                       calendar_id: str = "primary", 
                       time_min: datetime = None, 
                       time_max: datetime = None,
                       max_results: int = 10) -> List[Dict[str, Any]]:
        """
        Get events from a calendar.
        
        Args:
            calendar_id: Calendar ID (default is "primary")
            time_min: Start time for listing events
            time_max: End time for listing events
            max_results: Maximum number of results
            
        Returns:
            List of events
        """
        try:
            calendar_service = self._get_service(
                "calendar", "v3", ["https://www.googleapis.com/auth/calendar"]
            )
            
            # Default time range is today to two weeks from now if not specified
            if not time_min:
                time_min = datetime.utcnow()
            if not time_max:
                time_max = time_min + timedelta(days=14)
            
            def _get_events():
                return calendar_service.events().list(
                    calendarId=calendar_id,
                    timeMin=time_min.isoformat() + 'Z',
                    timeMax=time_max.isoformat() + 'Z',
                    maxResults=max_results,
                    singleEvents=True,
                    orderBy='startTime'
                ).execute()
                
            events_result = await self.run_in_executor(_get_events)
            return events_result.get('items', [])
        except Exception as e:
            logger.error("Error getting calendar events", error=str(e))
            return []
    
    # Drive API methods
    async def list_files(self, folder_id: str = None, query: str = None, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        List files in Google Drive.
        
        Args:
            folder_id: ID of the folder to list files from
            query: Custom query string for Drive
            max_results: Maximum number of results
            
        Returns:
            List of files
        """
        try:
            drive_service = self._get_service(
                "drive", "v3", ["https://www.googleapis.com/auth/drive"]
            )
            
            # Build the query
            query_parts = []
            if folder_id:
                query_parts.append(f"'{folder_id}' in parents")
            if query:
                query_parts.append(query)
                
            final_query = " and ".join(query_parts) if query_parts else None
            
            def _list_files():
                return drive_service.files().list(
                    q=final_query,
                    pageSize=max_results,
                    fields="files(id, name, mimeType, webViewLink, createdTime, modifiedTime, owners)"
                ).execute()
                
            files_result = await self.run_in_executor(_list_files)
            return files_result.get('files', [])
        except Exception as e:
            logger.error("Error listing Drive files", error=str(e))
            return []
            
    async def create_document(self, title: str, content: str = None, folder_id: str = None) -> Dict[str, Any]:
        """
        Create a Google Document.
        
        Args:
            title: Document title
            content: Initial document content
            folder_id: ID of the folder to create the document in
            
        Returns:
            Created document data
        """
        try:
            drive_service = self._get_service(
                "drive", "v3", ["https://www.googleapis.com/auth/drive"]
            )
            docs_service = self._get_service(
                "docs", "v1", ["https://www.googleapis.com/auth/documents"]
            )
            
            # Create an empty document
            def _create_doc():
                doc_metadata = {
                    'name': title,
                    'mimeType': 'application/vnd.google-apps.document'
                }
                
                if folder_id:
                    doc_metadata['parents'] = [folder_id]
                    
                return drive_service.files().create(body=doc_metadata, fields='id').execute()
                
            doc = await self.run_in_executor(_create_doc)
            doc_id = doc.get('id')
            
            # Add content if provided
            if content and doc_id:
                def _update_doc_content():
                    # The content needs to be added as requests for the Docs API
                    requests = [{
                        'insertText': {
                            'location': {
                                'index': 1
                            },
                            'text': content
                        }
                    }]
                    
                    return docs_service.documents().batchUpdate(
                        documentId=doc_id,
                        body={'requests': requests}
                    ).execute()
                
                await self.run_in_executor(_update_doc_content)
                
                # Get the complete document data
                def _get_doc():
                    return docs_service.documents().get(documentId=doc_id).execute()
                    
                doc_data = await self.run_in_executor(_get_doc)
                return doc_data
            else:
                # If no content was provided, just return the doc ID
                return {'id': doc_id, 'title': title}
                
        except Exception as e:
            logger.error("Error creating Google Document", error=str(e))
            raise e
    
    async def export_meeting_to_doc(self, 
                                 meeting_data: Dict[str, Any], 
                                 folder_id: str = None) -> Dict[str, Any]:
        """
        Export meeting data to a Google Doc.
        
        Args:
            meeting_data: Meeting data from MeetingAgent
            folder_id: Optional folder ID to create the document in
            
        Returns:
            Created document data
        """
        try:
            # Extract meeting information
            title = meeting_data.get("metadata", {}).get("title", f"Meeting Notes - {datetime.now().strftime('%Y-%m-%d')}")
            summary = meeting_data.get("summary", "No summary available.")
            
            # Get participants
            participants = meeting_data.get("metadata", {}).get("participants", [])
            if not isinstance(participants, list):
                participants = [str(participants)]
            
            # Format document content
            content = f"# {title}\n\n"
            content += f"Date: {datetime.now().strftime('%Y-%m-%d')}\n"
            
            # Add participants
            content += "\n## Participants\n"
            for participant in participants:
                content += f"- {participant}\n"
                
            # Add summary
            content += f"\n## Summary\n{summary}\n"
            
            # Add key points
            key_points = meeting_data.get("key_points", [])
            content += "\n## Key Points\n"
            for point in key_points:
                content += f"- {point}\n"
                
            # Add decisions
            decisions = meeting_data.get("decisions", [])
            content += "\n## Decisions\n"
            for decision in decisions:
                content += f"- {decision}\n"
                
            # Add action items
            action_items = meeting_data.get("action_items", [])
            content += "\n## Action Items\n"
            for item in action_items:
                if isinstance(item, dict):
                    task = item.get("task", "")
                    assignee = item.get("assignee", "Unassigned")
                    deadline = item.get("deadline", "No deadline")
                    content += f"- {task} (Assignee: {assignee}, Due: {deadline})\n"
                else:
                    content += f"- {item}\n"
                    
            # Add questions if available
            if meeting_data.get("questions"):
                content += "\n## Questions\n"
                for question in meeting_data.get("questions", []):
                    content += f"- {question}\n"
                    
            # Add concerns if available
            if meeting_data.get("concerns"):
                content += "\n## Concerns\n"
                for concern in meeting_data.get("concerns", []):
                    content += f"- {concern}\n"
            
            # Create the document
            return await self.create_document(title, content, folder_id)
            
        except Exception as e:
            logger.error("Error exporting meeting to Google Doc", error=str(e))
            raise e
    
    # Gmail API methods
    async def send_email(self, 
                       to: List[str], 
                       subject: str, 
                       body: str, 
                       cc: List[str] = None,
                       bcc: List[str] = None,
                       html_content: str = None) -> Dict[str, Any]:
        """
        Send an email using Gmail API.
        
        Args:
            to: List of recipient email addresses
            subject: Email subject
            body: Plain text email body
            cc: List of CC recipients
            bcc: List of BCC recipients
            html_content: HTML version of the email body
            
        Returns:
            Sent message data
        """
        try:
            gmail_service = self._get_service(
                "gmail", "v1", ["https://www.googleapis.com/auth/gmail.send"]
            )
            
            # Create the message
            message = MIMEMultipart('alternative')
            message['Subject'] = subject
            message['To'] = ', '.join(to)
            
            if cc:
                message['Cc'] = ', '.join(cc)
            if bcc:
                message['Bcc'] = ', '.join(bcc)
                
            # Add the sender only if using OAuth credentials
            if "type" not in self.credentials_info or self.credentials_info.get("type") != "service_account":
                message['From'] = self.user_email or self.credentials_info.get("client_email", "")
            
            # Attach plain text version
            message.attach(MIMEText(body, 'plain'))
            
            # Attach HTML version if provided
            if html_content:
                message.attach(MIMEText(html_content, 'html'))
                
            # Encode the message
            encoded_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
            
            def _send_message():
                return gmail_service.users().messages().send(
                    userId='me',
                    body={'raw': encoded_message}
                ).execute()
                
            result = await self.run_in_executor(_send_message)
            return result
        except Exception as e:
            logger.error("Error sending email", error=str(e))
            raise e
            
    async def list_messages(self, query: str = None, max_results: int = 10) -> List[Dict[str, Any]]:
        """
        List Gmail messages.
        
        Args:
            query: Gmail search query
            max_results: Maximum number of results
            
        Returns:
            List of messages
        """
        try:
            gmail_service = self._get_service(
                "gmail", "v1", ["https://www.googleapis.com/auth/gmail.readonly"]
            )
            
            def _list_messages():
                return gmail_service.users().messages().list(
                    userId='me',
                    q=query,
                    maxResults=max_results
                ).execute()
                
            messages_result = await self.run_in_executor(_list_messages)
            messages = messages_result.get('messages', [])
            
            # Get full message details
            detailed_messages = []
            for msg in messages:
                def _get_message(msg_id):
                    return gmail_service.users().messages().get(
                        userId='me',
                        id=msg_id
                    ).execute()
                    
                message_data = await self.run_in_executor(_get_message, msg['id'])
                detailed_messages.append(message_data)
                
            return detailed_messages
        except Exception as e:
            logger.error("Error listing Gmail messages", error=str(e))
            return []
    
    # Google Sheets methods
    async def create_spreadsheet(self, title: str, data: List[List[Any]] = None, folder_id: str = None) -> Dict[str, Any]:
        """
        Create a Google Spreadsheet.
        
        Args:
            title: Spreadsheet title
            data: Initial data for the first sheet
            folder_id: ID of the folder to create the spreadsheet in
            
        Returns:
            Created spreadsheet data
        """
        try:
            drive_service = self._get_service(
                "drive", "v3", ["https://www.googleapis.com/auth/drive"]
            )
            sheets_service = self._get_service(
                "sheets", "v4", ["https://www.googleapis.com/auth/spreadsheets"]
            )
            
            # Create the spreadsheet
            def _create_spreadsheet():
                spreadsheet_body = {
                    'properties': {
                        'title': title
                    }
                }
                return sheets_service.spreadsheets().create(body=spreadsheet_body).execute()
                
            spreadsheet = await self.run_in_executor(_create_spreadsheet)
            spreadsheet_id = spreadsheet['spreadsheetId']
            
            # Move to specified folder if provided
            if folder_id:
                def _move_file():
                    file = drive_service.files().get(
                        fileId=spreadsheet_id,
                        fields='parents'
                    ).execute()
                    
                    previous_parents = ",".join(file.get('parents', []))
                    
                    return drive_service.files().update(
                        fileId=spreadsheet_id,
                        addParents=folder_id,
                        removeParents=previous_parents,
                        fields='id, parents'
                    ).execute()
                    
                await self.run_in_executor(_move_file)
            
            # Add data if provided
            if data:
                def _update_values():
                    body = {
                        'values': data
                    }
                    return sheets_service.spreadsheets().values().update(
                        spreadsheetId=spreadsheet_id,
                        range='Sheet1!A1',
                        valueInputOption='RAW',
                        body=body
                    ).execute()
                    
                await self.run_in_executor(_update_values)
                
            return {
                'spreadsheetId': spreadsheet_id,
                'title': title,
                'url': f"https://docs.google.com/spreadsheets/d/{spreadsheet_id}"
            }
        except Exception as e:
            logger.error("Error creating Google Spreadsheet", error=str(e))
            raise e
            
    async def export_meeting_to_sheet(self, meeting_data: Dict[str, Any], folder_id: str = None) -> Dict[str, Any]:
        """
        Export meeting data to a Google Sheet.
        
        Args:
            meeting_data: Meeting data from MeetingAgent
            folder_id: Optional folder ID to create the spreadsheet in
            
        Returns:
            Created spreadsheet data
        """
        try:
            # Extract meeting information
            title = meeting_data.get("metadata", {}).get("title", f"Meeting Notes - {datetime.now().strftime('%Y-%m-%d')}")
            
            # Prepare data for the sheet
            data = [
                ["Meeting Notes"],
                [f"Date: {datetime.now().strftime('%Y-%m-%d')}"],
                []
            ]
            
            # Add participants
            participants = meeting_data.get("metadata", {}).get("participants", [])
            if not isinstance(participants, list):
                participants = [str(participants)]
                
            data.append(["Participants"])
            for participant in participants:
                data.append([participant])
            data.append([])
            
            # Add summary
            summary = meeting_data.get("summary", "No summary available.")
            data.append(["Summary"])
            data.append([summary])
            data.append([])
            
            # Add key points
            key_points = meeting_data.get("key_points", [])
            data.append(["Key Points"])
            for point in key_points:
                data.append([point])
            data.append([])
            
            # Add decisions
            decisions = meeting_data.get("decisions", [])
            data.append(["Decisions"])
            for decision in decisions:
                data.append([decision])
            data.append([])
            
            # Add action items with more structure
            action_items = meeting_data.get("action_items", [])
            data.append(["Action Items", "Assignee", "Deadline"])
            for item in action_items:
                if isinstance(item, dict):
                    task = item.get("task", "")
                    assignee = item.get("assignee", "Unassigned")
                    deadline = item.get("deadline", "No deadline")
                    data.append([task, assignee, deadline])
                else:
                    data.append([item, "", ""])
            data.append([])
            
            # Create the spreadsheet
            return await self.create_spreadsheet(title, data, folder_id)
            
        except Exception as e:
            logger.error("Error exporting meeting to Google Sheet", error=str(e))
            raise e
