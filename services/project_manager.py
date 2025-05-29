from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from backend.core.logging import get_logger
from backend.services.integration_service import integration_service
from backend.db.mongodb.repositories.integration_repository import integration_repository
from backend.models.integration import IntegrationType
from backend.core.config import settings

# Initialize logger
logger = get_logger("lumicoria.services.project_manager")

class ProjectManager:
    """
    Project Manager service that integrates with external project management tools
    like Notion to create and manage projects, tasks, and workflows.
    """
    
    def __init__(self):
        pass
        
    async def create_project(self, 
                           organization_id: str,
                           title: str, 
                           description: str, 
                           due_date: Optional[Union[datetime, str]] = None, 
                           status: str = "Not Started", 
                           integration_type: str = "notion",
                           integration_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a new project in the specified integration platform.
        
        Args:
            organization_id: Organization ID
            title: Project title
            description: Project description
            due_date: Due date (datetime or ISO format string)
            status: Initial status
            integration_type: Type of integration ("notion" by default)
            integration_id: Specific integration ID to use
            
        Returns:
            Project creation result
        """
        try:
            # Convert string date to datetime if needed
            if due_date and isinstance(due_date, str):
                try:
                    due_date = datetime.fromisoformat(due_date)
                except ValueError:
                    logger.warning("Invalid date format", date=due_date)
                    due_date = None
            
            # Find appropriate integration if not specified
            if not integration_id:
                # Get first active integration of the specified type
                integrations = await integration_repository.get_integrations_by_type_and_status(
                    organization_id=organization_id,
                    type=self._get_integration_type(integration_type),
                    status="active"
                )
                
                if not integrations:
                    logger.error(
                        "No active integration found",
                        organization_id=organization_id,
                        type=integration_type
                    )
                    return {
                        "status": "error",
                        "message": f"No active {integration_type} integration found for the organization"
                    }
                    
                integration_id = str(integrations[0].id)
            
            # Prepare action data for the integration service
            action_data = {
                "title": title,
                "description": description,
                "status": status
            }
            
            if due_date:
                action_data["due_date"] = due_date
            
            # Execute the integration action
            result = await integration_service.execute_integration_action(
                integration_id=integration_id,
                organization_id=organization_id,
                action="create_project",
                action_data=action_data
            )
            
            return {
                "status": "success",
                "message": f"Project '{title}' created successfully",
                "project_id": result.get("id"),
                "project_data": result
            }
            
        except Exception as e:
            logger.error(
                "Error creating project",
                error=str(e),
                organization_id=organization_id,
                title=title
            )
            return {"status": "error", "message": f"Failed to create project: {str(e)}"}
    
    async def create_project_database(self,
                                   organization_id: str,
                                   title: str,
                                   integration_type: str = "notion",
                                   integration_id: Optional[str] = None,
                                   parent_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Create a project management database/board in the specified integration platform.
        
        Args:
            organization_id: Organization ID
            title: Database/board title
            integration_type: Type of integration ("notion" by default)
            integration_id: Specific integration ID to use
            parent_id: Parent page/folder ID
            
        Returns:
            Database creation result
        """
        try:
            # Find appropriate integration if not specified
            if not integration_id:
                # Get first active integration of the specified type
                integrations = await integration_repository.get_integrations_by_type_and_status(
                    organization_id=organization_id,
                    type=self._get_integration_type(integration_type),
                    status="active"
                )
                
                if not integrations:
                    logger.error(
                        "No active integration found",
                        organization_id=organization_id,
                        type=integration_type
                    )
                    return {
                        "status": "error",
                        "message": f"No active {integration_type} integration found for the organization"
                    }
                    
                integration_id = str(integrations[0].id)
            
            # Prepare action data for the integration service
            action_data = {
                "title": title
            }
            
            if parent_id:
                action_data["parent_id"] = parent_id
            
            # Execute the integration action
            result = await integration_service.execute_integration_action(
                integration_id=integration_id,
                organization_id=organization_id,
                action="create_project_database",
                action_data=action_data
            )
            
            return {
                "status": "success",
                "message": f"Project database '{title}' created successfully",
                "database_id": result.get("id"),
                "database_data": result
            }
            
        except Exception as e:
            logger.error(
                "Error creating project database",
                error=str(e),
                organization_id=organization_id,
                title=title
            )
            return {"status": "error", "message": f"Failed to create project database: {str(e)}"}
    
    async def add_task(self,
                     organization_id: str,
                     database_id: str,
                     task_name: str,
                     description: str,
                     due_date: Optional[Union[datetime, str]] = None,
                     status: str = "Not Started",
                     priority: str = "Medium",
                     assigned_to: Optional[str] = None,
                     integration_type: str = "notion",
                     integration_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Add a task to a project database/board.
        
        Args:
            organization_id: Organization ID
            database_id: Database/board ID
            task_name: Task name/title
            description: Task description
            due_date: Due date (datetime or ISO format string)
            status: Task status
            priority: Task priority
            assigned_to: Person assigned to the task
            integration_type: Type of integration ("notion" by default)
            integration_id: Specific integration ID to use
            
        Returns:
            Task creation result
        """
        try:
            # Convert string date to datetime if needed
            if due_date and isinstance(due_date, str):
                try:
                    due_date = datetime.fromisoformat(due_date)
                except ValueError:
                    logger.warning("Invalid date format", date=due_date)
                    due_date = None
            
            # Find appropriate integration if not specified
            if not integration_id:
                # Get first active integration of the specified type
                integrations = await integration_repository.get_integrations_by_type_and_status(
                    organization_id=organization_id,
                    type=self._get_integration_type(integration_type),
                    status="active"
                )
                
                if not integrations:
                    logger.error(
                        "No active integration found",
                        organization_id=organization_id,
                        type=integration_type
                    )
                    return {
                        "status": "error",
                        "message": f"No active {integration_type} integration found for the organization"
                    }
                    
                integration_id = str(integrations[0].id)
            
            # Prepare action data for the integration service
            action_data = {
                "database_id": database_id,
                "task_name": task_name,
                "description": description,
                "status": status,
                "priority": priority
            }
            
            if due_date:
                action_data["due_date"] = due_date
                
            if assigned_to:
                action_data["assigned_to"] = assigned_to
            
            # Execute the integration action
            result = await integration_service.execute_integration_action(
                integration_id=integration_id,
                organization_id=organization_id,
                action="add_project_task",
                action_data=action_data
            )
            
            return {
                "status": "success",
                "message": f"Task '{task_name}' added successfully",
                "task_id": result.get("id"),
                "task_data": result
            }
            
        except Exception as e:
            logger.error(
                "Error adding task",
                error=str(e),
                organization_id=organization_id,
                database_id=database_id,
                task_name=task_name
            )            
            return {"status": "error", "message": f"Failed to add task: {str(e)}"}

    async def export_meeting_to_project(self,
                                       organization_id: str,
                                       meeting_data: Dict[str, Any],
                                       integration_type: str = "notion",
                                       integration_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Export meeting data to a project management tool.
        
        Args:
            organization_id: Organization ID
            meeting_data: Meeting data from the MeetingAgent
            integration_type: Type of integration ("notion" by default)
            integration_id: Specific integration ID to use
            
        Returns:
            Export result
        """
        try:
            # Find appropriate integration if not specified
            if not integration_id:
                # Get first active integration of the specified type
                integrations = await integration_repository.get_integrations_by_type_and_status(
                    organization_id=organization_id,
                    type=self._get_integration_type(integration_type),
                    status="active"
                )
                
                if not integrations:
                    logger.error(
                        "No active integration found",
                        organization_id=organization_id,
                        type=integration_type
                    )
                    return {
                        "status": "error",
                        "message": f"No active {integration_type} integration found for the organization"
                    }
                    
                integration_id = str(integrations[0].id)
            
            # Execute the integration action based on integration type
            action = None
            if integration_type.lower() == "notion":
                action = "export_meeting_to_notion"
            elif integration_type.lower() == "google_workspace":
                action = "export_meeting_to_google_workspace"
            else:
                logger.error(
                    "Unsupported integration type for meeting export",
                    integration_type=integration_type
                )
                return {
                    "status": "error",
                    "message": f"Unsupported integration type for meeting export: {integration_type}"
                }
                
            result = await integration_service.execute_integration_action(
                integration_id=integration_id,
                organization_id=organization_id,
                action=action,
                action_data={"meeting_data": meeting_data}
            )
            
            return {
                "status": "success",
                "message": "Meeting exported successfully",
                "page_id": result.get("id"),
                "page_data": result
            }
            
        except Exception as e:
            logger.error(
                "Error exporting meeting",
                error=str(e),
                organization_id=organization_id
            )
            return {"status": "error", "message": f"Failed to export meeting: {str(e)}"}
    
    async def get_tasks(self,
                      organization_id: str,
                      database_id: str,
                      filter_status: Optional[str] = None,
                      filter_priority: Optional[str] = None,
                      integration_type: str = "notion",
                      integration_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get tasks from a project database/board.
        
        Args:
            organization_id: Organization ID
            database_id: Database/board ID
            filter_status: Filter by status
            filter_priority: Filter by priority
            integration_type: Type of integration ("notion" by default)
            integration_id: Specific integration ID to use
            
        Returns:
            List of tasks
        """
        try:
            # Find appropriate integration if not specified
            if not integration_id:
                # Get first active integration of the specified type
                integrations = await integration_repository.get_integrations_by_type_and_status(
                    organization_id=organization_id,
                    type=self._get_integration_type(integration_type),
                    status="active"
                )
                
                if not integrations:
                    logger.error(
                        "No active integration found",
                        organization_id=organization_id,
                        type=integration_type
                    )
                    return {
                        "status": "error",
                        "message": f"No active {integration_type} integration found for the organization"
                    }
                    
                integration_id = str(integrations[0].id)
            
            # Prepare action data for the integration service
            action_data = {
                "database_id": database_id
            }
            
            if filter_status:
                action_data["filter_status"] = filter_status
                
            if filter_priority:
                action_data["filter_priority"] = filter_priority
            
            # Execute the integration action
            result = await integration_service.execute_integration_action(
                integration_id=integration_id,
                organization_id=organization_id,
                action="get_project_tasks",
                action_data=action_data
            )
            
            return {
                "status": "success",
                "tasks": result,
                "count": len(result)
            }
            
        except Exception as e:
            logger.error(
                "Error getting tasks",
                error=str(e),
                organization_id=organization_id,
                database_id=database_id
            )
            return {"status": "error", "message": f"Failed to get tasks: {str(e)}"}
            
    def _get_integration_type(self, type_name: str) -> IntegrationType:
        """Convert string type name to IntegrationType enum."""
        type_map = {
            "notion": IntegrationType.NOTION,
            "slack": IntegrationType.SLACK,
            "google_workspace": IntegrationType.GOOGLE_WORKSPACE,
            "salesforce": IntegrationType.SALESFORCE
        }
        
        return type_map.get(type_name.lower(), IntegrationType.NOTION)

# Create a singleton instance
project_manager = ProjectManager()
