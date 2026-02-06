from typing import Any, List, Optional, Dict, Union
from fastapi import APIRouter, Depends, HTTPException, status, Body
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import structlog

from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.models.mongodb_models import ComponentType, ComponentCategory
from backend.db.mongodb.repositories.component_repository import component_repository
from backend.agents.studio_service import StudioService, ComponentDefinition, ComponentInstance, AgentWorkflow
from backend.agents.security import AgentSecurityContext, AgentPermission
from backend.agents.factory import AgentFactory
from backend.agents.security import AgentSecurityManager
from backend.agents.cache import AgentCache

# Configure logger
logger = structlog.get_logger(__name__)

router = APIRouter()

# Initialize the StudioService - in production this should be handled via dependency injection
def get_studio_service() -> StudioService:
    agent_factory = AgentFactory()
    security_manager = AgentSecurityManager(jwt_secret="temp-secret-for-dev")  # In production, load from environment
    return StudioService(agent_factory, security_manager)

# Pydantic models for request/response
class ComponentDefinitionModel(BaseModel):
    id: str
    name: str
    type: str
    category: str
    description: str
    config_schema: Dict[str, Any]
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    icon: str
    version: str = "1.0.0"
    is_beta: bool = False
    requires_auth: bool = False

class ComponentInstanceModel(BaseModel):
    id: str
    component_id: str
    name: str
    config: Dict[str, Any]
    position: Dict[str, int]
    connections: List[Dict[str, str]]

class CreateWorkflowRequest(BaseModel):
    name: str = Field(..., description="Name of the workflow")
    description: str = Field(..., description="Description of the workflow")
    components: List[ComponentInstanceModel] = Field(..., description="Component instances in the workflow")
    is_public: bool = Field(False, description="Whether the workflow is public")
    tags: Optional[List[str]] = Field(None, description="Optional tags for the workflow")

class UpdateWorkflowRequest(BaseModel):
    name: Optional[str] = Field(None, description="Updated name of the workflow")
    description: Optional[str] = Field(None, description="Updated description of the workflow")
    components: Optional[List[ComponentInstanceModel]] = Field(None, description="Updated component instances")
    is_public: Optional[bool] = Field(None, description="Updated public status")
    tags: Optional[List[str]] = Field(None, description="Updated tags")

class WorkflowResponse(BaseModel):
    id: str
    name: str
    description: str
    components: List[ComponentInstanceModel]
    created_at: datetime
    updated_at: datetime
    created_by: str
    version: str
    is_public: bool
    tags: Optional[List[str]]

class DeploymentResponse(BaseModel):
    agent_id: str
    workflow_id: str
    status: str
    deployed_at: datetime

class ValidationErrorResponse(BaseModel):
    errors: List[str]

# Component Endpoints

@router.get("/components", response_model=List[ComponentDefinitionModel])
async def get_components(
    category: Optional[str] = None,
    component_type: Optional[str] = None,
    current_user: User = Depends(get_current_active_user),
    studio_service: StudioService = Depends(get_studio_service)
) -> Any:
    """
    Get available components, optionally filtered by category or type.
    """
    try:
        # Convert string parameters to enum values if provided
        category_enum = None
        if category:
            try:
                category_enum = ComponentCategory(category)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid category: {category}"
                )
                
        component_type_enum = None
        if component_type:
            try:
                component_type_enum = ComponentType(component_type)
            except ValueError:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid component type: {component_type}"
                )
        
        components = studio_service.get_available_components(
            category=category_enum,
            component_type=component_type_enum
        )
        
        # Convert to response model
        return [
            ComponentDefinitionModel(
                id=c.id,
                name=c.name,
                type=c.type.value,
                category=c.category.value,
                description=c.description,
                config_schema=c.config_schema,
                input_schema=c.input_schema,
                output_schema=c.output_schema,
                icon=c.icon,
                version=c.version,
                is_beta=c.is_beta,
                requires_auth=c.requires_auth
            )
            for c in components
        ]
        
    except Exception as e:
        logger.error("Error getting components", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get components: {str(e)}"
        )

@router.get("/components/{component_id}", response_model=ComponentDefinitionModel)
async def get_component(
    component_id: str,
    current_user: User = Depends(get_current_active_user),
    studio_service: StudioService = Depends(get_studio_service)
) -> Any:
    """
    Get a specific component by ID.
    """
    try:
        if component_id not in studio_service._component_definitions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Component not found: {component_id}"
            )
            
        c = studio_service._component_definitions[component_id]
        
        return ComponentDefinitionModel(
            id=c.id,
            name=c.name,
            type=c.type.value,
            category=c.category.value,
            description=c.description,
            config_schema=c.config_schema,
            input_schema=c.input_schema,
            output_schema=c.output_schema,
            icon=c.icon,
            version=c.version,
            is_beta=c.is_beta,
            requires_auth=c.requires_auth
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting component", error=str(e), component_id=component_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get component: {str(e)}"
        )

# Workflow Endpoints

@router.post("/workflows", response_model=WorkflowResponse)
async def create_workflow(
    request: CreateWorkflowRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    studio_service: StudioService = Depends(get_studio_service)
) -> Any:
    """
    Create a new agent workflow.
    """
    try:
        # Convert component instances
        components = [
            ComponentInstance(
                id=c.id,
                component_id=c.component_id,
                name=c.name,
                config=c.config,
                position=c.position,
                connections=c.connections
            )
            for c in request.components
        ]
        
        # Set up security context
        security_context = AgentSecurityContext(
            user_id=str(current_user.id),
            organization_id=str(current_user.organization_id) if hasattr(current_user, "organization_id") else None,
            permissions=[AgentPermission.CREATE, AgentPermission.CONFIGURE]
        )
        
        workflow = await studio_service.create_workflow(
            name=request.name,
            description=request.description,
            components=components,
            security_context=security_context,
            is_public=request.is_public,
            tags=request.tags
        )
        
        # Convert to response model
        return WorkflowResponse(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            components=[
                ComponentInstanceModel(
                    id=c.id,
                    component_id=c.component_id,
                    name=c.name,
                    config=c.config,
                    position=c.position,
                    connections=c.connections
                )
                for c in workflow.components
            ],
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            created_by=workflow.created_by,
            version=workflow.version,
            is_public=workflow.is_public,
            tags=workflow.tags
        )
        
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except Exception as e:
        logger.error("Error creating workflow", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create workflow: {str(e)}"
        )

@router.get("/workflows", response_model=List[WorkflowResponse])
async def get_workflows(
    current_user: User = Depends(get_current_active_user),
    studio_service: StudioService = Depends(get_studio_service)
) -> Any:
    """
    Get all workflows accessible to the current user.
    """
    try:
        # Set up security context
        security_context = AgentSecurityContext(
            user_id=str(current_user.id),
            organization_id=str(current_user.organization_id) if hasattr(current_user, "organization_id") else None,
            permissions=[AgentPermission.READ]
        )
        
        workflows = []
        
        # Filter workflows based on security context
        for workflow_id, workflow in studio_service._workflows.items():
            # Show public workflows or those created by the current user
            if (workflow.is_public or 
                workflow.created_by == security_context.user_id or 
                AgentPermission.ADMIN in security_context.permissions):
                workflows.append(workflow)
        
        # Convert to response model
        return [
            WorkflowResponse(
                id=workflow.id,
                name=workflow.name,
                description=workflow.description,
                components=[
                    ComponentInstanceModel(
                        id=c.id,
                        component_id=c.component_id,
                        name=c.name,
                        config=c.config,
                        position=c.position,
                        connections=c.connections
                    )
                    for c in workflow.components
                ],
                created_at=workflow.created_at,
                updated_at=workflow.updated_at,
                created_by=workflow.created_by,
                version=workflow.version,
                is_public=workflow.is_public,
                tags=workflow.tags
            )
            for workflow in workflows
        ]
        
    except Exception as e:
        logger.error("Error getting workflows", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflows: {str(e)}"
        )

@router.get("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_active_user),
    studio_service: StudioService = Depends(get_studio_service)
) -> Any:
    """
    Get a specific workflow by ID.
    """
    try:
        if workflow_id not in studio_service._workflows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow not found: {workflow_id}"
            )
            
        workflow = studio_service._workflows[workflow_id]
        
        # Check if user has permission to access this workflow
        security_context = AgentSecurityContext(
            user_id=str(current_user.id),
            organization_id=str(current_user.organization_id) if hasattr(current_user, "organization_id") else None,
            permissions=[AgentPermission.READ]
        )
        
        if (not workflow.is_public and 
            workflow.created_by != security_context.user_id and 
            AgentPermission.ADMIN not in security_context.permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this workflow"
            )
        
        return WorkflowResponse(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            components=[
                ComponentInstanceModel(
                    id=c.id,
                    component_id=c.component_id,
                    name=c.name,
                    config=c.config,
                    position=c.position,
                    connections=c.connections
                )
                for c in workflow.components
            ],
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            created_by=workflow.created_by,
            version=workflow.version,
            is_public=workflow.is_public,
            tags=workflow.tags
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error getting workflow", error=str(e), workflow_id=workflow_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get workflow: {str(e)}"
        )

@router.put("/workflows/{workflow_id}", response_model=WorkflowResponse)
async def update_workflow(
    workflow_id: str,
    request: UpdateWorkflowRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
    studio_service: StudioService = Depends(get_studio_service)
) -> Any:
    """
    Update an existing workflow.
    """
    try:
        if workflow_id not in studio_service._workflows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow not found: {workflow_id}"
            )
            
        # Set up security context
        security_context = AgentSecurityContext(
            user_id=str(current_user.id),
            organization_id=str(current_user.organization_id) if hasattr(current_user, "organization_id") else None,
            permissions=[AgentPermission.CONFIGURE]
        )
        
        # Prepare updates
        updates = {}
        if request.name is not None:
            updates["name"] = request.name
        if request.description is not None:
            updates["description"] = request.description
        if request.is_public is not None:
            updates["is_public"] = request.is_public
        if request.tags is not None:
            updates["tags"] = request.tags
        if request.components is not None:
            updates["components"] = [
                ComponentInstance(
                    id=c.id,
                    component_id=c.component_id,
                    name=c.name,
                    config=c.config,
                    position=c.position,
                    connections=c.connections
                )
                for c in request.components
            ]
            
        workflow = await studio_service.update_workflow(
            workflow_id=workflow_id,
            updates=updates,
            security_context=security_context
        )
        
        # Convert to response model
        return WorkflowResponse(
            id=workflow.id,
            name=workflow.name,
            description=workflow.description,
            components=[
                ComponentInstanceModel(
                    id=c.id,
                    component_id=c.component_id,
                    name=c.name,
                    config=c.config,
                    position=c.position,
                    connections=c.connections
                )
                for c in workflow.components
            ],
            created_at=workflow.created_at,
            updated_at=workflow.updated_at,
            created_by=workflow.created_by,
            version=workflow.version,
            is_public=workflow.is_public,
            tags=workflow.tags
        )
        
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error updating workflow", error=str(e), workflow_id=workflow_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update workflow: {str(e)}"
        )

@router.delete("/workflows/{workflow_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_active_user),
    studio_service: StudioService = Depends(get_studio_service)
) -> None:
    """
    Delete a workflow.
    """
    try:
        if workflow_id not in studio_service._workflows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow not found: {workflow_id}"
            )
            
        # Set up security context
        security_context = AgentSecurityContext(
            user_id=str(current_user.id),
            organization_id=str(current_user.organization_id) if hasattr(current_user, "organization_id") else None,
            permissions=[AgentPermission.DELETE]
        )
        
        workflow = studio_service._workflows[workflow_id]
        
        # Check if user has permission to delete this workflow
        if (workflow.created_by != security_context.user_id and 
            AgentPermission.ADMIN not in security_context.permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to delete this workflow"
            )
            
        # Remove the workflow
        async with studio_service._lock:
            del studio_service._workflows[workflow_id]
            
        logger.info(
            "workflow_deleted",
            workflow_id=workflow_id,
            user_id=security_context.user_id
        )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deleting workflow", error=str(e), workflow_id=workflow_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete workflow: {str(e)}"
        )

@router.post("/workflows/{workflow_id}/validate", response_model=Union[Dict[str, str], ValidationErrorResponse])
async def validate_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_active_user),
    studio_service: StudioService = Depends(get_studio_service)
) -> Any:
    """
    Validate a workflow for correctness.
    """
    try:
        if workflow_id not in studio_service._workflows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow not found: {workflow_id}"
            )
            
        workflow = studio_service._workflows[workflow_id]
        
        # Check if user has permission to access this workflow
        security_context = AgentSecurityContext(
            user_id=str(current_user.id),
            organization_id=str(current_user.organization_id) if hasattr(current_user, "organization_id") else None,
            permissions=[AgentPermission.READ]
        )
        
        if (not workflow.is_public and 
            workflow.created_by != security_context.user_id and 
            AgentPermission.ADMIN not in security_context.permissions):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You don't have permission to access this workflow"
            )
            
        errors = await studio_service.validate_workflow(workflow)
        
        if errors:
            return ValidationErrorResponse(errors=errors)
        else:
            return {"status": "valid"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error validating workflow", error=str(e), workflow_id=workflow_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to validate workflow: {str(e)}"
        )

@router.post("/workflows/{workflow_id}/deploy", response_model=DeploymentResponse)
async def deploy_workflow(
    workflow_id: str,
    current_user: User = Depends(get_current_active_user),
    studio_service: StudioService = Depends(get_studio_service)
) -> Any:
    """
    Deploy a workflow as an agent.
    """
    try:
        if workflow_id not in studio_service._workflows:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Workflow not found: {workflow_id}"
            )
            
        # Set up security context
        security_context = AgentSecurityContext(
            user_id=str(current_user.id),
            organization_id=str(current_user.organization_id) if hasattr(current_user, "organization_id") else None,
            permissions=[AgentPermission.CREATE]
        )
        
        # Validate workflow first
        workflow = studio_service._workflows[workflow_id]
        errors = await studio_service.validate_workflow(workflow)
        
        if errors:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Workflow validation failed: {', '.join(errors)}"
            )
            
        # Deploy the workflow
        agent = await studio_service.deploy_workflow(
            workflow_id=workflow_id,
            security_context=security_context
        )
        
        # Return deployment information
        now = datetime.utcnow()
        return DeploymentResponse(
            agent_id=str(id(agent)),
            workflow_id=workflow_id,
            status="deployed",
            deployed_at=now
        )
        
    except PermissionError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error deploying workflow", error=str(e), workflow_id=workflow_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to deploy workflow: {str(e)}"
        )
