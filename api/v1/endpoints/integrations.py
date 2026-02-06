from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Query
from pydantic import BaseModel
from datetime import datetime

from backend.api.deps import get_current_active_user
from backend.models.user import User
from backend.models.integration import Integration, IntegrationCreate, IntegrationUpdate, IntegrationType
from backend.db.mongodb.repositories.integration_repository import integration_repository
from backend.db.mongodb.repositories.permission_repository import permission_repository

router = APIRouter()

class IntegrationResponse(BaseModel):
    id: str
    name: str
    type: IntegrationType
    organization_id: str
    created_by: str
    created_at: datetime
    updated_at: Optional[datetime]
    status: str # e.g., 'active', 'inactive', 'configuration_required'
    metadata: Optional[Dict[str, Any]]
    # Note: We might not want to return credentials in the response for security reasons

@router.post("/", response_model=IntegrationResponse)
async def create_integration(
    integration_in: IntegrationCreate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Create a new integration configuration.
    """
    # Check if user has permission to create integrations
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="INTEGRATION",
        resource_id="*",
        permission_type="CREATE"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to create integrations"
        )

    try:
        integration = await integration_repository.create_integration(
            name=integration_in.name,
            type=integration_in.type,
            organization_id=current_user.organization_id,
            created_by=current_user.id,
            credentials=integration_in.credentials, # Securely handle credentials in repository
            settings=integration_in.settings,
            metadata=integration_in.metadata
        )
        return integration
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/{integration_id}", response_model=IntegrationResponse)
async def get_integration(
    integration_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get integration details by ID.
    """
    # Check if user has permission to view the integration
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="INTEGRATION",
        resource_id=integration_id,
        permission_type="VIEW"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to view this integration"
        )

    integration = await integration_repository.get_integration_by_id(
        integration_id=integration_id,
        organization_id=current_user.organization_id
    )
    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Integration not found"
        )
    return integration

@router.put("/{integration_id}", response_model=IntegrationResponse)
async def update_integration(
    integration_id: str,
    integration_in: IntegrationUpdate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Update an integration configuration.
    """
    # Check if user has permission to edit the integration
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="INTEGRATION",
        resource_id=integration_id,
        permission_type="EDIT"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to update this integration"
        )

    integration = await integration_repository.update_integration(
        integration_id=integration_id,
        organization_id=current_user.organization_id,
        update_data=integration_in.dict(exclude_unset=True)
    )
    if not integration:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Integration not found"
        )
    return integration

@router.delete("/{integration_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_integration(
    integration_id: str,
    current_user: User = Depends(get_current_active_user)
) -> None:
    """
    Delete an integration configuration.
    """
    # Check if user has permission to delete the integration
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="INTEGRATION",
        resource_id=integration_id,
        permission_type="DELETE"
    )
    if not has_permission:        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to delete this integration"
        )
    
    success = await integration_repository.delete_integration(
        integration_id=integration_id,
        organization_id=current_user.organization_id
    )
    
    if not success:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Integration not found"
        )

@router.get("/", response_model=List[IntegrationResponse])
async def list_integrations(
    type: Optional[IntegrationType] = None,
    status: Optional[str] = None,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    List integrations for the current organization.
    """
    # Check if user has permission to list integrations
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="INTEGRATION",
        resource_id="*",
        permission_type="VIEW"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to list integrations"
        )

    integrations = await integration_repository.list_integrations(
        organization_id=current_user.organization_id,
        type=type,
        status=status
    )
    return integrations 