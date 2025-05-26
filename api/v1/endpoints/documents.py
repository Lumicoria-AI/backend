from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Query, Form, Body
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import json

from api.deps import get_current_active_user
from db.mongodb.repositories.document_repository import document_repository
from db.mongodb.repositories.task_repository import task_repository
from db.mongodb.repositories.permission_repository import permission_repository
from services.ai_model_service import ai_model_service
from models.user import User
from models.document import (
    Document,
    DocumentCreate,
    DocumentUpdate,
    DocumentType,
    DocumentStatus,
    ExtractionResult
)
from models.task import TaskCreate

router = APIRouter()

class DocumentResponse(BaseModel):
    id: str
    name: str
    description: Optional[str]
    document_type: DocumentType
    status: DocumentStatus
    organization_id: str
    created_by: str
    created_at: datetime
    updated_at: Optional[datetime]
    metadata: Optional[Dict[str, Any]]
    file_url: Optional[str]
    file_type: Optional[str]
    file_size: Optional[int]
    extraction_status: Optional[str]
    extraction_result: Optional[Dict[str, Any]]

class RecentDocumentSummary(BaseModel):
    id: str
    name: str
    document_type: DocumentType
    created_at: datetime
    extracted_items_count: int = Field(..., description="Number of extracted items from the document.")
    tasks_created_count: int = Field(..., description="Number of tasks created from the document.")

class DocumentQuery(BaseModel):
    query: str = Field(..., description="Natural language query or keywords to search within documents.")
    filters: Optional[Dict[str, Any]] = Field(None, description="Optional filters (e.g., {'document_type': 'pdf'}).")
    include_extracted_data: bool = Field(False, description="Whether to include full extracted data in results.")

class DocumentSummaryResponse(BaseModel):
    total_count: int
    summary_by_status: List[Dict[str, Any]]
    summary_by_type: List[Dict[str, Any]]

@router.post("/upload", response_model=DocumentResponse, status_code=status.HTTP_201_CREATED)
async def upload_document(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    document_type: DocumentType = Form(...),
    metadata: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Upload and process a new document.
    """
    if not current_user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not belong to an organization"
        )

    file_location = f"uploads/{current_user.organization_id}/{file.filename}"
    await file.seek(0)
    file_content = await file.read()
    file_size = len(file_content)

    metadata_dict = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid metadata JSON format"
            )

    document_data = {
        "name": name,
        "description": description,
        "document_type": document_type.value,
        "organization_id": current_user.organization_id,
        "created_by": current_user.id,
        "file_url": file_location,
        "file_type": file.content_type,
        "file_size": file_size,
        "metadata": metadata_dict,
        "status": DocumentStatus.UPLOADED.value
    }

    try:
        document = await document_repository.create_document(document_data)
        await logger.info("Document uploaded and metadata created", document_id=document.id, filename=file.filename)
        return document
    except Exception as e:
        await logger.error("Error uploading document", error=str(e), filename=file.filename)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to upload document")

@router.get("/{document_id}", response_model=DocumentResponse)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get document by ID.
    """
    document = await document_repository.get_document_by_id(
        document_id,
        organization_id=current_user.organization_id
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    if str(document.organization_id) != current_user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this document"
        )

    return document

@router.put("/{document_id}", response_model=DocumentResponse)
async def update_document(
    document_id: str,
    document_in: DocumentUpdate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Update document metadata.
    """
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="DOCUMENT",
        resource_id=document_id,
        permission_type="EDIT"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to update this document"
        )

    document = await document_repository.update_document(
        document_id=document_id,
        organization_id=current_user.organization_id,
        update_data=document_in.dict(exclude_unset=True)
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return document

@router.post("/{document_id}/extract")
async def extract_document_data(
    document_id: str,
    extraction_config: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Extract data from a document.
    """
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="DOCUMENT",
        resource_id=document_id,
        permission_type="PROCESS"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to process this document"
        )

    try:
        result = await document_repository.extract_document_data(
            document_id=document_id,
            organization_id=current_user.organization_id,
            extraction_config=extraction_config
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/{document_id}/query")
async def query_document(
    document_id: str,
    query: DocumentQuery,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Query document content using natural language.
    """
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="DOCUMENT",
        resource_id=document_id,
        permission_type="QUERY"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to query this document"
        )

    try:
        result = await document_repository.query_document(
            document_id=document_id,
            organization_id=current_user.organization_id,
            query=query.query,
            filters=query.filters,
            include_extracted_data=query.include_extracted_data
        )
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.post("/{document_id}/create-tasks")
async def create_tasks_from_document(
    document_id: str,
    task_config: Optional[Dict[str, Any]] = None,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Create tasks from document content.
    """
    has_permission = await permission_repository.check_permission(
        user_id=current_user.id,
        organization_id=current_user.organization_id,
        resource_type="DOCUMENT",
        resource_id=document_id,
        permission_type="PROCESS"
    )
    if not has_permission:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to create tasks from this document"
        )

    try:
        tasks = await document_repository.create_tasks_from_document(
            document_id=document_id,
            organization_id=current_user.organization_id,
            created_by=current_user.id,
            task_config=task_config
        )
        return tasks
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

@router.get("/search", response_model=List[DocumentResponse])
async def search_documents(
    query: Optional[str] = Query(None, description="Text search query", min_length=1),
    status: Optional[DocumentStatus] = Query(None),
    document_type: Optional[DocumentType] = Query(None),
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Search documents using text search and filters.
    """
    if not current_user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not belong to an organization"
        )

    documents = await document_repository.get_organization_documents(
        organization_id=current_user.organization_id,
        status=status,
        document_type=document_type,
        search_query=query,
        skip=skip,
        limit=limit
    )
    return documents

@router.get("/summary", response_model=DocumentSummaryResponse)
async def get_document_summary(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get summary statistics for documents in the organization.
    """
    if not current_user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not belong to an organization"
        )

    summary = await document_repository.get_document_summary(organization_id=current_user.organization_id)
    return summary

@router.get("/analytics", response_model=Dict[str, Any])
async def get_document_analytics(
    time_range: str = Query("7d", regex="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get document processing analytics.
    """
    analytics = await document_repository.get_document_analytics(
        organization_id=current_user.organization_id,
        time_range=time_range
    )
    return analytics

@router.get("/recent", response_model=List[RecentDocumentSummary])
async def get_recent_documents(
    limit: int = Query(5, ge=1, le=100),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get a list of recent documents with summary counts.
    """
    if not current_user.organization_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User does not belong to an organization"
        )

    recent_documents_data = await document_repository.get_recent_documents_with_counts(
        organization_id=current_user.organization_id,
        limit=limit
    )

    # Map the data from the repository to the Pydantic model
    recent_documents = [RecentDocumentSummary(**doc_data) for doc_data in recent_documents_data]

    return recent_documents 