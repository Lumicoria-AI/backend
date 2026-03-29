from typing import Any, List, Optional, Dict
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, status, UploadFile, File, Query, Form, Body
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum
import json
import os
import tempfile
import uuid
import structlog

from backend.api.deps import get_current_active_user
from backend.db.mongodb.repositories.document_repository import get_document_repository
from backend.db.mongodb.repositories.task_repository import task_repository
from backend.db.mongodb.repositories.permission_repository import permission_repository
from backend.services.ai_model_service import ai_model_service
from backend.services.storage_service import storage_service
from backend.services.context_service import context_service
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.models.document import (
    Document,
    DocumentCreate,
    DocumentUpdate,
    DocumentType,
    DocumentStatus,
    ExtractionResult
)

# Configure logger
logger = structlog.get_logger(__name__)
from backend.models.task import TaskCreate

router = APIRouter()

async def _get_doc_repo():
    """Lazily resolve the document repository (needs async init)."""
    return await get_document_repository()


async def _check_doc_permission(
    current_user: User,
    document_id: str,
    permission_type: str,
) -> bool:
    """
    Check if the user has permission to act on a document.

    Owners (created_by) and org-members always have full access to their own
    documents.  Falls back to the permission_repository for cross-org/shared docs.
    """
    user_id = str(current_user.id)
    org_id = getattr(current_user, "organization_id", None) or user_id

    # Fetch the document to check ownership
    doc = await (await _get_doc_repo()).get_document_by_id(document_id, org_id)
    if not doc:
        return False

    # Owner always has permission
    if str(doc.created_by) == user_id:
        return True

    # Same org = access
    if str(doc.organization_id) == str(org_id):
        return True

    # Fall back to explicit permission entries
    return await permission_repository.check_permission(
        user_id=user_id,
        organization_id=org_id,
        resource_type="DOCUMENT",
        resource_id=document_id,
        permission_type=permission_type,
    )

def _serialize_doc(doc) -> dict:
    """Convert a Document model (with ObjectId fields) to a JSON-safe dict."""
    data = doc.model_dump(by_alias=True) if hasattr(doc, "model_dump") else dict(doc)
    # Convert ObjectId fields to strings
    for field in ("_id", "id", "organization_id", "created_by"):
        if field in data and data[field] is not None:
            data[field] = str(data[field])
    # Ensure 'id' is present
    if "_id" in data:
        data["id"] = data.pop("_id")
    return data

class DocumentResponse(BaseModel):
    id: str
    name: str
    description: Optional[str] = None
    document_type: DocumentType
    status: DocumentStatus
    organization_id: str
    created_by: str
    created_at: datetime
    updated_at: Optional[datetime] = None
    metadata: Optional[Dict[str, Any]] = None
    file_url: Optional[str] = None
    mime_type: Optional[str] = None
    file_size: Optional[int] = None
    extraction_status: Optional[str] = None
    extraction_result: Optional[Dict[str, Any]] = None

    model_config = {"from_attributes": True}

    @classmethod
    def from_document(cls, doc) -> "DocumentResponse":
        """Convert a Document model (with ObjectId fields) to a response."""
        data = doc.model_dump() if hasattr(doc, "model_dump") else doc.__dict__
        # Convert ObjectId fields to strings
        for field in ("id", "_id", "organization_id", "created_by"):
            if field in data and data[field] is not None:
                data[field] = str(data[field])
        # Map _id to id
        if "_id" in data and "id" not in data:
            data["id"] = data.pop("_id")
        elif "_id" in data:
            data.pop("_id")
        return cls(**data)

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

@router.get("", response_model=None)
@router.get("/", response_model=None)
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=500),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    List all documents owned by the current user.
    """
    documents = await (await _get_doc_repo()).get_documents_by_user(
        user_id=str(current_user.id),
        skip=skip,
        limit=limit
    )
    return [_serialize_doc(d) for d in documents]

@router.post("/upload", response_model=None, status_code=status.HTTP_201_CREATED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    name: str = Form(...),
    description: Optional[str] = Form(None),
    document_type: DocumentType = Form(...),
    metadata: Optional[str] = Form(None),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """
    Upload a document to S3 (MinIO + R2) and trigger background processing
    (chunking, embedding, vector store ingestion).
    """
    user_id = str(current_user.id)
    org_id = getattr(current_user, "organization_id", None) or user_id

    # Read file content
    await file.seek(0)
    file_content = await file.read()
    file_size = len(file_content)

    # Parse optional metadata JSON
    metadata_dict: Dict[str, Any] = {}
    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except json.JSONDecodeError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid metadata JSON")

    # Generate document ID and S3 key
    doc_id = str(uuid.uuid4())
    safe_filename = file.filename or "document"
    s3_key = f"{user_id}/{doc_id}/{safe_filename}"

    # 1. Upload to S3 (dual-write MinIO + R2)
    try:
        await storage_service.upload_file(
            file_content=file_content,
            key=s3_key,
            content_type=file.content_type or "application/octet-stream",
        )
    except Exception as e:
        logger.error("S3 upload failed", error=str(e), key=s3_key)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="File storage failed")

    # 2. Create document record in MongoDB
    document_data = {
        "name": name,
        "description": description,
        "document_type": document_type.value,
        "organization_id": org_id,
        "created_by": user_id,
        "file_url": s3_key,
        "mime_type": file.content_type or "application/octet-stream",
        "file_size": file_size,
        "metadata": {**metadata_dict, "s3_key": s3_key, "original_filename": safe_filename},
        "status": DocumentStatus.UPLOADED.value,
    }

    try:
        document = await (await _get_doc_repo()).create_document(document_data)
    except Exception as e:
        logger.error("Failed to create document record", error=str(e))
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to save document metadata")

    # 3. Background task: process document (chunk → embed → vector store → auto-extract → auto-generate tasks)
    async def _process_upload(doc_id_str: str, org_id_str: str, uid: str, content: bytes, filename: str, doc_name: str):
        from backend.services.notification_service import notification_service
        from backend.db.mongodb.models.notification import NotificationType, NotificationPriority
        temp_path = None
        repo = await _get_doc_repo()
        try:
            # ── Step 1: Mark as processing ──
            await repo.update_document(
                doc_id_str,
                {"$set": {"status": DocumentStatus.PROCESSING.value}},
                organization_id=org_id_str,
            )

            # Notify: processing started
            try:
                await notification_service.send_document_notification(
                    user_id=uid, document_id=doc_id_str,
                    action="processing", document_name=doc_name,
                )
            except Exception:
                pass  # Non-fatal

            # ── Step 2: Chunk → embed → vector store ──
            suffix = os.path.splitext(filename)[1] or ".pdf"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(content)
                temp_path = tmp.name

            result = await context_service.add_document_from_file(
                file_path=temp_path,
                user_id=uid,
                title=doc_name,
                tags=metadata_dict.get("tags", []),
            )

            if result.get("status") != "success":
                await repo.update_document(
                    doc_id_str,
                    {"$set": {
                        "status": DocumentStatus.FAILED.value,
                        "extraction_status": "failed",
                        "extraction_error": result.get("error", "Unknown processing error"),
                    }},
                    organization_id=org_id_str,
                )
                try:
                    await notification_service.send_document_notification(
                        user_id=uid, document_id=doc_id_str,
                        action="failed", document_name=doc_name,
                    )
                except Exception:
                    pass
                return

            # ── Step 3: Auto-extract structured data via DocumentAgent ──
            extraction_result = {}
            auto_tasks = []
            try:
                from backend.agents.document_agent import DocumentAgent

                agent = DocumentAgent({
                    "type": "document",
                    "model_config": {"model": "sonar-large-online"},
                })

                # Read text from the file for the LLM
                doc_obj = await repo.get_document_by_id(doc_id_str, org_id_str)
                if doc_obj:
                    doc_text = await repo._get_document_content(doc_obj)
                else:
                    doc_text = content.decode("utf-8", errors="replace")[:12000]

                agent_result = await agent.process_async({
                    "text": doc_text,
                    "metadata": {"id": doc_id_str, "name": doc_name},
                })

                extraction_result = agent_result
                auto_tasks = agent_result.get("tasks", [])

            except Exception as extract_err:
                logger.warning("Auto-extraction failed (non-fatal)", error=str(extract_err))
                extraction_result = {"extraction_error": str(extract_err)}

            # ── Step 4: Save everything to the document record ──
            await repo.update_document(
                doc_id_str,
                {"$set": {
                    "status": DocumentStatus.PROCESSED.value,
                    "extraction_status": "completed",
                    "extraction_result": extraction_result,
                    "metadata.chunk_count": result.get("chunk_count", 0),
                    "metadata.vector_ids": result.get("vector_ids", []),
                    "metadata.auto_tasks": auto_tasks,
                }},
                organization_id=org_id_str,
            )
            logger.info("Document processed", document_id=doc_id_str, chunks=result.get("chunk_count"), tasks=len(auto_tasks))

            # Notify: processing completed (in-app + push + WebSocket + email)
            try:
                task_count = len(auto_tasks)
                task_summary = f" — {task_count} task{'s' if task_count != 1 else ''} extracted" if auto_tasks else ""
                await notification_service.create_in_app_notification(
                    user_id=uid,
                    title="Document Processed",
                    content=f"'{doc_name}' has been processed successfully{task_summary}. Review the results in your dashboard.",
                    notification_type=NotificationType.DOCUMENT,
                    priority=NotificationPriority.NORMAL,
                    metadata={"document_id": doc_id_str, "action": "processed", "task_count": task_count},
                )
                # Send email using document_processed template
                try:
                    from backend.db.mongodb.repositories.user_repository import get_user_repository
                    user_repo = await get_user_repository()
                    user_obj = await user_repo.get_user_by_id(uid)
                    if user_obj and user_obj.email:
                        # Build task list for email template
                        created_tasks_for_email = [
                            {"title": t.get("title", "Untitled"), "due_date": t.get("deadline", "")}
                            for t in auto_tasks[:5]
                        ] if auto_tasks else []
                        await notification_service.send_email_notification(
                            to_email=user_obj.email,
                            template_name="document_processed",
                            template_data={
                                "subject": f"Document Processed: {doc_name}",
                                "user_name": getattr(user_obj, "full_name", None) or user_obj.email.split("@")[0],
                                "document_name": doc_name,
                                "document_type": os.path.splitext(filename)[1].upper().lstrip(".") or "Document",
                                "processed_at": datetime.utcnow().strftime("%B %d, %Y at %I:%M %p UTC"),
                                "tasks_created": task_count if task_count > 0 else None,
                                "created_tasks": created_tasks_for_email or None,
                                "document_url": f"https://lumicoria.ai/agents/document",
                                "tasks_url": f"https://lumicoria.ai/tasks" if task_count > 0 else None,
                            },
                        )
                except Exception as email_err:
                    logger.debug("Email notification failed (non-fatal)", error=str(email_err))
            except Exception:
                pass  # Non-fatal

        except Exception as proc_err:
            logger.error("Background processing failed", document_id=doc_id_str, error=str(proc_err))
            try:
                await repo.update_document(
                    doc_id_str,
                    {"$set": {
                        "status": DocumentStatus.FAILED.value,
                        "extraction_error": str(proc_err),
                    }},
                    organization_id=org_id_str,
                )
            except Exception:
                pass
            # Notify: processing failed
            try:
                await notification_service.send_document_notification(
                    user_id=uid, document_id=doc_id_str,
                    action="failed", document_name=doc_name,
                )
            except Exception:
                pass
        finally:
            if temp_path and os.path.exists(temp_path):
                os.unlink(temp_path)

    background_tasks.add_task(
        _process_upload,
        str(document.id), str(org_id), user_id,
        file_content, safe_filename, name,
    )

    logger.info("Document uploaded to S3", document_id=str(document.id), s3_key=s3_key)

    # Log activity
    await log_activity(
        user_id=user_id,
        organization_id=str(org_id),
        activity_type="document.uploaded",
        details={"name": name, "file_size": file_size, "mime_type": file.content_type, "document_type": document_type.value},
        related_resource_type="DOCUMENT",
        related_resource_id=str(document.id),
        agent_name="Document Agent",
    )

    return _serialize_doc(document)


@router.get("/{document_id}/presigned-url")
async def get_presigned_url(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """
    Get a presigned URL to view/download a document directly from S3.
    """
    document = await (await _get_doc_repo()).get_document_by_id(
        document_id, organization_id=getattr(current_user, "organization_id", None) or str(current_user.id)
    )
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    s3_key = document.file_url
    if not s3_key:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No file associated with this document")

    try:
        url = await storage_service.get_presigned_url(s3_key)
        return {"url": url, "document_id": document_id, "expires_in": storage_service._primary and storage_service._primary.bucket}
    except Exception as e:
        logger.error("Failed to generate presigned URL", error=str(e), key=s3_key)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Failed to generate download URL")

@router.get("/{document_id}", response_model=None)
async def get_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get document by ID.
    """
    document = await (await _get_doc_repo()).get_document_by_id(
        document_id,
        organization_id=getattr(current_user, "organization_id", None) or str(current_user.id)
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )

    user_org = getattr(current_user, "organization_id", None) or str(current_user.id)
    if str(document.organization_id) != str(user_org):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not authorized to access this document"
        )

    return _serialize_doc(document)

@router.put("/{document_id}", response_model=None)
async def update_document(
    document_id: str,
    document_in: DocumentUpdate,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Update document metadata.
    """
    if not await _check_doc_permission(current_user, document_id, "EDIT"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to update this document"
        )

    document = await (await _get_doc_repo()).update_document(
        document_id=document_id,
        organization_id=getattr(current_user, "organization_id", None) or str(current_user.id),
        update_data=document_in.dict(exclude_unset=True)
    )
    if not document:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document not found"
        )
    return _serialize_doc(document)

@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(
    document_id: str,
    current_user: User = Depends(get_current_active_user),
) -> None:
    """
    Delete a document — removes from S3 (both MinIO and R2) and MongoDB.
    """
    org_id = getattr(current_user, "organization_id", None) or str(current_user.id)

    if not await _check_doc_permission(current_user, document_id, "DELETE"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to delete this document"
        )

    # Fetch the document to get the S3 key
    document = await (await _get_doc_repo()).get_document_by_id(document_id, org_id)
    if not document:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    # Delete from S3
    s3_key = document.file_url
    if s3_key:
        try:
            await storage_service.delete_file(s3_key)
        except Exception as e:
            logger.error("Failed to delete file from S3", error=str(e), key=s3_key)

    # Delete from MongoDB
    deleted = await (await _get_doc_repo()).delete_document(document_id, org_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found")

    await log_activity(
        user_id=str(current_user.id),
        organization_id=org_id,
        activity_type="document.deleted",
        details={"name": document.name, "document_id": document_id},
        related_resource_type="DOCUMENT",
        related_resource_id=document_id,
        agent_name="Document Agent",
    )


@router.post("/{document_id}/extract")
async def extract_document_data(
    document_id: str,
    extraction_config: Optional[Dict[str, Any]] = Body(None),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Extract data from a document using Perplexity AI.
    
    This endpoint uses the Perplexity-powered document agent to analyze documents
    and extract key information, including dates, names, organizations, monetary amounts,
    action items, and key points.
    """
    if not await _check_doc_permission(current_user, document_id, "PROCESS"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to process this document"
        )

    try:
        document = await (await _get_doc_repo()).get_document_by_id(
            document_id=document_id,
            organization_id=getattr(current_user, "organization_id", None) or str(current_user.id)
        )

        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )

        # Check if document is in a state that can be processed
        if document.status not in [DocumentStatus.UPLOADED, DocumentStatus.FAILED, DocumentStatus.PROCESSED]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Document cannot be processed in its current state: {document.status}"
            )
        
        # Update document status to PROCESSING
        await (await _get_doc_repo()).update_document(
            document_id=document_id,
            organization_id=getattr(current_user, "organization_id", None) or str(current_user.id),
            update_data={"$set": {"status": DocumentStatus.PROCESSING}}
        )
        
        # Extract data using Perplexity-powered document agent
        result = await (await _get_doc_repo()).extract_document_data(
            document_id=document_id,
            organization_id=getattr(current_user, "organization_id", None) or str(current_user.id),
            extraction_config=extraction_config
        )
        
        # Update document status to PROCESSED
        await (await _get_doc_repo()).update_document(
            document_id=document_id,
            organization_id=getattr(current_user, "organization_id", None) or str(current_user.id),
            update_data={"$set": {"status": DocumentStatus.PROCESSED}}
        )

        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None) or str(current_user.id),
            activity_type="document.extracted",
            details={"document_id": document_id, "document_name": getattr(document, "name", "")},
            related_resource_type="DOCUMENT",
            related_resource_id=document_id,
            agent_name="Document Agent",
        )

        return result
    except Exception as e:
        # Update document status to PROCESSING_FAILED
        await (await _get_doc_repo()).update_document(
            document_id=document_id,
            organization_id=getattr(current_user, "organization_id", None) or str(current_user.id),
            update_data={"$set": {"status": DocumentStatus.PROCESSING_FAILED}}
        )
        
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document processing failed: {str(e)}"
        )

class DocumentQueryResponse(BaseModel):
    """Response model for document queries."""
    query: str
    response: str
    document_id: str
    document_name: str
    citations: Optional[List[Dict[str, Any]]] = None
    extracted_data: Optional[Dict[str, Any]] = None
    search_queries: Optional[List[str]] = None

@router.post("/{document_id}/query", response_model=DocumentQueryResponse)
async def query_document(
    document_id: str,
    query: DocumentQuery,
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Query document content using natural language with Perplexity AI.
    
    This endpoint uses Perplexity's powerful search and document analysis capabilities
    to answer questions about documents using natural language. Results include
    citations to relevant parts of the document and additional context from online
    sources when using online-enabled models.
    """
    if not await _check_doc_permission(current_user, document_id, "QUERY"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to query this document"
        )

    try:
        # Get document to check its status first
        document = await (await _get_doc_repo()).get_document_by_id(
            document_id=document_id,
            organization_id=getattr(current_user, "organization_id", None) or str(current_user.id)
        )
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Query document using Perplexity-powered document agent
        result = await (await _get_doc_repo()).query_document(
            document_id=document_id,
            organization_id=getattr(current_user, "organization_id", None) or str(current_user.id),
            query=query.query,
            filters=query.filters,
            include_extracted_data=query.include_extracted_data
        )
        
        await log_activity(
            user_id=str(current_user.id),
            organization_id=getattr(current_user, "organization_id", None) or str(current_user.id),
            activity_type="document.queried",
            details={"document_id": document_id, "query": query.query},
            related_resource_type="DOCUMENT",
            related_resource_id=document_id,
            agent_name="Document Agent",
        )

        return result
    except Exception as e:
        await logger.error("Error querying document", error=str(e), document_id=document_id)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Document query failed: {str(e)}"
        )

class TaskConfig(BaseModel):
    """Task generation configuration options."""
    max_tasks: Optional[int] = Field(None, description="Maximum number of tasks to generate")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus on when generating tasks")
    priority_threshold: Optional[str] = Field(None, description="Minimum priority level for tasks (e.g., 'medium')")
    due_date_required: Optional[bool] = Field(None, description="Whether tasks must have due dates")
    assignees: Optional[List[str]] = Field(None, description="List of potential assignees to consider")
    user_context: Optional[Dict[str, Any]] = Field(None, description="Additional context about the user/organization")

@router.post("/{document_id}/create-tasks")
async def create_tasks_from_document(
    document_id: str,
    task_config: Optional[TaskConfig] = Body(None),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Create tasks from document content using Perplexity AI.
    
    This endpoint uses Perplexity's powerful document analysis capabilities to identify
    actionable items in documents and convert them to tasks. Tasks include title,
    priority, deadlines (when available), and suggested assignees.
    """
    if not await _check_doc_permission(current_user, document_id, "PROCESS"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to create tasks from this document"
        )

    try:
        # Get document to check its status first
        document = await (await _get_doc_repo()).get_document_by_id(
            document_id=document_id,
            organization_id=getattr(current_user, "organization_id", None) or str(current_user.id)
        )
        
        if not document:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Document not found"
            )
        
        # Convert task config to dict if provided
        config_dict = task_config.dict() if task_config else {}
        
        # Create tasks using Perplexity-powered document agent
        tasks = await (await _get_doc_repo()).create_tasks_from_document(
            document_id=document_id,
            organization_id=getattr(current_user, "organization_id", None) or str(current_user.id),
            created_by=current_user.id,
            task_config=config_dict
        )
        
        # Notify user about created tasks (in-app + push + WebSocket + email)
        try:
            from backend.services.notification_service import notification_service
            from backend.db.mongodb.models.notification import NotificationType, NotificationPriority
            doc_name = document.name if document else "document"
            task_count = len(tasks)
            await notification_service.create_in_app_notification(
                user_id=str(current_user.id),
                title=f"{task_count} Task{'s' if task_count != 1 else ''} Created",
                content=f"{task_count} task{'s' if task_count != 1 else ''} created from '{doc_name}'. Review and manage them in your task dashboard.",
                notification_type=NotificationType.TASK,
                priority=NotificationPriority.HIGH,
                metadata={"document_id": document_id, "action": "tasks_created", "task_count": task_count},
            )
            # Send email using document_processed template (with task details)
            try:
                user_email = getattr(current_user, "email", None)
                if user_email:
                    created_tasks_for_email = []
                    for t in tasks[:5]:
                        t_dict = t if isinstance(t, dict) else (t.dict() if hasattr(t, "dict") else {"title": str(t)})
                        created_tasks_for_email.append({
                            "title": t_dict.get("title", "Untitled"),
                            "due_date": t_dict.get("due_date", "") or t_dict.get("deadline", ""),
                        })
                    user_name = getattr(current_user, "full_name", None) or user_email.split("@")[0]
                    await notification_service.send_email_notification(
                        to_email=user_email,
                        template_name="document_processed",
                        template_data={
                            "subject": f"{task_count} Task{'s' if task_count != 1 else ''} Created from '{doc_name}'",
                            "user_name": user_name,
                            "document_name": doc_name,
                            "tasks_created": task_count,
                            "created_tasks": created_tasks_for_email,
                            "document_url": f"https://lumicoria.ai/agents/document",
                            "tasks_url": "https://lumicoria.ai/tasks",
                        },
                    )
            except Exception as email_err:
                logger.debug("Email notification for tasks failed (non-fatal)", error=str(email_err))
        except Exception:
            pass  # Non-fatal — tasks were still created

        # Return created tasks
        return {
            "document_id": document_id,
            "tasks_created": len(tasks),
            "tasks": tasks
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create tasks: {str(e)}"
        )

@router.get("/search", response_model=None)
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
    documents = await (await _get_doc_repo()).get_organization_documents(
        organization_id=getattr(current_user, "organization_id", None) or str(current_user.id),
        status=status,
        document_type=document_type,
        search_query=query,
        skip=skip,
        limit=limit
    )
    return [_serialize_doc(d) for d in documents]

@router.get("/summary", response_model=DocumentSummaryResponse)
async def get_document_summary(
    current_user: User = Depends(get_current_active_user)
) -> Dict[str, Any]:
    """
    Get summary statistics for documents in the organization.
    """
    summary = await (await _get_doc_repo()).get_document_summary(organization_id=getattr(current_user, "organization_id", None) or str(current_user.id))
    return summary

@router.get("/analytics", response_model=Dict[str, Any])
async def get_document_analytics(
    time_range: str = Query("7d", pattern="^(1d|7d|30d|90d|1y)$"),
    current_user: User = Depends(get_current_active_user)
) -> Any:
    """
    Get document processing analytics.
    """
    analytics = await (await _get_doc_repo()).get_document_analytics(
        organization_id=getattr(current_user, "organization_id", None) or str(current_user.id),
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
    recent_documents_data = await (await _get_doc_repo()).get_recent_documents_with_counts(
        organization_id=getattr(current_user, "organization_id", None) or str(current_user.id),
        limit=limit
    )

    # Map the data from the repository to the Pydantic model
    recent_documents = [RecentDocumentSummary(**doc_data) for doc_data in recent_documents_data]

    return recent_documents 