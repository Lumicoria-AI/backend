"""
API endpoints for the main Ask Lumicoria.ai chat feature.

This implements the core RAG-based chat experience that combines context from multiple sources.
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile, Form
from fastapi.responses import JSONResponse
from typing import List, Dict, Any, Optional
import uuid
from pydantic import BaseModel, Field
from datetime import datetime
import os
import shutil
from pathlib import Path

from ....core.auth import get_current_user
from ....core.config import settings
from ....services.context_service import context_service
from ....agents.agent_service import AgentService
from ....core.dependencies import get_agent_service

router = APIRouter()

# Request and Response Models
class LumicoriaChatRequest(BaseModel):
    """Request model for Lumicoria.ai chat queries."""
    query: str = Field(..., description="The user's question or query")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context continuity")
    save_to_context: bool = Field(True, description="Whether to save this interaction to context")
    include_sources: Optional[List[str]] = Field(None, description="Source types to include (e.g., upload, drive, chat_history)")
    max_sources_per_type: Optional[int] = Field(3, description="Maximum number of sources per type to include")

class DocumentUploadRequest(BaseModel):
    """Request model for document upload metadata."""
    title: Optional[str] = Field(None, description="Document title")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing the document") 
    source: str = Field("upload", description="Source of the document")

class DocumentUrlRequest(BaseModel):
    """Request model for adding a document from URL."""
    url: str = Field(..., description="URL to process")
    title: Optional[str] = Field(None, description="Document title")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing the document")

class DocumentTextRequest(BaseModel):
    """Request model for adding text content."""
    text: str = Field(..., description="Text content to process")
    title: Optional[str] = Field(None, description="Document title")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing the document")
    source: str = Field("manual_entry", description="Source of the document")

class GoogleDriveRequest(BaseModel):
    """Request model for adding Google Drive document."""
    drive_file_id: str = Field(..., description="Google Drive file ID")
    title: Optional[str] = Field(None, description="Document title")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing the document")

class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    response: str = Field(..., description="The AI's response to the query")
    conversation_id: str = Field(..., description="ID of the conversation")
    sources: List[Dict[str, Any]] = Field([], description="Sources used for generating the response")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    context_used: int = Field(0, description="Number of context chunks used")
    success: bool = Field(True, description="Whether the request was successful")

class DocumentListRequest(BaseModel):
    """Request model for listing documents."""
    source_types: Optional[List[str]] = Field(None, description="Source types to include")
    tags: Optional[List[str]] = Field(None, description="Tags to filter by")
    limit: int = Field(100, description="Maximum number of documents to return")
    offset: int = Field(0, description="Offset for pagination")

@router.post("/chat", response_model=ChatResponse)
async def ask_lumicoria(
    request: LumicoriaChatRequest,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Ask a question to Lumicoria.ai with context from multiple sources.
    """
    # Get the user information
    user_id = current_user["id"]
    organization_id = current_user.get("organization_id")
    
    # Create conversation ID if not provided
    conversation_id = request.conversation_id or str(uuid.uuid4())
    
    # Prepare request data for the agent
    request_data = {
        "query": request.query,
        "user_id": user_id,
        "organization_id": organization_id,
        "conversation_id": conversation_id,
        "save_to_context": request.save_to_context,
        "include_sources": request.include_sources,
        "max_sources_per_type": request.max_sources_per_type
    }
    
    try:
        # Get the RAG agent
        rag_agent = agent_service.get_agent("rag")
        
        # Process the request
        start_time = datetime.utcnow()
        result = await rag_agent.process_async(request_data)
        end_time = datetime.utcnow()
        processing_time = (end_time - start_time).total_seconds()
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500, 
                detail=f"Processing error: {result.get('error', 'Unknown error')}"
            )
            
        return ChatResponse(
            response=result["response"],
            conversation_id=result.get("conversation_id", conversation_id),
            sources=result.get("sources", []),
            processing_time_seconds=processing_time,
            context_used=result.get("context_used", 0),
            success=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/upload", status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Upload a document to be processed and stored for context.
    """
    user_id = current_user["id"]
    organization_id = current_user.get("organization_id")
    
    # Parse tags if provided
    parsed_tags = []
    if tags:
        try:
            import json
            parsed_tags = json.loads(tags)
        except:
            parsed_tags = [tag.strip() for tag in tags.split(',') if tag.strip()]
    
    # Create a unique filename
    file_ext = os.path.splitext(file.filename)[1]
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    
    # Create user's upload directory if it doesn't exist
    user_upload_dir = settings.UPLOAD_DIR / user_id
    user_upload_dir.mkdir(exist_ok=True, parents=True)
    
    # Save the file
    file_path = user_upload_dir / unique_filename
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    # Process document in background
    background_tasks.add_task(
        context_service.add_document_from_file,
        file_path=str(file_path),
        user_id=user_id,
        organization_id=organization_id,
        title=title or file.filename,
        tags=parsed_tags
    )
    
    return {
        "message": "Document upload received and processing started",
        "status": "processing",
        "filename": file.filename,
        "saved_as": unique_filename
    }

@router.post("/documents/url", status_code=201)
async def add_document_url(
    document: DocumentUrlRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Add a document from a URL to be processed and stored for context.
    """
    user_id = current_user["id"]
    organization_id = current_user.get("organization_id")
    
    # Process URL in background
    background_tasks.add_task(
        context_service.add_document_from_url,
        url=document.url,
        user_id=user_id,
        organization_id=organization_id,
        tags=document.tags,
        title=document.title
    )
    
    return {"message": "URL processing started", "status": "processing", "url": document.url}

@router.post("/documents/text", status_code=201)
async def add_document_text(
    document: DocumentTextRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Add text content to be processed and stored for context.
    """
    user_id = current_user["id"]
    organization_id = current_user.get("organization_id")
    
    # Process text in background
    background_tasks.add_task(
        context_service.add_document_from_text,
        text=document.text,
        user_id=user_id,
        title=document.title,
        organization_id=organization_id,
        source=document.source,
        tags=document.tags
    )
    
    return {"message": "Text processing started", "status": "processing"}

@router.post("/documents/google-drive", status_code=201)
async def add_google_drive_document(
    document: GoogleDriveRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Add a Google Drive document to be processed and stored for context.
    """
    user_id = current_user["id"]
    organization_id = current_user.get("organization_id")
    
    # Process Google Drive document in background
    background_tasks.add_task(
        context_service.add_document_from_google_drive,
        drive_file_id=document.drive_file_id,
        user_id=user_id,
        organization_id=organization_id,
        tags=document.tags,
        title=document.title
    )
    
    return {
        "message": "Google Drive document processing started", 
        "status": "processing", 
        "drive_file_id": document.drive_file_id
    }

@router.post("/documents/list")
async def list_documents(
    request: DocumentListRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    List documents for the current user.
    """
    user_id = current_user["id"]
    organization_id = current_user.get("organization_id")
    
    result = await context_service.get_user_documents(
        user_id=user_id,
        organization_id=organization_id,
        source_types=request.source_types,
        tags=request.tags,
        limit=request.limit,
        offset=request.offset
    )
    
    return result

@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete a document and all its chunks from the vector store.
    """
    user_id = current_user["id"]
    
    result = await context_service.delete_user_context(
        user_id=user_id,
        document_id=document_id
    )
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete document: {result.get('error', 'Unknown error')}"
        )
        
    return {"message": f"Document {document_id} deleted successfully"}

@router.post("/context/search")
async def search_context(
    query: str,
    include_sources: Optional[List[str]] = None,
    max_results: int = 5,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Search user's context using semantic similarity.
    """
    user_id = current_user["id"]
    organization_id = current_user.get("organization_id")
    
    result = await context_service.get_context_for_query(
        query=query,
        user_id=user_id,
        organization_id=organization_id,
        k=max_results,
        include_sources=include_sources
    )
    
    return result

@router.post("/context/combined-search")
async def combined_context_search(
    query: str,
    max_results_per_source: int = 3,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Search across all context sources, returning results categorized by source.
    This is useful for the primary Ask Lumicoria AI feature.
    """
    user_id = current_user["id"]
    organization_id = current_user.get("organization_id")
    
    result = await context_service.get_all_context_sources(
        query=query,
        user_id=user_id,
        organization_id=organization_id,
        max_results_per_source=max_results_per_source
    )
    
    return result
