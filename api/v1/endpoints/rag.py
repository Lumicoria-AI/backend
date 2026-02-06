"""
API endpoints for the RAG system in Lumicoria.ai
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import uuid
from pydantic import BaseModel, Field
from datetime import datetime

from ....core.auth import get_current_user
from ....services.context_service import context_service
from ....agents.agent_service import AgentService
from ....core.dependencies import get_agent_service

router = APIRouter()

# Request and Response Models
class RAGQueryRequest(BaseModel):
    """Request model for RAG queries."""
    query: str = Field(..., description="The user's question or query")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context continuity")
    save_to_context: bool = Field(True, description="Whether to save this interaction to context")
    include_sources: Optional[List[str]] = Field(None, description="Source types to include (e.g., upload, drive, chat_history)")
    
class DocumentRequest(BaseModel):
    """Request model for adding a document to context."""
    url: Optional[str] = Field(None, description="URL to process")
    text: Optional[str] = Field(None, description="Text content to process")
    title: Optional[str] = Field(None, description="Document title")
    source: str = Field("manual_entry", description="Source of the document")
    tags: Optional[List[str]] = Field(None, description="Tags for categorizing the document")

class RAGResponse(BaseModel):
    """Response model for RAG system responses."""
    response: str = Field(..., description="The AI's response to the query")
    conversation_id: str = Field(..., description="ID of the conversation")
    sources: List[Dict[str, Any]] = Field([], description="Sources used for generating the response")
    processing_time_seconds: float = Field(..., description="Processing time in seconds")
    context_used: int = Field(0, description="Number of context chunks used")
    success: bool = Field(True, description="Whether the request was successful")

@router.post("/ask", response_model=RAGResponse)
async def ask_rag(
    request: RAGQueryRequest,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Ask a question using the RAG system with relevant context.
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
        "include_sources": request.include_sources
    }
    
    try:
        # Get the RAG agent
        rag_agent = agent_service.get_agent("rag")
        
        # Process the request
        result = await rag_agent.process_async(request_data)
        
        if not result.get("success", False):
            raise HTTPException(
                status_code=500, 
                detail=f"RAG processing error: {result.get('error', 'Unknown error')}"
            )
            
        return RAGResponse(
            response=result["response"],
            conversation_id=result.get("conversation_id", conversation_id),
            sources=result.get("sources", []),
            processing_time_seconds=result.get("processing_time_seconds", 0.0),
            context_used=result.get("context_used", 0),
            success=True
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-document-url", status_code=201)
async def add_document_url(
    document: DocumentRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Add a document from a URL to the user's context.
    """
    if not document.url:
        raise HTTPException(status_code=400, detail="URL is required")
    
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
    
    return {"message": "Document processing started", "status": "processing"}

@router.post("/add-document-text", status_code=201)
async def add_document_text(
    document: DocumentRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Add text content to the user's context.
    """
    if not document.text:
        raise HTTPException(status_code=400, detail="Text content is required")
    
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
    
    return {"message": "Document processing started", "status": "processing"}

@router.delete("/user-context")
async def delete_user_context(
    source: Optional[str] = None,
    document_id: Optional[str] = None,
    older_than_days: Optional[int] = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """
    Delete context for the current user based on filters.
    """
    user_id = current_user["id"]
    
    result = await context_service.delete_user_context(
        user_id=user_id,
        source=source,
        document_id=document_id,
        older_than_days=older_than_days
    )
    
    if not result.get("success", False):
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete context: {result.get('error', 'Unknown error')}"
        )
        
    return {"message": "Context deleted successfully", "filters": result.get("filters", {})}
