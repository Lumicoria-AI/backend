"""
API endpoints for the RAG system in Lumicoria.ai
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, Query
from fastapi.responses import StreamingResponse
from typing import List, Dict, Any, Optional
import uuid
import structlog
from pydantic import BaseModel, Field
from datetime import datetime

from ....api.deps import get_current_active_user
from ....models.user import User
from ....services.context_service import context_service
from ....agents.agent_service import AgentService
from ....core.dependencies import get_agent_service
from ....db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)

router = APIRouter()

# ── MongoDB collection ─────────────────────────────────────────────
RAG_COLLECTION = "rag_sessions"


async def _save_rag_session(
    user_id: str,
    query: str,
    response: str,
    sources: List[Dict[str, Any]],
    conversation_id: str,
    context_used: int,
    processing_time: float,
) -> str:
    """Persist RAG Q&A session to MongoDB. Returns the document _id."""
    col = await MongoDB.get_collection(RAG_COLLECTION)
    doc_id = str(uuid.uuid4())
    doc = {
        "_id": doc_id,
        "user_id": user_id,
        "query": query,
        "response": response,
        "sources": sources,
        "conversation_id": conversation_id,
        "context_used": context_used,
        "processing_time": processing_time,
        "created_at": datetime.utcnow().isoformat(),
    }
    await col.insert_one(doc)
    return doc_id

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
    current_user: User = Depends(get_current_active_user),
):
    """
    Ask a question using the RAG system with relevant context.
    """
    user_id = str(current_user.id)
    organization_id = getattr(current_user, "organization_id", None)

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
    }

    try:
        # Get the RAG agent
        rag_agent = agent_service.get_agent("rag")

        # Process the request
        result = await rag_agent.process_async(request_data)

        if not result.get("success", False):
            raise HTTPException(
                status_code=500,
                detail=f"RAG processing error: {result.get('error', 'Unknown error')}",
            )

        ai_response = result["response"]
        sources = result.get("sources", [])
        context_used = result.get("context_used", 0)
        processing_time = result.get("processing_time_seconds", 0.0)

        # Save to MongoDB
        doc_id = await _save_rag_session(
            user_id=user_id,
            query=request.query,
            response=ai_response,
            sources=sources,
            conversation_id=result.get("conversation_id", conversation_id),
            context_used=context_used,
            processing_time=processing_time,
        )

        return RAGResponse(
            response=ai_response,
            conversation_id=result.get("conversation_id", conversation_id),
            sources=sources,
            processing_time_seconds=processing_time,
            context_used=context_used,
            success=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error("ask_rag_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-document-url", status_code=201)
async def add_document_url(
    document: DocumentRequest,
    background_tasks: BackgroundTasks,
    current_user: User = Depends(get_current_active_user)
):
    """
    Add a document from a URL to the user's context.
    """
    if not document.url:
        raise HTTPException(status_code=400, detail="URL is required")
    
    user_id = str(current_user.id)
    organization_id = getattr(current_user, "organization_id", None)
    
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Add text content to the user's context.
    """
    if not document.text:
        raise HTTPException(status_code=400, detail="Text content is required")
    
    user_id = str(current_user.id)
    organization_id = getattr(current_user, "organization_id", None)
    
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
    current_user: User = Depends(get_current_active_user)
):
    """
    Delete context for the current user based on filters.
    """
    user_id = str(current_user.id)
    
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


# ── History / Detail / Stats / Delete ──────────────────────────────

@router.get("/history")
async def get_rag_history(
    limit: int = Query(default=20, le=50),
    skip: int = Query(default=0, ge=0),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get the current user's RAG Q&A history from MongoDB."""
    try:
        col = await MongoDB.get_collection(RAG_COLLECTION)
        query_filter: Dict[str, Any] = {"user_id": str(current_user.id)}

        cursor = col.find(query_filter).sort("created_at", -1).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)

        return [
            {
                "id": doc["_id"],
                "query": doc.get("query", ""),
                "response_preview": (doc.get("response", ""))[:200],
                "context_used": doc.get("context_used", 0),
                "sources_count": len(doc.get("sources", [])),
                "created_at": doc.get("created_at", ""),
            }
            for doc in docs
        ]
    except Exception as e:
        logger.error("rag_history_fetch_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


@router.get("/history/{session_id}")
async def get_rag_detail(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get full details of a specific RAG session."""
    try:
        col = await MongoDB.get_collection(RAG_COLLECTION)
        doc = await col.find_one({"_id": session_id, "user_id": str(current_user.id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Session not found")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("rag_detail_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch session: {str(e)}")


@router.get("/stats")
async def get_rag_stats(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get aggregated stats for the current user's RAG usage."""
    try:
        col = await MongoDB.get_collection(RAG_COLLECTION)
        user_id = str(current_user.id)

        pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {
                "_id": None,
                "total_sessions": {"$sum": 1},
                "avg_context_used": {"$avg": "$context_used"},
                "avg_processing_time": {"$avg": "$processing_time"},
            }},
        ]
        results = await col.aggregate(pipeline).to_list(length=1)

        if results:
            r = results[0]
            return {
                "total_sessions": r.get("total_sessions", 0),
                "avg_context_used": round(r.get("avg_context_used", 0), 1),
                "avg_processing_time": round(r.get("avg_processing_time", 0), 2),
            }
        return {"total_sessions": 0, "avg_context_used": 0, "avg_processing_time": 0}
    except Exception as e:
        logger.error("rag_stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


@router.delete("/history/{session_id}")
async def delete_rag_session(
    session_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Delete a specific RAG session from history."""
    try:
        col = await MongoDB.get_collection(RAG_COLLECTION)
        result = await col.delete_one({"_id": session_id, "user_id": str(current_user.id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_rag_session_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete session: {str(e)}")
