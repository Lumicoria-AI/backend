"""
API endpoints for the main Ask Lumicoria.ai chat feature.

Now powered by:
- Intent Router        — LLM-based classification to 21 specialized agents
- Conversation Memory  — MongoDB-persisted chat history
- Response Normalizer  — Consistent output from diverse agent shapes
- Rate Limiter         — In-memory sliding-window (10 req/min per user)
- Security             — Basic prompt injection check
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Optional, AsyncGenerator
import json
import uuid
import re
import time
import structlog
from collections import defaultdict
from pydantic import BaseModel, Field
from datetime import datetime, timezone
import os
import shutil
from pathlib import Path

from ....core.auth import get_current_user
from ....core.config import settings
from ....services.context_service import context_service
from ....agents.agent_service import AgentService
from ....core.dependencies import get_agent_service
from ....agents.router import get_router
from ....agents import memory as conversation_memory
from ....agents.response_normalizer import normalize_agent_response

router = APIRouter()
logger = structlog.get_logger(__name__)

# ═══════════════════════════════════════════════════════════════════
#  Rate Limiter (in-memory sliding window — 10 req/min per user)
# ═══════════════════════════════════════════════════════════════════
_rate_limit_store: Dict[str, list] = defaultdict(list)
RATE_LIMIT_WINDOW = 60    # seconds
RATE_LIMIT_MAX = 10       # max requests per window


def _check_rate_limit(user_id: str) -> bool:
    """Returns True if request is allowed, False if rate-limited."""
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW
    # Prune old entries
    _rate_limit_store[user_id] = [t for t in _rate_limit_store[user_id] if t > window_start]
    if len(_rate_limit_store[user_id]) >= RATE_LIMIT_MAX:
        return False
    _rate_limit_store[user_id].append(now)
    return True


# ═══════════════════════════════════════════════════════════════════
#  Prompt Injection Check
# ═══════════════════════════════════════════════════════════════════
INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?previous\s+instructions",
    r"you\s+are\s+now\s+a",
    r"system\s*:\s*",
    r"<\|im_start\|>",
    r"<\|system\|>",
    r"override\s+your\s+programming",
    r"disregard\s+your\s+instructions",
]
_INJECTION_RE = re.compile("|".join(INJECTION_PATTERNS), re.IGNORECASE)


def _is_prompt_injection(text: str) -> bool:
    return bool(_INJECTION_RE.search(text))


# ═══════════════════════════════════════════════════════════════════
#  Request / Response Models
# ═══════════════════════════════════════════════════════════════════

class LumicoriaChatRequest(BaseModel):
    """Request model for Lumicoria.ai chat queries."""
    query: str = Field(..., description="The user's question or query")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context continuity")
    save_to_context: bool = Field(True, description="Whether to save this interaction to context")
    include_sources: Optional[List[str]] = Field(None, description="Source types to include")
    max_sources_per_type: Optional[int] = Field(3, description="Max sources per type")

class ChatResponse(BaseModel):
    """Response model for chat interactions."""
    response: str = Field(..., description="The AI's response")
    conversation_id: str = Field(..., description="Conversation ID")
    agent_used: str = Field("general", description="Which agent handled the request")
    route_confidence: float = Field(0.0, description="Router confidence in the choice")
    sources: List[Dict[str, Any]] = Field([], description="Sources used")
    processing_time_seconds: float = Field(..., description="Processing time")
    context_used: int = Field(0, description="Number of context chunks used")
    success: bool = Field(True, description="Whether the request was successful")

class ConversationSummary(BaseModel):
    conversation_id: str
    title: str = ""
    preview: str = ""
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    agents_used: List[str] = []


# ═══════════════════════════════════════════════════════════════════
#  Main Chat Endpoint
# ═══════════════════════════════════════════════════════════════════

@router.post("/chat", response_model=ChatResponse)
async def ask_lumicoria(
    request: LumicoriaChatRequest,
    background_tasks: BackgroundTasks,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Ask a question to Lumicoria.ai.
    
    1. Rate limit check
    2. Prompt injection check
    3. Route to the best agent via LLM
    4. Execute the agent
    5. Normalize the response
    6. Save to conversation memory (background)
    """
    user_id = current_user["id"]
    
    # ── Rate Limit ──
    if not _check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait a moment before sending another message."
        )
    
    # ── Prompt Injection ──
    if _is_prompt_injection(request.query):
        logger.warning("prompt_injection_blocked", user_id=user_id, query_preview=request.query[:60])
        raise HTTPException(status_code=400, detail="Your message was flagged for safety. Please rephrase.")
    
    conversation_id = request.conversation_id or str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # ── 1. Get conversation history for context ──
        history = await conversation_memory.get_conversation_history(conversation_id, limit=6)
        
        # ── 2. Route to the right agent ──
        intent_router = await get_router()
        route_result = await intent_router.route(
            message=request.query,
            conversation_history=history,
        )
        agent_key = route_result["agent"]
        confidence = route_result["confidence"]
        
        logger.info(
            "chat_routed",
            user_id=user_id,
            conversation_id=conversation_id,
            agent=agent_key,
            confidence=confidence,
            query_preview=request.query[:80],
        )
        
        # ── 3. Get the agent and execute ──
        try:
            agent = agent_service.get_agent(agent_key)
        except (ValueError, KeyError):
            # Agent not loaded — try the general agent first, then instantiate on-demand
            logger.warning("agent_not_loaded_falling_back", agent=agent_key)
            try:
                agent = agent_service.get_agent("general")
            except (ValueError, KeyError):
                # general agent also not loaded — create a minimal instance on-demand
                logger.warning("general_agent_not_loaded_instantiating", agent=agent_key)
                from ....agents.general_agent import GeneralAgent
                agent = GeneralAgent({})
            agent_key = "general"
        
        # Build agent input — include conversation history for context
        agent_input = {
            "query": request.query,
            "content": request.query,
            "prompt": request.query,
            "user_id": user_id,
            "conversation_id": conversation_id,
        }
        
        # Inject conversation history if the agent supports it
        if history:
            agent_input["conversation_history"] = history
        
        raw_result = await agent.process_async(agent_input)
        
        # ── 4. Normalize ──
        normalized = normalize_agent_response(raw_result, agent_key=agent_key)
        
        processing_time = time.time() - start_time
        
        # ── 5. Save to memory (background — don't block the response) ──
        background_tasks.add_task(
            conversation_memory.save_message,
            conversation_id=conversation_id,
            user_id=user_id,
            role="user",
            content=request.query,
        )
        background_tasks.add_task(
            conversation_memory.save_message,
            conversation_id=conversation_id,
            user_id=user_id,
            role="assistant",
            content=normalized["response"],
            agent=agent_key,
        )
        
        # Auto-generate title on first message
        if not request.conversation_id:
            background_tasks.add_task(
                conversation_memory.generate_conversation_title,
                conversation_id=conversation_id,
                first_message=request.query,
            )
        
        return ChatResponse(
            response=normalized["response"],
            conversation_id=conversation_id,
            agent_used=agent_key,
            route_confidence=confidence,
            sources=normalized.get("sources", []),
            processing_time_seconds=round(processing_time, 2),
            context_used=normalized.get("context_used", 0),
            success=normalized.get("success", True),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error("chat_error", error=str(e), user_id=user_id)
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")


# ═══════════════════════════════════════════════════════════════════
#  Streaming Chat Endpoint  (POST /chat/stream  →  text/event-stream)
# ═══════════════════════════════════════════════════════════════════

@router.post("/stream")
async def ask_lumicoria_stream(
    request: LumicoriaChatRequest,
    agent_service: AgentService = Depends(get_agent_service),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Streaming version of the chat endpoint.
    Returns text/event-stream with JSON-encoded SSE frames.

    Frame types:
      {"type": "meta",  "conversation_id": "...", "agent_used": "..."}  (first frame)
      {"type": "delta", "text": "..."}  (one per token batch)
      {"type": "done",  "processing_time": 1.23}  (final frame)
      {"type": "error", "message": "..."}  (on failure)
    """
    user_id = current_user["id"]

    if not _check_rate_limit(user_id):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Please wait a moment."
        )

    if _is_prompt_injection(request.query):
        raise HTTPException(status_code=400, detail="Your message was flagged for safety.")

    conversation_id = request.conversation_id or str(uuid.uuid4())
    start_time = time.time()

    async def event_stream() -> AsyncGenerator[str, None]:
        full_response = ""
        agent_key = "general"

        # ── Step 1: fetch history BEFORE saving this turn (avoids it appearing in LLM context twice) ──
        try:
            history = await conversation_memory.get_conversation_history(conversation_id, limit=6)
        except Exception:
            history = []

        # ── Step 2: persist user message immediately — guaranteed even if streaming fails ──
        await conversation_memory.save_message(
            conversation_id=conversation_id, user_id=user_id,
            role="user", content=request.query,
        )

        try:
            # ── Step 3: Route intent ──
            intent_router = await get_router()
            route_result = await intent_router.route(
                message=request.query,
                conversation_history=history,
            )
            agent_key = route_result["agent"]
            confidence = route_result["confidence"]

            # First frame: metadata
            yield f"data: {json.dumps({'type': 'meta', 'conversation_id': conversation_id, 'agent_used': agent_key, 'confidence': confidence})}\n\n"

            # ── Step 4: Resolve agent ──
            try:
                agent = agent_service.get_agent(agent_key)
            except (ValueError, KeyError):
                try:
                    agent = agent_service.get_agent("general")
                except (ValueError, KeyError):
                    from ....agents.general_agent import GeneralAgent
                    agent = GeneralAgent({})
                agent_key = "general"

            # ── Step 5: Stream or fallback ──
            from ....ai_models.base import LLMConfig
            llm = getattr(agent, "llm_client", None) or getattr(agent, "perplexity_client", None)

            if llm and hasattr(llm, "stream"):
                system_prompt = getattr(agent, "system_prompt", None) or ""
                messages = []
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                if history:
                    for m in history[-8:]:
                        messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
                messages.append({"role": "user", "content": request.query})

                cfg = LLMConfig(temperature=0.7, max_tokens=None)  # None = model's native max (65k for gemini-2.5-flash)
                async for chunk in llm.stream(messages, config=cfg):
                    if chunk.content:
                        full_response += chunk.content
                        yield f"data: {json.dumps({'type': 'delta', 'text': chunk.content})}\n\n"
            else:
                # Fallback: process_async → single delta
                agent_input = {
                    "query": request.query,
                    "content": request.query,
                    "prompt": request.query,
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                }
                if history:
                    agent_input["conversation_history"] = history
                raw_result = await agent.process_async(agent_input)
                normalized = normalize_agent_response(raw_result, agent_key=agent_key)
                full_response = normalized["response"]
                yield f"data: {json.dumps({'type': 'delta', 'text': full_response})}\n\n"

            # ── Step 6: Persist assistant message with await (NOT create_task) before done frame ──
            if full_response:
                await conversation_memory.save_message(
                    conversation_id=conversation_id, user_id=user_id,
                    role="assistant", content=full_response, agent=agent_key,
                )
            if not request.conversation_id:
                import asyncio as _asyncio
                _asyncio.create_task(conversation_memory.generate_conversation_title(
                    conversation_id=conversation_id, first_message=request.query,
                ))

            # ── Step 7: Done frame ──
            processing_time = round(time.time() - start_time, 2)
            yield f"data: {json.dumps({'type': 'done', 'processing_time': processing_time})}\n\n"

        except Exception as e:
            logger.error("stream_chat_error", error=str(e), user_id=user_id)
            # Save any partial assistant content so the conversation is still recoverable
            if full_response:
                try:
                    await conversation_memory.save_message(
                        conversation_id=conversation_id, user_id=user_id,
                        role="assistant", content=full_response, agent=agent_key,
                    )
                except Exception:
                    pass
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


# ═══════════════════════════════════════════════════════════════════
#  Conversation History Endpoints
# ═══════════════════════════════════════════════════════════════════

@router.get("/conversations", response_model=List[ConversationSummary])
async def list_conversations(
    limit: int = Query(50, ge=1, le=100),
    offset: int = Query(0, ge=0),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List all conversations for the current user."""
    user_id = current_user["id"]
    conversations = await conversation_memory.list_user_conversations(user_id, limit=limit, offset=offset)
    return conversations


@router.get("/conversations/{conversation_id}")
async def get_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Get full conversation history by ID."""
    doc = await conversation_memory.get_full_conversation(conversation_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Conversation not found")
    # Security: verify ownership
    if doc.get("user_id") != current_user["id"]:
        raise HTTPException(status_code=403, detail="Access denied")
    # Remove MongoDB _id (not JSON serializable)
    doc.pop("_id", None)
    return doc


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(
    conversation_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete a conversation."""
    deleted = await conversation_memory.delete_conversation(conversation_id, current_user["id"])
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"message": "Conversation deleted"}


# ═══════════════════════════════════════════════════════════════════
#  Document Models & Endpoints
# ═══════════════════════════════════════════════════════════════════

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

class DocumentListRequest(BaseModel):
    """Request model for listing documents."""
    source_types: Optional[List[str]] = Field(None, description="Source types to include")
    tags: Optional[List[str]] = Field(None, description="Tags to filter by")
    limit: int = Field(100, description="Maximum number of documents to return")
    offset: int = Field(0, description="Offset for pagination")

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
