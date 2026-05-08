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
import hashlib
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

from ....core.auth import get_current_user, get_current_user_sse
from ....core.config import settings
from ....services.context_service import context_service
from ....services.storage_service import storage_service
from ....services import rag_document_registry as rag_registry
from ....services.document_processor import document_processor
from ....agents.agent_service import AgentService
from ....services.activity_logger import log_activity
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
    agent_override: Optional[str] = Field(None, description="Explicitly route to this agent (skip intent router)")
    document_ids: Optional[List[str]] = Field(None, description="Specific document IDs to use as context (@-mentioned docs)")

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

        # Agent hint from /command — only override when auto-router is unsure
        if request.agent_override:
            if confidence < 0.7 or agent_key == "general":
                agent_key = request.agent_override
                confidence = max(confidence, 0.9)
        
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
        
        # Log activity (fire-and-forget via background task)
        background_tasks.add_task(
            log_activity,
            user_id=user_id,
            organization_id=getattr(current_user, "organization_id", user_id),
            activity_type="chat.message_sent",
            details={
                "query_preview": request.query[:100],
                "agent_used": agent_key,
                "route_confidence": confidence,
                "processing_time": round(processing_time, 2),
                "conversation_id": conversation_id,
            },
            related_resource_type="CONVERSATION",
            related_resource_id=conversation_id,
            agent_name=agent_key,
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
            # ── Step 3: Route intent (always runs the auto-router) ──
            intent_router = await get_router()
            route_result = await intent_router.route(
                message=request.query,
                conversation_history=history,
            )
            agent_key = route_result["agent"]
            confidence = route_result["confidence"]

            # If user explicitly tagged an agent via /, use it as a hint:
            # override only when auto-router has low confidence or picked "general"
            if request.agent_override:
                if confidence < 0.7 or agent_key == "general":
                    agent_key = request.agent_override
                    confidence = max(confidence, 0.9)

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

            # ── Step 5: Retrieve RAG context before streaming ──
            from ....ai_models.base import LLMConfig
            from ....services.context_service import context_service as _ctx_svc

            sources = []
            context_used = 0

            # Fetch relevant document chunks from vector store
            try:
                # If user @-mentioned specific documents, fetch their chunks first
                mentioned_chunks = []
                if request.document_ids:
                    for doc_id in request.document_ids:
                        doc_result = await _ctx_svc.get_context_for_query(
                            query=request.query,
                            user_id=user_id,
                            organization_id=current_user.get("organization_id"),
                            k=4,
                            filters={"document_id": doc_id},
                        )
                        mentioned_chunks.extend(doc_result.get("context", []))

                # Then get general RAG context (reduced k if we already have @-mentioned docs)
                general_k = max(2, 8 - len(mentioned_chunks))
                ctx_result = await _ctx_svc.get_context_for_query(
                    query=request.query,
                    user_id=user_id,
                    organization_id=current_user.get("organization_id"),
                    k=general_k,
                )
                general_chunks = ctx_result.get("context", [])

                # Merge: @-mentioned docs first, then general (deduplicated)
                seen_ids = set()
                context_chunks = []
                for chunk in mentioned_chunks + general_chunks:
                    chunk_key = (chunk.get("metadata", {}).get("document_id", ""), chunk.get("text", chunk.get("content", ""))[:80])
                    if chunk_key not in seen_ids:
                        seen_ids.add(chunk_key)
                        context_chunks.append(chunk)
                context_used = len(context_chunks)

                # Format context into numbered citations
                if context_chunks:
                    formatted_parts = []
                    for i, chunk in enumerate(context_chunks):
                        text = chunk.get("text", chunk.get("content", ""))
                        meta = chunk.get("metadata", {})
                        source_title = meta.get("title", meta.get("filename", "Document"))
                        formatted_parts.append(f"[{i+1}] {text}\n(Source: {source_title})")
                        sources.append({
                            "index": i + 1,
                            "title": source_title,
                            "type": meta.get("source", "upload"),
                            "document_id": meta.get("document_id", ""),
                            "page_number": meta.get("page_number"),
                            "bbox": meta.get("bbox"),
                            "page_width": meta.get("page_width"),
                            "page_height": meta.get("page_height"),
                            "chunk_text": text[:200] if text else "",
                        })
                    rag_context = "\n\n".join(formatted_parts)
                else:
                    rag_context = ""
            except Exception as ctx_err:
                logger.warning("rag_context_fetch_failed", error=str(ctx_err))
                rag_context = ""

            # ── Step 6: Stream with RAG context ──
            llm = getattr(agent, "llm_client", None) or getattr(agent, "perplexity_client", None)

            if llm and hasattr(llm, "stream"):
                # Build system prompt with RAG context
                base_system = getattr(agent, "system_prompt", None) or (
                    "You are Lumicoria.ai, a helpful AI assistant. "
                    "Answer the user's question accurately and helpfully."
                )

                if rag_context:
                    system_prompt = (
                        f"{base_system}\n\n"
                        "Use the following context from the user's documents to help answer their question. "
                        "When you use information from the context, cite the source using its number "
                        "in square brackets, e.g. [1], [2]. Place citations inline right after the relevant statement.\n\n"
                        f"{rag_context}"
                    )
                else:
                    system_prompt = base_system

                messages = [{"role": "system", "content": system_prompt}]
                if history:
                    for m in history[-8:]:
                        messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
                messages.append({"role": "user", "content": request.query})

                cfg = LLMConfig(temperature=0.7, max_tokens=None)
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
                sources = normalized.get("sources", sources)
                context_used = normalized.get("context_used", context_used)
                yield f"data: {json.dumps({'type': 'delta', 'text': full_response})}\n\n"

            # ── Step 7: Persist assistant message ──
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

            # ── Step 7b: Upsert conversation into the RAG registry ──
            # Runs as a background task so the stream finishes immediately.
            # Full conversation is fetched from MongoDB inside add_chat_context,
            # so we pass the latest turn only as a hint.
            if full_response:
                import asyncio as _asyncio
                _asyncio.create_task(_ctx_svc.add_chat_context(
                    messages=[
                        {"role": "user", "content": request.query},
                        {"role": "assistant", "content": full_response},
                    ],
                    user_id=user_id,
                    organization_id=current_user.get("organization_id"),
                    conversation_id=conversation_id,
                ))

            # ── Step 8: Done frame with sources ──
            processing_time = round(time.time() - start_time, 2)
            yield f"data: {json.dumps({'type': 'done', 'processing_time': processing_time, 'sources': sources, 'context_used': context_used})}\n\n"

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

# ── Background workers for RAG ingestion ───────────────────────────

async def _chunk_uploaded_file(
    file_bytes: bytes,
    document_id: str,
    user_id: str,
    organization_id: Optional[str],
    s3_key: str,
    stored_filename: str,
    original_filename: str,
    title: Optional[str],
    content_type: str,
    tags: List[str],
) -> None:
    """Background: write to temp file → chunk → Weaviate → update Postgres."""
    import tempfile
    tmp_dir = Path(tempfile.gettempdir()) / "rag_ingest"
    tmp_dir.mkdir(exist_ok=True, parents=True)
    tmp_path = tmp_dir / stored_filename

    try:
        tmp_path.write_bytes(file_bytes)

        metadata: Dict[str, Any] = {
            "document_id": document_id,
            "user_id": user_id,
            "source": "upload",
            "s3_key": s3_key,
            "filename": stored_filename,
            "original_filename": original_filename,
            "title": title or original_filename,
            "mime_type": content_type,
            "tags": tags,
            "created_at": datetime.utcnow().isoformat(),
        }
        if organization_id:
            metadata["organization_id"] = organization_id

        # Best-effort preview artifact (DOCX/PPTX/XLSX).
        try:
            from ....services.ingest.preview import render_preview, preview_artifact_key
            rendered = render_preview(str(tmp_path), content_type)
            if rendered is not None:
                artifact_bytes, artifact_ct = rendered
                artifact_key = preview_artifact_key(s3_key, content_type)
                if artifact_key:
                    await storage_service.upload_file(artifact_bytes, artifact_key, artifact_ct)
        except Exception as e:
            logger.warning("preview_artifact_failed", document_id=document_id, error=str(e))

        result = await document_processor.process_file(str(tmp_path), metadata)

        if result.status == "error":
            await rag_registry.update(document_id, status="error", error_message=result.error)
        else:
            await rag_registry.update(document_id, chunk_count=result.chunk_count, status="ready")

    except Exception as e:
        logger.error("chunk_uploaded_file_failed", error=str(e), document_id=document_id)
        await rag_registry.update(document_id, status="error", error_message=str(e))
    finally:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except Exception:
            pass


async def _process_url_document(
    document_id: str,
    url: str,
    user_id: str,
    organization_id: Optional[str],
    s3_key: str,
    stored_filename: str,
    title: Optional[str],
    tags: List[str],
) -> None:
    """Background: fetch URL → MinIO snapshot → chunk → Weaviate → update Postgres."""
    import httpx

    try:
        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            response = await client.get(
                url,
                headers={"User-Agent": "Mozilla/5.0 (compatible; Lumicoria.ai/1.0)"},
            )
            response.raise_for_status()
            content = response.text

        content_bytes = content.encode("utf-8")
        await storage_service.upload_file(content_bytes, s3_key, "text/html; charset=utf-8")

        await rag_registry.update(document_id, size_bytes=len(content_bytes))

        metadata: Dict[str, Any] = {
            "document_id": document_id,
            "user_id": user_id,
            "source": "web",
            "url": url,
            "source_url": url,
            "s3_key": s3_key,
            "filename": stored_filename,
            "title": title or url,
            "mime_type": "text/html",
            "tags": tags,
            "created_at": datetime.utcnow().isoformat(),
        }
        if organization_id:
            metadata["organization_id"] = organization_id

        # Strip HTML tags before chunking — otherwise <script>/<style>/markup
        # end up embedded verbatim in the vector store and tank retrieval quality.
        stripped = re.sub(r"<[^>]+>", " ", content)
        stripped = re.sub(r"\s+", " ", stripped).strip()

        result = await document_processor.process_text(stripped, metadata)

        if result.status == "error":
            await rag_registry.update(document_id, status="error", error_message=result.error)
        else:
            await rag_registry.update(document_id, chunk_count=result.chunk_count, status="ready")

    except Exception as e:
        logger.error("process_url_document_failed", error=str(e), document_id=document_id, url=url)
        await rag_registry.update(document_id, status="error", error_message=str(e))


async def _chunk_text_document(
    document_id: str,
    user_id: str,
    organization_id: Optional[str],
    s3_key: str,
    stored_filename: str,
    text: str,
    title: Optional[str],
    source: str,
    tags: List[str],
) -> None:
    """Background: chunk text → Weaviate → update Postgres."""
    try:
        metadata: Dict[str, Any] = {
            "document_id": document_id,
            "user_id": user_id,
            "source": source,
            "s3_key": s3_key,
            "filename": stored_filename,
            "title": title or f"Note {document_id[:8]}",
            "mime_type": "text/markdown",
            "tags": tags,
            "created_at": datetime.utcnow().isoformat(),
        }
        if organization_id:
            metadata["organization_id"] = organization_id

        result = await document_processor.process_text(text, metadata)

        if result.status == "error":
            await rag_registry.update(document_id, status="error", error_message=result.error)
        else:
            await rag_registry.update(document_id, chunk_count=result.chunk_count, status="ready")

    except Exception as e:
        logger.error("chunk_text_document_failed", error=str(e), document_id=document_id)
        await rag_registry.update(document_id, status="error", error_message=str(e))


# ── Document endpoints ─────────────────────────────────────────────

@router.post("/documents/upload", status_code=201)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    tags: Optional[str] = Form(None),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Upload a document → MinIO (+ R2) → Postgres registry → Weaviate chunks (background)."""
    user_id = current_user["id"]
    organization_id = current_user.get("organization_id")

    # Parse tags if provided
    parsed_tags: List[str] = []
    if tags:
        try:
            parsed_tags = json.loads(tags)
        except Exception:
            parsed_tags = [t.strip() for t in tags.split(",") if t.strip()]

    # Generate IDs / keys
    document_id = str(uuid.uuid4())
    original_filename = file.filename or "upload"
    file_ext = os.path.splitext(original_filename)[1]
    stored_filename = f"{document_id}{file_ext}"
    s3_key = f"rag/{user_id}/{stored_filename}"

    # Read bytes + content type
    file_bytes = await file.read()
    content_type = file.content_type or "application/octet-stream"

    # SHA256 dedup: if this user already ingested the same bytes, alias
    # rather than re-uploading to MinIO and re-chunking.
    content_sha256 = hashlib.sha256(file_bytes).hexdigest()
    dedup_enabled = getattr(settings, "INGEST_DOC_DEDUP_ENABLED", True)
    existing = None
    if dedup_enabled:
        try:
            existing = await rag_registry.find_by_content_sha256(user_id, content_sha256)
        except Exception as e:
            logger.warning("dedup_lookup_failed", error=str(e), user_id=user_id)

    if existing:
        try:
            await rag_registry.create(
                document_id=document_id,
                user_id=user_id,
                organization_id=organization_id,
                s3_key=existing["s3_key"],
                filename=existing.get("filename") or stored_filename,
                original_filename=original_filename,
                title=title or original_filename,
                mime_type=content_type,
                source="upload",
                size_bytes=len(file_bytes),
                tags=parsed_tags,
                status="ready",
                chunk_count=existing.get("chunk_count") or 0,
                content_sha256=content_sha256,
                aliased_document_id=existing["document_id"],
            )
        except Exception as e:
            logger.error("dedup_alias_create_failed", error=str(e), document_id=document_id)
            raise HTTPException(status_code=500, detail=f"Registry write failed: {e}")
        try:
            from ....services.ingest.metrics import record_dedup_hit
            record_dedup_hit("upload")
        except Exception:
            pass
        return {
            "message": "Document already ingested — aliased existing copy",
            "status": "ready",
            "document_id": document_id,
            "aliased_document_id": existing["document_id"],
            "filename": original_filename,
            "s3_key": existing["s3_key"],
            "deduplicated": True,
        }

    # Upload to MinIO/R2 synchronously so preview is available immediately
    try:
        await storage_service.upload_file(file_bytes, s3_key, content_type)
    except Exception as e:
        logger.error("minio_upload_failed", error=str(e), key=s3_key)
        raise HTTPException(status_code=500, detail=f"Storage upload failed: {e}")

    # Create the authoritative Postgres row
    try:
        await rag_registry.create(
            document_id=document_id,
            user_id=user_id,
            organization_id=organization_id,
            s3_key=s3_key,
            filename=stored_filename,
            original_filename=original_filename,
            title=title or original_filename,
            mime_type=content_type,
            source="upload",
            size_bytes=len(file_bytes),
            tags=parsed_tags,
            status="processing",
            content_sha256=content_sha256,
        )
    except Exception as e:
        # Best-effort rollback of the MinIO object to avoid orphans
        try:
            await storage_service.delete_file(s3_key)
        except Exception:
            pass
        logger.error("rag_registry_create_failed", error=str(e), document_id=document_id)
        raise HTTPException(status_code=500, detail=f"Registry write failed: {e}")

    # Chunk + embed — Celery when a worker is running, BackgroundTasks otherwise
    if getattr(settings, "CELERY_ENABLED", False):
        from ....tasks.document_tasks import ingest_file as _ingest_file_task
        _ingest_file_task.delay(
            document_id=document_id,
            user_id=user_id,
            s3_key=s3_key,
            stored_filename=stored_filename,
            original_filename=original_filename,
            content_type=content_type,
            organization_id=organization_id,
            title=title or original_filename,
            tags=parsed_tags,
        )
    else:
        background_tasks.add_task(
            _chunk_uploaded_file,
            file_bytes=file_bytes,
            document_id=document_id,
            user_id=user_id,
            organization_id=organization_id,
            s3_key=s3_key,
            stored_filename=stored_filename,
            original_filename=original_filename,
            title=title or original_filename,
            content_type=content_type,
            tags=parsed_tags,
        )

    await log_activity(
        user_id=user_id,
        organization_id=organization_id or user_id,
        activity_type="chat.document_uploaded",
        details={
            "filename": original_filename,
            "title": title or original_filename,
            "document_id": document_id,
        },
        related_resource_type="DOCUMENT",
        agent_name="Document Agent",
    )

    return {
        "message": "Document uploaded and processing started",
        "status": "processing",
        "document_id": document_id,
        "filename": original_filename,
        "s3_key": s3_key,
    }

@router.post("/documents/url", status_code=201)
async def add_document_url(
    document: DocumentUrlRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Add a URL → fetch + MinIO snapshot → Postgres + Weaviate (background)."""
    user_id = current_user["id"]
    organization_id = current_user.get("organization_id")

    document_id = str(uuid.uuid4())
    stored_filename = f"{document_id}.html"
    s3_key = f"rag/{user_id}/{stored_filename}"

    # URL dedup key: hash the normalized URL.  Two users ingesting the same
    # URL each get their own canonical copy; a single user re-adding the
    # same URL aliases.
    url_key = document.url.strip()
    content_sha256 = hashlib.sha256(url_key.encode("utf-8")).hexdigest()
    dedup_enabled = getattr(settings, "INGEST_DOC_DEDUP_ENABLED", True)
    existing = None
    if dedup_enabled:
        try:
            existing = await rag_registry.find_by_content_sha256(user_id, content_sha256)
        except Exception as e:
            logger.warning("dedup_lookup_failed", error=str(e), user_id=user_id)

    if existing:
        try:
            await rag_registry.create(
                document_id=document_id,
                user_id=user_id,
                organization_id=organization_id,
                s3_key=existing["s3_key"],
                filename=existing.get("filename") or stored_filename,
                original_filename=None,
                title=document.title or document.url,
                mime_type="text/html",
                source="web",
                source_url=document.url,
                size_bytes=existing.get("size_bytes") or 0,
                tags=document.tags or [],
                status="ready",
                chunk_count=existing.get("chunk_count") or 0,
                content_sha256=content_sha256,
                aliased_document_id=existing["document_id"],
            )
        except Exception as e:
            logger.error("dedup_alias_create_failed", error=str(e), document_id=document_id)
            raise HTTPException(status_code=500, detail=f"Registry write failed: {e}")
        try:
            from ....services.ingest.metrics import record_dedup_hit
            record_dedup_hit("web")
        except Exception:
            pass
        return {
            "message": "URL already ingested — aliased existing copy",
            "status": "ready",
            "document_id": document_id,
            "aliased_document_id": existing["document_id"],
            "url": document.url,
            "deduplicated": True,
        }

    # Create a placeholder Postgres row so the doc shows up in the UI immediately
    try:
        await rag_registry.create(
            document_id=document_id,
            user_id=user_id,
            organization_id=organization_id,
            s3_key=s3_key,
            filename=stored_filename,
            original_filename=None,
            title=document.title or document.url,
            mime_type="text/html",
            source="web",
            source_url=document.url,
            size_bytes=0,
            tags=document.tags or [],
            status="processing",
            content_sha256=content_sha256,
        )
    except Exception as e:
        logger.error("rag_registry_create_failed", error=str(e), document_id=document_id)
        raise HTTPException(status_code=500, detail=f"Registry write failed: {e}")

    if getattr(settings, "CELERY_ENABLED", False):
        from ....tasks.document_tasks import ingest_url as _ingest_url_task
        _ingest_url_task.delay(
            document_id=document_id,
            user_id=user_id,
            url=document.url,
            s3_key=s3_key,
            stored_filename=stored_filename,
            organization_id=organization_id,
            title=document.title,
            tags=document.tags or [],
        )
    else:
        background_tasks.add_task(
            _process_url_document,
            document_id=document_id,
            url=document.url,
            user_id=user_id,
            organization_id=organization_id,
            s3_key=s3_key,
            stored_filename=stored_filename,
            title=document.title,
            tags=document.tags or [],
        )

    return {
        "message": "URL processing started",
        "status": "processing",
        "document_id": document_id,
        "url": document.url,
    }

@router.post("/documents/text", status_code=201)
async def add_document_text(
    document: DocumentTextRequest,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Add a note/text → MinIO (.md) → Postgres + Weaviate (background)."""
    user_id = current_user["id"]
    organization_id = current_user.get("organization_id")

    document_id = str(uuid.uuid4())
    stored_filename = f"{document_id}.md"
    s3_key = f"rag/{user_id}/{stored_filename}"

    text_bytes = document.text.encode("utf-8")
    content_sha256 = hashlib.sha256(text_bytes).hexdigest()
    dedup_enabled = getattr(settings, "INGEST_DOC_DEDUP_ENABLED", True)
    existing = None
    if dedup_enabled:
        try:
            existing = await rag_registry.find_by_content_sha256(user_id, content_sha256)
        except Exception as e:
            logger.warning("dedup_lookup_failed", error=str(e), user_id=user_id)

    if existing:
        try:
            await rag_registry.create(
                document_id=document_id,
                user_id=user_id,
                organization_id=organization_id,
                s3_key=existing["s3_key"],
                filename=existing.get("filename") or stored_filename,
                original_filename=None,
                title=document.title or f"Note {document_id[:8]}",
                mime_type="text/markdown",
                source=document.source or "manual_entry",
                size_bytes=len(text_bytes),
                tags=document.tags or [],
                status="ready",
                chunk_count=existing.get("chunk_count") or 0,
                content_sha256=content_sha256,
                aliased_document_id=existing["document_id"],
            )
        except Exception as e:
            logger.error("dedup_alias_create_failed", error=str(e), document_id=document_id)
            raise HTTPException(status_code=500, detail=f"Registry write failed: {e}")
        try:
            from ....services.ingest.metrics import record_dedup_hit
            record_dedup_hit(document.source or "manual_entry")
        except Exception:
            pass
        return {
            "message": "Note already ingested — aliased existing copy",
            "status": "ready",
            "document_id": document_id,
            "aliased_document_id": existing["document_id"],
            "deduplicated": True,
        }

    try:
        await storage_service.upload_file(text_bytes, s3_key, "text/markdown; charset=utf-8")
    except Exception as e:
        logger.error("minio_upload_failed", error=str(e), key=s3_key)
        raise HTTPException(status_code=500, detail=f"Storage upload failed: {e}")

    try:
        await rag_registry.create(
            document_id=document_id,
            user_id=user_id,
            organization_id=organization_id,
            s3_key=s3_key,
            filename=stored_filename,
            original_filename=None,
            title=document.title or f"Note {document_id[:8]}",
            mime_type="text/markdown",
            source=document.source or "manual_entry",
            size_bytes=len(text_bytes),
            tags=document.tags or [],
            status="processing",
            content_sha256=content_sha256,
        )
    except Exception as e:
        try:
            await storage_service.delete_file(s3_key)
        except Exception:
            pass
        logger.error("rag_registry_create_failed", error=str(e), document_id=document_id)
        raise HTTPException(status_code=500, detail=f"Registry write failed: {e}")

    if getattr(settings, "CELERY_ENABLED", False):
        from ....tasks.document_tasks import ingest_text as _ingest_text_task
        _ingest_text_task.delay(
            document_id=document_id,
            user_id=user_id,
            text=document.text,
            s3_key=s3_key,
            stored_filename=stored_filename,
            source=document.source or "manual_entry",
            organization_id=organization_id,
            title=document.title,
            tags=document.tags or [],
        )
    else:
        background_tasks.add_task(
            _chunk_text_document,
            document_id=document_id,
            user_id=user_id,
            organization_id=organization_id,
            s3_key=s3_key,
            stored_filename=stored_filename,
            text=document.text,
            title=document.title,
            source=document.source or "manual_entry",
            tags=document.tags or [],
        )

    return {
        "message": "Text processing started",
        "status": "processing",
        "document_id": document_id,
    }

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
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """List documents for the current user — reads from the Postgres registry."""
    user_id = current_user["id"]
    organization_id = current_user.get("organization_id")

    logger.info("list_documents_request", user_id=user_id, organization_id=organization_id,
                source_types=request.source_types)

    # Self-healing backfill for legacy chat_history conversations indexed
    # before the Postgres registry existed.  Runs only when the user is
    # actually looking at the Chat History tab (or an unfiltered list).
    wants_chat = (not request.source_types) or ("chat_history" in request.source_types)
    if wants_chat:
        try:
            backfilled = await rag_registry.backfill_chat_history_from_weaviate(user_id)
            if backfilled:
                logger.info("chat_history backfilled", user_id=user_id, rows=backfilled)
        except Exception as e:
            logger.warning("chat_history backfill skipped", error=str(e))

    result = await rag_registry.list_documents(
        user_id=user_id,
        organization_id=organization_id,
        source_types=request.source_types,
        tags=request.tags,
        limit=request.limit,
        offset=request.offset,
    )

    logger.info("list_documents_result", doc_count=len(result.get("documents", [])),
                total=result.get("total", 0))

    return result


@router.get("/documents/{document_id}/progress")
async def stream_document_progress(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user_sse),
):
    """Server-sent events stream for ingest progress.

    Emits JSON events keyed by the Celery / background worker pipeline:
      - {'stage':'parsing', 'processed':N, 'total':M}
      - {'stage':'chunking', 'chunks':N}
      - {'stage':'embedding', 'processed':N, 'total':M}
      - {'stage':'storing', 'processed':N, 'total':M}
      - {'stage':'ready', 'chunk_count':N}                          terminal
      - {'stage':'error', 'error_code':'...', 'message':'...'}      terminal
      - {'stage':'cancelled'}                                       terminal
      - {'stage':'heartbeat'}                                       keep-alive
    """
    from ....services.ingest.progress import subscribe as _subscribe

    user_id = current_user["id"]
    doc = await rag_registry.get(document_id, user_id=user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    async def event_stream() -> AsyncGenerator[str, None]:
        try:
            async for event in _subscribe(document_id):
                yield f"data: {json.dumps(event)}\n\n"
        except Exception as e:
            logger.error("progress_stream_error", error=str(e), document_id=document_id)
            yield f"data: {json.dumps({'stage':'error','message':str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


_CODE_EXTS = {
    ".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs", ".rb", ".php",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".cs", ".swift", ".kt", ".scala",
    ".sh", ".bash", ".zsh", ".sql", ".yaml", ".yml", ".json", ".toml", ".xml",
}


_OFFICE_HTML_MIMES = {
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    "application/msword",
    "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}
_XLSX_MIMES = {
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    "application/vnd.ms-excel",
}


def _pick_preview_type(mime: str, filename: str) -> str:
    mime = (mime or "").lower()
    ext = os.path.splitext(filename or "")[1].lower()
    if mime == "application/pdf":
        return "pdf"
    if mime.startswith("image/"):
        return "image"
    if mime in _XLSX_MIMES:
        return "xlsx"
    if mime in _OFFICE_HTML_MIMES:
        return "html"
    if mime == "text/html" or mime == "application/xhtml+xml":
        return "html"
    if mime in {"text/markdown", "text/x-markdown"} or ext in {".md", ".markdown"}:
        return "markdown"
    if ext in _CODE_EXTS or mime.startswith("text/x-"):
        return "code"
    if mime == "text/plain":
        return "text"
    return "download"


@router.post("/documents/{document_id}/regenerate-preview")
async def regenerate_document_preview(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Re-render the office-format preview artifact for an existing doc.

    Useful when the artifact was missing (e.g. mammoth/pptx/openpyxl not
    installed at ingest time) or corrupt.  Downloads the original from
    MinIO, runs render_preview(), uploads `{s3_key}.preview.html|json`.
    """
    import tempfile
    from ....services.ingest.preview import (
        preview_artifact_key, preview_kind, render_preview,
    )

    user_id = current_user["id"]
    doc = await rag_registry.get(document_id, user_id=user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    source_key = doc["s3_key"]
    mime = doc.get("mime_type") or ""

    kind = preview_kind(mime)
    if kind is None:
        raise HTTPException(
            status_code=400,
            detail=f"No preview-artifact rendering for mime type {mime!r}",
        )

    # Pull the original into a tempfile so the (synchronous) preview
    # libraries can read it from disk.
    try:
        content = await storage_service.download_file(source_key)
    except Exception as e:
        logger.error("regenerate_preview_download_failed",
                     document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Download failed: {e}")

    suffix = os.path.splitext(doc.get("original_filename") or doc.get("filename") or "")[1]
    tmp_dir = Path(tempfile.gettempdir()) / "rag_preview"
    tmp_dir.mkdir(exist_ok=True, parents=True)
    tmp_path = tmp_dir / f"{document_id}{suffix}"

    try:
        tmp_path.write_bytes(content)
        rendered = render_preview(str(tmp_path), mime)
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    if rendered is None:
        raise HTTPException(
            status_code=500,
            detail=(
                "Preview rendering returned no output — the required library "
                f"(mammoth / python-pptx / openpyxl) for {kind} may be missing."
            ),
        )

    artifact_bytes, artifact_ct = rendered
    artifact_key = preview_artifact_key(source_key, mime)
    if not artifact_key:
        raise HTTPException(status_code=500, detail="Could not compute artifact key")

    try:
        await storage_service.upload_file(artifact_bytes, artifact_key, artifact_ct)
    except Exception as e:
        logger.error("regenerate_preview_upload_failed",
                     document_id=document_id, error=str(e))
        raise HTTPException(status_code=500, detail=f"Upload failed: {e}")

    logger.info("preview_artifact_regenerated",
                document_id=document_id, kind=kind,
                artifact_key=artifact_key, bytes=len(artifact_bytes))

    return {
        "document_id": document_id,
        "artifact_key": artifact_key,
        "kind": kind,
        "bytes": len(artifact_bytes),
        "status": "ready",
    }


@router.post("/documents/{document_id}/cancel")
async def cancel_document_ingest(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Mark an in-progress ingest as cancelled.  Celery tasks check status
    between stages and exit early when they see this."""
    user_id = current_user["id"]

    doc = await rag_registry.get(document_id, user_id=user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    if doc["status"] in {"ready", "error", "cancelled"}:
        return {"status": doc["status"], "message": "Already terminal", "document_id": document_id}

    await rag_registry.update(document_id, status="cancelled",
                               error_message="cancelled_by_user")
    try:
        from ....services.ingest.progress import stage as _stage
        _stage(document_id, "cancelled", message="cancelled_by_user")
    except Exception:
        pass
    return {"status": "cancelled", "document_id": document_id}


@router.get("/documents/{document_id}/preview")
async def get_document_preview(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Return a discriminated union describing how the frontend should render
    this document. Shapes:
        {type: "pdf"|"image"|"html"|"xlsx", url, ...}
        {type: "markdown"|"text"|"code", data, language?, ...}
        {type: "download", url, ...}   (fallback)
    """
    user_id = current_user["id"]

    doc = await rag_registry.get(document_id, user_id=user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Dedup aliases resolve to the canonical copy's storage.
    source_key = doc["s3_key"]
    canonical_id = doc.get("aliased_document_id") or document_id

    mime = doc.get("mime_type") or "application/octet-stream"
    filename = doc.get("original_filename") or doc.get("filename") or ""
    ptype = _pick_preview_type(mime, filename)

    base: Dict[str, Any] = {
        "document_id": document_id,
        "canonical_document_id": canonical_id,
        "type": ptype,
        "mime_type": mime,
        "title": doc.get("title"),
        "filename": filename,
        "source_url": doc.get("source_url"),
    }

    # Inline-data types: fetch and decode the stored bytes once.
    if ptype in {"markdown", "text", "code"}:
        try:
            raw = await storage_service.download_file(source_key)
            data = raw.decode("utf-8", errors="replace")
        except Exception as e:
            logger.warning("preview_fetch_failed", error=str(e), document_id=document_id)
            data = ""
        base["data"] = data
        if ptype == "code":
            ext = os.path.splitext(filename)[1].lower().lstrip(".")
            base["language"] = ext or "text"
        return base

    # Office formats have a pre-rendered preview artifact stored next to
    # the original (see backend/services/ingest/preview.py).  Prefer that.
    artifact_key: Optional[str] = None
    if mime in _OFFICE_HTML_MIMES:
        artifact_key = f"{source_key}.preview.html"
    elif mime in _XLSX_MIMES:
        artifact_key = f"{source_key}.preview.json"

    if artifact_key:
        try:
            if await storage_service.file_exists(artifact_key):
                if ptype == "xlsx":
                    # Inline the JSON so the frontend doesn't need a second
                    # cross-origin fetch.
                    try:
                        raw = await storage_service.download_file(artifact_key)
                        base["data"] = json.loads(raw.decode("utf-8"))
                    except Exception as e:
                        logger.warning("xlsx_artifact_fetch_failed", error=str(e))
                        base["url"] = await storage_service.get_presigned_url(artifact_key)
                    return base
                base["url"] = await storage_service.get_presigned_url(artifact_key)
                return base
        except Exception as e:
            logger.warning("preview_artifact_lookup_failed", error=str(e),
                           document_id=document_id)
            # Fall through to the download fallback below.

        # Office format without a rendered artifact (e.g. mammoth /
        # python-pptx / openpyxl not installed, or rendering failed).
        # Don't point the iframe at the raw .docx/.pptx/.xlsx — the
        # browser can't render office binaries and it only produces a
        # confusing blank iframe.  Serve a download link instead.
        try:
            url = await storage_service.get_presigned_url(source_key)
        except Exception as e:
            logger.error("preview_url_failed", error=str(e), document_id=document_id)
            raise HTTPException(status_code=500, detail=f"Preview URL failed: {e}")
        base["type"] = "download"
        base["url"] = url
        return base

    # URL-rendered types, fallback for unsupported formats.
    try:
        url = await storage_service.get_presigned_url(source_key)
    except Exception as e:
        logger.error("preview_url_failed", error=str(e), document_id=document_id)
        raise HTTPException(status_code=500, detail=f"Preview URL failed: {e}")
    base["url"] = url
    return base


@router.get("/documents/{document_id}/presigned-url")
async def get_document_presigned_url(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Return a time-limited signed URL for the original file stored in MinIO."""
    user_id = current_user["id"]

    doc = await rag_registry.get(document_id, user_id=user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        url = await storage_service.get_presigned_url(doc["s3_key"])
    except Exception as e:
        logger.error("presigned_url_failed", error=str(e), document_id=document_id)
        raise HTTPException(status_code=500, detail=f"Failed to generate URL: {e}")

    return {
        "url": url,
        "document_id": document_id,
        "filename": doc["filename"],
        "original_filename": doc.get("original_filename"),
        "mime_type": doc["mime_type"],
        "title": doc["title"],
        "source": doc["source"],
        "source_url": doc.get("source_url"),
    }


@router.get("/documents/{document_id}/content")
async def get_document_content(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Return the extracted text content (concatenated chunks from Weaviate).

    Used as a preview fallback for binary formats (docx, xlsx, pptx, …) that
    can't be displayed in-browser directly.
    """
    user_id = current_user["id"]

    doc = await rag_registry.get(document_id, user_id=user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        from ....db.vector_stores import get_vector_store
        vector_store = get_vector_store()
        chunks = await vector_store.get_documents(
            filters={"user_id": user_id, "document_id": document_id},
            limit=500,
        )
    except Exception as e:
        logger.error("content_fetch_failed", error=str(e), document_id=document_id)
        raise HTTPException(status_code=500, detail=f"Failed to fetch content: {e}")

    # Preserve chunk ordering when available
    def _chunk_sort_key(c: Dict[str, Any]) -> Any:
        meta = c.get("metadata", {}) or {}
        return (
            meta.get("page_number", 0) or 0,
            meta.get("block_index", 0) or 0,
            meta.get("chunk_id", 0) or 0,
        )

    chunks = sorted(chunks, key=_chunk_sort_key)
    content = "\n\n".join((c.get("content") or "").strip() for c in chunks if c.get("content"))

    return {
        "document_id": document_id,
        "title": doc["title"],
        "mime_type": doc["mime_type"],
        "source": doc["source"],
        "chunk_count": len(chunks),
        "content": content,
    }


@router.delete("/documents/{document_id}")
async def delete_document(
    document_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """Delete a document from Postgres + Weaviate + MinIO."""
    user_id = current_user["id"]

    s3_key = await rag_registry.soft_delete(document_id, user_id=user_id)
    if s3_key is None:
        raise HTTPException(status_code=404, detail="Document not found")

    # Best-effort: delete chunks from Weaviate
    try:
        await context_service.delete_user_context(user_id=user_id, document_id=document_id)
    except Exception as e:
        logger.warning("weaviate_delete_failed", error=str(e), document_id=document_id)

    # Best-effort: delete the object from MinIO + R2
    try:
        await storage_service.delete_file(s3_key)
    except Exception as e:
        logger.warning("minio_delete_failed", error=str(e), key=s3_key)

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
