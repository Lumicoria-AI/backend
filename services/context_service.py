"""
Context Service for Lumicoria.ai RAG System

This service manages the retrieval and formatting of context from various sources,
including the vector database, recent conversations, and user-specific data.
"""

import asyncio
from typing import Dict, Any, List, Optional, Union, Tuple
import structlog
from datetime import datetime, timedelta
import json

from ..db.vector_stores import get_vector_store
from ..services.document_processor import document_processor
from ..services import rag_document_registry as rag_registry
from ..services.storage_service import storage_service
from ..core.config import settings
from ..ai_models import get_embedding_client, LLMClient

logger = structlog.get_logger(__name__)

class ContextService:
    """
    Service for retrieving and managing context for RAG from multiple sources.
    """
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize the context service."""
        self.llm_client = llm_client
        
    async def initialize(self):
        """Ensure client is initialized."""
        if not self.llm_client:
            self.llm_client = get_embedding_client()
            
    async def get_context_for_query(
        self,
        query: str,
        user_id: str,
        organization_id: Optional[str] = None,
        k: int = 8,
        filters: Optional[Dict[str, Any]] = None,
        include_sources: Optional[List[str]] = None,
        *,
        token_budget: Optional[int] = None,
        recency_half_life_days: float = 7.0,
        rerank: Optional[bool] = None,
        diversity: bool = True,
        hybrid_alpha: float = 0.5,
    ) -> Dict[str, Any]:
        """Retrieve grounded context for ``query`` from the user's vector store.

        The retrieval pipeline (all in this one method):

          1. Generate the query embedding once.
          2. **Hybrid search**: Weaviate's native vector + BM25 fusion when
             the store supports it; pure vector otherwise. ``hybrid_alpha``
             0=BM25-only, 1=vector-only, 0.5=even mix (default).
          3. Filter by ``user_id`` (Weaviate) + post-filter by
             ``organization_id`` (so personal docs without an org still
             show up to their owner).
          4. **Recency boost**: ``score *= 0.5 ** (age_days / half_life)``.
             Newer chunks rank higher without crowding out high-relevance
             old ones.
          5. **Source diversity**: at most 2 chunks per document_id, 3 per
             source. Stops the top-k from being 8 chunks of the same email
             thread.
          6. **Optional rerank**: cross-encoder rerank if
             ``settings.CONTEXT_RERANK_ENABLED`` (or per-call ``rerank``)
             and a rerank provider is configured. Behind a flag because
             it's a per-call cost.
          7. **Token-budget trim**: if ``token_budget`` is set, pack chunks
             until the budget is hit (rough 4 chars / token).
          8. **Provenance**: response carries ``sources`` listing every
             chunk's document_id, source, chunk_id, score, score_components.

        Args:
            query: The natural-language query.
            user_id: Owner filter (always applied).
            organization_id: Optional org filter (post-filtered).
            k: Target number of context chunks to return.
            filters: Extra metadata filters to AND into the search.
            include_sources: Restrict to specific source types
                (e.g. ["upload", "gmail", "drive", "chat_history"]).
            token_budget: If set, trim chunks until total tokens ≤ budget.
            recency_half_life_days: Newer chunks get a boost; 7 means a
                chunk's effective score halves every week.
            rerank: Force rerank on/off. ``None`` = follow settings flag.
            diversity: Apply source diversity. Set False for "give me
                everything from this one doc" queries.
            hybrid_alpha: Weight between BM25 (0) and vector (1).

        Returns:
            ``{"context": [...], "sources": [...], "query": str,
               "timestamp": iso, "stats": {...}}`` — additive return
            shape; old callers reading only ``context`` keep working.
        """
        await self.initialize()

        # 1. Embed the query.
        query_embedding = await self.llm_client.generate_embeddings(texts=[query])
        if not query_embedding or len(query_embedding) == 0:
            logger.error("Failed to generate query embedding")
            return {"context": [], "sources": [], "error": "Failed to generate query embedding"}

        # 2. Build filters. Weaviate filters at index time; org is
        #    post-filtered to admit pre-org personal docs.
        search_filters: Dict[str, Any] = dict(filters or {})
        search_filters["user_id"] = user_id
        if include_sources:
            search_filters["source"] = include_sources

        if not settings.db.VECTOR_STORE_ENABLED:
            return {"context": [], "sources": [], "error": "Vector store disabled"}

        # Overshoot k by 4× so diversity + org filter + recency reordering
        # still produce a full k at the end. Cap at 64 to keep latency sane.
        fetch_k = min(64, k * 4)

        try:
            vector_store = get_vector_store()

            # 3. Hybrid if the adapter supports it; vector-only otherwise.
            results: List[Dict[str, Any]] = []
            if hasattr(vector_store, "similarity_search_hybrid"):
                results = await vector_store.similarity_search_hybrid(
                    query=query,
                    query_vector=query_embedding[0],
                    k=fetch_k,
                    filters=search_filters,
                    alpha=hybrid_alpha,
                )
            if not results:
                # Either hybrid isn't available or it returned nothing.
                results = await vector_store.similarity_search(
                    query_vector=query_embedding[0],
                    k=fetch_k,
                    filters=search_filters,
                )

            # 4. Post-filter by org.
            if organization_id:
                results = [
                    r for r in results
                    if not r.get("metadata", {}).get("organization_id")
                    or r["metadata"]["organization_id"] == organization_id
                ]

            stats: Dict[str, Any] = {
                "raw_hits": len(results),
                "fetch_k": fetch_k,
                "hybrid": hasattr(vector_store, "similarity_search_hybrid"),
            }

            # 5. Apply recency boost — operates in place, returns sorted.
            results = self._apply_recency_boost(results, recency_half_life_days)

            # 6. Source diversity.
            if diversity:
                results = self._apply_diversity(
                    results, max_per_doc=2, max_per_source=3,
                )
                stats["after_diversity"] = len(results)

            # 7. Trim to k before the (expensive) rerank.
            results = results[:k]

            # 8. Optional rerank.
            rerank_enabled = rerank if rerank is not None else getattr(
                settings, "CONTEXT_RERANK_ENABLED", False,
            )
            if rerank_enabled:
                results = await self._rerank(query, results)
                stats["reranked"] = True

            # 9. Token-budget trimming.
            if token_budget:
                results = self._trim_to_token_budget(results, token_budget)
                stats["token_budget"] = token_budget

            # 10. Format + provenance.
            formatted_context = self._format_context(results)
            sources = self._build_sources_provenance(results)

            stats["returned"] = len(formatted_context)

            return {
                "context": formatted_context,
                "sources": sources,
                "query": query,
                "timestamp": datetime.utcnow().isoformat(),
                "stats": stats,
            }

        except Exception as e:
            logger.error("Error retrieving context", error=str(e))
            return {"context": [], "sources": [], "error": str(e)}

    # ─────────────────────────────────────────────────────────────────
    # Retrieval pipeline helpers
    # ─────────────────────────────────────────────────────────────────

    def _apply_recency_boost(
        self,
        results: List[Dict[str, Any]],
        half_life_days: float,
    ) -> List[Dict[str, Any]]:
        """Multiply each result's score by an exponential recency decay.

        ``half_life_days`` controls how aggressive the boost is — 7 means
        a chunk's effective score halves every week. Chunks with no
        ``created_at`` get a neutral factor of 1.0. The original score
        is preserved in ``score_components["raw_score"]`` so callers can
        debug ranking changes.
        """
        if not results or half_life_days <= 0:
            return results

        now = datetime.utcnow()
        decay_lambda = 0.6931471805599453 / half_life_days  # ln(2) / half_life

        boosted = []
        for r in results:
            md = r.get("metadata") or {}
            raw_score = float(r.get("score") or 0.0)
            created_at = md.get("created_at")

            recency_factor = 1.0
            if created_at:
                try:
                    if isinstance(created_at, str):
                        ts = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
                    else:
                        ts = created_at
                    # Make naive so subtraction works.
                    if ts.tzinfo is not None:
                        ts = ts.replace(tzinfo=None)
                    age_days = max(0.0, (now - ts).total_seconds() / 86400.0)
                    import math
                    recency_factor = math.exp(-decay_lambda * age_days)
                except Exception:
                    recency_factor = 1.0

            new_score = raw_score * recency_factor
            components = dict(r.get("score_components") or {})
            components["raw_score"] = raw_score
            components["recency_factor"] = round(recency_factor, 4)

            boosted.append({
                **r,
                "score": new_score,
                "score_components": components,
            })

        boosted.sort(key=lambda r: r["score"], reverse=True)
        return boosted

    def _apply_diversity(
        self,
        results: List[Dict[str, Any]],
        *,
        max_per_doc: int = 2,
        max_per_source: int = 3,
    ) -> List[Dict[str, Any]]:
        """Cap chunks-per-document and chunks-per-source so the top-k
        doesn't collapse onto one email thread or one PDF."""
        kept: List[Dict[str, Any]] = []
        seen_per_doc: Dict[str, int] = {}
        seen_per_source: Dict[str, int] = {}

        for r in results:
            md = r.get("metadata") or {}
            doc_id = str(md.get("document_id") or "")
            source = str(md.get("source") or "unknown")

            if doc_id and seen_per_doc.get(doc_id, 0) >= max_per_doc:
                continue
            if seen_per_source.get(source, 0) >= max_per_source:
                continue

            kept.append(r)
            if doc_id:
                seen_per_doc[doc_id] = seen_per_doc.get(doc_id, 0) + 1
            seen_per_source[source] = seen_per_source.get(source, 0) + 1

        return kept

    def _trim_to_token_budget(
        self,
        results: List[Dict[str, Any]],
        token_budget: int,
    ) -> List[Dict[str, Any]]:
        """Pack chunks until the cumulative token estimate hits the
        budget. Uses a 4-chars-per-token approximation — fast and
        accurate enough for budgeting; precise tokenisation is a Phase 8
        upgrade once we lock the embedding/inference model mix."""
        if token_budget <= 0:
            return results

        kept: List[Dict[str, Any]] = []
        used = 0
        for r in results:
            content = r.get("content") or r.get("text") or ""
            tokens = max(1, len(content) // 4)
            if used + tokens > token_budget and kept:
                break
            kept.append(r)
            used += tokens
        return kept

    async def _rerank(
        self,
        query: str,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Cross-encoder rerank. Off unless a provider key is set —
        wired here so the pipeline shape doesn't change; the actual
        provider call goes in once we pick between Voyage / Cohere /
        local cross-encoder. For now we no-op and tag the results so
        downstream telemetry can tell rerank was *attempted*."""
        # Provider selection happens in Phase 8 (eval-driven).
        # Returning the input unchanged keeps the API stable.
        for r in results:
            components = dict(r.get("score_components") or {})
            components["rerank_attempted"] = True
            r["score_components"] = components
        return results

    def _build_sources_provenance(
        self,
        results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """One row per returned chunk — what powers the "Sources" footer
        in agent proposals and the digest email."""
        out: List[Dict[str, Any]] = []
        for r in results:
            md = r.get("metadata") or {}
            out.append({
                "document_id": md.get("document_id"),
                "source": md.get("source"),
                "chunk_id": md.get("chunk_id"),
                "title": md.get("title") or md.get("filename"),
                "score": r.get("score"),
                "score_components": r.get("score_components") or {},
            })
        return out
    
    async def add_chat_context(
        self,
        messages: List[Dict[str, Any]],
        user_id: str,
        organization_id: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Store a conversation as a RAG document so it's searchable alongside
        uploaded files, URLs, and manual notes.

        Flow (Option A — upsert by conversation_id):
          1. Fetch the FULL conversation from MongoDB (not just current turn).
          2. Render as a markdown transcript.
          3. Upload transcript to MinIO (+ R2) at rag/{user_id}/chat_{id}.md.
          4. Delete any existing Weaviate chunks for this document_id.
          5. Upsert the Postgres registry row so the chat shows up on the
             `/documents` listing with source="chat_history".
          6. Re-chunk + re-embed the transcript into Weaviate.

        Args:
            messages: Most recent turn(s) — kept for backward-compat but
                      ignored when conversation_id lets us fetch the full log.
            user_id: Owner of the chat.
            organization_id: Optional org.
            conversation_id: Required for upsert; if missing we fall back to
                             a one-shot text ingestion.
        """
        # No conversation_id → legacy one-shot path (no registry, no upsert)
        if not conversation_id:
            metadata = {
                "user_id": user_id,
                "source": "chat_history",
                "created_at": datetime.utcnow().isoformat(),
            }
            if organization_id:
                metadata["organization_id"] = organization_id
            result = await document_processor.process_chat_history(
                messages=messages, metadata=metadata
            )
            return {
                "document_id": result.document_id,
                "status": result.status,
                "chunk_count": result.chunk_count,
                "error": result.error,
            }

        # Lazy import to avoid circular dependency (memory → services → context)
        from ..agents import memory as chat_memory

        document_id = f"chat_{conversation_id}"

        # 1. Fetch the full conversation from MongoDB
        conversation = await chat_memory.get_full_conversation(conversation_id)
        full_messages = (
            conversation.get("messages", []) if conversation else messages
        )

        if not full_messages:
            return {
                "document_id": document_id,
                "status": "skipped",
                "chunk_count": 0,
                "error": "no messages to index",
            }

        # 2. Build markdown transcript + derive title from first user msg
        first_user_msg = next(
            (m.get("content", "") for m in full_messages if m.get("role") == "user"),
            "Conversation",
        )
        turn_count = sum(1 for m in full_messages if m.get("role") == "user")
        short_title = (first_user_msg[:80] + "…") if len(first_user_msg) > 80 else first_user_msg
        title = f"{short_title} ({turn_count} turn{'s' if turn_count != 1 else ''})"

        transcript_lines = [f"# {title}", ""]
        for msg in full_messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            heading = "User" if role == "user" else ("Assistant" if role == "assistant" else role.title())
            transcript_lines.append(f"### {heading}")
            transcript_lines.append("")
            transcript_lines.append(content)
            transcript_lines.append("")
        transcript = "\n".join(transcript_lines)
        transcript_bytes = transcript.encode("utf-8")

        # 3. Upload transcript to MinIO (+ R2)
        s3_key = f"rag/{user_id}/chat_{conversation_id}.md"
        try:
            await storage_service.upload_file(
                file_content=transcript_bytes,
                key=s3_key,
                content_type="text/markdown; charset=utf-8",
            )
        except Exception as e:
            logger.warning("chat_history transcript upload failed", error=str(e), conversation_id=conversation_id)

        # 4. Delete stale Weaviate chunks so we don't accumulate duplicates
        if settings.db.VECTOR_STORE_ENABLED:
            try:
                vector_store = get_vector_store()
                await vector_store.delete_documents(
                    filters={"user_id": user_id, "document_id": document_id}
                )
            except Exception as e:
                logger.warning("chat_history old chunk delete failed", error=str(e), conversation_id=conversation_id)

        # 5. Upsert Postgres registry row
        try:
            await rag_registry.upsert(
                document_id=document_id,
                user_id=user_id,
                organization_id=organization_id,
                s3_key=s3_key,
                filename=f"chat_{conversation_id}.md",
                original_filename=f"{short_title}.md",
                title=title,
                mime_type="text/markdown",
                source="chat_history",
                conversation_id=conversation_id,
                size_bytes=len(transcript_bytes),
                status="processing",
            )
        except Exception as e:
            logger.warning("chat_history registry upsert failed", error=str(e), conversation_id=conversation_id)

        # 6. Re-chunk + re-embed transcript into Weaviate
        metadata = {
            "user_id": user_id,
            "source": "chat_history",
            "document_id": document_id,
            "conversation_id": conversation_id,
            "title": title,
            "s3_key": s3_key,
            "mime_type": "text/markdown",
            "created_at": datetime.utcnow().isoformat(),
        }
        if organization_id:
            metadata["organization_id"] = organization_id

        result = await document_processor.process_text(
            text=transcript, metadata=metadata
        )

        # Update registry with final chunk_count + status
        try:
            await rag_registry.update(
                document_id,
                chunk_count=result.chunk_count or 0,
                status="ready" if result.status == "success" else "error",
                error_message=result.error,
            )
        except Exception as e:
            logger.warning("chat_history registry status update failed", error=str(e), conversation_id=conversation_id)

        return {
            "document_id": document_id,
            "status": result.status,
            "chunk_count": result.chunk_count,
            "error": result.error,
        }
    
    async def add_document_from_url(
        self,
        url: str,
        user_id: str,
        organization_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process and add a document from a URL to the vector store.
        
        Args:
            url: URL to process
            user_id: User ID who owns the document
            organization_id: Optional organization ID
            tags: Optional tags for the document
            title: Optional document title
            
        Returns:
            Processing result
        """
        # Build metadata
        metadata = {
            "user_id": user_id,
            "source": "web",
            "url": url,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if organization_id:
            metadata["organization_id"] = organization_id
            
        if tags:
            metadata["tags"] = tags
            
        if title:
            metadata["title"] = title
            
        # Process URL
        result = await document_processor.process_url(
            url=url,
            metadata=metadata
        )
        
        return {
            "document_id": result.document_id,
            "status": result.status,
            "chunk_count": result.chunk_count,
            "error": result.error
        }
    
    async def add_document_from_text(
        self,
        text: str,
        user_id: str,
        title: Optional[str] = None,
        organization_id: Optional[str] = None,
        source: str = "manual_entry",
        tags: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Process and add a text document to the vector store.
        
        Args:
            text: Text content to process
            user_id: User ID who owns the document
            title: Optional document title
            organization_id: Optional organization ID
            source: Source of the document
            tags: Optional tags for the document
            
        Returns:
            Processing result
        """
        # Build metadata
        metadata = {
            "user_id": user_id,
            "source": source,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if organization_id:
            metadata["organization_id"] = organization_id
            
        if title:
            metadata["title"] = title
            
        if tags:
            metadata["tags"] = tags
            
        # Process text
        result = await document_processor.process_text(
            text=text,
            metadata=metadata
        )
        
        return {
            "document_id": result.document_id,
            "status": result.status,
            "chunk_count": result.chunk_count,
            "error": result.error
        }
    
    async def delete_user_context(
        self,
        user_id: str,
        source: Optional[str] = None,
        document_id: Optional[str] = None,
        older_than_days: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Delete context for a specific user.
        
        Args:
            user_id: User ID whose context to delete
            source: Optional source filter
            document_id: Optional document ID filter
            older_than_days: Optional age filter
            
        Returns:
            Result of deletion operation
        """
        filters = {"user_id": user_id}
        
        if source:
            filters["source"] = source
            
        if document_id:
            filters["document_id"] = document_id
            
        if older_than_days:
            cutoff_date = (datetime.utcnow() - timedelta(days=older_than_days)).isoformat()
            filters["created_before"] = cutoff_date
            
        try:
            if not settings.db.VECTOR_STORE_ENABLED:
                return {"success": False, "error": "Vector store disabled", "filters": filters}

            vector_store = get_vector_store()
            success = await vector_store.delete_documents(filters=filters)
            
            return {
                "success": success,
                "filters": filters            }
            
        except Exception as e:
            logger.error("Error deleting user context", error=str(e), user_id=user_id)
            return {
                "success": False,
                "error": str(e)
            }
    
    async def add_document_from_file(
        self,
        file_path: str,
        user_id: str,
        organization_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        title: Optional[str] = None,
        source: str = "upload"
    ) -> Dict[str, Any]:
        """
        Process and add a document from a file path to the vector store.
        
        Args:
            file_path: Path to the file
            user_id: User ID who owns the document
            organization_id: Optional organization ID
            tags: Optional tags for the document
            title: Optional document title
            source: Source of the document
            
        Returns:
            Processing result
        """
        # Build metadata
        metadata = {
            "user_id": user_id,
            "source": source,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if organization_id:
            metadata["organization_id"] = organization_id
            
        if tags:
            metadata["tags"] = tags
            
        if title:
            metadata["title"] = title
            
        # Process file
        result = await document_processor.process_file(
            file_path=file_path,
            metadata=metadata
        )
        
        return {
            "document_id": result.document_id,
            "status": result.status,
            "chunk_count": result.chunk_count,
            "error": result.error
        }
    
    async def add_document_from_google_drive(
        self,
        drive_file_id: str,
        user_id: str,
        organization_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        title: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process and add a document from Google Drive to the vector store.
        
        Args:
            drive_file_id: Google Drive file ID
            user_id: User ID who owns the document
            organization_id: Optional organization ID
            tags: Optional tags for the document
            title: Optional document title
            
        Returns:
            Processing result
        """
        # Build metadata
        metadata = {
            "user_id": user_id,
            "source": "drive",
            "drive_file_id": drive_file_id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if organization_id:
            metadata["organization_id"] = organization_id
            
        if tags:
            metadata["tags"] = tags
            
        if title:
            metadata["title"] = title
        
        # Process Google Drive document
        result = await document_processor.process_google_drive(
            drive_file_id=drive_file_id,
            metadata=metadata
        )
        
        return {
            "document_id": result.document_id,
            "status": result.status,            "chunk_count": result.chunk_count,
            "error": result.error
        }
    
    async def get_user_documents(
        self,
        user_id: str,
        organization_id: Optional[str] = None,
        source_types: Optional[List[str]] = None,
        tags: Optional[List[str]] = None,
        limit: int = 100,
        offset: int = 0
    ) -> Dict[str, Any]:
        """
        Get a list of documents for a user.
        
        Args:
            user_id: User ID to filter by
            organization_id: Optional organization ID to filter by
            source_types: Optional list of source types to include
            tags: Optional list of tags to filter by
            limit: Maximum number of documents to return
            offset: Offset for pagination
            
        Returns:
            List of documents and metadata
        """
        # Build filters — always filter by user_id.
        # organization_id filtering is done post-query because we need OR logic:
        # include docs matching the org OR docs with no org (personal uploads).
        filters = {"user_id": user_id}
            
        # Note: don't filter by source in Weaviate query when source_types
        # is provided — old data may have file paths instead of type labels.
        # We'll filter after reading instead.
        _source_filter = source_types
            
        if tags:
            filters["tags"] = tags
            
        try:
            # Get document count for pagination info
            if not settings.db.VECTOR_STORE_ENABLED:
                return {
                    "documents": [],
                    "total": 0,
                    "unique_count": 0,
                    "limit": limit,
                    "offset": offset,
                    "error": "Vector store disabled"
                }

            vector_store = get_vector_store()
            total_count = await vector_store.get_document_count(filters=filters)
            
            # Get document metadata from vector store
            documents = await vector_store.get_documents(
                filters=filters,
                limit=limit,
                offset=offset
            )
            
            # Normalize source values — old data may have file paths instead of type labels
            def _normalize_source(raw: str) -> str:
                if not raw or raw in ("upload", "web", "manual_entry", "chat_history", "drive", "direct_text"):
                    return raw or "unknown"
                # File path → "upload"
                if "/" in raw or "\\" in raw:
                    return "upload"
                return raw

            # Group by document_id and extract summary data
            document_map = {}
            for doc in documents:
                doc_id = doc["metadata"].get("document_id")
                if not doc_id:
                    continue

                source = _normalize_source(doc["metadata"].get("source", "unknown"))

                if doc_id not in document_map:
                    document_map[doc_id] = {
                        "document_id": doc_id,
                        "title": doc["metadata"].get("title", "Unnamed document"),
                        "source": source,
                        "created_at": doc["metadata"].get("created_at"),
                        "tags": doc["metadata"].get("tags", []),
                        "chunk_count": 1,
                        "url": doc["metadata"].get("url", ""),
                        "mime_type": doc["metadata"].get("mime_type", ""),
                        "organization_id": doc["metadata"].get("organization_id", ""),
                        "summary": doc["content"][:150] + "..." if len(doc["content"]) > 150 else doc["content"]
                    }
                else:
                    document_map[doc_id]["chunk_count"] += 1

            # Get unique documents
            unique_documents = list(document_map.values())

            # Post-filter by organization: include docs that match the org OR have no org set
            if organization_id:
                unique_documents = [
                    d for d in unique_documents
                    if not d.get("organization_id") or d["organization_id"] == organization_id
                ]

            # Post-filter by source types if requested
            if _source_filter:
                unique_documents = [d for d in unique_documents if d["source"] in _source_filter]
            
            # Sort by created_at (newest first)
            unique_documents.sort(
                key=lambda x: x.get("created_at", ""), 
                reverse=True
            )
            
            return {
                "documents": unique_documents,
                "total": sum(d["chunk_count"] for d in unique_documents),
                "unique_count": len(unique_documents),
                "limit": limit,
                "offset": offset
            }
            
        except Exception as e:
            logger.error("Error retrieving user documents", error=str(e), user_id=user_id)
            return {
                "documents": [],
                "total": 0,
                "limit": limit,
                "offset": offset,
                "error": str(e)
            }
    
    async def get_all_context_sources(
        self,
        query: str,
        user_id: str, 
        organization_id: Optional[str] = None,
        max_results_per_source: int = 3
    ) -> Dict[str, Any]:
        """
        Get context from all available sources for a comprehensive answer.
        This is used for the main 'Ask Lumicoria AI' feature.
        
        Args:
            query: The user's question
            user_id: User ID for filtering context
            organization_id: Optional organization ID for filtering
            max_results_per_source: Maximum results per source type
            
        Returns:
            Dict with context from different sources
        """
        await self.initialize()
        
        # Generate embedding for the query
        query_embedding = await self.llm_client.generate_embeddings(texts=[query])
        if not query_embedding or len(query_embedding) == 0:
            logger.error("Failed to generate query embedding")
            return {"context": [], "error": "Failed to generate query embedding"}
        
        # Define source types to search from
        source_types = ["upload", "drive", "web", "chat_history"]
        
        all_context = []
        total_chunks = 0
        context_by_source = {}
        
        # Build base filters — org_id is post-filtered to support personal + org docs
        base_filters = {"user_id": user_id}
        
        # Query each source type
        for source in source_types:
            try:
                # Create source-specific filters
                source_filters = base_filters.copy()
                source_filters["source"] = source
                
                # Search vector store
                if not settings.db.VECTOR_STORE_ENABLED:
                    continue

                vector_store = get_vector_store()
                results = await vector_store.similarity_search(
                    query_vector=query_embedding[0],
                    k=max_results_per_source,
                    filters=source_filters
                )
                
                if results:
                    # Post-filter by org: keep docs matching the org OR with no org
                    if organization_id:
                        results = [
                            r for r in results
                            if not r.get("metadata", {}).get("organization_id")
                            or r["metadata"]["organization_id"] == organization_id
                        ]
                    # Format results
                    formatted_results = self._format_context(results)
                    all_context.extend(formatted_results)
                    context_by_source[source] = formatted_results
                    total_chunks += len(formatted_results)
            
            except Exception as e:
                logger.error(f"Error retrieving context from {source}", error=str(e))
        
        return {
            "context": all_context,
            "context_by_source": context_by_source,
            "total_chunks": total_chunks,
            "query": query,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _format_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Format vector store results for context inclusion.

        Backward-compatible shape — adds ``score_components`` for callers
        that want to inspect ranking, but old code that only reads
        ``text`` / ``score`` / ``source`` keeps working.
        """
        formatted_results: List[Dict[str, Any]] = []

        for result in results:
            metadata_in = result.get("metadata") or {}
            formatted_result: Dict[str, Any] = {
                "text": result.get("content", "") or result.get("text", ""),
                "score": result.get("score"),
                "source": metadata_in.get("source", "unknown"),
                "score_components": result.get("score_components") or {},
                "metadata": {},
            }

            # Include relevant metadata but filter out internal fields.
            for key, value in metadata_in.items():
                if key not in ("user_id", "organization_id", "chunk_id"):
                    formatted_result["metadata"][key] = value

            formatted_results.append(formatted_result)

        return formatted_results

# Create a singleton instance
context_service = ContextService()

# Initialize the service asynchronously - this needs to be called at application startup
async def initialize_context_service():
    """Initialize the context service and its dependencies."""
    await context_service.initialize()
