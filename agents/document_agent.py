from .base_agent import BaseAgent
from backend.ai_models import LLMConfig
from typing import Dict, Any, List, Optional, Tuple
import hashlib
import json
import structlog
import asyncio
from datetime import datetime, timedelta
import re

# Configure logger
logger = structlog.get_logger(__name__)


# Bump this when the prompts or schema change — cached results carrying
# the old version are ignored and re-extracted on next read.
EXTRACTOR_VERSION = "1.0.0"

# Pipeline tuning constants — adjust to balance latency / cost / quality.
_STAGE_A_INPUT_CHARS = 4000      # Stage A only needs a sample
_CHUNK_SIZE_CHARS = 7200          # ~1800 tokens at 4 chars/token
_CHUNK_OVERLAP_CHARS = 240
_MAX_CHUNKS = 40                  # hard cap so cost is bounded on huge docs
_PARALLEL_CHUNKS = 8              # asyncio semaphore for Stage B
_SKIP_CHUNK_BELOW_CHARS = 4000    # below this we send the whole doc to Stage C
_STAGE_D_PASS_FLOOR = 0.7         # below → retry Stage C once with strict mode

class DocumentAgent(BaseAgent):
    """Agent for processing documents using LLM providers.
    
    This agent extracts key information, tasks, dates, and insights from 
    documents using the provider-agnostic LLM interface.
    """
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        # Default extraction targets if not specified in config
        self.extraction_targets = config.get("extraction_targets", [
            "tasks", "dates", "names", "organizations", 
            "monetary amounts", "action items", "key points"
        ])
        
        # Configure with default model if not specified
        if "model" not in self.model_config:
            self.model_config["model"] = "sonar-large-online"

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the document with natural language questions.
        
        Args:
            query: The natural language query about the document
            context: Optional context including document content and metadata
            
        Returns:
            Dictionary containing the response and any relevant extracted information
        """
        if not context or not context.get("document_text"):
            return {"error": "No document content provided in context"}
            
        document_text = context["document_text"]
        prompt = (
            f"Using the following document content, answer this question: {query}\n\n"
            f"Document content:\n{document_text[:8000]}..."  # Limit document length
        )
        
        try:
            if self.llm_client:
                messages = [{"role": "user", "content": prompt}]
                response = await self.llm_client.generate(messages)
                return {
                    "response": response.content,
                    "query": query,
                    "document_id": context.get("document_id"),
                    "confidence": 0.0
                }
            else:
                return {"error": "LLM client not initialized"}
        except Exception as e:
            logger.error(f"Error querying document: {str(e)}")
            return {"error": f"Failed to process query: {str(e)}"}

    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document end-to-end through the 4-stage pipeline.

        Stage A: classify the document (type, sensitivity, urgency,
                 language, expected action count).
        Stage B: chunk via RecursiveCharacterTextSplitter, summarise
                 every chunk in parallel (Semaphore(8)).
        Stage C: extract structured items — action items, decisions,
                 dates, people — citing chunk_ids for provenance.
        Stage D: LLM-as-judge self-evaluation. Below the pass floor →
                 retry Stage C once with strict mode (temperature=0,
                 stricter prompt). Still below → flag low_confidence.

        Args:
            data: ``{"text": str, "metadata": dict, "user_context": dict}``.

        Returns:
            Backward-compatible dict — every key the old shape produced
            still ships ("analysis", "tasks", "metadata",
            "extraction_targets", "model_used", "timestamp"). Plus new
            keys ("extraction_id", "classification", "extraction",
            "confidence", "low_confidence", "chunk_count",
            "duration_ms", "cached", "sources") that downstream
            consumers can opt into without breaking anything that
            reads only the old keys.
        """
        try:
            document_text: str = data.get("text", "") or ""
            document_metadata: Dict[str, Any] = data.get("metadata") or {}
            user_context: Dict[str, Any] = data.get("user_context") or {}

            if not document_text.strip():
                return {"error": "No document text provided"}
            if not self.llm_client:
                return {"error": "LLM client not initialized"}

            start_ms = self._now_ms()

            # ── Cache lookup ─────────────────────────────────────────
            content_hash = self._hash_text(document_text)
            cached = await self._cache_get(content_hash)
            if cached:
                logger.info(
                    "document_agent.cache_hit",
                    content_hash=content_hash[:12],
                    document_id=document_metadata.get("document_id"),
                )
                return self._compose_public_result(
                    cached=cached,
                    document_metadata=document_metadata,
                    user_context=user_context,
                    was_cached=True,
                )

            # ── Stage A — classify ──────────────────────────────────
            classification = await self._stage_a_classify(
                document_text=document_text,
                user_context=user_context,
            )

            # ── Stage B — chunk + summarise (parallel, semaphore-capped)
            chunk_summaries, raw_chunks = await self._stage_b_chunk_and_summarise(
                document_text=document_text,
                classification=classification,
            )

            # ── Stage C — extract (one or two passes)
            extraction, parsed_ok = await self._stage_c_extract(
                document_text=document_text,
                classification=classification,
                chunk_summaries=chunk_summaries,
                raw_chunks=raw_chunks,
                user_context=user_context,
                strict=False,
            )

            # ── Stage D — self-evaluate
            self_eval = await self._stage_d_self_evaluate(
                classification=classification,
                extraction=extraction,
                chunk_summaries=chunk_summaries,
            )

            low_confidence = False
            if self_eval is not None and self_eval.get("score", 0.0) < _STAGE_D_PASS_FLOOR:
                # One strict retry of Stage C — temperature=0, tighter prompt.
                logger.info(
                    "document_agent.stage_d_retry",
                    score=self_eval.get("score"),
                )
                extraction_retry, _ok = await self._stage_c_extract(
                    document_text=document_text,
                    classification=classification,
                    chunk_summaries=chunk_summaries,
                    raw_chunks=raw_chunks,
                    user_context=user_context,
                    strict=True,
                )
                self_eval_retry = await self._stage_d_self_evaluate(
                    classification=classification,
                    extraction=extraction_retry,
                    chunk_summaries=chunk_summaries,
                )
                if self_eval_retry and self_eval_retry.get("score", 0.0) >= _STAGE_D_PASS_FLOOR:
                    extraction = extraction_retry
                    self_eval = self_eval_retry
                else:
                    # Keep whichever pass had the higher score.
                    if (
                        self_eval_retry
                        and self_eval_retry.get("score", 0.0)
                        > self_eval.get("score", 0.0)
                    ):
                        extraction = extraction_retry
                        self_eval = self_eval_retry
                    low_confidence = True

            confidence = float(
                (self_eval or {}).get("score")
                or extraction.get("overall_confidence")
                or 0.0
            )

            cache_payload = {
                "content_hash": content_hash,
                "extractor_version": EXTRACTOR_VERSION,
                "classification": classification,
                "chunk_summaries": chunk_summaries,
                "extraction": extraction,
                "self_eval": self_eval,
                "confidence": confidence,
                "low_confidence": low_confidence,
                "chunk_count": len(chunk_summaries),
                "duration_ms": self._now_ms() - start_ms,
            }
            await self._cache_set(
                content_hash=content_hash,
                payload=cache_payload,
                document_id=document_metadata.get("document_id"),
            )

            return self._compose_public_result(
                cached=cache_payload,
                document_metadata=document_metadata,
                user_context=user_context,
                was_cached=False,
            )

        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {"error": f"Failed to process document: {str(e)}"}

    # ─────────────────────────────────────────────────────────────────
    # Public-result shaping — preserves the legacy keys + adds new ones
    # ─────────────────────────────────────────────────────────────────

    def _compose_public_result(
        self,
        *,
        cached: Dict[str, Any],
        document_metadata: Dict[str, Any],
        user_context: Dict[str, Any],
        was_cached: bool,
    ) -> Dict[str, Any]:
        """Map the structured cache payload onto the legacy return shape
        plus the new fields. Keeping the legacy keys untouched means
        every existing caller (task_executor, chat, document_tasks)
        continues to work without a single line changed."""
        extraction = cached.get("extraction") or {}
        action_items = extraction.get("action_items") or []
        # Normalise each ExtractedActionItem dict through the existing
        # `_normalize_task_record` so downstream task-creation code sees
        # the canonical shape it always saw — title, description,
        # priority, due_date (mandatory + capped), assignee, agent_key.
        tasks: List[Dict[str, Any]] = []
        user_tz = (user_context or {}).get("timezone", "UTC")
        for ai in action_items:
            try:
                normalised = self._normalize_task_record({
                    "title": ai.get("title", ""),
                    "description": ai.get("description", ""),
                    "priority": ai.get("priority", "medium"),
                    "due_date": (
                        ai.get("due_date").isoformat()
                        if hasattr(ai.get("due_date"), "isoformat")
                        else ai.get("due_date")
                    ),
                    "deadline": ai.get("deadline_phrase"),
                    "inferred_due_date": ai.get("inferred_due_date", False),
                    "assignee": ai.get("assignee"),
                    "assigned_to_agent": ai.get("assigned_to_agent"),
                })
                tasks.append(normalised)
            except Exception as e:  # noqa: BLE001
                logger.debug("document_agent.task_normalise_skip", error=str(e))

        # `sources` powers the "Sources" footer in agent proposals + the
        # digest — one row per chunk that grounded an action item.
        sources: List[Dict[str, Any]] = []
        for ai in action_items:
            for cid in ai.get("cite_chunk_ids", []) or []:
                sources.append({
                    "chunk_id": cid,
                    "kind": "action_item",
                    "title": ai.get("title"),
                })

        return {
            # ── Legacy shape (unchanged) ────────────────────────────
            "analysis": extraction.get("summary", "") or "",
            "tasks": tasks,
            "metadata": document_metadata,
            "extraction_targets": self.extraction_targets,
            "model_used": self.model_config.get("model", "unknown"),
            "timestamp": datetime.utcnow().isoformat(),
            # ── New, additive fields ────────────────────────────────
            "extraction_id": f"{cached.get('content_hash', '')}:{cached.get('extractor_version', '')}",
            "content_hash": cached.get("content_hash"),
            "extractor_version": cached.get("extractor_version"),
            "classification": cached.get("classification"),
            "extraction": extraction,
            "self_eval": cached.get("self_eval"),
            "confidence": cached.get("confidence"),
            "low_confidence": bool(cached.get("low_confidence")),
            "chunk_count": cached.get("chunk_count"),
            "duration_ms": cached.get("duration_ms"),
            "cached": was_cached,
            "sources": sources,
        }

    # ─────────────────────────────────────────────────────────────────
    # Stage A — classify
    # ─────────────────────────────────────────────────────────────────

    async def _stage_a_classify(
        self,
        *,
        document_text: str,
        user_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """One small LLM call. Returns a dict matching DocumentClassification.

        Validated via schema_eval — on parse fail we fall back to a
        conservative classification (type=other, urgency=low) so the
        pipeline keeps moving.
        """
        from .document_agent_schemas import DocumentClassification
        from backend.services.brain.evals import check_schema

        sample = document_text[:_STAGE_A_INPUT_CHARS]
        prompt = (
            "Classify this document.\n\n"
            "Return STRICT JSON only (no prose, no markdown). Schema:\n"
            '  {"document_type": "contract|invoice|meeting_notes|proposal|email_thread|'
            'report|spec|policy|presentation|letter|form|research_paper|other",\n'
            '   "language": "ISO-639-1 code (en, es, fr, ...)",\n'
            '   "sensitivity": "public|internal|confidential|restricted",\n'
            '   "urgency": "critical|high|medium|low",\n'
            '   "estimated_action_count": 0..50,\n'
            '   "short_title": "≤120 chars summary or null",\n'
            '   "detected_parties": ["name1", "name2", ...]}\n\n'
            "Sample:\n```\n"
            f"{sample}\n```"
        )
        try:
            response = await self.llm_client.generate(
                [{"role": "user", "content": prompt}],
            )
            raw = getattr(response, "content", None) or str(response)
        except Exception as e:  # noqa: BLE001
            logger.warning("document_agent.stage_a_llm_failed", error=str(e))
            return DocumentClassification().model_dump()

        items, eval_result = check_schema(
            raw, DocumentClassification, pass_floor=0.9,
        )
        if not eval_result.passed or not items:
            logger.warning(
                "document_agent.stage_a_parse_failed",
                reason=eval_result.reason,
            )
            return DocumentClassification().model_dump()
        return items[0].model_dump()

    # ─────────────────────────────────────────────────────────────────
    # Stage B — chunk + summarise (parallel, Semaphore-capped)
    # ─────────────────────────────────────────────────────────────────

    async def _stage_b_chunk_and_summarise(
        self,
        *,
        document_text: str,
        classification: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Returns ``(chunk_summaries, raw_chunks)``. If the doc is below
        the chunk threshold, returns ``([], [document_text])`` so
        Stage C can operate on the full text directly.
        """
        from .document_agent_schemas import ChunkSummary
        from backend.services.brain.evals import check_schema

        # Short docs: skip chunking entirely.
        if len(document_text) <= _SKIP_CHUNK_BELOW_CHARS:
            return [], [document_text]

        # Use LangChain's splitter for structure-aware boundaries.
        try:
            from langchain_text_splitters import RecursiveCharacterTextSplitter
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=_CHUNK_SIZE_CHARS,
                chunk_overlap=_CHUNK_OVERLAP_CHARS,
                separators=["\n\n\n", "\n\n", "\n", ". ", " ", ""],
            )
            raw_chunks = splitter.split_text(document_text)
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "document_agent.chunk_split_failed",
                error=str(e), fallback="naive_slices",
            )
            raw_chunks = [
                document_text[i : i + _CHUNK_SIZE_CHARS]
                for i in range(0, len(document_text), _CHUNK_SIZE_CHARS)
            ]

        if len(raw_chunks) > _MAX_CHUNKS:
            logger.info(
                "document_agent.chunks_capped",
                original=len(raw_chunks), cap=_MAX_CHUNKS,
            )
            raw_chunks = raw_chunks[:_MAX_CHUNKS]

        # Summarise each chunk in parallel with a semaphore.
        semaphore = asyncio.Semaphore(_PARALLEL_CHUNKS)
        doc_type = classification.get("document_type", "other")

        async def _summarise_one(idx: int, chunk: str) -> Optional[Dict[str, Any]]:
            async with semaphore:
                prompt = (
                    f"Document type: {doc_type}\n"
                    f"Chunk index: {idx} of {len(raw_chunks) - 1}\n\n"
                    "Summarise this chunk in ≤200 words. Mark has_action=true if it "
                    "contains a verb-led action item, deadline, or decision.\n\n"
                    "Return STRICT JSON only. Schema:\n"
                    '  {"chunk_id": int, "summary": "≤2000 char string", '
                    '"key_terms": ["..."], "has_action": bool}\n\n'
                    "Chunk:\n```\n"
                    f"{chunk}\n```"
                )
                try:
                    response = await self.llm_client.generate(
                        [{"role": "user", "content": prompt}],
                    )
                    raw = getattr(response, "content", None) or str(response)
                except Exception as e:  # noqa: BLE001
                    logger.debug(
                        "document_agent.stage_b_chunk_failed",
                        chunk=idx, error=str(e),
                    )
                    return None

                items, ev = check_schema(raw, ChunkSummary, pass_floor=0.9)
                if not items:
                    return None
                # Force chunk_id alignment with our index.
                item = items[0]
                item.chunk_id = idx
                return item.model_dump()

        summaries = await asyncio.gather(
            *[_summarise_one(i, c) for i, c in enumerate(raw_chunks)],
            return_exceptions=False,
        )
        chunk_summaries = [s for s in summaries if s]
        return chunk_summaries, raw_chunks

    # ─────────────────────────────────────────────────────────────────
    # Stage C — structured extract
    # ─────────────────────────────────────────────────────────────────

    async def _stage_c_extract(
        self,
        *,
        document_text: str,
        classification: Dict[str, Any],
        chunk_summaries: List[Dict[str, Any]],
        raw_chunks: List[str],
        user_context: Dict[str, Any],
        strict: bool,
    ) -> Tuple[Dict[str, Any], bool]:
        """Run the structured extract pass. ``strict=True`` tightens the
        prompt and asks the LLM for higher confidence — used by the
        Stage D retry path. Returns ``(extraction_dict, parsed_ok)``.
        """
        from .document_agent_schemas import DocumentExtraction
        from backend.services.brain.evals import check_schema

        # Build the input. When Stage B ran, we feed it summaries; when
        # the doc was short and Stage B was skipped, we feed it the
        # full text.
        if chunk_summaries:
            stage_b_block = "\n\n".join(
                f"[chunk {s['chunk_id']}] {s['summary']}"
                for s in chunk_summaries
            )
            stage_b_intro = (
                "Below are summaries of every chunk in the document. Use them "
                "to extract structured items. Cite chunk_ids in each item's "
                "cite_chunk_ids list so we can render provenance."
            )
        else:
            stage_b_block = raw_chunks[0]
            stage_b_intro = (
                "The full document text follows. Treat it as a single chunk "
                "with chunk_id=0 — all citations should reference 0."
            )

        try:
            from backend.agents.router import AGENT_REGISTRY as _AGENT_REGISTRY
            agent_lines = "\n".join(
                f'  - "{k}": {v}' for k, v in _AGENT_REGISTRY.items()
            )
        except Exception:
            agent_lines = "  (agent registry unavailable)"

        now_utc = datetime.utcnow()
        five_days = (now_utc + timedelta(days=5)).replace(microsecond=0).isoformat() + "Z"
        week_cap = (now_utc + timedelta(days=7)).replace(microsecond=0).isoformat() + "Z"
        user_tz = (user_context or {}).get("timezone", "UTC")

        strict_clause = ""
        if strict:
            strict_clause = (
                "\nSTRICT MODE: temperature 0, no speculation. Only output "
                "items that have a direct sentence-level citation in the "
                "source. Set confidence ≥ 0.85 or omit the item entirely.\n"
            )

        prompt = (
            "Extract structured items from this document.\n"
            "Return STRICT JSON only (no prose, no markdown).\n\n"
            f"Classification: type={classification.get('document_type')}, "
            f"urgency={classification.get('urgency')}, "
            f"language={classification.get('language')}.\n"
            f"Current time (UTC): {now_utc.replace(microsecond=0).isoformat()}Z\n"
            f"User timezone:      {user_tz}\n"
            f"Default due_date if none explicit: {five_days}\n"
            f"Maximum allowed due_date:          {week_cap}\n"
            f"{strict_clause}\n"
            "Available specialist agents (key: capability) — assigned_to_agent "
            "must be one of these keys or null:\n"
            f"{agent_lines}\n\n"
            "Schema (return EXACTLY this shape):\n"
            "{\n"
            '  "summary": "≤4000 chars — 1-2 paragraph executive summary",\n'
            '  "action_items": [\n'
            "    {\n"
            '      "title": "≤200 chars, verb-led",\n'
            '      "description": "≤1000 chars",\n'
            '      "priority": "critical|high|medium|low",\n'
            '      "due_date": "ISO-8601 UTC or null",\n'
            '      "inferred_due_date": bool,\n'
            '      "deadline_phrase": "exact phrase from doc or null",\n'
            '      "assignee": "person name or null",\n'
            '      "assigned_to_agent": "agent key from list above or null",\n'
            '      "cite_chunk_ids": [int, ...],\n'
            '      "confidence": 0..1\n'
            "    }\n"
            "  ],\n"
            '  "key_decisions":  [{"text": "...", "cite_chunk_ids": [...], "confidence": 0..1}],\n'
            '  "key_dates":      [{"date": "ISO-8601 or null", "raw_phrase": "...", "what": "...", '
            '"cite_chunk_ids": [...], "confidence": 0..1}],\n'
            '  "key_people":     [{"name": "...", "role": "..." or null, "email": "..." or null, '
            '"cite_chunk_ids": [...]}],\n'
            '  "sentiment":      "positive|neutral|negative",\n'
            '  "overall_confidence": 0..1\n'
            "}\n\n"
            f"{stage_b_intro}\n\n"
            f"{stage_b_block}"
        )

        try:
            response = await self.llm_client.generate(
                [{"role": "user", "content": prompt}],
            )
            raw = getattr(response, "content", None) or str(response)
        except Exception as e:  # noqa: BLE001
            logger.warning("document_agent.stage_c_llm_failed", error=str(e))
            return DocumentExtraction(summary="").model_dump(), False

        items, ev = check_schema(raw, DocumentExtraction, pass_floor=0.7)
        if not items:
            logger.warning(
                "document_agent.stage_c_parse_failed",
                reason=ev.reason, strict=strict,
            )
            return DocumentExtraction(summary="").model_dump(), False
        return items[0].model_dump(), True

    # ─────────────────────────────────────────────────────────────────
    # Stage D — self-evaluate (LLM-as-judge)
    # ─────────────────────────────────────────────────────────────────

    async def _stage_d_self_evaluate(
        self,
        *,
        classification: Dict[str, Any],
        extraction: Dict[str, Any],
        chunk_summaries: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Have the LLM grade its own extraction. Returns None on hard
        failure (caller then proceeds without self_eval gating — the
        extraction still ships, but flagged low_confidence)."""
        from .document_agent_schemas import SelfEvalResult
        from backend.services.brain.evals import check_schema

        # Compact the chunk summaries for the judge prompt.
        summaries_block = "\n".join(
            f"[chunk {s['chunk_id']}] {s['summary']}"
            for s in chunk_summaries[:20]
        ) or "(no chunks — document was short, extraction ran on full text)"

        prompt = (
            "You are an evaluator. Grade the extraction below against the "
            "source chunk summaries.\n"
            "Return STRICT JSON only matching this schema:\n"
            '  {"score": 0..1, "grounded": bool, "issues": ["..."], '
            '"recommendations": ["..."], "flagged_action_indices": [int, ...]}\n\n'
            "Score rubric:\n"
            "  1.0 — every claim has a clear citation in the chunks.\n"
            "  0.7 — most claims grounded; minor unsupported items.\n"
            "  0.4 — meaningful hallucination or fabricated dates/names.\n"
            "  0.0 — extraction unrelated to the source.\n\n"
            f"Classification: {json.dumps(classification, default=str)}\n\n"
            f"Extraction:\n{json.dumps(extraction, default=str)[:6000]}\n\n"
            f"Source chunks:\n{summaries_block}"
        )

        try:
            response = await self.llm_client.generate(
                [{"role": "user", "content": prompt}],
            )
            raw = getattr(response, "content", None) or str(response)
        except Exception as e:  # noqa: BLE001
            logger.debug("document_agent.stage_d_llm_failed", error=str(e))
            return None

        items, ev = check_schema(raw, SelfEvalResult, pass_floor=0.9)
        if not items:
            logger.debug(
                "document_agent.stage_d_parse_failed", reason=ev.reason,
            )
            return None
        return items[0].model_dump()

    # ─────────────────────────────────────────────────────────────────
    # Mongo cache — read-through, write on success
    # ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()

    @staticmethod
    def _now_ms() -> int:
        return int(datetime.utcnow().timestamp() * 1000)

    async def _cache_get(self, content_hash: str) -> Optional[Dict[str, Any]]:
        try:
            from backend.db.mongodb.mongodb import MongoDB
            db = await MongoDB.get_database()
            doc = await db.document_extractions.find_one({
                "content_hash": content_hash,
                "extractor_version": EXTRACTOR_VERSION,
            })
            if not doc:
                return None
            doc.pop("_id", None)
            return doc
        except Exception as e:  # noqa: BLE001
            logger.debug("document_agent.cache_get_failed", error=str(e))
            return None

    async def _cache_set(
        self,
        *,
        content_hash: str,
        payload: Dict[str, Any],
        document_id: Optional[str],
    ) -> None:
        """Upsert the extraction. Best-effort — a Mongo blip never
        breaks the public return; the caller already has the result."""
        try:
            from backend.db.mongodb.mongodb import MongoDB
            db = await MongoDB.get_database()
            try:
                # First write also creates the unique index. Repeated
                # creates are cheap; Mongo ignores the dupe.
                await db.document_extractions.create_index(
                    [("content_hash", 1), ("extractor_version", 1)],
                    unique=True, background=True,
                )
            except Exception:
                pass

            doc = {
                **payload,
                "document_id": document_id,
                "cached_at": datetime.utcnow(),
            }
            await db.document_extractions.update_one(
                {
                    "content_hash": content_hash,
                    "extractor_version": EXTRACTOR_VERSION,
                },
                {"$set": doc},
                upsert=True,
            )
        except Exception as e:  # noqa: BLE001
            logger.debug("document_agent.cache_set_failed", error=str(e))

    def _normalize_task_record(self, raw: Dict[str, Any]) -> Dict[str, Any]:
        """Phase 4: normalise one extracted task into the canonical schema.

        Coerces `due_date` to ISO-8601 UTC, falls back to a +5-day default
        when missing/unparseable, caps at +7 days, and lifts `deadline`
        into `due_date` when the LLM put a structured value there instead.
        """
        if not isinstance(raw, dict):
            return raw  # let the caller drop non-dict entries

        now = datetime.utcnow().replace(microsecond=0)
        max_due = now + timedelta(days=7)
        default_due = now + timedelta(days=5, hours=0)

        # 1. Resolve a candidate datetime from due_date OR deadline.
        candidate_str = raw.get("due_date") or raw.get("deadline")
        parsed_due: Optional[datetime] = None
        if candidate_str:
            parsed_due = self._coerce_to_datetime(str(candidate_str))

        inferred = bool(raw.get("inferred_due_date"))
        if parsed_due is None:
            parsed_due = default_due
            inferred = True

        # 2. Cap at +7 days (the system promises max-7-day due dates).
        if parsed_due > max_due:
            parsed_due = max_due
            inferred = True

        # 3. Never schedule in the past — bump to +1 hour from now.
        if parsed_due < now:
            parsed_due = now + timedelta(hours=1)
            inferred = True

        priority = str(raw.get("priority", "medium")).lower()
        if priority not in {"low", "medium", "high", "critical"}:
            priority = "medium"

        # Phase 6: validate `assigned_to_agent` against the registry; drop
        # anything the LLM hallucinated.
        suggested_agent = raw.get("assigned_to_agent")
        if isinstance(suggested_agent, str):
            suggested_agent = suggested_agent.strip().lower() or None
            try:
                from backend.agents.router import AGENT_REGISTRY as _AGENT_REGISTRY
                if suggested_agent not in _AGENT_REGISTRY:
                    suggested_agent = None
            except Exception:
                suggested_agent = None
        else:
            suggested_agent = None

        normalized: Dict[str, Any] = {
            "title": str(raw.get("title") or "Untitled task").strip()[:280],
            "description": str(raw.get("description") or "").strip(),
            "priority": priority,
            # `deadline` stays as the human-readable original phrase for UI
            "deadline": raw.get("deadline") if isinstance(raw.get("deadline"), str) else None,
            "due_date": parsed_due.isoformat() + "Z",
            "inferred_due_date": inferred,
            "assignee": raw.get("assignee") if isinstance(raw.get("assignee"), str) else None,
            "assigned_to_agent": suggested_agent,
        }
        return normalized

    def _coerce_to_datetime(self, value: str) -> Optional[datetime]:
        """Parse a wide range of date formats into a naive UTC datetime.

        Order of attempts:
          1. ISO-8601 (with or without trailing Z).
          2. python-dateutil best-effort (handles 'May 1, 2026', 'Mon 14:00', etc.).
          3. Embedded YYYY-MM-DD substring.
          4. None — caller will fall back to the +5-day default.
        """
        s = (value or "").strip()
        if not s:
            return None
        # Strip trailing 'Z' since fromisoformat (≤3.10) doesn't grok it
        iso_candidate = s[:-1] if s.endswith("Z") else s
        try:
            dt = datetime.fromisoformat(iso_candidate)
            if dt.tzinfo is not None:
                # Drop tz and treat as UTC for downstream consistency.
                dt = dt.replace(tzinfo=None)
            return dt
        except ValueError:
            pass
        # dateutil
        try:
            from dateutil import parser as date_parser  # type: ignore
            return date_parser.parse(s, fuzzy=True, default=datetime.utcnow().replace(hour=9, minute=0, second=0, microsecond=0))
        except Exception:
            pass
        # Embedded YYYY-MM-DD
        m = re.search(r"(\d{4})-(\d{2})-(\d{2})", s)
        if m:
            try:
                return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), 9, 0, 0)
            except ValueError:
                pass
        return None

    def _parse_tasks_json(self, raw_text: str) -> List[Dict[str, Any]]:
        """Best-effort parse of an LLM response into a list of task dicts.

        Phase 4: every returned task is normalised via `_normalize_task_record`
        so the downstream consumer (task creation + reminders) gets a
        consistent ISO `due_date` and bounded fields.
        """
        candidates: List[Any] = []

        # Try direct JSON parse first
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                candidates = parsed
            elif isinstance(parsed, dict) and "tasks" in parsed:
                candidates = parsed["tasks"]
        except json.JSONDecodeError:
            pass

        # Try extracting JSON array from markdown code fences
        if not candidates:
            json_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", raw_text)
            if json_match:
                try:
                    candidates = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

        # Try finding any JSON array in the text
        if not candidates:
            bracket_match = re.search(r"\[[\s\S]*\]", raw_text)
            if bracket_match:
                try:
                    candidates = json.loads(bracket_match.group(0))
                except json.JSONDecodeError:
                    pass

        # Last resort: fall back to the existing regex parser, then normalise.
        if not candidates:
            candidates = self._parse_tasks(raw_text)

        # Normalise each entry — drops malformed dicts silently.
        normalized: List[Dict[str, Any]] = []
        for entry in candidates or []:
            if not isinstance(entry, dict):
                continue
            try:
                normalized.append(self._normalize_task_record(entry))
            except Exception:  # pragma: no cover — defensive
                continue
        return normalized

    # NB: a second copy of the original method body lives below for
    # backwards-compatibility with any caller that imports it directly.
    # The new implementation above supersedes it.
    def _parse_tasks_json_legacy(self, raw_text: str) -> List[Dict[str, Any]]:
        """Original parser preserved for back-compat (no normalisation)."""
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict) and "tasks" in parsed:
                return parsed["tasks"]
        except json.JSONDecodeError:
            pass
        json_match = re.search(r"```(?:json)?\s*(\[[\s\S]*?\])\s*```", raw_text)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding any JSON array in the text
        bracket_match = re.search(r"\[[\s\S]*\]", raw_text)
        if bracket_match:
            try:
                return json.loads(bracket_match.group(0))
            except json.JSONDecodeError:
                pass

        # Last resort: fall back to the existing regex parser
        return self._parse_tasks(raw_text)
            
    def process(self, document_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a document to extract key information and generate tasks.
        
        Args:
            document_data: Dictionary containing document text, metadata, and context.
            
        Returns:
            Dictionary with extracted information, tasks, and insights.
        """
        # Extract document content and metadata
        document_text = document_data.get("text", "")
        document_metadata = document_data.get("metadata", {})
        user_context = document_data.get("user_context", {})
        
        if not document_text:
            return {"error": "No document text provided"}
            
        # Use the configured model to process the document
        prompt = (
            f"Analyze the following document and extract key information including "
            f"{', '.join(self.extraction_targets)}. For each extracted item, include "
            f"the exact text from the document and the relevant context."
        )
        
        try:
            # Get basic extraction first
            extraction_result = self._call_model(
                prompt=prompt, 
                document=document_text,
                model=self.model_config.get("model")
            )
            
            # Generate tasks separately for better specialization
            tasks_prompt = (
                f"Extract all tasks, deadlines, and action items from this document. "
                f"Format them as a list where each task includes a title, priority (High/Medium/Low), "
                f"deadline (if available), and responsible party (if mentioned). "
                f"Only include actionable items that require someone to do something."
            )
            
            # This could run in parallel with extraction in a real implementation
            tasks_result = self._call_model(
                prompt=tasks_prompt,
                document=document_text,
                model=self.model_config.get("model")
            )
            
            # Parse tasks into structured format
            parsed_tasks = self._parse_tasks(tasks_result)
            
            # Create comprehensive response
            result = {
                "document_id": document_metadata.get("id", ""),
                "extracted_info": self._parse_extraction(extraction_result),
                "tasks": parsed_tasks,
                "processed_at": datetime.utcnow().isoformat(),
                "model_used": self.model_config.get("model"),
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing document: {str(e)}")
            return {"error": f"Document processing failed: {str(e)}"}

    def _parse_extraction(self, extraction_text: str) -> Dict[str, Any]:
        """Parse the extraction results into a structured format.
        
        Args:
            extraction_text: Raw text from the model extraction
            
        Returns:
            Structured dictionary with categorized extractions
        """
        try:
            # Try to parse as JSON first (in case model returned JSON)
            try:
                return json.loads(extraction_text)
            except json.JSONDecodeError:
                # If not JSON, use regex-based parsing
                pass
            
            # Initialize categories
            extraction_dict = {
                "dates": [],
                "names": [],
                "organizations": [],
                "monetary_amounts": [],
                "action_items": [],
                "key_points": [],
                "raw_extraction": extraction_text
            }
            
            # Extract dates with regex
            date_patterns = [
                r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",  # MM/DD/YYYY or DD/MM/YYYY
                r"\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b",  # 1 Jan 2024
                r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{2,4})\b",  # January 1st, 2024
            ]
            
            for pattern in date_patterns:
                for match in re.finditer(pattern, extraction_text, re.IGNORECASE):
                    if match.group(1) not in [d["text"] for d in extraction_dict["dates"]]:
                        extraction_dict["dates"].append({"text": match.group(1)})
            
            # Use section-based parsing for the rest
            sections = {
                "Names:": "names",
                "Organizations:": "organizations",
                "Monetary Amounts:": "monetary_amounts",
                "Action Items:": "action_items",
                "Key Points:": "key_points",
                "Tasks:": "action_items"
            }
            
            lines = extraction_text.split('\n')
            current_section = None
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                    
                # Check if this line is a section header
                for header, section_name in sections.items():
                    if line.startswith(header) or line == header:
                        current_section = section_name
                        break
                
                # If we're in a section and this isn't a header, it's an item
                if current_section and not any(line.startswith(h) for h in sections.keys()):
                    # Clean up bullet points and other markers
                    item_text = re.sub(r"^[-•*]\s*", "", line)
                    if item_text and len(item_text) > 3:  # Avoid very short items
                        if item_text not in [i["text"] for i in extraction_dict[current_section]]:
                            extraction_dict[current_section].append({"text": item_text})
            
            return extraction_dict
            
        except Exception as e:
            logger.error(f"Error parsing extraction: {str(e)}")
            return {"raw_extraction": extraction_text, "parsing_error": str(e)}

    def _parse_tasks(self, tasks_text: str) -> List[Dict[str, Any]]:
        """Parse the tasks extraction into a structured list.
        
        Args:
            tasks_text: Raw text from the model's task extraction
            
        Returns:
            List of structured task dictionaries
        """
        try:
            # Try to parse as JSON first
            try:
                parsed_json = json.loads(tasks_text)
                if isinstance(parsed_json, list):
                    return parsed_json
                elif isinstance(parsed_json, dict) and "tasks" in parsed_json:
                    return parsed_json["tasks"]
            except json.JSONDecodeError:
                # If not JSON, use regex-based parsing
                pass
                
            # Parse free-text format
            tasks = []
            current_task = {}
            
            # Detect bullet points or numbered items
            tasks_pattern = r"(?:^|\n)(?:\d+[\.\)]\s*|[-•*]\s*|Task\s*\d+:\s*|)([A-Z].*?)(?:\n|$)"
            
            for match in re.finditer(tasks_pattern, tasks_text, re.MULTILINE):
                task_text = match.group(1).strip()
                
                # Skip empty or very short tasks
                if not task_text or len(task_text) < 5:
                    continue
                    
                task = {"title": task_text}
                
                # Extract priority if present
                priority_match = re.search(r"\b(High|Medium|Low)\s+priority\b", task_text, re.IGNORECASE)
                if priority_match:
                    task["priority"] = priority_match.group(1).lower()
                
                # Extract deadline if present
                deadline_patterns = [
                    r"due\s+by\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                    r"due\s+on\s+(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
                    r"deadline:\s+(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
                    r"by\s+(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})",
                ]
                
                for pattern in deadline_patterns:
                    deadline_match = re.search(pattern, task_text, re.IGNORECASE)
                    if deadline_match:
                        task["deadline"] = deadline_match.group(1)
                        break
                
                # Extract responsible party if present
                assignee_patterns = [
                    r"assigned\s+to\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                    r"responsible:\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
                ]
                
                for pattern in assignee_patterns:
                    assignee_match = re.search(pattern, task_text, re.IGNORECASE)
                    if assignee_match:
                        task["assignee"] = assignee_match.group(1)
                        break
                
                tasks.append(task)
            
            return tasks
            
        except Exception as e:
            logger.error(f"Error parsing tasks: {str(e)}")
            return [{"title": tasks_text, "parsing_error": str(e)}]
