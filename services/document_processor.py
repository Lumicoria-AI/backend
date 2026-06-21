"""
Document Processing Pipeline for Lumicoria.ai RAG

This module handles the processing of documents for the RAG system, 
including chunking, embedding, and storage in the vector database.
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, Tuple, Union
from pathlib import Path
import structlog
import asyncio
import uuid
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import httpx
from pydantic import BaseModel, Field

# Import text processing utilities
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    TextLoader, 
    PyPDFLoader, 
    Docx2txtLoader,
    CSVLoader,
    UnstructuredHTMLLoader,
    UnstructuredMarkdownLoader,
    UnstructuredEmailLoader
)
from langchain_core.documents import Document as LangchainDocument

# PyMuPDF for position-aware PDF extraction (optional — graceful fallback)
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False

from ..db.vector_stores import get_vector_store
from ..core.config import settings
from ..ai_models import get_embedding_client, LLMClient
from .ingest import (
    ParsedBlock,
    ParsedDocument,
    chunk_document,
    get_parser,
)

logger = structlog.get_logger(__name__)


def _extract_pdf_page_range(
    file_path: str, start_page: int, end_page: int
) -> List[Dict[str, Any]]:
    """Worker for ProcessPoolExecutor: extract positional blocks from a PDF
    page range [start_page, end_page). Runs in a child process — imports fitz
    locally so the parent process doesn't need it loaded when the pool idles.

    Returns plain dicts (cheap to pickle); the caller reconstructs
    LangchainDocument objects.
    """
    import fitz  # imported in-process to keep pickling trivial

    results: List[Dict[str, Any]] = []
    doc = fitz.open(file_path)
    try:
        for page_idx in range(start_page, min(end_page, len(doc))):
            page = doc[page_idx]
            page_dict = page.get_text("dict", sort=True)
            page_width = page_dict.get("width", page.rect.width)
            page_height = page_dict.get("height", page.rect.height)

            blocks = page_dict.get("blocks", [])
            char_offset = 0

            for block_idx, block in enumerate(blocks):
                if block.get("type", 0) != 0:
                    continue

                parts: List[str] = []
                for line in block.get("lines", []):
                    line_text = "".join(
                        span.get("text", "") for span in line.get("spans", [])
                    )
                    parts.append(line_text)

                block_text = "\n".join(parts).strip()
                if not block_text:
                    continue

                bbox = block.get("bbox", [0, 0, page_width, page_height])
                results.append({
                    "page_content": block_text,
                    "metadata": {
                        "page_number": page_idx + 1,
                        "block_index": block_idx,
                        "start_char": char_offset,
                        "end_char": char_offset + len(block_text),
                        "bbox": list(bbox),
                        "page_width": page_width,
                        "page_height": page_height,
                    },
                })
                char_offset += len(block_text) + 1
    finally:
        doc.close()
    return results


_pdf_process_pool: Optional[ProcessPoolExecutor] = None


def _get_pdf_process_pool() -> Optional[ProcessPoolExecutor]:
    """Lazily build a shared ProcessPoolExecutor for PDF extraction.
    Returns None when workers are set to 1 (disabled)."""
    global _pdf_process_pool
    workers = max(1, int(getattr(settings, "INGEST_PROCESS_POOL_WORKERS", 4)))
    if workers <= 1:
        return None
    if _pdf_process_pool is None:
        _pdf_process_pool = ProcessPoolExecutor(max_workers=workers)
    return _pdf_process_pool


def _langchain_to_parsed(
    documents: List[LangchainDocument], metadata: Dict[str, Any]
) -> ParsedDocument:
    """Adapter: wrap loaded LangchainDocuments as a ParsedDocument so the new
    chunker can process them. Each input doc becomes one paragraph block,
    preserving any page_number / bbox / page_width / page_height metadata.
    """
    blocks: List[ParsedBlock] = []
    for i, ld in enumerate(documents):
        md = ld.metadata or {}
        bbox = md.get("bbox")
        if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
            bbox_tuple = tuple(float(x) for x in bbox)  # type: ignore[arg-type]
        else:
            bbox_tuple = None
        blocks.append(ParsedBlock(
            type="paragraph",
            text=ld.page_content,
            page_number=md.get("page_number"),
            bbox=bbox_tuple,
            page_width=md.get("page_width"),
            page_height=md.get("page_height"),
            order=i,
        ))
    mime = metadata.get("mime_type", "")
    if mime == "application/pdf":
        source_type = "pdf"
    elif mime.startswith("text/html"):
        source_type = "html"
    elif mime == "text/markdown":
        source_type = "markdown"
    elif metadata.get("source") in {"chat_history", "chat"}:
        source_type = "chat"
    else:
        source_type = "text"
    return ParsedDocument(
        blocks=blocks,
        metadata={},  # caller-supplied metadata is passed as user_metadata to the chunker
        source_type=source_type,
        title=metadata.get("title"),
    )


class DocumentChunk(BaseModel):
    """A single document chunk for processing."""
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class PositionMetadata(BaseModel):
    """Position data for a text chunk — enables citation linking."""
    page_number: int = 0
    block_index: int = 0
    start_char: int = 0
    end_char: int = 0
    bbox: Optional[List[float]] = None  # [x0, y0, x1, y1]
    page_width: Optional[float] = None
    page_height: Optional[float] = None

class ProcessedDocument(BaseModel):
    """Result of document processing."""
    document_id: str
    chunk_count: int
    metadata: Dict[str, Any]
    vector_ids: List[str]
    status: str = "success"
    error: Optional[str] = None

class DocumentProcessor:
    """
    Process documents for RAG by chunking, embedding, and storing them.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        """Initialize the document processor."""
        self.llm_client = llm_client
        self.loaders = {
            "text/plain": TextLoader,
            "application/pdf": PyPDFLoader,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": Docx2txtLoader,
            "text/csv": CSVLoader,
            "text/html": UnstructuredHTMLLoader,
            "text/markdown": UnstructuredMarkdownLoader,
            "message/rfc822": UnstructuredEmailLoader
        }
        
        # Default chunking parameters
        self.default_chunk_size = 1000
        self.default_chunk_overlap = 100
        
    async def initialize(self):
        """Ensure client is initialized."""
        if not self.llm_client:
            self.llm_client = get_embedding_client()
            
    async def process_file(
        self, 
        file_path: str,
        metadata: Dict[str, Any],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> ProcessedDocument:
        """
        Process a file by loading, chunking, embedding, and storing it.
        
        Args:
            file_path: Path to the file
            metadata: Document metadata including user_id and organization_id
            chunk_size: Size of text chunks (default: 1000 chars)
            chunk_overlap: Overlap between chunks (default: 100 chars)
            
        Returns:
            ProcessedDocument object with processing results
        """
        try:
            # Get document ID or generate a new one
            document_id = metadata.get("document_id", str(uuid.uuid4()))
            metadata["document_id"] = document_id
            
            # Add source information if not provided
            if "source" not in metadata:
                metadata["source"] = "upload"
                
            # Determine MIME type if not provided
            if "mime_type" not in metadata:
                metadata["mime_type"] = self._detect_mime_type(file_path)
            
            # Add file info to metadata
            file_path_obj = Path(file_path)
            metadata["filename"] = metadata.get("filename", file_path_obj.name)
            metadata["title"] = metadata.get("title", file_path_obj.stem)
            metadata["created_at"] = metadata.get("created_at", datetime.utcnow().isoformat())

            # Preferred path: parser factory → ParsedDocument → chunker.
            parser = get_parser(metadata["mime_type"], metadata)
            if parser.name in {"pymupdf", "docling"}:
                parsed = await parser.parse(file_path, metadata)
                return await self._process_parsed_document(parsed, metadata)

            # Fallback: legacy LangChain loaders (keeps CSV / email support
            # unchanged until those parsers are written).
            documents = await self._load_document(file_path, metadata["mime_type"])
            return await self._process_documents(
                documents=documents,
                metadata=metadata,
                chunk_size=chunk_size or self.default_chunk_size,
                chunk_overlap=chunk_overlap or self.default_chunk_overlap,
            )
            
        except Exception as e:
            logger.error("Error processing document", error=str(e), file_path=file_path)
            return ProcessedDocument(
                document_id=metadata.get("document_id", str(uuid.uuid4())),
                chunk_count=0,
                metadata=metadata,
                vector_ids=[],
                status="error",
                error=str(e)
            )
    
    async def process_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> ProcessedDocument:
        """
        Process text directly without a file.
        
        Args:
            text: Text content to process
            metadata: Document metadata
            chunk_size: Size of text chunks
            chunk_overlap: Overlap between chunks
            
        Returns:
            ProcessedDocument object with processing results
        """
        try:
            # Get document ID or generate a new one
            document_id = metadata.get("document_id", str(uuid.uuid4()))
            metadata["document_id"] = document_id
            
            # Add source information if not provided
            if "source" not in metadata:
                metadata["source"] = "direct_text"
                
            # Set default MIME type
            metadata["mime_type"] = metadata.get("mime_type", "text/plain")

            # Add metadata
            metadata["filename"] = metadata.get("filename", f"{document_id}.txt")
            metadata["title"] = metadata.get("title", document_id)
            metadata["created_at"] = metadata.get("created_at", datetime.utcnow().isoformat())

            # Route through the plaintext parser — handles markdown headings,
            # list blocks, and code fences cleanly before chunking.
            parser = get_parser(metadata["mime_type"], metadata)
            parsed = await parser.parse(text, metadata)
            return await self._process_parsed_document(parsed, metadata)

        except Exception as e:
            logger.error("Error processing text", error=str(e))
            return ProcessedDocument(
                document_id=metadata.get("document_id", str(uuid.uuid4())),
                chunk_count=0,
                metadata=metadata,
                vector_ids=[],
                status="error",
                error=str(e)
            )
    
    async def process_chat_history(
        self,
        messages: List[Dict[str, Any]],
        metadata: Dict[str, Any]
    ) -> ProcessedDocument:
        """
        Process chat history for context retention.
        
        Args:
            messages: List of chat messages in {role, content} format
            metadata: Document metadata
            
        Returns:
            ProcessedDocument object with processing results
        """
        try:
            # Format chat history as text
            text = self._format_chat_history(messages)
            
            # Add chat-specific metadata
            metadata["source"] = "chat_history"
            metadata["message_count"] = len(messages)
            
            # Process as text
            return await self.process_text(
                text=text,
                metadata=metadata,
                chunk_size=1500,  # Larger chunks for chat history
                chunk_overlap=150
            )
            
        except Exception as e:
            logger.error("Error processing chat history", error=str(e))
            return ProcessedDocument(
                document_id=metadata.get("document_id", str(uuid.uuid4())),
                chunk_count=0,
                metadata=metadata,
                vector_ids=[],
                status="error",
                error=str(e)
            )
    
    async def _load_document(self, file_path: str, mime_type: str) -> List[LangchainDocument]:
        """Load a document using appropriate loader based on MIME type."""
        try:
            # For PDFs, prefer PyMuPDF for position-aware extraction
            if mime_type == "application/pdf" and HAS_PYMUPDF:
                return await self._load_pdf_with_positions(file_path)

            # Fallback to LangChain loaders
            loader_class = self.loaders.get(mime_type)
            if not loader_class:
                raise ValueError(f"Unsupported MIME type: {mime_type}")

            loader = loader_class(file_path)
            documents = loader.load()
            return documents

        except Exception as e:
            logger.error("Error loading document", error=str(e), file_path=file_path, mime_type=mime_type)
            raise

    async def _load_pdf_with_positions(self, file_path: str) -> List[LangchainDocument]:
        """Load a PDF with position metadata per block.

        Splits the page range across a ProcessPoolExecutor so a 100-page PDF
        parses in parallel across CPU cores. Small PDFs (<= pages_per_worker)
        stay in a single thread to avoid pickling overhead.
        """
        doc = fitz.open(file_path)
        total_pages = len(doc)
        doc.close()

        pages_per_worker = max(1, int(getattr(settings, "INGEST_PDF_PAGES_PER_WORKER", 25)))
        pool = _get_pdf_process_pool()

        # Fast path: small PDFs or pool disabled → run in a worker thread.
        if pool is None or total_pages <= pages_per_worker:
            raw_blocks = await asyncio.to_thread(
                _extract_pdf_page_range, file_path, 0, total_pages
            )
        else:
            loop = asyncio.get_running_loop()
            ranges = [
                (start, min(start + pages_per_worker, total_pages))
                for start in range(0, total_pages, pages_per_worker)
            ]
            futures = [
                loop.run_in_executor(pool, _extract_pdf_page_range, file_path, s, e)
                for s, e in ranges
            ]
            results = await asyncio.gather(*futures)
            # Results come back in dispatch order; flatten preserves page order.
            raw_blocks = [b for chunk in results for b in chunk]

        documents = [
            LangchainDocument(
                page_content=b["page_content"], metadata=b["metadata"]
            )
            for b in raw_blocks
        ]
        logger.info(
            "PDF loaded with positions",
            pages=total_pages,
            blocks=len(documents),
            file=file_path,
            parallel=pool is not None and total_pages > pages_per_worker,
        )
        return documents
    
    def _detect_mime_type(self, file_path: str) -> str:
        """Detect MIME type of a file based on extension."""
        # Simple extension-based detection
        ext = Path(file_path).suffix.lower()
        
        if ext in ['.txt']:
            return "text/plain"
        elif ext in ['.pdf']:
            return "application/pdf"
        elif ext in ['.docx']:
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif ext in ['.csv']:
            return "text/csv"
        elif ext in ['.html', '.htm']:
            return "text/html"
        elif ext in ['.md', '.markdown']:
            return "text/markdown"
        elif ext in ['.eml']:
            return "message/rfc822"
        else:
            # Default to plain text
            return "text/plain"
    
    def _split_documents_sync(
        self,
        documents: List[LangchainDocument],
        metadata: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int,
    ) -> List["DocumentChunk"]:
        """Synchronous chunking — called via asyncio.to_thread so the splitter
        never blocks the event loop on large documents."""
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            length_function=len,
        )

        chunked_docs: List[DocumentChunk] = []
        global_chunk_idx = 0
        for doc in documents:
            # Our metadata takes precedence (e.g. source="upload" over LangChain's file path)
            combined_metadata = {**doc.metadata, **metadata}

            chunks = text_splitter.split_text(doc.page_content)

            search_start = 0
            for chunk_text in chunks:
                chunk_metadata = combined_metadata.copy()
                chunk_metadata["chunk_id"] = global_chunk_idx

                pos = doc.page_content.find(chunk_text, search_start)
                if pos != -1:
                    base_start = combined_metadata.get("start_char", 0)
                    chunk_metadata["start_char"] = base_start + pos
                    chunk_metadata["end_char"] = base_start + pos + len(chunk_text)
                    search_start = pos + len(chunk_text)

                chunked_docs.append(DocumentChunk(
                    text=chunk_text,
                    metadata=chunk_metadata,
                ))
                global_chunk_idx += 1

        return chunked_docs

    async def _process_documents(
        self,
        documents: List[LangchainDocument],
        metadata: Dict[str, Any],
        chunk_size: int,
        chunk_overlap: int
    ) -> ProcessedDocument:
        """Adapter for legacy callers that produce LangchainDocuments.

        Each LangchainDocument becomes one ParsedBlock, preserving any
        bbox / page_number metadata that PyMuPDF / loaders attached. The
        structure-aware chunker then takes over.

        `chunk_size` / `chunk_overlap` are accepted for signature compatibility
        but ignored — the new chunker uses token-based policies from settings.
        """
        parsed = _langchain_to_parsed(documents, metadata)
        return await self._process_parsed_document(parsed, metadata)

    async def _process_parsed_document(
        self,
        parsed: ParsedDocument,
        metadata: Dict[str, Any],
    ) -> ProcessedDocument:
        """Full pipeline: chunker → embedder → vector store.

        Embed(batch N+1) runs concurrently with store(batch N) so the first
        chunks land in the vector store before the last batch has finished
        embedding. Publishes progress events to Redis along the way.
        """
        from .ingest import progress as _progress
        from .ingest import metrics as _metrics

        document_id = metadata["document_id"]
        mime = metadata.get("mime_type")
        ocr_pages = int(parsed.metadata.get("ocr_pages") or 0) if parsed.metadata else 0
        if ocr_pages:
            _metrics.record_ocr_fallback(mime, ocr_pages)

        try:
            _progress.stage(document_id, "chunking")

            with _metrics.record_stage(mime, "chunk"):
                chunks = await asyncio.to_thread(chunk_document, parsed, metadata)

            if not chunks:
                logger.warning("no_chunks_produced", document_id=document_id)
                _progress.stage(document_id, "ready", chunk_count=0)
                return ProcessedDocument(
                    document_id=document_id,
                    chunk_count=0,
                    metadata=metadata,
                    vector_ids=[],
                    status="success",
                )

            _progress.stage(document_id, "embedding", total=len(chunks), processed=0)

            await self.initialize()
            texts = [c.text for c in chunks]
            all_metadata = [c.metadata for c in chunks]
            provider_name = getattr(self.llm_client, "provider_name", "")

            # Local provider: one call (FastEmbed parallelizes internally).
            # API providers: batch with pipelined store.
            vector_ids: List[str] = []
            vector_store = get_vector_store() if settings.db.VECTOR_STORE_ENABLED else None

            if provider_name == "local":
                with _metrics.record_stage(mime, "embed"), _metrics.record_embed(provider_name):
                    all_embeddings = await self.llm_client.generate_embeddings(texts=texts)
                _progress.stage(document_id, "embedding", total=len(chunks), processed=len(chunks))

                if vector_store is not None:
                    _progress.stage(document_id, "storing", total=len(chunks), processed=0)
                    with _metrics.record_stage(mime, "store"):
                        vector_ids = await vector_store.add_documents(
                            texts=texts, embeddings=all_embeddings, metadatas=all_metadata,
                        )
                    _progress.stage(document_id, "storing", total=len(chunks), processed=len(chunks))
            else:
                batch_size = 20
                total = len(chunks)
                # Pipeline: kick off store(batch N-1) while embed(batch N) runs.
                pending_store: Optional[asyncio.Task] = None
                processed = 0

                for i in range(0, total, batch_size):
                    embed_slice = slice(i, i + batch_size)
                    with _metrics.record_embed(provider_name):
                        batch_embeddings = await self.llm_client.generate_embeddings(
                            texts=texts[embed_slice]
                        )
                    processed += len(batch_embeddings)
                    _progress.stage(
                        document_id, "embedding",
                        total=total, processed=processed,
                    )

                    if vector_store is not None:
                        if pending_store is not None:
                            ids = await pending_store
                            vector_ids.extend(ids)
                            _progress.stage(
                                document_id, "storing",
                                total=total, processed=len(vector_ids),
                            )
                        pending_store = asyncio.create_task(vector_store.add_documents(
                            texts=texts[embed_slice],
                            embeddings=batch_embeddings,
                            metadatas=all_metadata[embed_slice],
                        ))

                    if i + batch_size < total:
                        await asyncio.sleep(0.5)

                if pending_store is not None:
                    ids = await pending_store
                    vector_ids.extend(ids)
                    _progress.stage(
                        document_id, "storing",
                        total=total, processed=len(vector_ids),
                    )

            _progress.stage(document_id, "ready", chunk_count=len(chunks))

            return ProcessedDocument(
                document_id=document_id,
                chunk_count=len(chunks),
                metadata=metadata,
                vector_ids=vector_ids,
                status="success",
            )
        except Exception as e:
            logger.error("Error processing documents", error=str(e))
            _progress.stage(document_id, "error", message=str(e))
            raise
    
    def _format_chat_history(self, messages: List[Dict[str, Any]]) -> str:
        """Format chat history as text for embedding."""
        formatted = []
        
        for msg in messages:
            role = msg.get("role", "user").upper()
            content = msg.get("content", "")
            formatted.append(f"{role}: {content}")
            
        return "\n\n".join(formatted)
    
    def _extract_title_from_html(self, html_content: str) -> Optional[str]:
        """Extract title from HTML content."""
        title_match = re.search(r"<title>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
        if title_match:
            return title_match.group(1).strip()
        return None
        
    async def process_google_drive(
        self,
        drive_file_id: str,
        metadata: Dict[str, Any],
        chunk_size: int = None,
        chunk_overlap: int = None,
        *,
        drive_client: Optional[Any] = None,
        user_id: Optional[str] = None,
    ) -> ProcessedDocument:
        """Process a document from Google Drive end-to-end.

        Args:
            drive_file_id: Google Drive file ID
            metadata: Document metadata. Should include `user_id` if no
                `drive_client` is passed — we look up the user's
                Google integration to build a client.
            chunk_size: Override chunker size.
            chunk_overlap: Override chunker overlap.
            drive_client: Optional pre-built `GoogleWorkspaceClient`.
                The brain pipeline reuses a single client across many
                files in one run; ad-hoc callers can omit this and the
                method will build one from the user's stored credentials.
            user_id: Override for the user whose Google integration to
                use. Defaults to `metadata['user_id']`.

        Returns:
            ProcessedDocument with chunk_count, vector_ids, and status.
        """
        try:
            document_id = metadata.get("document_id", str(uuid.uuid4()))
            metadata["document_id"] = document_id
            metadata.setdefault("source", "drive")
            metadata["drive_file_id"] = drive_file_id
            metadata["created_at"] = metadata.get("created_at", datetime.utcnow().isoformat())

            # Resolve a Drive client. The brain pipeline passes one in
            # explicitly to amortise OAuth refresh across many files.
            if drive_client is None:
                resolved_user_id = user_id or metadata.get("user_id")
                if not resolved_user_id:
                    raise ValueError(
                        "process_google_drive needs a user_id (in metadata or kwarg) "
                        "or a pre-built drive_client",
                    )
                from backend.services.integration_service import integration_service
                integ = await integration_service.get_user_integration(
                    str(resolved_user_id), provider="google_workspace",
                )
                if not integ or not integ.get("credentials"):
                    raise ValueError(
                        f"User {resolved_user_id} has no active Google Workspace integration",
                    )
                from backend.services.ai_clients.google_workspace_client import (
                    GoogleWorkspaceClient,
                )
                drive_client = GoogleWorkspaceClient(integ["credentials"])

            # Download via the typed Drive helper. Native Google docs
            # get exported to a portable MIME automatically.
            payload = await drive_client.download_drive_file(drive_file_id)
            if payload is None:
                logger.warning("drive.file_missing", drive_file_id=drive_file_id)
                return ProcessedDocument(
                    document_id=document_id,
                    chunk_count=0,
                    metadata=metadata,
                    vector_ids=[],
                    status="error",
                    error="Drive file not found",
                )

            file_bytes: bytes = payload["bytes"]
            file_name: str = payload.get("name") or f"{drive_file_id}.bin"
            file_mime: str = payload.get("mime_type") or "application/octet-stream"

            metadata.setdefault("filename", file_name)
            metadata.setdefault("title", file_name)
            metadata["mime_type"] = file_mime
            metadata.setdefault("file_size", payload.get("size", len(file_bytes)))
            if "original_mime_type" in payload:
                metadata["drive_native_mime_type"] = payload["original_mime_type"]

            # Persist to a temp file so the same parser used for uploads
            # handles the bytes. Cleaned up in `finally`.
            import tempfile as _tempfile
            tmp_path: Optional[str] = None
            try:
                # Pick a sensible suffix so PyMuPDF / Docling can sniff.
                _SUFFIX_FROM_MIME = {
                    "application/pdf": ".pdf",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
                    "application/vnd.openxmlformats-officedocument.presentationml.presentation": ".pptx",
                    "text/plain": ".txt",
                    "text/markdown": ".md",
                    "text/csv": ".csv",
                    "image/png": ".png",
                    "image/jpeg": ".jpg",
                }
                suffix = (
                    _SUFFIX_FROM_MIME.get(file_mime)
                    or os.path.splitext(file_name)[1]
                    or ".bin"
                )
                with _tempfile.NamedTemporaryFile(
                    suffix=suffix, delete=False,
                ) as tmp:
                    tmp.write(file_bytes)
                    tmp_path = tmp.name

                return await self.process_file(
                    file_path=tmp_path,
                    metadata=metadata,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                )
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass

        except Exception as e:
            logger.error(
                "Error processing Google Drive document",
                error=str(e), drive_file_id=drive_file_id,
            )
            return ProcessedDocument(
                document_id=metadata.get("document_id", str(uuid.uuid4())),
                chunk_count=0,
                metadata=metadata,
                vector_ids=[],
                status="error",
                error=str(e),
            )
    
    async def process_chat_history(
        self,
        messages: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        chunk_size: int = None,
        chunk_overlap: int = None
    ) -> ProcessedDocument:
        """
        Process chat history for context storage.
        
        Args:
            messages: List of chat messages
            metadata: Document metadata
            chunk_size: Size of text chunks (default: 1000 chars)
            chunk_overlap: Overlap between chunks (default: 100 chars)
            
        Returns:
            ProcessedDocument object with processing results
        """
        try:
            # Format chat history as text
            text = self._format_chat_history(messages)
            
            # Set document ID if not provided
            if "document_id" not in metadata and "conversation_id" in metadata:
                metadata["document_id"] = f"chat_{metadata['conversation_id']}"
            
            # Make sure source is set
            if "source" not in metadata:
                metadata["source"] = "chat_history"
                
            # Process the text
            return await self.process_text(
                text=text,
                metadata=metadata,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
            
        except Exception as e:
            logger.error("Error processing chat history", error=str(e))
            return ProcessedDocument(
                document_id=metadata.get("document_id", str(uuid.uuid4())),
                chunk_count=0,
                metadata=metadata,
                vector_ids=[],
                status="error",
                error=str(e)
            )
    
    async def process_url(
        self,
        url: str,
        metadata: Dict[str, Any],
        chunk_size: int = None,
        chunk_overlap: int = None,
    ) -> ProcessedDocument:
        """Fetch a URL and chunk its main content.

        Uses TrafilaturaParser when available (discards nav/ads/script/style);
        falls back to a safe regex tag-strip otherwise. `chunk_size` /
        `chunk_overlap` are accepted for signature compatibility but ignored —
        the structure-aware chunker picks policy from settings.
        """
        try:
            document_id = metadata.get("document_id", str(uuid.uuid4()))
            metadata["document_id"] = document_id
            metadata["url"] = url
            metadata.setdefault("source", "web")
            metadata.setdefault("mime_type", "text/html")
            metadata.setdefault("created_at", datetime.utcnow().isoformat())

            parser = get_parser("text/html", metadata)
            parsed = await parser.parse(url, metadata)

            # Persist the parser-resolved title back onto metadata so the
            # registry row + vector store share the same value.
            if parsed.title:
                metadata["title"] = metadata.get("title") or parsed.title

            return await self._process_parsed_document(parsed, metadata)

        except Exception as e:
            logger.error("Error processing URL", error=str(e), url=url)
            return ProcessedDocument(
                document_id=metadata.get("document_id", str(uuid.uuid4())),
                chunk_count=0,
                metadata=metadata,
                vector_ids=[],
                status="error",
                error=str(e),
            )

# Create a singleton instance
document_processor = DocumentProcessor()
