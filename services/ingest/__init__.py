"""Structure-aware ingest pipeline for Lumicoria RAG.

Public surface:
    - ParsedBlock, ParsedDocument, BlockType   (base.py)
    - DocumentParser                            (base.py)
    - get_parser                                (parsers/factory.py)
    - Chunk, chunk_document                     (chunker.py)
"""

from .base import BlockType, DocumentParser, ParsedBlock, ParsedDocument
from .chunker import Chunk, chunk_document
from .parsers.factory import get_parser

__all__ = [
    "BlockType",
    "Chunk",
    "DocumentParser",
    "ParsedBlock",
    "ParsedDocument",
    "chunk_document",
    "get_parser",
]
