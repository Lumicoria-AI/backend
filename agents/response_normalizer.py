"""
Response Normalizer — Extracts a clean text response from any agent's output dict.

Each agent returns a different dict shape. This module provides a single
function that reliably extracts the main text content.
"""

import structlog
from typing import Dict, Any, List

logger = structlog.get_logger(__name__)

# Keys to try, in priority order. First match wins.
RESPONSE_KEYS = [
    "response",
    "content",
    "result",
    "answer",
    "analysis",
    "generated_content",
    "research_findings",
    "model_response",
    "summary",
    "text",
    "output",
    "translation",
    "translated_text",
]

# Keys that typically contain source/metadata info
SOURCE_KEYS = ["sources", "citations", "references"]
METADATA_KEYS = ["confidence", "sentiment", "score", "processing_time"]


def normalize_agent_response(
    raw: Dict[str, Any],
    agent_key: str = "unknown",
) -> Dict[str, Any]:
    """
    Normalize any agent's response dict into a consistent shape.
    
    Returns:
        {
            "response": str,          # The main text content
            "sources": List[dict],    # Source citations if any
            "metadata": dict,         # Extra metadata
            "success": bool,
            "context_used": int,
        }
    """
    # Check for error first
    if "error" in raw and raw.get("error"):
        return {
            "response": f"I encountered an issue: {raw['error']}",
            "sources": [],
            "metadata": {"error": raw["error"]},
            "success": False,
            "context_used": 0,
        }
    
    # Extract main text content
    response_text = None
    for key in RESPONSE_KEYS:
        val = raw.get(key)
        if val and isinstance(val, str) and val.strip():
            response_text = val.strip()
            break
    
    # If no text key found, try to build from nested dicts
    if not response_text:
        for key, val in raw.items():
            if isinstance(val, dict):
                for subkey in RESPONSE_KEYS:
                    subval = val.get(subkey)
                    if subval and isinstance(subval, str) and subval.strip():
                        response_text = subval.strip()
                        break
            if response_text:
                break
    
    # Last resort: stringify the whole thing (but clean it up)
    if not response_text:
        content_dict = {k: v for k, v in raw.items() 
                       if k not in ("success", "error", "timestamp", "processing_time")}
        if content_dict:
            response_text = str(content_dict)
        else:
            response_text = "I processed your request but couldn't generate a text response."
        logger.warning("normalizer_fallback_to_str", agent=agent_key, keys=list(raw.keys()))
    
    # Extract sources
    sources = []
    for key in SOURCE_KEYS:
        if key in raw and isinstance(raw[key], list):
            sources = raw[key]
            break
    
    # Extract metadata
    metadata = {}
    for key in METADATA_KEYS:
        if key in raw:
            metadata[key] = raw[key]
    
    return {
        "response": response_text,
        "sources": sources,
        "metadata": metadata,
        "success": raw.get("success", True),
        "context_used": raw.get("context_used", 0),
    }
