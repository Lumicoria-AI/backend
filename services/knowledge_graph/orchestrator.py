"""Service-layer orchestrator for KnowledgeGraphAgent.

The router gives us a request + the caller's `organization_id`.  This
module:
  1. Loads the org's graph from Postgres into a fresh DiGraph.
  2. Attaches it to a per-org agent instance.
  3. Calls the agent's existing `process_async`.
  4. Persists any new nodes / edges back through the repository.
  5. Writes an audit row into `kg_extractions`.

Concurrency: a per-organization asyncio lock prevents two simultaneous
requests on the same org from clobbering each other's in-memory graph.
This is a single-process guarantee; horizontal scaling across workers
still relies on the database to serialize writes via the unique
(org, lower(label), type) index.
"""

from __future__ import annotations

import asyncio
import dataclasses
import time
from enum import Enum
from typing import Any, Dict, Optional, Tuple

import structlog

from .repository import (
    find_node_id_by_label_type,
    load_org_graph,
    upsert_edges,
    upsert_nodes,
)
from .sanitize import clean_label, clean_text
from . import extractions as extractions_svc

logger = structlog.get_logger(__name__)


# Per-organization async locks.  Workers handle low-traffic agents
# fine; the lock keeps within-process correctness if two browser tabs
# fire simultaneously.
_org_locks: Dict[str, asyncio.Lock] = {}


def _lock_for(org_id: str) -> asyncio.Lock:
    if org_id not in _org_locks:
        _org_locks[org_id] = asyncio.Lock()
    return _org_locks[org_id]


def _truncate(text: str, n: int = 240) -> str:
    if not text:
        return ""
    text = text.strip()
    if len(text) <= n:
        return text
    return text[: n - 1].rstrip() + "…"


# ── Agent factory ───────────────────────────────────────────────────


def _build_agent_config() -> Dict[str, Any]:
    """Build the agent config from settings so the LLM provider /
    model match the rest of the platform.  Mirrors the pattern in
    customer_service_tickets.ai_draft_for_ticket."""
    from ...core.config import settings
    provider = (settings.DEFAULT_LLM_PROVIDER or "gemini").lower()
    model = {
        "gemini": getattr(settings, "GEMINI_MODEL", None) or "gemini-2.5-flash",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-haiku-4-5-20251001",
        "mistral": "mistral-small-latest",
        "perplexity": "sonar",
    }.get(provider, "sonar")
    return {
        "provider": provider,
        "model": model,
        # BaseAgent reads from `agent_model_config`, not `model_config`.
        "agent_model_config": {
            "model": model,
            "temperature": 0.2,
            "max_tokens": 4096,
        },
    }


def _new_agent():
    """Instantiate a fresh KnowledgeGraphAgent.  We do not reuse a
    singleton because the agent holds the in-memory graph as state."""
    from ...agents.knowledge_graph_agent import KnowledgeGraphAgent
    return KnowledgeGraphAgent(_build_agent_config())


# ── Result parsing ─────────────────────────────────────────────────


def _to_plain(item: Any) -> Optional[Dict[str, Any]]:
    """Normalize a single node/edge into a plain dict regardless of
    whether the agent returned a dataclass, an Enum-typed field, or an
    already-flat dict.  Returns None for anything we cannot coerce."""
    if item is None:
        return None
    if dataclasses.is_dataclass(item):
        item = dataclasses.asdict(item)
    if not isinstance(item, dict):
        return None
    coerced: Dict[str, Any] = {}
    for k, v in item.items():
        if isinstance(v, Enum):
            coerced[k] = v.value
        else:
            coerced[k] = v
    return coerced


def _extract_added_payload(result: Dict[str, Any], action: str) -> Tuple[list, list]:
    """Pull the (added_nodes, added_edges) lists out of the agent's
    response shape and convert every entry to a plain dict so the
    persistence layer can treat them uniformly.  Different actions
    return slightly different shapes; we normalize them here.
    """
    if not isinstance(result, dict):
        return [], []
    inner = result.get("results") or {}

    raw_nodes: list = []
    raw_edges: list = []

    if action == "extract":
        extracted = inner.get("extracted") or {}
        raw_nodes = list(extracted.get("nodes") or [])
        raw_edges = list(extracted.get("relations") or [])
    elif action == "discover_relations":
        discovered = inner.get("discovered") or []
        raw_edges = list(discovered or [])
    elif action == "fill_gaps":
        filled = inner.get("filled_gaps") or []
        for chunk in filled:
            if dataclasses.is_dataclass(chunk):
                chunk = dataclasses.asdict(chunk)
            if isinstance(chunk, dict):
                raw_nodes.extend(chunk.get("nodes") or [])
                raw_edges.extend(chunk.get("relations") or [])

    nodes = [n for n in (_to_plain(x) for x in raw_nodes) if n]
    edges = [e for e in (_to_plain(x) for x in raw_edges) if e]
    return nodes, edges


# ── Public entry points called by the router ────────────────────────


async def run(
    *,
    organization_id: str,
    user_id: str,
    request: Dict[str, Any],
    extraction_title: Optional[str] = None,
    source_kind: Optional[str] = None,
    source_ref: Optional[str] = None,
    content_preview: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute one extract / discover / fill_gaps / query / visualize
    request against the caller's tenant graph.  Returns the agent's
    result dict augmented with the new extraction id when applicable."""
    if not organization_id:
        raise ValueError("organization_id is required")

    action = (request.get("action") or "extract").lower()
    started = time.perf_counter()

    # Mutating actions get an audit row.  Read-only actions (query,
    # visualize) skip the audit log so we don't fill it with noise.
    extraction: Optional[Dict[str, Any]] = None
    if action in ("extract", "discover_relations", "fill_gaps"):
        action_label = {"discover_relations": "discover"}.get(action, action)
        extraction = await extractions_svc.create_extraction(
            organization_id=organization_id,
            user_id=user_id,
            action=action_label,
            title=extraction_title,
            source_kind=source_kind,
            source_ref=source_ref,
            content_preview=content_preview,
        )

    extraction_id = (extraction or {}).get("id")

    async with _lock_for(organization_id):
        graph = await load_org_graph(organization_id)
        agent = _new_agent()
        agent.attach_graph(graph)

        try:
            result = await agent.process_async(request)
        except Exception as e:  # noqa: BLE001
            logger.error("kg_agent_failed", error=str(e), action=action)
            if extraction_id:
                await extractions_svc.finalize_extraction(
                    organization_id,
                    extraction_id,
                    status="error",
                    error_message=str(e),
                    processing_time_ms=int((time.perf_counter() - started) * 1000),
                )
            raise

        # Persist new nodes / edges to Postgres.  Read-only actions
        # produce empty payloads so this is a no-op for them.
        added_node_ids: list[str] = []
        added_edge_ids: list[str] = []
        if action in ("extract", "discover_relations", "fill_gaps"):
            nodes_to_persist, edges_to_persist = _extract_added_payload(result, action)
            try:
                added_node_ids = await upsert_nodes(
                    organization_id,
                    nodes_to_persist,
                    extraction_id=extraction_id,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("kg_upsert_nodes_failed", error=str(e))

            # The LLM emits edges that reference nodes by label rather
            # than uuid.  Resolve labels to ids before insertion.  We
            # pass `source_type` / `target_type` only when the LLM
            # actually supplied one; otherwise the repository falls
            # back to a label-only match so a mis-typed edge still
            # links instead of dropping silently.
            resolved_edges: list[Dict[str, Any]] = []
            for edge in edges_to_persist:
                if not isinstance(edge, dict):
                    continue
                source_ref_id = edge.get("source_id") or edge.get("source")
                target_ref_id = edge.get("target_id") or edge.get("target")
                if source_ref_id and not _looks_like_uuid(source_ref_id):
                    found = await find_node_id_by_label_type(
                        organization_id,
                        str(source_ref_id),
                        edge.get("source_type"),
                    )
                    if found:
                        edge["source_id"] = found
                if target_ref_id and not _looks_like_uuid(target_ref_id):
                    found = await find_node_id_by_label_type(
                        organization_id,
                        str(target_ref_id),
                        edge.get("target_type"),
                    )
                    if found:
                        edge["target_id"] = found
                resolved_edges.append(edge)
            try:
                added_edge_ids = await upsert_edges(
                    organization_id,
                    resolved_edges,
                    extraction_id=extraction_id,
                )
            except Exception as e:  # noqa: BLE001
                logger.warning("kg_upsert_edges_failed", error=str(e))

        # Drop the agent's in-memory graph reference so it doesn't
        # leak across requests if the agent instance is reused.
        agent.attach_graph(None)

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    if extraction_id:
        await extractions_svc.finalize_extraction(
            organization_id,
            extraction_id,
            status="ready",
            node_ids=added_node_ids,
            edge_ids=added_edge_ids,
            processing_time_ms=elapsed_ms,
        )

    # Augment the return so the frontend gets the audit id immediately
    # and can scope the visualization to what this call just produced.
    # Cap the id lists so we never ship a huge payload back to the UI.
    if isinstance(result, dict):
        meta = result.setdefault("metadata", {})
        if extraction_id:
            meta["extraction_id"] = extraction_id
        meta["added_node_count"] = len(added_node_ids)
        meta["added_edge_count"] = len(added_edge_ids)
        meta["added_node_ids"] = added_node_ids[:200]
        meta["added_edge_ids"] = added_edge_ids[:200]
        meta["processing_time_ms"] = elapsed_ms
    return result


def _looks_like_uuid(value: Any) -> bool:
    """Light heuristic: a UUID4 string has 36 chars with hyphens, our
    KG ids use the same format.  Anything else is a label."""
    if not isinstance(value, str):
        return False
    return len(value) == 36 and value.count("-") == 4
