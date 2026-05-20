"""Knowledge Graph endpoints.

All routes are tenant-scoped on `organization_id` and gated by the
platform permission system.  Mutating routes are rate-limited and
audit-logged.  The agent runs through `services.knowledge_graph.orchestrator`
which loads / persists each tenant's graph in Postgres around every
call — never a global singleton.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, Field, field_validator

from backend.agents.knowledge_graph_agent import GraphNodeType, GraphRelationType
from backend.api.deps import get_current_active_user
from backend.core.security import rate_limit
from backend.db.mongodb.repositories.permission_repository import permission_repository
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.services.knowledge_graph import (
    analytics as analytics_svc,
    extractions as extractions_svc,
    orchestrator,
)
from backend.services.knowledge_graph.sanitize import (
    clean_focus_list,
    clean_label,
    clean_text,
)

logger = structlog.get_logger(__name__)

router = APIRouter(
    responses={404: {"description": "Not found"}},
)


# ── Permission helper ───────────────────────────────────────────────


async def _require_kg_permission(current_user: User) -> str:
    """Resolve the tenant scope id, or raise 403 if the caller doesn't
    have AGENT/knowledge_graph/EXECUTE.  Same pattern Customer Service
    and Data Analysis use."""
    user_id = str(current_user.id)
    permission_org = getattr(current_user, "organization_id", None)
    has = await permission_repository.check_permission(
        user_id=user_id,
        organization_id=permission_org,
        resource_type="AGENT",
        resource_id="knowledge_graph",
        permission_type="EXECUTE",
    )
    if not has:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to use the knowledge graph agent",
        )
    # personal accounts: organization_id is None; fall back to user_id
    # for tenant scoping so each personal account gets its own graph.
    return permission_org or user_id


# ── Pydantic models ─────────────────────────────────────────────────


class KnowledgeExtractionRequest(BaseModel):
    """Either `content` OR `rag_document_id` must be set.  Inline
    content has a hard cap of 50k characters; the rag document path
    re-fetches text from object storage."""
    content: Optional[str] = Field(None, max_length=50_000)
    rag_document_id: Optional[str] = Field(None, max_length=64)
    source: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    @classmethod
    def _validate_content(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        return clean_text(v, max_len=50_000)


class RelationDiscoveryRequest(BaseModel):
    focus: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    constraints: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class GapFillingRequest(BaseModel):
    focus: List[str] = Field(default_factory=list)
    context: Dict[str, Any] = Field(default_factory=dict)
    parameters: Dict[str, Any] = Field(default_factory=dict)


class GraphQueryRequest(BaseModel):
    query_type: str = Field(..., min_length=1, max_length=32)
    query: Dict[str, Any] = Field(...)
    parameters: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("query_type")
    @classmethod
    def _validate_query_type(cls, v: str) -> str:
        allowed = {"path", "neighbors", "search", "subgraph"}
        if v not in allowed:
            raise ValueError(f"query_type must be one of {sorted(allowed)}")
        return v


class VisualizationRequest(BaseModel):
    focus: List[str] = Field(default_factory=list)
    parameters: Dict[str, Any] = Field(default_factory=dict)


# ── /extract ───────────────────────────────────────────────────────


async def _load_text_from_rag(rag_document_id: str, user_id: str, org_id: str) -> str:
    """Resolve a RAG document id to its plain text.  Verifies that the
    document belongs to the caller's organization before loading."""
    try:
        from backend.services import rag_document_registry
        from backend.services.storage_service import storage_service
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"Storage unavailable: {e}")

    doc = await rag_document_registry.get(rag_document_id, user_id=user_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    # Belt-and-braces tenant check.  The registry already scopes by
    # user_id, but if a user belongs to an org we ensure the document's
    # org matches the caller's scope.
    doc_org = doc.get("organization_id")
    if doc_org and doc_org != org_id and doc.get("user_id") != org_id:
        raise HTTPException(status_code=404, detail="Document not found")

    try:
        raw = await storage_service.download_file(doc["s3_key"])
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=502, detail=f"Could not fetch document: {e}")

    try:
        text = raw.decode("utf-8", errors="replace")
    except Exception:
        text = str(raw)
    return clean_text(text, max_len=50_000)


@router.post("/extract")
@rate_limit(limit=10, window=900)
async def extract_knowledge(
    request: Request,
    payload: KnowledgeExtractionRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    """Extract nodes / relations from text and persist into the org's graph."""
    org_id = await _require_kg_permission(current_user)
    user_id = str(current_user.id)

    # Resolve content source.
    content = payload.content or ""
    source_kind = "inline" if content else None
    source_ref: Optional[str] = None

    if payload.rag_document_id:
        content = await _load_text_from_rag(payload.rag_document_id, user_id, org_id)
        source_kind = "rag_document"
        source_ref = payload.rag_document_id

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Either `content` or `rag_document_id` must be provided.",
        )

    # The agent expects its data dict to carry `content` regardless of
    # source.  We populate source metadata for the LLM context.
    agent_request = {
        "action": "extract",
        "data": {
            "content": content,
            "source": {**(payload.source or {}), "kind": source_kind, "ref": source_ref},
            "metadata": payload.metadata or {},
        },
        "context": payload.context or {},
        "parameters": payload.parameters or {},
    }

    title = (
        clean_label(payload.metadata.get("title"))
        or clean_label((payload.source or {}).get("title"))
        or (f"Extraction from document {source_ref[:8]}" if source_ref else None)
        or "Knowledge extraction"
    )

    result = await orchestrator.run(
        organization_id=org_id,
        user_id=user_id,
        request=agent_request,
        extraction_title=title,
        source_kind=source_kind,
        source_ref=source_ref,
        content_preview=content[:240],
    )

    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="knowledge_graph.extracted",
        details={
            "source_kind": source_kind,
            "added_nodes": (result.get("metadata") or {}).get("added_node_count"),
            "added_edges": (result.get("metadata") or {}).get("added_edge_count"),
            "extraction_id": (result.get("metadata") or {}).get("extraction_id"),
        },
        agent_name="Knowledge Graph Agent",
    )
    return result


@router.post("/discover-relations")
@rate_limit(limit=10, window=900)
async def discover_relations(
    request: Request,
    payload: RelationDiscoveryRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    org_id = await _require_kg_permission(current_user)
    user_id = str(current_user.id)

    focus = clean_focus_list(payload.focus or [])
    agent_request = {
        "action": "discover_relations",
        "data": {
            "focus": focus,
            "context": payload.context or {},
            "constraints": payload.constraints or {},
        },
        "context": payload.context or {},
        "parameters": payload.parameters or {},
    }
    result = await orchestrator.run(
        organization_id=org_id,
        user_id=user_id,
        request=agent_request,
        extraction_title=("Relations: " + ", ".join(focus[:3])) if focus else "Relation discovery",
        source_kind="inline",
        content_preview=", ".join(focus[:5]) if focus else None,
    )
    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="knowledge_graph.discovered_relations",
        details={
            "focus": focus,
            "added_edges": (result.get("metadata") or {}).get("added_edge_count"),
            "extraction_id": (result.get("metadata") or {}).get("extraction_id"),
        },
        agent_name="Knowledge Graph Agent",
    )
    return result


@router.post("/fill-gaps")
@rate_limit(limit=5, window=900)
async def fill_knowledge_gaps(
    request: Request,
    payload: GapFillingRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    org_id = await _require_kg_permission(current_user)
    user_id = str(current_user.id)

    focus = clean_focus_list(payload.focus or [])
    agent_request = {
        "action": "fill_gaps",
        "data": {
            "focus": focus,
            "context": payload.context or {},
        },
        "context": payload.context or {},
        "parameters": payload.parameters or {},
    }
    result = await orchestrator.run(
        organization_id=org_id,
        user_id=user_id,
        request=agent_request,
        extraction_title=("Fill gaps: " + ", ".join(focus[:3])) if focus else "Gap filling",
        source_kind="inline",
        content_preview=", ".join(focus[:5]) if focus else None,
    )
    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="knowledge_graph.filled_gaps",
        details={
            "focus": focus,
            "added_nodes": (result.get("metadata") or {}).get("added_node_count"),
            "added_edges": (result.get("metadata") or {}).get("added_edge_count"),
            "extraction_id": (result.get("metadata") or {}).get("extraction_id"),
        },
        agent_name="Knowledge Graph Agent",
    )
    return result


@router.post("/query")
@rate_limit(limit=60, window=900)
async def query_graph(
    request: Request,
    payload: GraphQueryRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    org_id = await _require_kg_permission(current_user)
    user_id = str(current_user.id)

    agent_request = {
        "action": "query",
        "data": {
            "query_type": payload.query_type,
            "query": payload.query,
        },
        "parameters": payload.parameters or {},
    }
    return await orchestrator.run(
        organization_id=org_id,
        user_id=user_id,
        request=agent_request,
    )


@router.post("/visualize")
@rate_limit(limit=60, window=900)
async def visualize_graph(
    request: Request,
    payload: VisualizationRequest,
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    org_id = await _require_kg_permission(current_user)
    user_id = str(current_user.id)

    focus = clean_focus_list(payload.focus or [])
    agent_request = {
        "action": "visualize",
        "data": {"focus": focus},
        "parameters": payload.parameters or {},
    }
    return await orchestrator.run(
        organization_id=org_id,
        user_id=user_id,
        request=agent_request,
    )


@router.get("/stats")
async def get_graph_stats(
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    org_id = await _require_kg_permission(current_user)
    return await analytics_svc.get_stats(org_id)


@router.get("/node-types", response_model=List[str])
async def get_node_types() -> List[str]:
    return [t.value for t in GraphNodeType]


@router.get("/relation-types", response_model=List[str])
async def get_relation_types() -> List[str]:
    return [t.value for t in GraphRelationType]


@router.get("/extractions")
async def list_kg_extractions(
    action: Optional[str] = Query(None, max_length=32),
    time_range: Optional[str] = Query(None, pattern="^(1d|7d|30d|90d|1y)$"),
    limit: int = Query(30, ge=1, le=200),
    offset: int = Query(0, ge=0),
    current_user: User = Depends(get_current_active_user),
) -> Dict[str, Any]:
    org_id = await _require_kg_permission(current_user)
    return await extractions_svc.list_extractions(
        org_id,
        action=action,
        time_range=time_range,
        limit=limit,
        offset=offset,
    )


@router.delete("/extractions/{extraction_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_kg_extraction(
    extraction_id: str,
    current_user: User = Depends(get_current_active_user),
) -> None:
    org_id = await _require_kg_permission(current_user)
    user_id = str(current_user.id)
    ok = await extractions_svc.soft_delete_extraction(org_id, extraction_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Extraction not found")
    await log_activity(
        user_id=user_id,
        organization_id=org_id,
        activity_type="knowledge_graph.extraction_deleted",
        details={"extraction_id": extraction_id},
        agent_name="Knowledge Graph Agent",
    )
