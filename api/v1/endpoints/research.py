"""
Research Agent API — web research, academic, fact-check, source comparison.

Persists all research results to MongoDB for history/stats retrieval.
"""

from typing import Any, List, Optional, Dict
from fastapi import APIRouter, Depends, HTTPException, status, Body, Query
from pydantic import BaseModel, Field
from datetime import datetime
import uuid
import structlog

from backend.api.deps import get_current_active_user
from backend.agents.research_agent import ResearchAgent
from backend.models.user import User
from backend.services.activity_logger import log_activity
from backend.core.config import settings
from backend.db.mongodb.mongodb import MongoDB

logger = structlog.get_logger(__name__)

router = APIRouter()

# ── MongoDB collection ─────────────────────────────────────────────
RESEARCH_COLLECTION = "research_queries"

# ── Pydantic models ────────────────────────────────────────────────

class ResearchContext(BaseModel):
    domain: Optional[str] = Field(None)
    purpose: Optional[str] = Field(None)
    background_knowledge: Optional[str] = Field(None)
    time_scope: Optional[str] = Field(None)
    geographic_scope: Optional[str] = Field(None)
    previous_findings: Optional[str] = Field(None)

class ResearchRequest(BaseModel):
    query: str = Field(..., description="The research query or topic to investigate")
    context: Optional[ResearchContext] = Field(None)
    research_type: str = Field("general")
    depth: str = Field("comprehensive")
    focus_areas: Optional[List[str]] = Field(None)
    model: Optional[str] = Field(None)
    max_sources: Optional[int] = Field(None)

class ResearchResponse(BaseModel):
    id: Optional[str] = None
    findings: Dict[str, Any]
    raw_response: Optional[str] = None
    processed_at: str
    model_used: str
    research_type: str
    query: str
    sub_questions: Optional[List[str]] = None
    citations: Optional[List[Dict[str, Any]]] = None

class ResearchHistoryItem(BaseModel):
    id: str
    query: str
    research_type: str
    depth: str
    sources_count: int
    created_at: str

class ResearchStatsResponse(BaseModel):
    total_researches: int
    total_sources: int
    research_types: Dict[str, int]


# ── MongoDB helpers ────────────────────────────────────────────────

async def _save_research(
    user_id: str,
    query: str,
    research_type: str,
    depth: str,
    result: Dict[str, Any],
) -> str:
    """Persist research result to MongoDB. Returns the document _id."""
    col = await MongoDB.get_collection(RESEARCH_COLLECTION)
    doc_id = str(uuid.uuid4())

    citations = result.get("citations", [])
    sources = result.get("findings", {}).get("sources", [])
    sources_count = len(citations) if citations else len(sources) if sources else 0

    doc = {
        "_id": doc_id,
        "user_id": user_id,
        "query": query,
        "research_type": research_type,
        "depth": depth,
        "findings": result.get("findings", {}),
        "raw_response": result.get("raw_response", ""),
        "model_used": result.get("model_used", ""),
        "citations": citations,
        "sub_questions": result.get("sub_questions", []),
        "sources_count": sources_count,
        "processed_at": result.get("processed_at", datetime.utcnow().isoformat()),
        "created_at": datetime.utcnow().isoformat(),
    }
    await col.insert_one(doc)
    return doc_id


async def _run_research(request: ResearchRequest, current_user: User, activity_type: str) -> Dict[str, Any]:
    """Shared logic for all research endpoints: run agent, save to MongoDB, log activity."""
    research_agent_config = {
        "type": "research",
        "provider": "perplexity",
        "agent_model_config": {
            "model": request.model or settings.PERPLEXITY_MODEL
        },
        "research_depth": request.depth,
        "require_citations": True,
    }

    research_agent = ResearchAgent(research_agent_config)

    research_data = {
        "query": request.query,
        "context": request.context.dict() if request.context else {},
        "research_type": request.research_type,
        "depth": request.depth,
        "focus_areas": request.focus_areas or [],
    }

    result = await research_agent.process_async(research_data)

    if "error" in result:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=result["error"],
        )

    # Save to MongoDB
    doc_id = await _save_research(
        user_id=str(current_user.id),
        query=request.query,
        research_type=request.research_type,
        depth=request.depth,
        result=result,
    )
    result["id"] = doc_id

    await log_activity(
        user_id=str(current_user.id),
        organization_id=getattr(current_user, "organization_id", None),
        activity_type=activity_type,
        details={"query": request.query, "depth": request.depth, "research_type": request.research_type},
        related_resource_type="AGENT",
        agent_name="Research Agent",
    )

    return result


# ── Research endpoints ─────────────────────────────────────────────

@router.post("/query", response_model=ResearchResponse)
async def research_query(
    request: ResearchRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Conduct comprehensive research on a specific query using Perplexity AI."""
    try:
        return await _run_research(request, current_user, "research.query")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error processing research query", error=str(e))
        raise HTTPException(status_code=500, detail=f"Research processing failed: {str(e)}")


@router.post("/topic", response_model=ResearchResponse)
async def research_topic(
    request: ResearchRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Conduct topic-based research using Perplexity AI."""
    try:
        request.research_type = "topic_research"
        return await _run_research(request, current_user, "research.topic")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error researching topic", error=str(e))
        raise HTTPException(status_code=500, detail=f"Topic research failed: {str(e)}")


@router.post("/literature-review", response_model=ResearchResponse)
async def conduct_literature_review(
    request: ResearchRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Conduct a literature review on a specific topic using Perplexity AI."""
    try:
        request.research_type = "literature_review"
        return await _run_research(request, current_user, "research.literature_review")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error conducting literature review", error=str(e))
        raise HTTPException(status_code=500, detail=f"Literature review failed: {str(e)}")


@router.post("/fact-check", response_model=ResearchResponse)
async def fact_check(
    request: ResearchRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Fact-check statements or claims using Perplexity AI."""
    try:
        request.research_type = "fact_checking"
        return await _run_research(request, current_user, "research.fact_check")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error fact-checking", error=str(e))
        raise HTTPException(status_code=500, detail=f"Fact-checking failed: {str(e)}")


@router.post("/compare-sources", response_model=ResearchResponse)
async def compare_sources(
    request: ResearchRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Compare information from multiple sources on a topic using Perplexity AI."""
    try:
        request.research_type = "source_comparison"
        return await _run_research(request, current_user, "research.compare_sources")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error comparing sources", error=str(e))
        raise HTTPException(status_code=500, detail=f"Source comparison failed: {str(e)}")


@router.post("/comprehensive", response_model=ResearchResponse)
async def comprehensive_research(
    request: ResearchRequest = Body(...),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Conduct deep, comprehensive research on a topic using Perplexity AI."""
    try:
        request.research_type = "comprehensive"
        request.depth = "deep"
        return await _run_research(request, current_user, "research.comprehensive")
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Error conducting comprehensive research", error=str(e))
        raise HTTPException(status_code=500, detail=f"Comprehensive research failed: {str(e)}")


# ── History / Stats / Delete endpoints ─────────────────────────────

@router.get("/history", response_model=List[ResearchHistoryItem])
async def get_research_history(
    limit: int = Query(default=20, le=50),
    skip: int = Query(default=0, ge=0),
    research_type: Optional[str] = Query(default=None),
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get the current user's research history from MongoDB."""
    try:
        col = await MongoDB.get_collection(RESEARCH_COLLECTION)
        query: Dict[str, Any] = {"user_id": str(current_user.id)}
        if research_type:
            query["research_type"] = research_type

        cursor = col.find(query).sort("created_at", -1).skip(skip).limit(limit)
        docs = await cursor.to_list(length=limit)

        return [
            ResearchHistoryItem(
                id=doc["_id"],
                query=doc.get("query", ""),
                research_type=doc.get("research_type", "general"),
                depth=doc.get("depth", "comprehensive"),
                sources_count=doc.get("sources_count", 0),
                created_at=doc.get("created_at", ""),
            )
            for doc in docs
        ]
    except Exception as e:
        logger.error("research_history_fetch_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch history: {str(e)}")


@router.get("/history/{research_id}")
async def get_research_detail(
    research_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get full details of a specific research query."""
    try:
        col = await MongoDB.get_collection(RESEARCH_COLLECTION)
        doc = await col.find_one({"_id": research_id, "user_id": str(current_user.id)})
        if not doc:
            raise HTTPException(status_code=404, detail="Research not found")
        return doc
    except HTTPException:
        raise
    except Exception as e:
        logger.error("research_detail_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch research: {str(e)}")


@router.get("/stats", response_model=ResearchStatsResponse)
async def get_research_stats(
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Get aggregated stats for the current user's research usage."""
    try:
        col = await MongoDB.get_collection(RESEARCH_COLLECTION)
        user_id = str(current_user.id)

        pipeline = [
            {"$match": {"user_id": user_id}},
            {
                "$group": {
                    "_id": None,
                    "total_researches": {"$sum": 1},
                    "total_sources": {"$sum": "$sources_count"},
                }
            },
        ]
        results = await col.aggregate(pipeline).to_list(length=1)

        # Get per-type counts
        type_pipeline = [
            {"$match": {"user_id": user_id}},
            {"$group": {"_id": "$research_type", "count": {"$sum": 1}}},
        ]
        type_results = await col.aggregate(type_pipeline).to_list(length=20)
        research_types = {r["_id"]: r["count"] for r in type_results if r["_id"]}

        if results:
            r = results[0]
            return ResearchStatsResponse(
                total_researches=r.get("total_researches", 0),
                total_sources=r.get("total_sources", 0),
                research_types=research_types,
            )
        return ResearchStatsResponse(total_researches=0, total_sources=0, research_types={})
    except Exception as e:
        logger.error("research_stats_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to fetch stats: {str(e)}")


@router.delete("/history/{research_id}")
async def delete_research(
    research_id: str,
    current_user: User = Depends(get_current_active_user),
) -> Any:
    """Delete a specific research from history."""
    try:
        col = await MongoDB.get_collection(RESEARCH_COLLECTION)
        result = await col.delete_one({"_id": research_id, "user_id": str(current_user.id)})
        if result.deleted_count == 0:
            raise HTTPException(status_code=404, detail="Research not found")
        return {"deleted": True}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("delete_research_failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to delete research: {str(e)}")
