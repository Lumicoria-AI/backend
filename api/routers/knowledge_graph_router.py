from typing import Dict, Any, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from datetime import datetime
import logging

from agents.knowledge_graph_agent import KnowledgeGraphAgent, GraphNodeType, GraphRelationType
from api.dependencies import get_agent_service

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/knowledge-graph",
    tags=["knowledge-graph"],
    responses={404: {"description": "Not found"}},
)

# Request/Response Models
class KnowledgeExtractionRequest(BaseModel):
    """Request model for knowledge extraction."""
    content: str = Field(..., description="Content to extract knowledge from")
    source: Dict[str, Any] = Field(default_factory=dict, description="Source information")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for extraction")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Extraction parameters")

class RelationDiscoveryRequest(BaseModel):
    """Request model for relation discovery."""
    focus: List[str] = Field(..., description="Focus areas for discovery")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for discovery")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Discovery constraints")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Discovery parameters")

class GapFillingRequest(BaseModel):
    """Request model for gap filling."""
    focus: List[str] = Field(..., description="Focus areas for gap filling")
    context: Dict[str, Any] = Field(default_factory=dict, description="Context for gap filling")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Gap filling parameters")

class GraphQueryRequest(BaseModel):
    """Request model for graph queries."""
    query_type: str = Field(..., description="Type of query (path, neighbors, search, subgraph)")
    query: Dict[str, Any] = Field(..., description="Query parameters")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Query parameters")

class VisualizationRequest(BaseModel):
    """Request model for graph visualization."""
    focus: List[str] = Field(default_factory=list, description="Focus areas for visualization")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Visualization parameters")

# API Endpoints
@router.post("/extract")
async def extract_knowledge(
    request: KnowledgeExtractionRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Extract knowledge from content and add to graph."""
    try:
        agent = agent_service.get_agent("knowledge_graph")
        if not isinstance(agent, KnowledgeGraphAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "extract",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error extracting knowledge: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/discover-relations")
async def discover_relations(
    request: RelationDiscoveryRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Discover new relationships between nodes."""
    try:
        agent = agent_service.get_agent("knowledge_graph")
        if not isinstance(agent, KnowledgeGraphAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "discover_relations",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error discovering relations: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/fill-gaps")
async def fill_knowledge_gaps(
    request: GapFillingRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Fill gaps in the knowledge graph."""
    try:
        agent = agent_service.get_agent("knowledge_graph")
        if not isinstance(agent, KnowledgeGraphAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "fill_gaps",
            "data": request.dict(),
            "context": request.context,
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error filling knowledge gaps: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query")
async def query_graph(
    request: GraphQueryRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Query the knowledge graph."""
    try:
        agent = agent_service.get_agent("knowledge_graph")
        if not isinstance(agent, KnowledgeGraphAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "query",
            "data": request.dict(),
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error querying graph: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/visualize")
async def visualize_graph(
    request: VisualizationRequest,
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Generate visualization data for the graph."""
    try:
        agent = agent_service.get_agent("knowledge_graph")
        if not isinstance(agent, KnowledgeGraphAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        result = await agent.process_async({
            "action": "visualize",
            "data": request.dict(),
            "parameters": request.parameters
        })
        
        return result
    except Exception as e:
        logger.error(f"Error generating visualization: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stats")
async def get_graph_stats(
    agent_service = Depends(get_agent_service)
) -> Dict[str, Any]:
    """Get statistics about the knowledge graph."""
    try:
        agent = agent_service.get_agent("knowledge_graph")
        if not isinstance(agent, KnowledgeGraphAgent):
            raise HTTPException(status_code=500, detail="Invalid agent type")
        
        return agent._get_graph_stats()
    except Exception as e:
        logger.error(f"Error getting graph stats: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/node-types")
async def get_node_types() -> List[str]:
    """Get available node types."""
    return [t.value for t in GraphNodeType]

@router.get("/relation-types")
async def get_relation_types() -> List[str]:
    """Get available relation types."""
    return [t.value for t in GraphRelationType] 