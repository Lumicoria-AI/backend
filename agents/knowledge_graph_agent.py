from typing import Dict, Any, List, Optional, Set
from enum import Enum
import logging
from datetime import datetime
import json
import networkx as nx
from dataclasses import dataclass
from uuid import uuid4

from .base_agent import BaseAgent
# Removing circular import - agent_service already imports knowledge_graph_agent

logger = logging.getLogger(__name__)

class GraphNodeType(str, Enum):
    """Types of nodes in the knowledge graph."""
    CONCEPT = "concept"
    PERSON = "person"
    PROJECT = "project"
    DOCUMENT = "document"
    EVENT = "event"
    ORGANIZATION = "organization"
    LOCATION = "location"
    RESOURCE = "resource"

class GraphRelationType(str, Enum):
    """Types of relationships in the knowledge graph."""
    RELATED_TO = "related_to"
    PART_OF = "part_of"
    CREATED_BY = "created_by"
    MENTIONS = "mentions"
    INFLUENCES = "influences"
    COLLABORATES_WITH = "collaborates_with"
    OCCURS_IN = "occurs_in"
    REFERENCES = "references"
    SIMILAR_TO = "similar_to"
    DEPENDS_ON = "depends_on"

@dataclass
class GraphNode:
    """Represents a node in the knowledge graph."""
    id: str
    type: GraphNodeType
    label: str
    properties: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    confidence: float = 1.0

@dataclass
class GraphRelation:
    """Represents a relationship in the knowledge graph."""
    id: str
    source_id: str
    target_id: str
    type: GraphRelationType
    properties: Dict[str, Any]
    created_at: datetime
    updated_at: datetime
    confidence: float = 1.0

class KnowledgeGraphAgent(BaseAgent):
    """Agent specialized in building and maintaining personal knowledge graphs."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize the Knowledge Graph Agent with specific capabilities."""
        super().__init__(config)
        
        # Set default capabilities
        self.capabilities = {
            "graph_building": True,
            "concept_extraction": True,
            "relation_discovery": True,
            "gap_filling": True,
            "query_answering": True,
            "visualization": True
        }
        
        # Initialize graph
        self.graph = nx.DiGraph()
        
        # Configure model for knowledge graph tasks
        self.model_config.update({
            "temperature": 0.2,  # Lower temperature for more precise extraction
            "max_tokens": 4096,
            "top_p": 0.9,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3
        })
        
        # Load existing graph if available
        self._load_graph()

    async def process_async(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Process a knowledge graph request asynchronously."""
        try:
            action = request.get("action", "extract")
            data = request.get("data", {})
            context = request.get("context", {})
            parameters = request.get("parameters", {})
            
            # Process based on action
            if action == "extract":
                result = await self._extract_knowledge(data, context, parameters)
            elif action == "discover_relations":
                result = await self._discover_relations(data, context, parameters)
            elif action == "fill_gaps":
                result = await self._fill_knowledge_gaps(data, context, parameters)
            elif action == "query":
                result = await self._query_graph(data, context, parameters)
            elif action == "visualize":
                result = await self._generate_visualization(data, context, parameters)
            else:
                raise ValueError(f"Unsupported action: {action}")
            
            # Save graph after modifications
            if action in ["extract", "discover_relations", "fill_gaps"]:
                self._save_graph()
            
            return {
                "results": result,
                "metadata": {
                    "action": action,
                    "timestamp": datetime.utcnow().isoformat(),
                    "graph_stats": self._get_graph_stats(),
                    "parameters": parameters
                }
            }
            
        except Exception as e:
            logger.error(f"Error processing knowledge graph request: {str(e)}")
            return {"error": str(e)}

    async def _extract_knowledge(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract knowledge from input data and add to graph."""
        try:
            # Prepare system prompt for knowledge extraction
            system_prompt = self._create_extraction_prompt(context, parameters)
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "content": data.get("content", ""),
                    "source": data.get("source", {}),
                    "metadata": data.get("metadata", {})
                }),
                parameters
            )
            
            # Parse extracted knowledge
            extracted = self._parse_extraction(response)
            
            # Add to graph
            added_nodes, added_relations = self._add_to_graph(extracted)
            
            return {
                "extracted": extracted,
                "added": {
                    "nodes": [node.id for node in added_nodes],
                    "relations": [rel.id for rel in added_relations]
                }
            }
            
        except Exception as e:
            logger.error(f"Error extracting knowledge: {str(e)}")
            raise

    async def _discover_relations(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Discover new relationships between existing nodes."""
        try:
            # Prepare system prompt for relation discovery
            system_prompt = self._create_discovery_prompt(context, parameters)
            
            # Get relevant nodes
            nodes = self._get_relevant_nodes(data.get("focus", []))
            
            # Process the request
            response = await self._process_with_model(
                system_prompt,
                json.dumps({
                    "nodes": [self._node_to_dict(node) for node in nodes],
                    "context": data.get("context", {}),
                    "constraints": data.get("constraints", {})
                }),
                parameters
            )
            
            # Parse discovered relations
            discovered = self._parse_discovery(response)
            
            # Add to graph
            added_relations = self._add_relations_to_graph(discovered)
            
            return {
                "discovered": discovered,
                "added": [rel.id for rel in added_relations]
            }
            
        except Exception as e:
            logger.error(f"Error discovering relations: {str(e)}")
            raise

    async def _fill_knowledge_gaps(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Fill gaps in the knowledge graph using Perplexity."""
        try:
            # Identify gaps
            gaps = self._identify_gaps(data.get("focus", []))
            
            # Prepare system prompt for gap filling
            system_prompt = self._create_gap_filling_prompt(context, parameters)
            
            # Process each gap
            filled = []
            for gap in gaps:
                response = await self._process_with_model(
                    system_prompt,
                    json.dumps({
                        "gap": gap,
                        "context": data.get("context", {}),
                        "existing_knowledge": self._get_relevant_knowledge(gap)
                    }),
                    parameters
                )
                
                # Parse filled gap
                filled_gap = self._parse_gap_filling(response)
                filled.append(filled_gap)
                
                # Add to graph
                self._add_to_graph(filled_gap)
            
            return {
                "filled_gaps": filled,
                "total_gaps_filled": len(filled)
            }
            
        except Exception as e:
            logger.error(f"Error filling knowledge gaps: {str(e)}")
            raise

    async def _query_graph(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Query the knowledge graph."""
        try:
            query_type = data.get("query_type", "path")
            query = data.get("query", {})
            
            if query_type == "path":
                result = self._find_paths(query)
            elif query_type == "neighbors":
                result = self._find_neighbors(query)
            elif query_type == "search":
                result = self._search_graph(query)
            elif query_type == "subgraph":
                result = self._extract_subgraph(query)
            else:
                raise ValueError(f"Unsupported query type: {query_type}")
            
            return {
                "query_type": query_type,
                "results": result,
                "metadata": {
                    "result_count": len(result),
                    "query_time": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error querying graph: {str(e)}")
            raise

    async def _generate_visualization(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate visualization data for the graph."""
        try:
            # Get subgraph to visualize
            subgraph = self._get_visualization_subgraph(data.get("focus", []))
            
            # Generate visualization data
            viz_data = self._create_visualization_data(subgraph, parameters)
            
            return {
                "visualization": viz_data,
                "metadata": {
                    "node_count": len(subgraph.nodes),
                    "edge_count": len(subgraph.edges),
                    "generated_at": datetime.utcnow().isoformat()
                }
            }
            
        except Exception as e:
            logger.error(f"Error generating visualization: {str(e)}")
            raise

    def _create_extraction_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create system prompt for knowledge extraction."""
        return f"""You are a specialized knowledge extraction AI. Your task is to identify key concepts, entities, and relationships from the input content.
        
        Context:
        - Domain: {context.get('domain', 'general')}
        - Extraction Focus: {context.get('focus', 'all')}
        - Confidence Threshold: {parameters.get('confidence_threshold', 0.7)}
        
        Extract:
        1. Key concepts and entities
        2. Relationships between entities
        3. Properties and attributes
        4. Temporal information
        5. Source citations
        
        Format the output as structured JSON with nodes and relationships.
        Include confidence scores and supporting evidence.
        """

    def _create_discovery_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create system prompt for relation discovery."""
        return f"""You are a specialized relationship discovery AI. Your task is to identify meaningful connections between entities in the knowledge graph.
        
        Context:
        - Domain: {context.get('domain', 'general')}
        - Discovery Focus: {context.get('focus', 'all')}
        - Minimum Confidence: {parameters.get('min_confidence', 0.6)}
        
        Discover:
        1. Direct relationships
        2. Indirect connections
        3. Hierarchical structures
        4. Temporal relationships
        5. Causal links
        
        Provide evidence and confidence scores for each discovered relationship.
        """

    def _create_gap_filling_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create system prompt for gap filling."""
        return f"""You are a specialized knowledge gap filling AI. Your task is to identify and fill gaps in the knowledge graph using reliable sources.
        
        Context:
        - Domain: {context.get('domain', 'general')}
        - Gap Types: {context.get('gap_types', 'all')}
        - Source Quality: {parameters.get('source_quality', 'high')}
        
        For each gap:
        1. Identify missing information
        2. Research reliable sources
        3. Extract relevant knowledge
        4. Validate against existing graph
        5. Provide citations and confidence scores
        
        Ensure all new information is well-supported and consistent with existing knowledge.
        """

    def _parse_extraction(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured knowledge."""
        try:
            data = json.loads(response)
            return {
                "nodes": [
                    GraphNode(
                        id=str(uuid4()),
                        type=GraphNodeType(node["type"]),
                        label=node["label"],
                        properties=node["properties"],
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                        confidence=node.get("confidence", 1.0)
                    )
                    for node in data.get("nodes", [])
                ],
                "relations": [
                    GraphRelation(
                        id=str(uuid4()),
                        source_id=rel["source"],
                        target_id=rel["target"],
                        type=GraphRelationType(rel["type"]),
                        properties=rel["properties"],
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                        confidence=rel.get("confidence", 1.0)
                    )
                    for rel in data.get("relations", [])
                ]
            }
        except Exception as e:
            logger.error(f"Error parsing extraction: {str(e)}")
            raise

    def _parse_discovery(self, response: str) -> List[GraphRelation]:
        """Parse the model's response into discovered relationships."""
        try:
            data = json.loads(response)
            return [
                GraphRelation(
                    id=str(uuid4()),
                    source_id=rel["source"],
                    target_id=rel["target"],
                    type=GraphRelationType(rel["type"]),
                    properties=rel["properties"],
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    confidence=rel.get("confidence", 1.0)
                )
                for rel in data.get("relations", [])
            ]
        except Exception as e:
            logger.error(f"Error parsing discovery: {str(e)}")
            raise

    def _parse_gap_filling(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into filled gap knowledge."""
        try:
            data = json.loads(response)
            return {
                "nodes": [
                    GraphNode(
                        id=str(uuid4()),
                        type=GraphNodeType(node["type"]),
                        label=node["label"],
                        properties=node["properties"],
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                        confidence=node.get("confidence", 1.0)
                    )
                    for node in data.get("nodes", [])
                ],
                "relations": [
                    GraphRelation(
                        id=str(uuid4()),
                        source_id=rel["source"],
                        target_id=rel["target"],
                        type=GraphRelationType(rel["type"]),
                        properties=rel["properties"],
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow(),
                        confidence=rel.get("confidence", 1.0)
                    )
                    for rel in data.get("relations", [])
                ],
                "metadata": {
                    "gap_type": data.get("gap_type"),
                    "sources": data.get("sources", []),
                    "confidence": data.get("confidence", 1.0)
                }
            }
        except Exception as e:
            logger.error(f"Error parsing gap filling: {str(e)}")
            raise

    def _add_to_graph(
        self,
        knowledge: Dict[str, Any]
    ) -> tuple[List[GraphNode], List[GraphRelation]]:
        """Add extracted knowledge to the graph."""
        added_nodes = []
        added_relations = []
        
        # Add nodes
        for node in knowledge["nodes"]:
            if not self.graph.has_node(node.id):
                self.graph.add_node(
                    node.id,
                    type=node.type,
                    label=node.label,
                    properties=node.properties,
                    created_at=node.created_at,
                    updated_at=node.updated_at,
                    confidence=node.confidence
                )
                added_nodes.append(node)
        
        # Add relations
        for relation in knowledge["relations"]:
            if (relation.source_id in self.graph and
                relation.target_id in self.graph and
                not self.graph.has_edge(relation.source_id, relation.target_id)):
                self.graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    type=relation.type,
                    properties=relation.properties,
                    created_at=relation.created_at,
                    updated_at=relation.updated_at,
                    confidence=relation.confidence
                )
                added_relations.append(relation)
        
        return added_nodes, added_relations

    def _add_relations_to_graph(
        self,
        relations: List[GraphRelation]
    ) -> List[GraphRelation]:
        """Add discovered relations to the graph."""
        added = []
        for relation in relations:
            if (relation.source_id in self.graph and
                relation.target_id in self.graph and
                not self.graph.has_edge(relation.source_id, relation.target_id)):
                self.graph.add_edge(
                    relation.source_id,
                    relation.target_id,
                    type=relation.type,
                    properties=relation.properties,
                    created_at=relation.created_at,
                    updated_at=relation.updated_at,
                    confidence=relation.confidence
                )
                added.append(relation)
        return added

    def _get_relevant_nodes(
        self,
        focus: List[str]
    ) -> List[GraphNode]:
        """Get nodes relevant to the focus areas."""
        relevant = []
        for node_id in self.graph.nodes:
            node_data = self.graph.nodes[node_id]
            if any(f.lower() in node_data["label"].lower() for f in focus):
                relevant.append(
                    GraphNode(
                        id=node_id,
                        type=GraphNodeType(node_data["type"]),
                        label=node_data["label"],
                        properties=node_data["properties"],
                        created_at=node_data["created_at"],
                        updated_at=node_data["updated_at"],
                        confidence=node_data["confidence"]
                    )
                )
        return relevant

    def _identify_gaps(self, focus: List[str]) -> List[Dict[str, Any]]:
        """Identify gaps in the knowledge graph."""
        gaps = []
        
        # Get relevant subgraph
        subgraph = self._get_relevant_subgraph(focus)
        
        # Analyze connectivity
        for node in subgraph.nodes:
            # Check for isolated nodes
            if subgraph.degree(node) == 0:
                gaps.append({
                    "type": "isolated_node",
                    "node_id": node,
                    "node_data": subgraph.nodes[node]
                })
            
            # Check for missing relationships
            for other in subgraph.nodes:
                if node != other and not subgraph.has_edge(node, other):
                    # Check if relationship might be expected
                    if self._should_have_relation(
                        subgraph.nodes[node],
                        subgraph.nodes[other]
                    ):
                        gaps.append({
                            "type": "missing_relation",
                            "source_id": node,
                            "target_id": other,
                            "source_data": subgraph.nodes[node],
                            "target_data": subgraph.nodes[other]
                        })
        
        return gaps

    def _should_have_relation(
        self,
        node1_data: Dict[str, Any],
        node2_data: Dict[str, Any]
    ) -> bool:
        """Determine if two nodes should likely have a relationship."""
        # Implement logic to determine if nodes should be related
        # based on their types, properties, and domain knowledge
        return False  # Placeholder

    def _get_relevant_knowledge(
        self,
        gap: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get knowledge relevant to a gap for context."""
        relevant = {
            "nodes": [],
            "relations": []
        }
        
        # Get nodes within 2 hops of gap nodes
        if gap["type"] == "isolated_node":
            node_id = gap["node_id"]
            for n in nx.single_source_shortest_path_length(
                self.graph,
                node_id,
                cutoff=2
            ).keys():
                if n != node_id:
                    relevant["nodes"].append(self.graph.nodes[n])
                    for _, _, edge_data in self.graph.edges(n, data=True):
                        relevant["relations"].append(edge_data)
        
        elif gap["type"] == "missing_relation":
            source_id = gap["source_id"]
            target_id = gap["target_id"]
            for n in set(
                list(nx.single_source_shortest_path_length(
                    self.graph,
                    source_id,
                    cutoff=1
                ).keys()) +
                list(nx.single_source_shortest_path_length(
                    self.graph,
                    target_id,
                    cutoff=1
                ).keys())
            ):
                if n not in [source_id, target_id]:
                    relevant["nodes"].append(self.graph.nodes[n])
                    for _, _, edge_data in self.graph.edges(n, data=True):
                        relevant["relations"].append(edge_data)
        
        return relevant

    def _find_paths(self, query: Dict[str, Any]) -> List[List[str]]:
        """Find paths between nodes in the graph."""
        source = query.get("source")
        target = query.get("target")
        max_length = query.get("max_length", 3)
        
        if not (source and target and
                source in self.graph and
                target in self.graph):
            return []
        
        paths = []
        for path in nx.all_simple_paths(
            self.graph,
            source=source,
            target=target,
            cutoff=max_length
        ):
            paths.append(path)
        
        return paths

    def _find_neighbors(
        self,
        query: Dict[str, Any]
    ) -> Dict[str, List[str]]:
        """Find neighboring nodes in the graph."""
        node_id = query.get("node_id")
        depth = query.get("depth", 1)
        
        if not (node_id and node_id in self.graph):
            return {}
        
        neighbors = {
            "incoming": [],
            "outgoing": []
        }
        
        # Get incoming neighbors
        for pred in self.graph.predecessors(node_id):
            if depth == 1:
                neighbors["incoming"].append(pred)
            else:
                for path in nx.all_simple_paths(
                    self.graph,
                    source=pred,
                    target=node_id,
                    cutoff=depth
                ):
                    neighbors["incoming"].extend(path[:-1])
        
        # Get outgoing neighbors
        for succ in self.graph.successors(node_id):
            if depth == 1:
                neighbors["outgoing"].append(succ)
            else:
                for path in nx.all_simple_paths(
                    self.graph,
                    source=node_id,
                    target=succ,
                    cutoff=depth
                ):
                    neighbors["outgoing"].extend(path[1:])
        
        return neighbors

    def _search_graph(
        self,
        query: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Search the graph for nodes and relationships."""
        search_term = query.get("term", "").lower()
        node_type = query.get("node_type")
        min_confidence = query.get("min_confidence", 0.0)
        
        results = []
        for node_id, node_data in self.graph.nodes(data=True):
            # Check search term
            if search_term not in node_data["label"].lower():
                continue
            
            # Check node type
            if node_type and node_data["type"] != node_type:
                continue
            
            # Check confidence
            if node_data["confidence"] < min_confidence:
                continue
            
            # Get node relationships
            relationships = {
                "incoming": [
                    {
                        "node": pred,
                        "edge": self.graph.edges[pred, node_id]
                    }
                    for pred in self.graph.predecessors(node_id)
                ],
                "outgoing": [
                    {
                        "node": succ,
                        "edge": self.graph.edges[node_id, succ]
                    }
                    for succ in self.graph.successors(node_id)
                ]
            }
            
            results.append({
                "node": node_data,
                "relationships": relationships
            })
        
        return results

    def _extract_subgraph(
        self,
        query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Extract a subgraph based on query parameters."""
        focus_nodes = query.get("focus_nodes", [])
        max_depth = query.get("max_depth", 2)
        
        if not focus_nodes:
            return {"nodes": [], "edges": []}
        
        # Get nodes within max_depth of focus nodes
        subgraph_nodes = set()
        for node in focus_nodes:
            if node in self.graph:
                subgraph_nodes.add(node)
                for n in nx.single_source_shortest_path_length(
                    self.graph,
                    node,
                    cutoff=max_depth
                ).keys():
                    subgraph_nodes.add(n)
        
        # Create subgraph
        subgraph = self.graph.subgraph(subgraph_nodes)
        
        return {
            "nodes": [
                {
                    "id": node,
                    **self.graph.nodes[node]
                }
                for node in subgraph.nodes
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    **self.graph.edges[u, v]
                }
                for u, v in subgraph.edges
            ]
        }

    def _get_visualization_subgraph(
        self,
        focus: List[str]
    ) -> nx.DiGraph:
        """Get subgraph for visualization."""
        if not focus:
            return self.graph
        
        # Get nodes within 2 hops of focus nodes
        subgraph_nodes = set()
        for node in focus:
            if node in self.graph:
                subgraph_nodes.add(node)
                for n in nx.single_source_shortest_path_length(
                    self.graph,
                    node,
                    cutoff=2
                ).keys():
                    subgraph_nodes.add(n)
        
        return self.graph.subgraph(subgraph_nodes)

    def _create_visualization_data(
        self,
        subgraph: nx.DiGraph,
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Create data for graph visualization."""
        layout = nx.spring_layout(subgraph)
        
        return {
            "nodes": [
                {
                    "id": node,
                    "label": subgraph.nodes[node]["label"],
                    "type": subgraph.nodes[node]["type"],
                    "properties": subgraph.nodes[node]["properties"],
                    "position": {
                        "x": float(layout[node][0]),
                        "y": float(layout[node][1])
                    },
                    "confidence": subgraph.nodes[node]["confidence"]
                }
                for node in subgraph.nodes
            ],
            "edges": [
                {
                    "source": u,
                    "target": v,
                    "type": subgraph.edges[u, v]["type"],
                    "properties": subgraph.edges[u, v]["properties"],
                    "confidence": subgraph.edges[u, v]["confidence"]
                }
                for u, v in subgraph.edges
            ]
        }

    def _get_graph_stats(self) -> Dict[str, Any]:
        """Get statistics about the knowledge graph."""
        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "node_types": {
                node_type: len([
                    n for n, d in self.graph.nodes(data=True)
                    if d["type"] == node_type
                ])
                for node_type in GraphNodeType
            },
            "relation_types": {
                rel_type: len([
                    (u, v) for u, v, d in self.graph.edges(data=True)
                    if d["type"] == rel_type
                ])
                for rel_type in GraphRelationType
            },
            "average_degree": sum(dict(self.graph.degree()).values()) / self.graph.number_of_nodes() if self.graph.number_of_nodes() > 0 else 0,
            "density": nx.density(self.graph),
            "is_directed": self.graph.is_directed(),
            "is_dag": nx.is_directed_acyclic_graph(self.graph)
        }

    def _load_graph(self) -> None:
        """Load the knowledge graph from storage."""
        try:
            graph_path = self.config.get("graph_storage_path")
            if graph_path and os.path.exists(graph_path):
                self.graph = nx.read_gpickle(graph_path)
                logger.info(f"Loaded knowledge graph from {graph_path}")
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {str(e)}")

    def _save_graph(self) -> None:
        """Save the knowledge graph to storage."""
        try:
            graph_path = self.config.get("graph_storage_path")
            if graph_path:
                nx.write_gpickle(self.graph, graph_path)
                logger.info(f"Saved knowledge graph to {graph_path}")
        except Exception as e:
            logger.error(f"Error saving knowledge graph: {str(e)}")

    def _node_to_dict(self, node: GraphNode) -> Dict[str, Any]:
        """Convert a GraphNode to a dictionary."""
        return {
            "id": node.id,
            "type": node.type,
            "label": node.label,
            "properties": node.properties,
            "created_at": node.created_at.isoformat(),
            "updated_at": node.updated_at.isoformat(),
            "confidence": node.confidence
        } 