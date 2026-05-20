
from typing import Dict, Any, List, Optional, Set
from enum import Enum
import logging
import re
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
        """Initialize the Knowledge Graph Agent.

        IMPORTANT: this agent no longer owns a global, shared graph.  The
        previous design loaded a single pickle file and let every
        organization on the platform read from / write to the same
        DiGraph, which was a hard data-isolation leak.  Now the per-org
        graph is injected before every call via `attach_graph`, the
        service layer wrapping the agent persists changes back through
        `backend.services.knowledge_graph.repository`, and the agent
        itself never touches SQLAlchemy.
        """
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

        # Start with an empty in-memory graph.  The service layer calls
        # `attach_graph(org_graph)` before each request to swap in the
        # caller's tenant-scoped graph.
        self.graph = nx.DiGraph()

        # Configure model for knowledge graph tasks.  16k token ceiling
        # because Gemini 2.5 spends part of its budget on internal
        # reasoning before any text reaches the wire — 4k was too tight
        # and produced truncated JSON.
        self.model_config.update({
            "temperature": 0.2,
            "max_tokens": 16384,
            "top_p": 0.9,
            "frequency_penalty": 0.3,
            "presence_penalty": 0.3
        })

    def attach_graph(self, graph: "nx.DiGraph") -> None:
        """Replace the agent's in-memory graph with the caller's
        tenant-scoped one.  Called by the service layer at the start of
        every request and reset to an empty DiGraph at the end."""
        self.graph = graph if graph is not None else nx.DiGraph()

    async def _process_with_model(
        self,
        system_prompt: str,
        user_payload: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Adapter onto BaseAgent._call_model_async with KG-tuned defaults.

        Knowledge extraction wants near-deterministic JSON output, so the
        temperature defaults stay low.  Gemini 2.5 consumes part of its
        `max_output_tokens` budget on internal reasoning before emitting
        any user-visible text, so we give it a generous ceiling — small
        budgets truncate the JSON mid-string and break parsing.
        """
        parameters = parameters or {}
        temperature = parameters.get("temperature", self.model_config.get("temperature", 0.2))
        max_tokens = parameters.get("max_tokens", self.model_config.get("max_tokens", 16384))
        return await self._call_model_async(
            prompt=user_payload,
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

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
            
            # Persistence is owned by the service-layer wrapper, not the
            # agent.  The wrapper inspects the result, persists added
            # nodes / edges to Postgres via the repository module, and
            # then resets the agent's in-memory graph.

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

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the knowledge graph agent asynchronously."""
        return await self.process_async({
            "action": "query",
            "data": {
                "query": {"term": query},
                "query_type": "search"
            },
            "context": context or {}
        })

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
        node_types = ", ".join(t.value for t in GraphNodeType)
        relation_types = ", ".join(t.value for t in GraphRelationType)
        return f"""You are a knowledge extraction engine.  Read the user's content and identify the entities and the connections between them.

Domain: {context.get('domain', 'general')}
Focus: {context.get('focus', 'all')}
Confidence threshold: {parameters.get('confidence_threshold', 0.7)}

Allowed node types: {node_types}
Allowed relation types: {relation_types}

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  No prose, no markdown, no code fences, no commentary.  The shape MUST be:

{{
  "nodes": [
    {{"label": "<short name>", "type": "<one of the allowed node types>", "properties": {{}}, "confidence": 0.0}}
  ],
  "relations": [
    {{"source": "<label of source node>", "target": "<label of target node>", "type": "<one of the allowed relation types>", "properties": {{}}, "confidence": 0.0}}
  ]
}}

Rules:
- Use only the allowed type values, lowercase, exactly as listed above.
- Reference relation endpoints by the same `label` you used in `nodes`.
- Confidence is a number between 0 and 1.
- BE CONCISE: emit at most 25 nodes and 40 relations total.  Pick the most important entities and drop trivia.  Leave `properties` as {{}} unless a property is genuinely informative.
- If you cannot find any entities, return {{"nodes": [], "relations": []}}.
- Do NOT wrap the JSON in ```json``` fences.
"""

    def _create_discovery_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create system prompt for relation discovery."""
        relation_types = ", ".join(t.value for t in GraphRelationType)
        return f"""You are a relationship discovery engine.  Given a list of existing nodes, identify meaningful connections between them.

Domain: {context.get('domain', 'general')}
Focus: {context.get('focus', 'all')}
Minimum confidence: {parameters.get('min_confidence', 0.6)}

Allowed relation types: {relation_types}

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  No prose, no markdown, no fences.  The shape MUST be:

{{
  "relations": [
    {{"source": "<label of source node>", "target": "<label of target node>", "type": "<one of the allowed relation types>", "properties": {{}}, "confidence": 0.0}}
  ]
}}

Rules:
- Use only the allowed relation types, lowercase, exactly as listed above.
- Reference nodes by the labels that appeared in the user's input.
- Confidence is a number between 0 and 1.
- BE CONCISE: emit at most 40 relations total.  Pick the strongest, most meaningful connections.  Leave `properties` as {{}} unless genuinely informative.
- If you cannot find any connections, return {{"relations": []}}.
"""

    def _create_gap_filling_prompt(
        self,
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> str:
        """Create system prompt for gap filling."""
        node_types = ", ".join(t.value for t in GraphNodeType)
        relation_types = ", ".join(t.value for t in GraphRelationType)
        return f"""You are a knowledge gap-filling engine.  Given a description of a gap in the user's knowledge graph and the surrounding context, propose new nodes and relations that would plausibly fill the gap.

Domain: {context.get('domain', 'general')}
Gap types: {context.get('gap_types', 'all')}

Allowed node types: {node_types}
Allowed relation types: {relation_types}

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  No prose, no markdown, no fences.  The shape MUST be:

{{
  "gap_type": "<short label for the gap>",
  "sources": ["<optional citation strings>"],
  "confidence": 0.0,
  "nodes": [
    {{"label": "<short name>", "type": "<one of the allowed node types>", "properties": {{}}, "confidence": 0.0}}
  ],
  "relations": [
    {{"source": "<label>", "target": "<label>", "type": "<one of the allowed relation types>", "properties": {{}}, "confidence": 0.0}}
  ]
}}

Rules:
- Use only the allowed type values, lowercase, exactly as listed.
- Reference relation endpoints by the same label used in `nodes`.
- Be conservative: only propose nodes/relations that you are reasonably confident about.
- If you cannot fill the gap, return {{"gap_type": "<label>", "sources": [], "confidence": 0.0, "nodes": [], "relations": []}}.
"""

    @staticmethod
    def _strip_fences(text: str) -> str:
        """Strip ```json ... ``` fences if present; tolerate truncated
        responses that lack the closing fence."""
        m = re.search(
            r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if m:
            return m.group(1).strip()
        return re.sub(r"^```(?:json)?\s*", "", text).strip()

    @staticmethod
    def _salvage_truncated_json(text: str) -> Dict[str, Any]:
        """Recover a usable prefix of a JSON object truncated mid-output.

        Walks the string, tracks brace / bracket / string state, and
        remembers the last position at which a top-level array element
        (`nodes` / `relations` / `edges`) was fully closed.  Slices up
        to that point and appends the trailing `]}` to make valid JSON.
        Returns {} if no usable prefix exists.
        """
        if not text or not text.lstrip().startswith("{"):
            return {}
        start = text.find("{")

        depth_stack: List[str] = []
        in_string = False
        escape = False
        reading_key = False
        key_buffer: List[str] = []
        current_key: Optional[str] = None
        in_top_array = False
        last_top_safe_cut = -1

        i = start
        while i < len(text):
            ch = text[i]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                    if reading_key:
                        current_key = "".join(key_buffer)
                        key_buffer = []
                        reading_key = False
                elif reading_key:
                    key_buffer.append(ch)
                i += 1
                continue

            if ch == '"':
                in_string = True
                # A string is a key when the previous non-space char is
                # `{` or `,` and we're directly inside an object.
                if depth_stack and depth_stack[-1] == "{":
                    j = i - 1
                    while j >= 0 and text[j] in " \t\r\n":
                        j -= 1
                    if j >= 0 and text[j] in "{,":
                        reading_key = True
                        key_buffer = []
            elif ch == "{":
                depth_stack.append("{")
            elif ch == "[":
                depth_stack.append("[")
                if len(depth_stack) == 2 and current_key in ("nodes", "relations", "edges"):
                    in_top_array = True
            elif ch == "}":
                if depth_stack and depth_stack[-1] == "{":
                    depth_stack.pop()
                    if in_top_array and len(depth_stack) == 2:
                        last_top_safe_cut = i
            elif ch == "]":
                if depth_stack and depth_stack[-1] == "[":
                    depth_stack.pop()
                    if len(depth_stack) == 1:
                        in_top_array = False
            i += 1

        if not depth_stack or last_top_safe_cut <= start:
            return {}

        prefix = text[start : last_top_safe_cut + 1]
        repaired = prefix + "]}"
        try:
            return json.loads(repaired)
        except Exception:
            return {}

    @classmethod
    def _extract_json(cls, response: Any) -> Dict[str, Any]:
        """Best-effort JSON extraction from an LLM response.

        Pipeline: try `json.loads` -> strip ```json``` fences -> first
        balanced `{...}` block -> structural salvage for token-truncated
        responses.  Returns {} if nothing usable is found.
        """
        if not response:
            return {}
        text = response.strip() if isinstance(response, str) else str(response)

        try:
            return json.loads(text)
        except Exception:
            pass

        unfenced = cls._strip_fences(text)
        try:
            return json.loads(unfenced)
        except Exception:
            pass

        s = unfenced.find("{")
        e = unfenced.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(unfenced[s : e + 1])
            except Exception:
                pass

        salvaged = cls._salvage_truncated_json(unfenced)
        if salvaged:
            logger.info(
                "kg_llm_json_salvaged nodes=%s relations=%s",
                len(salvaged.get("nodes") or []),
                len(salvaged.get("relations") or []),
            )
            return salvaged

        logger.warning(
            "kg_llm_non_json len=%s head=%r tail=%r",
            len(text),
            text[:300],
            text[-300:],
        )
        return {}

    @staticmethod
    def _coerce_node_type(value: Any) -> Optional[GraphNodeType]:
        if not value:
            return None
        try:
            return GraphNodeType(str(value).strip().lower())
        except Exception:
            return None

    @staticmethod
    def _coerce_relation_type(value: Any) -> Optional[GraphRelationType]:
        if not value:
            return None
        try:
            return GraphRelationType(str(value).strip().lower().replace(" ", "_"))
        except Exception:
            return None

    def _parse_extraction(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into structured knowledge.  Skips
        malformed entries instead of failing the whole extraction."""
        data = self._extract_json(response)
        nodes: List[GraphNode] = []
        for raw in data.get("nodes", []) or []:
            if not isinstance(raw, dict):
                continue
            ntype = self._coerce_node_type(raw.get("type"))
            label = (raw.get("label") or "").strip()
            if not ntype or not label:
                continue
            nodes.append(
                GraphNode(
                    id=str(uuid4()),
                    type=ntype,
                    label=label,
                    properties=raw.get("properties") or {},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    confidence=float(raw.get("confidence", 1.0) or 1.0),
                )
            )

        relations: List[GraphRelation] = []
        for raw in (data.get("relations") or data.get("edges") or []):
            if not isinstance(raw, dict):
                continue
            rtype = self._coerce_relation_type(raw.get("type"))
            source = raw.get("source") or raw.get("source_id")
            target = raw.get("target") or raw.get("target_id")
            if not rtype or not source or not target:
                continue
            relations.append(
                GraphRelation(
                    id=str(uuid4()),
                    source_id=str(source),
                    target_id=str(target),
                    type=rtype,
                    properties=raw.get("properties") or {},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    confidence=float(raw.get("confidence", 1.0) or 1.0),
                )
            )

        return {"nodes": nodes, "relations": relations}

    def _parse_discovery(self, response: str) -> List[GraphRelation]:
        """Parse the model's response into discovered relationships."""
        data = self._extract_json(response)
        out: List[GraphRelation] = []
        for raw in (data.get("relations") or data.get("edges") or []):
            if not isinstance(raw, dict):
                continue
            rtype = self._coerce_relation_type(raw.get("type"))
            source = raw.get("source") or raw.get("source_id")
            target = raw.get("target") or raw.get("target_id")
            if not rtype or not source or not target:
                continue
            out.append(
                GraphRelation(
                    id=str(uuid4()),
                    source_id=str(source),
                    target_id=str(target),
                    type=rtype,
                    properties=raw.get("properties") or {},
                    created_at=datetime.utcnow(),
                    updated_at=datetime.utcnow(),
                    confidence=float(raw.get("confidence", 1.0) or 1.0),
                )
            )
        return out

    def _parse_gap_filling(self, response: str) -> Dict[str, Any]:
        """Parse the model's response into filled gap knowledge."""
        data = self._extract_json(response)
        parsed = self._parse_extraction(response)
        return {
            "nodes": parsed["nodes"],
            "relations": parsed["relations"],
            "metadata": {
                "gap_type": data.get("gap_type"),
                "sources": data.get("sources") or [],
                "confidence": float(data.get("confidence", 1.0) or 1.0),
            },
        }

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

    def _node_to_brief(self, node_id: str) -> Optional[Dict[str, Any]]:
        """Return a {id, label, type} brief for a graph node, or None
        if the node is missing.  Used by query helpers so the frontend
        receives renderable objects instead of bare ids."""
        if node_id not in self.graph:
            return None
        data = self.graph.nodes[node_id]
        return {
            "id": node_id,
            "label": data.get("label") or node_id,
            "type": data.get("type") or "concept",
        }

    def _find_paths(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Find shortest paths between two nodes.  Returns paths as
        lists of {id, label, type} objects so the UI can render labels
        directly.
        """
        source = query.get("source")
        target = query.get("target")
        max_length = int(query.get("max_length", 4) or 4)
        max_paths = int(query.get("max_paths", 5) or 5)

        if not (source and target and source in self.graph and target in self.graph):
            return {"paths": []}

        paths: List[List[Dict[str, Any]]] = []
        try:
            # Undirected search so we don't miss paths because the
            # LLM happened to orient one edge the "wrong" way.
            undirected = self.graph.to_undirected(as_view=True)
            for raw_path in nx.shortest_simple_paths(
                undirected, source=source, target=target
            ):
                if len(raw_path) - 1 > max_length:
                    break
                resolved = [self._node_to_brief(n) for n in raw_path]
                resolved = [n for n in resolved if n]
                if resolved:
                    paths.append(resolved)
                if len(paths) >= max_paths:
                    break
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            pass
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Path search failed: {e}")

        return {"paths": paths}

    def _find_neighbors(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Return the direct neighbors of a node as a flat list of
        {id, label, type, edge_type, direction} objects so the
        frontend can render the row without an extra fetch.
        """
        node_id = query.get("node_id")
        if not (node_id and node_id in self.graph):
            return {"neighbors": []}

        out: List[Dict[str, Any]] = []
        seen: Set[str] = set()

        for pred in self.graph.predecessors(node_id):
            brief = self._node_to_brief(pred)
            if not brief or brief["id"] in seen:
                continue
            seen.add(brief["id"])
            try:
                edge_attrs = self.graph.edges[pred, node_id]
            except Exception:
                edge_attrs = {}
            brief["edge_type"] = edge_attrs.get("type") or "related_to"
            brief["direction"] = "incoming"
            out.append(brief)

        for succ in self.graph.successors(node_id):
            brief = self._node_to_brief(succ)
            if not brief or brief["id"] in seen:
                continue
            seen.add(brief["id"])
            try:
                edge_attrs = self.graph.edges[node_id, succ]
            except Exception:
                edge_attrs = {}
            brief["edge_type"] = edge_attrs.get("type") or "related_to"
            brief["direction"] = "outgoing"
            out.append(brief)

        return {"neighbors": out, "center": self._node_to_brief(node_id)}

    def _search_graph(
        self,
        query: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Search the graph for nodes whose label contains `term`.
        Returns a flat list of {id, label, type, confidence} matches
        so the frontend autocomplete can render them directly.  Caps
        results so an LLM-prompted search never balloons the response.
        """
        search_term = (query.get("term") or "").strip().lower()
        node_type = query.get("node_type")
        min_confidence = float(query.get("min_confidence", 0.0) or 0.0)
        limit = int(query.get("limit", 25) or 25)

        matches: List[Dict[str, Any]] = []
        for node_id, node_data in self.graph.nodes(data=True):
            label = (node_data.get("label") or "").lower()
            if search_term and search_term not in label:
                continue
            if node_type and node_data.get("type") != node_type:
                continue
            if float(node_data.get("confidence", 1.0) or 1.0) < min_confidence:
                continue
            matches.append({
                "id": node_id,
                "label": node_data.get("label") or node_id,
                "type": node_data.get("type") or "concept",
                "confidence": float(node_data.get("confidence", 1.0) or 1.0),
            })
            if len(matches) >= limit:
                break
        return {"matches": matches, "total": len(matches)}

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
        """Create data for graph visualization.  Coords are emitted as
        flat `x` / `y` fields on the node so the frontend SVG renderer
        can read them directly without an extra `position` indirection."""
        if subgraph.number_of_nodes() == 0:
            return {"nodes": [], "edges": []}
        try:
            layout = nx.spring_layout(subgraph, seed=42)
        except Exception:
            # Spring layout can rarely fail on disconnected single-node
            # subgraphs; fall back to a circular placement.
            layout = nx.circular_layout(subgraph)

        nodes_payload = []
        for node in subgraph.nodes:
            attrs = subgraph.nodes[node]
            pos = layout.get(node, (0.0, 0.0))
            nodes_payload.append({
                "id": node,
                "label": attrs.get("label") or str(node),
                "type": attrs.get("type") or "concept",
                "properties": attrs.get("properties") or {},
                "x": float(pos[0]),
                "y": float(pos[1]),
                "confidence": float(attrs.get("confidence", 1.0) or 1.0),
            })

        edges_payload = []
        for u, v in subgraph.edges:
            attrs = subgraph.edges[u, v]
            edges_payload.append({
                "source": u,
                "target": v,
                "type": attrs.get("type") or "related_to",
                "properties": attrs.get("properties") or {},
                "confidence": float(attrs.get("confidence", 1.0) or 1.0),
            })

        return {"nodes": nodes_payload, "edges": edges_payload}

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

    # NOTE: the previous `_load_graph` / `_save_graph` implementations
    # used `nx.read_gpickle` / `nx.write_gpickle`, both removed in
    # NetworkX 3.0, and shared a single global graph across all tenants.
    # Persistence is now per-organization in Postgres via
    # `backend.services.knowledge_graph.repository`, which the service
    # wrapper invokes around every `process_async` call.

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