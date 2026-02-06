"""
Workflow Orchestration Service for Agent Studio

This module handles the orchestration of components in an agent workflow,
including graph traversal, execution, and error handling.
"""

import asyncio
from typing import Dict, Any, List, Optional, Set
import structlog
from datetime import datetime
import time
import uuid
from enum import Enum

from .components.base_component import BaseComponent, ComponentResult, ComponentStatus, ComponentConfig
from .components.input_components import (
    DocumentUploadComponent,
    LiveCameraComponent,
    VoiceInputComponent,
    TextInputComponent
)
from .components.processor_components import (
    PerplexityResearchComponent,
    ChainOfThoughtComponent,
    DataExtractionComponent, 
    SummarizationComponent,
    TaskGeneratorComponent,
    LiveEnvironmentAnalyzerComponent,
    TranslatorComponent,
    CitationManagerComponent
)
from .components.output_components import (
    CalendarIntegrationComponent,
    AgentDeploymentComponent,
    WellbeingCoachComponent
)
from .studio_service import AgentWorkflow, ComponentInstance

# Configure logger
logger = structlog.get_logger(__name__)

class WorkflowStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"

class WorkflowNode:
    """Representation of a component in a workflow execution graph"""
    def __init__(self, component: BaseComponent, instance: ComponentInstance):
        self.component = component
        self.instance = instance
        self.id = instance.id
        self.input_connections: List[str] = []  # IDs of nodes that feed into this node
        self.output_connections: List[str] = []  # IDs of nodes that this node feeds into
        self.result: Optional[ComponentResult] = None
        self.status = ComponentStatus.IDLE
        self.execution_order: int = -1  # Will be set during topological sort
        
    @property
    def is_input_node(self) -> bool:
        """Check if this is an input node (no incoming connections)"""
        return len(self.input_connections) == 0
        
    @property
    def is_output_node(self) -> bool:
        """Check if this is an output node (no outgoing connections)"""
        return len(self.output_connections) == 0
        
    @property
    def has_completed(self) -> bool:
        """Check if this node has completed execution"""
        return self.status in (ComponentStatus.COMPLETED, ComponentStatus.ERROR)
        
    @property
    def can_execute(self, completed_nodes: Set[str]) -> bool:
        """Check if this node can execute based on its dependencies"""
        # A node can execute if all its input nodes have completed
        return all(node_id in completed_nodes for node_id in self.input_connections)

class WorkflowOrchestrator:
    """Orchestrates the execution of components in an agent workflow"""
    
    def __init__(self):
        self.nodes: Dict[str, WorkflowNode] = {}
        self.execution_order: List[str] = []
        self.status = WorkflowStatus.IDLE
        self.workflow_id: Optional[str] = None
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.execution_id: Optional[str] = None
        self._lock = asyncio.Lock()
        self._component_factory = ComponentFactory()
        
    async def load_workflow(self, workflow: AgentWorkflow) -> None:
        """
        Load a workflow for orchestration.
        
        Args:
            workflow: The workflow to orchestrate
        """
        async with self._lock:
            self.workflow_id = workflow.id
            self.nodes = {}
            self.execution_order = []
            self.status = WorkflowStatus.IDLE
            
            # Create nodes for each component instance in the workflow
            for instance in workflow.components:
                component = await self._component_factory.create_component(
                    instance.component_id,
                    ComponentConfig(
                        component_id=instance.component_id,
                        name=instance.name,
                        settings=instance.config,
                        position=instance.position,
                        connections=[c["target"] for c in instance.connections] if instance.connections else []
                    )
                )
                
                node = WorkflowNode(component, instance)
                self.nodes[instance.id] = node
            
            # Set up connections between nodes
            for instance in workflow.components:
                node = self.nodes[instance.id]
                
                for connection in instance.connections:
                    target_id = connection["target"]
                    if target_id in self.nodes:
                        node.output_connections.append(target_id)
                        self.nodes[target_id].input_connections.append(instance.id)
            
            # Perform topological sort to determine execution order
            self._topological_sort()
            
            logger.info(
                "workflow_loaded",
                workflow_id=workflow.id,
                nodes=len(self.nodes),
                execution_order=self.execution_order
            )
    
    def _topological_sort(self) -> None:
        """
        Perform a topological sort of the workflow nodes to determine execution order.
        This handles cycles by treating nodes with no remaining unvisited inputs as ready.
        """
        # Reset execution order
        self.execution_order = []
        
        # Find nodes with no incoming edges (input nodes)
        no_incoming = [node_id for node_id, node in self.nodes.items() if not node.input_connections]
        
        visited = set()
        temp_visited = set()
        
        def visit(node_id):
            if node_id in temp_visited:
                # This is a cycle, handle gracefully
                logger.warning(f"Cycle detected in workflow at node {node_id}")
                return
                
            if node_id in visited:
                return
                
            temp_visited.add(node_id)
            
            node = self.nodes[node_id]
            for out_id in node.output_connections:
                visit(out_id)
                
            temp_visited.remove(node_id)
            visited.add(node_id)
            self.execution_order.insert(0, node_id)
        
        # Visit all nodes
        for node_id in list(self.nodes.keys()):
            if node_id not in visited:
                visit(node_id)
                
        # Set execution order index in each node
        for index, node_id in enumerate(self.execution_order):
            self.nodes[node_id].execution_order = index
    
    async def execute_workflow(self, initial_input: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the workflow with the given input.
        
        Args:
            initial_input: Initial input data for the workflow
            
        Returns:
            Final output data from the workflow
        """
        if not self.workflow_id:
            raise ValueError("No workflow loaded")
            
        self.execution_id = str(uuid.uuid4())
        self.start_time = datetime.utcnow()
        self.status = WorkflowStatus.RUNNING
        
        logger.info(
            "workflow_execution_started",
            workflow_id=self.workflow_id,
            execution_id=self.execution_id
        )
        
        try:
            # Reset all nodes
            for node in self.nodes.values():
                node.status = ComponentStatus.IDLE
                node.result = None
                
            # Track completed nodes
            completed_nodes = set()
            
            # Find input nodes and start execution
            input_nodes = [node for node in self.nodes.values() if node.is_input_node]
            
            if not input_nodes:
                raise ValueError("Workflow has no input nodes")
                
            # Execute in topological order
            results = {}
            for node_id in self.execution_order:
                node = self.nodes[node_id]
                
                # Determine input for this node
                node_input = {}
                
                # If this is an input node, use the initial input
                if node.is_input_node:
                    node_input = initial_input
                else:
                    # Otherwise, combine input from all incoming connections
                    for input_id in node.input_connections:
                        if input_id in completed_nodes and self.nodes[input_id].result:
                            # Merge the data from the input node
                            node_input.update(self.nodes[input_id].result.data)
                
                # Execute the component
                try:
                    start_time = time.time()
                    node.status = ComponentStatus.PROCESSING
                    
                    logger.debug(
                        "executing_component",
                        node_id=node.id,
                        component_name=node.component.name,
                        execution_id=self.execution_id
                    )
                    
                    result = await node.component.execute(node_input)
                    execution_time = time.time() - start_time
                    
                    # Update node status and result
                    node.result = result
                    node.status = result.status
                    
                    logger.debug(
                        "component_executed",
                        node_id=node.id,
                        component_name=node.component.name,
                        status=result.status,
                        execution_time=execution_time,
                        execution_id=self.execution_id
                    )
                    
                    # Store results
                    results[node.id] = result.data
                    
                    # Mark node as completed
                    completed_nodes.add(node.id)
                    
                    # If node failed, log error but continue execution
                    if node.status == ComponentStatus.ERROR:
                        logger.error(
                            "component_error",
                            node_id=node.id,
                            component_name=node.component.name,
                            error=result.error,
                            execution_id=self.execution_id
                        )
                        
                except Exception as e:
                    logger.error(
                        "component_execution_failed",
                        node_id=node.id,
                        component_name=node.component.name,
                        error=str(e),
                        execution_id=self.execution_id
                    )
                    
                    # Create error result
                    node.result = ComponentResult(
                        component_id=node.component.id,
                        status=ComponentStatus.ERROR,
                        error=str(e)
                    )
                    node.status = ComponentStatus.ERROR
            
            # Combine results from output nodes
            output_nodes = [node for node in self.nodes.values() if node.is_output_node]
            final_output = {}
            
            for node in output_nodes:
                if node.result and node.status == ComponentStatus.COMPLETED:
                    final_output.update(node.result.data)
            
            self.status = WorkflowStatus.COMPLETED
            self.end_time = datetime.utcnow()
            
            logger.info(
                "workflow_execution_completed",
                workflow_id=self.workflow_id,
                execution_id=self.execution_id,
                duration=(self.end_time - self.start_time).total_seconds()
            )
            
            return final_output
            
        except Exception as e:
            self.status = WorkflowStatus.FAILED
            self.end_time = datetime.utcnow()
            
            logger.error(
                "workflow_execution_failed",
                workflow_id=self.workflow_id,
                execution_id=self.execution_id,
                error=str(e),
                duration=(self.end_time - self.start_time).total_seconds()
            )
            
            raise

    async def cancel_execution(self) -> None:
        """Cancel the current workflow execution"""
        if self.status == WorkflowStatus.RUNNING:
            self.status = WorkflowStatus.CANCELED
            self.end_time = datetime.utcnow()
            
            logger.info(
                "workflow_execution_canceled",
                workflow_id=self.workflow_id,
                execution_id=self.execution_id,
                duration=(self.end_time - self.start_time).total_seconds()
            )
            
            # Clean up any running components
            for node in self.nodes.values():
                if node.status == ComponentStatus.PROCESSING:
                    await node.component.cleanup()

    def get_execution_status(self) -> Dict[str, Any]:
        """Get the current status of workflow execution"""
        node_statuses = {
            node_id: {
                "name": node.component.name,
                "status": node.status,
                "execution_order": node.execution_order,
                "has_result": node.result is not None
            }
            for node_id, node in self.nodes.items()
        }
        
        return {
            "workflow_id": self.workflow_id,
            "execution_id": self.execution_id,
            "status": self.status,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "nodes": node_statuses
        }

class ComponentFactory:
    """Factory for creating component instances based on component ID"""
    
    def __init__(self):
        # Register all available components
        self._component_registry = {
            # Input components
            "document_upload": DocumentUploadComponent,
            "live_camera": LiveCameraComponent,
            "voice_input": VoiceInputComponent,
            "text_input": TextInputComponent,
            
            # Processor components
            "perplexity_research": PerplexityResearchComponent,
            "chain_of_thought": ChainOfThoughtComponent,
            "data_extraction": DataExtractionComponent,
            "summarization": SummarizationComponent,
            "task_generator": TaskGeneratorComponent,
            "live_environment_analyzer": LiveEnvironmentAnalyzerComponent,
            "translator": TranslatorComponent,
            "citation_manager": CitationManagerComponent,
            
            # Output components
            "calendar_integration": CalendarIntegrationComponent,
            "agent_deployment": AgentDeploymentComponent,
            "wellbeing_coach": WellbeingCoachComponent
        }
        
    async def create_component(self, component_id: str, config: ComponentConfig) -> BaseComponent:
        """Create a component instance based on component ID"""
        if component_id not in self._component_registry:
            raise ValueError(f"Unknown component: {component_id}")
            
        component_class = self._component_registry[component_id]
        return component_class(config)
