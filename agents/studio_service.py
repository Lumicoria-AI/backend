from typing import Dict, Any, Optional, List, Union
from enum import Enum
import structlog
from dataclasses import dataclass
from datetime import datetime
import json
import asyncio
from .factory import AgentFactory, AgentType, AgentConfig
from .security import AgentSecurityManager, AgentSecurityContext, AgentPermission
from .cache import AgentCache
from .base_agent import BaseAgent

# Configure logging
logger = structlog.get_logger(__name__)

class ComponentType(Enum):
    """Types of components available in the studio."""
    INPUT = "input"  # Input components (text, file, camera, etc.)
    PROCESSOR = "processor"  # Processing components (NLP, OCR, etc.)
    OUTPUT = "output"  # Output components (task, calendar, etc.)
    INTEGRATION = "integration"  # Integration components (Slack, Google, etc.)
    CONDITION = "condition"  # Conditional logic components
    LOOP = "loop"  # Loop/repeat components
    TRANSFORM = "transform"  # Data transformation components

class ComponentCategory(Enum):
    """Categories of components for organization."""
    DOCUMENT = "document"  # Document processing components
    WELLBEING = "wellbeing"  # Well-being related components
    VISION = "vision"  # Computer vision components
    MEETING = "meeting"  # Meeting management components
    CREATIVE = "creative"  # Creative content components
    STUDENT = "student"  # Student-focused components
    GENERAL = "general"  # General purpose components

@dataclass
class ComponentDefinition:
    """Definition of a studio component."""
    id: str
    name: str
    type: ComponentType
    category: ComponentCategory
    description: str
    config_schema: Dict[str, Any]  # JSON Schema for component configuration
    input_schema: Dict[str, Any]  # JSON Schema for component inputs
    output_schema: Dict[str, Any]  # JSON Schema for component outputs
    icon: str  # Icon identifier
    version: str = "1.0.0"
    is_beta: bool = False
    requires_auth: bool = False

@dataclass
class ComponentInstance:
    """Instance of a component in an agent workflow."""
    id: str
    component_id: str
    name: str
    config: Dict[str, Any]
    position: Dict[str, int]  # x, y coordinates in the studio
    connections: List[Dict[str, str]]  # List of input/output connections

@dataclass
class AgentWorkflow:
    """Definition of an agent workflow created in the studio."""
    id: str
    name: str
    description: str
    components: List[ComponentInstance]
    created_at: datetime
    updated_at: datetime
    created_by: str
    version: str = "1.0.0"
    is_public: bool = False
    tags: List[str] = None

class StudioService:
    """Service for managing the No-Code Agent Studio."""
    
    def __init__(
        self,
        agent_factory: AgentFactory,
        security_manager: AgentSecurityManager,
        cache: Optional[AgentCache] = None
    ):
        """
        Initialize the studio service.
        
        Args:
            agent_factory: Factory for creating agents
            security_manager: Security manager for access control
            cache: Optional cache for component definitions
        """
        self.agent_factory = agent_factory
        self.security_manager = security_manager
        self.cache = cache
        self._component_definitions: Dict[str, ComponentDefinition] = {}
        self._workflows: Dict[str, AgentWorkflow] = {}
        self._lock = asyncio.Lock()
        
        # Register default components
        self._register_default_components()
    
    def _register_default_components(self):
        """Register default components available in the studio."""
        # Document components
        self.register_component(ComponentDefinition(
            id="document_ocr",
            name="Document OCR",
            type=ComponentType.PROCESSOR,
            category=ComponentCategory.DOCUMENT,
            description="Extract text from documents using OCR",
            config_schema={
                "type": "object",
                "properties": {
                    "language": {"type": "string", "enum": ["en", "es", "fr"]},
                    "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1}
                }
            },
            input_schema={
                "type": "object",
                "properties": {
                    "document": {"type": "string", "format": "binary"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "confidence": {"type": "number"}
                }
            },
            icon="document-scan"
        ))
        
        # Well-being components
        self.register_component(ComponentDefinition(
            id="break_reminder",
            name="Break Reminder",
            type=ComponentType.PROCESSOR,
            category=ComponentCategory.WELLBEING,
            description="Monitor activity and suggest breaks",
            config_schema={
                "type": "object",
                "properties": {
                    "check_interval": {"type": "integer", "minimum": 60},
                    "break_duration": {"type": "integer", "minimum": 1}
                }
            },
            input_schema={
                "type": "object",
                "properties": {
                    "activity_data": {"type": "object"}
                }
            },
            output_schema={
                "type": "object",
                "properties": {
                    "should_break": {"type": "boolean"},
                    "break_type": {"type": "string", "enum": ["short", "long"]}
                }
            },
            icon="timer"
        ))
        
        # Add more default components...
    
    def register_component(self, component: ComponentDefinition):
        """
        Register a new component in the studio.
        
        Args:
            component: Component definition to register
        """
        self._component_definitions[component.id] = component
        logger.info(
            "component_registered",
            component_id=component.id,
            component_name=component.name,
            component_type=component.type.value
        )
    
    async def create_workflow(
        self,
        name: str,
        description: str,
        components: List[ComponentInstance],
        security_context: AgentSecurityContext,
        is_public: bool = False,
        tags: List[str] = None
    ) -> AgentWorkflow:
        """
        Create a new agent workflow.
        
        Args:
            name: Workflow name
            description: Workflow description
            components: List of component instances
            security_context: Security context
            is_public: Whether the workflow is public
            tags: List of tags
            
        Returns:
            Created workflow
        """
        if not self.security_manager.has_permission(security_context, AgentPermission.CREATE):
            raise PermissionError("Missing permission to create workflows")
        
        async with self._lock:
            workflow = AgentWorkflow(
                id=f"wf_{datetime.utcnow().timestamp()}",
                name=name,
                description=description,
                components=components,
                created_at=datetime.utcnow(),
                updated_at=datetime.utcnow(),
                created_by=security_context.user_id,
                is_public=is_public,
                tags=tags or []
            )
            
            self._workflows[workflow.id] = workflow
            
            logger.info(
                "workflow_created",
                workflow_id=workflow.id,
                workflow_name=workflow.name,
                user_id=security_context.user_id
            )
            
            return workflow
    
    async def update_workflow(
        self,
        workflow_id: str,
        updates: Dict[str, Any],
        security_context: AgentSecurityContext
    ) -> AgentWorkflow:
        """
        Update an existing workflow.
        
        Args:
            workflow_id: Workflow identifier
            updates: Dictionary of updates
            security_context: Security context
            
        Returns:
            Updated workflow
        """
        if not self.security_manager.has_permission(security_context, AgentPermission.CONFIGURE):
            raise PermissionError("Missing permission to update workflows")
        
        async with self._lock:
            if workflow_id not in self._workflows:
                raise ValueError(f"Workflow not found: {workflow_id}")
            
            workflow = self._workflows[workflow_id]
            
            # Check ownership or admin status
            if (workflow.created_by != security_context.user_id and 
                AgentPermission.ADMIN not in security_context.permissions):
                raise PermissionError("Not authorized to update this workflow")
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(workflow, key):
                    setattr(workflow, key, value)
            
            workflow.updated_at = datetime.utcnow()
            
            logger.info(
                "workflow_updated",
                workflow_id=workflow_id,
                user_id=security_context.user_id
            )
            
            return workflow
    
    async def deploy_workflow(
        self,
        workflow_id: str,
        security_context: AgentSecurityContext
    ) -> BaseAgent:
        """
        Deploy a workflow as an agent.
        
        Args:
            workflow_id: Workflow identifier
            security_context: Security context
            
        Returns:
            Deployed agent instance
        """
        if not self.security_manager.has_permission(security_context, AgentPermission.CREATE):
            raise PermissionError("Missing permission to deploy workflows")
        
        async with self._lock:
            if workflow_id not in self._workflows:
                raise ValueError(f"Workflow not found: {workflow_id}")
            
            workflow = self._workflows[workflow_id]
            
            # Create agent configuration
            config = AgentConfig(
                agent_type=AgentType.DOCUMENT,  # Default type, can be customized
                model="perplexity-sonar",
                cache_enabled=True,
                additional_config={
                    "workflow_id": workflow_id,
                    "components": [c.__dict__ for c in workflow.components]
                }
            )
            
            # Create agent instance
            agent = self.agent_factory.create_agent(config)
            
            logger.info(
                "workflow_deployed",
                workflow_id=workflow_id,
                agent_id=id(agent),
                user_id=security_context.user_id
            )
            
            return agent
    
    def get_available_components(
        self,
        category: Optional[ComponentCategory] = None,
        component_type: Optional[ComponentType] = None
    ) -> List[ComponentDefinition]:
        """
        Get available components, optionally filtered by category or type.
        
        Args:
            category: Optional category filter
            component_type: Optional type filter
            
        Returns:
            List of component definitions
        """
        components = self._component_definitions.values()
        
        if category:
            components = [c for c in components if c.category == category]
        
        if component_type:
            components = [c for c in components if c.type == component_type]
        
        return list(components)
    
    async def validate_workflow(self, workflow: AgentWorkflow) -> List[str]:
        """
        Validate a workflow for correctness.
        
        Args:
            workflow: Workflow to validate
            
        Returns:
            List of validation errors, empty if valid
        """
        errors = []
        
        # Check component existence
        for component in workflow.components:
            if component.component_id not in self._component_definitions:
                errors.append(f"Unknown component: {component.component_id}")
                continue
            
            # Validate component configuration
            definition = self._component_definitions[component.component_id]
            try:
                # Validate config against schema
                # You would use a JSON Schema validator here
                pass
            except Exception as e:
                errors.append(f"Invalid config for {component.name}: {str(e)}")
        
        # Check connections
        for component in workflow.components:
            for connection in component.connections:
                # Validate connection endpoints exist
                # Validate connection types match
                pass
        
        return errors

# Example usage:
"""
# Initialize studio service
studio = StudioService(
    agent_factory=AgentFactory(),
    security_manager=AgentSecurityManager(jwt_secret="your-secret"),
    cache=AgentCache(CacheManager("redis"))
)

# Create a workflow
workflow = await studio.create_workflow(
    name="Document Processor",
    description="Process documents and create tasks",
    components=[
        ComponentInstance(
            id="comp1",
            component_id="document_ocr",
            name="OCR Processor",
            config={"language": "en", "confidence_threshold": 0.8},
            position={"x": 100, "y": 100},
            connections=[]
        ),
        ComponentInstance(
            id="comp2",
            component_id="break_reminder",
            name="Break Monitor",
            config={"check_interval": 300, "break_duration": 5},
            position={"x": 300, "y": 100},
            connections=[]
        )
    ],
    security_context=security_context,
    is_public=False,
    tags=["document", "automation"]
)

# Deploy workflow as agent
agent = await studio.deploy_workflow(workflow.id, security_context)

# Use the agent
result = await agent.process_async({
    "document": "base64_encoded_document",
    "activity_data": {"typing_speed": 60}
})
"""
