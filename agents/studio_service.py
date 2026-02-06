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
        cache: Optional[AgentCache] = None,
        workflow_repo: Optional[Any] = None,
        mongo_workflow_repo: Optional[Any] = None,
        dual_write: bool = False
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
        self._workflow_repo = workflow_repo
        self._mongo_workflow_repo = mongo_workflow_repo
        self._dual_write = dual_write
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

    def _components_to_dicts(self, components: List[ComponentInstance]) -> List[Dict[str, Any]]:
        return [c.__dict__ for c in components]

    def _components_from_dicts(self, components: List[Dict[str, Any]]) -> List[ComponentInstance]:
        return [
            ComponentInstance(
                id=c.get("id"),
                component_id=c.get("component_id"),
                name=c.get("name"),
                config=c.get("config", {}),
                position=c.get("position", {}),
                connections=c.get("connections", []),
            )
            for c in (components or [])
        ]

    def _workflow_from_record(self, record: Any) -> AgentWorkflow:
        if not isinstance(record, dict):
            record = {
                k: v for k, v in getattr(record, "__dict__", {}).items()
                if not k.startswith("_sa_")
            }
        return AgentWorkflow(
            id=str(record.get("id") or record.get("_id") or record.get("workflow_id")),
            name=record.get("name"),
            description=record.get("description"),
            components=self._components_from_dicts(record.get("components", [])),
            created_at=record.get("created_at") or datetime.utcnow(),
            updated_at=record.get("updated_at") or datetime.utcnow(),
            created_by=record.get("created_by"),
            version=record.get("version", "1.0.0"),
            is_public=record.get("is_public", False),
            tags=record.get("tags", []) or [],
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

            # Persist to Postgres if configured (primary), otherwise Mongo
            if self._workflow_repo:
                try:
                    db_record = await self._workflow_repo.create_workflow({
                        "name": workflow.name,
                        "description": workflow.description,
                        "components": self._components_to_dicts(workflow.components),
                        "nodes": [],
                        "connections": [],
                        "organization_id": security_context.organization_id,
                        "created_by": workflow.created_by,
                        "version": workflow.version,
                        "is_public": workflow.is_public,
                        "tags": workflow.tags or [],
                        "status": "draft",
                    })
                    # Sync IDs and timestamps with DB record
                    workflow.id = str(db_record.id)
                    workflow.created_at = db_record.created_at
                    workflow.updated_at = db_record.updated_at
                except Exception as e:
                    logger.error("workflow_persist_failed", error=str(e))
            elif self._mongo_workflow_repo:
                try:
                    mongo_record = await self._mongo_workflow_repo.create_workflow(
                        {
                            "name": workflow.name,
                            "description": workflow.description,
                            "components": self._components_to_dicts(workflow.components),
                            "nodes": [],
                            "connections": [],
                            "version": workflow.version,
                            "is_public": workflow.is_public,
                            "tags": workflow.tags or [],
                            "status": "draft",
                        },
                        organization_id=security_context.organization_id,
                        created_by=workflow.created_by
                    )
                    workflow.id = str(mongo_record.get("id"))
                    workflow.created_at = mongo_record.get("created_at") or workflow.created_at
                    workflow.updated_at = mongo_record.get("updated_at") or workflow.updated_at
                except Exception as e:
                    logger.error("workflow_persist_failed_mongo", error=str(e))

            # Dual-write to Mongo when Postgres is primary
            if self._workflow_repo and self._dual_write and self._mongo_workflow_repo:
                try:
                    await self._mongo_workflow_repo.create_workflow(
                        {
                            "name": workflow.name,
                            "description": workflow.description,
                            "components": self._components_to_dicts(workflow.components),
                            "nodes": [],
                            "connections": [],
                            "version": workflow.version,
                            "is_public": workflow.is_public,
                            "tags": workflow.tags or [],
                            "status": "draft",
                        },
                        organization_id=security_context.organization_id,
                        created_by=workflow.created_by,
                        postgres_id=workflow.id
                    )
                except Exception as e:
                    logger.error("workflow_dual_write_failed", error=str(e))

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
            workflow = self._workflows.get(workflow_id)
            if workflow is None and self._workflow_repo:
                record = await self._workflow_repo.get_workflow_by_id(workflow_id)
                if record:
                    workflow = self._workflow_from_record(record)
                    self._workflows[workflow.id] = workflow
            if workflow is None and self._mongo_workflow_repo:
                record = await self._mongo_workflow_repo.get_workflow_by_id(
                    workflow_id, organization_id=security_context.organization_id
                )
                if record:
                    workflow = self._workflow_from_record(record)
                    self._workflows[workflow.id] = workflow
            if workflow is None:
                raise ValueError(f"Workflow not found: {workflow_id}")
            
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

            if self._workflow_repo:
                try:
                    db_updates = dict(updates)
                    if "components" in db_updates:
                        db_updates["components"] = self._components_to_dicts(db_updates["components"])
                    await self._workflow_repo.update_workflow(
                        workflow_id=workflow.id,
                        updates=db_updates
                    )
                except Exception as e:
                    logger.error("workflow_update_persist_failed", error=str(e))
            elif self._mongo_workflow_repo:
                try:
                    mongo_updates = dict(updates)
                    if "components" in mongo_updates:
                        mongo_updates["components"] = self._components_to_dicts(mongo_updates["components"])
                    await self._mongo_workflow_repo.update_workflow(
                        workflow_id=workflow.id,
                        update_data=mongo_updates,
                        organization_id=security_context.organization_id
                    )
                except Exception as e:
                    logger.error("workflow_update_persist_failed_mongo", error=str(e))

            if self._workflow_repo and self._dual_write and self._mongo_workflow_repo:
                try:
                    mongo_updates = dict(updates)
                    if "components" in mongo_updates:
                        mongo_updates["components"] = self._components_to_dicts(mongo_updates["components"])
                    await self._mongo_workflow_repo.update_workflow_by_postgres_id(
                        postgres_id=workflow.id,
                        update_data=mongo_updates,
                        organization_id=security_context.organization_id
                    )
                except Exception as e:
                    logger.error("workflow_dual_write_update_failed", error=str(e))
            
            return workflow

    async def list_workflows(
        self,
        security_context: AgentSecurityContext,
        skip: int = 0,
        limit: int = 100
    ) -> List[AgentWorkflow]:
        if not self.security_manager.has_permission(security_context, AgentPermission.READ):
            raise PermissionError("Missing permission to read workflows")

        if self._workflow_repo:
            try:
                is_admin = AgentPermission.ADMIN in security_context.permissions
                records = await self._workflow_repo.list_accessible_workflows(
                    organization_id=security_context.organization_id,
                    user_id=security_context.user_id,
                    is_admin=is_admin,
                    skip=skip,
                    limit=limit,
                )
                workflows = [self._workflow_from_record(r) for r in records]
                for wf in workflows:
                    self._workflows[wf.id] = wf
                return workflows
            except Exception as e:
                logger.error("workflow_list_failed", error=str(e))
                raise

        if self._mongo_workflow_repo:
            records = await self._mongo_workflow_repo.list_workflows(
                organization_id=security_context.organization_id,
                created_by=None,
                include_public=True,
                skip=skip,
                limit=limit
            )
            workflows = [self._workflow_from_record(r) for r in records]
            for wf in workflows:
                self._workflows[wf.id] = wf
            return workflows

        # Fallback to in-memory
        workflows = []
        for workflow_id, workflow in self._workflows.items():
            if (workflow.is_public or
                workflow.created_by == security_context.user_id or
                AgentPermission.ADMIN in security_context.permissions):
                workflows.append(workflow)
        return workflows[skip:skip + limit]

    async def get_workflow(
        self,
        workflow_id: str,
        security_context: AgentSecurityContext
    ) -> AgentWorkflow:
        if not self.security_manager.has_permission(security_context, AgentPermission.READ):
            raise PermissionError("Missing permission to read workflows")

        workflow = self._workflows.get(workflow_id)
        if workflow is None and self._workflow_repo:
            record = await self._workflow_repo.get_workflow_by_id(workflow_id)
            if record:
                workflow = self._workflow_from_record(record)
                self._workflows[workflow.id] = workflow
        if workflow is None and self._mongo_workflow_repo:
            record = await self._mongo_workflow_repo.get_workflow_by_id(
                workflow_id, organization_id=security_context.organization_id
            )
            if record:
                workflow = self._workflow_from_record(record)
                self._workflows[workflow.id] = workflow
        if workflow is None:
            raise ValueError(f"Workflow not found: {workflow_id}")

        if (not workflow.is_public and
            workflow.created_by != security_context.user_id and
            AgentPermission.ADMIN not in security_context.permissions):
            raise PermissionError("Not authorized to access this workflow")

        return workflow

    async def delete_workflow(
        self,
        workflow_id: str,
        security_context: AgentSecurityContext
    ) -> None:
        if not self.security_manager.has_permission(security_context, AgentPermission.DELETE):
            raise PermissionError("Missing permission to delete workflows")

        async with self._lock:
            workflow = self._workflows.get(workflow_id)
            if workflow is None and self._workflow_repo:
                record = await self._workflow_repo.get_workflow_by_id(workflow_id)
                if record:
                    workflow = self._workflow_from_record(record)
            if workflow is None and self._mongo_workflow_repo:
                record = await self._mongo_workflow_repo.get_workflow_by_id(
                    workflow_id, organization_id=security_context.organization_id
                )
                if record:
                    workflow = self._workflow_from_record(record)
            if workflow is None:
                raise ValueError(f"Workflow not found: {workflow_id}")

            if (workflow.created_by != security_context.user_id and
                AgentPermission.ADMIN not in security_context.permissions):
                raise PermissionError("Not authorized to delete this workflow")

            # Persist delete
            if self._workflow_repo:
                await self._workflow_repo.delete_workflow(workflow_id)
            elif self._mongo_workflow_repo:
                await self._mongo_workflow_repo.delete_workflow(
                    workflow_id, organization_id=security_context.organization_id
                )

            if self._workflow_repo and self._dual_write and self._mongo_workflow_repo:
                try:
                    await self._mongo_workflow_repo.delete_workflow_by_postgres_id(
                        postgres_id=workflow_id,
                        organization_id=security_context.organization_id
                    )
                except Exception as e:
                    logger.error("workflow_dual_write_delete_failed", error=str(e))

            if workflow_id in self._workflows:
                del self._workflows[workflow_id]
    
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
