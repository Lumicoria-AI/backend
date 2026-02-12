"""
Base Component Class

This module defines the base component class that all Agent Studio components inherit from.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import structlog
from enum import Enum

logger = structlog.get_logger(__name__)

class ComponentStatus(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    COMPLETED = "completed"
    ERROR = "error"

@dataclass
class ComponentConfig:
    """Configuration for a component instance"""
    component_id: str
    name: str
    description: Optional[str] = None
    settings: Dict[str, Any] = field(default_factory=dict)
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0, "y": 0})
    connections: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ComponentResult:
    """Result from component execution"""
    component_id: str
    status: ComponentStatus
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)

class BaseComponent(ABC):
    """
    Base class for all Agent Studio components.
    
    Components are the building blocks that users can drag and drop
    to create custom agents.
    """
    
    def __init__(self, config: ComponentConfig):
        self.config = config
        self.id = config.component_id
        self.name = config.name
        self.description = config.description
        self.settings = config.settings
        self.position = config.position
        self.connections = config.connections
        self.metadata = config.metadata
        self.status = ComponentStatus.IDLE
        self._execution_count = 0
        self._last_execution = None
        # LLM client for components that need AI capabilities
        self._llm_client = None

    @property
    def llm_client(self):
        """Lazy-initialize the LLM client for components that need it."""
        if self._llm_client is None:
            try:
                from backend.ai_models import get_llm_client
                self._llm_client = get_llm_client()
            except Exception as e:
                logger.error(f"Failed to initialize LLM client for component {self.name}: {e}")
        return self._llm_client
        
    @property
    @abstractmethod
    def component_type(self) -> str:
        """Return the type of component (input, processor, output, integration)"""
        pass
        
    @property
    @abstractmethod
    def category(self) -> str:
        """Return the category of component (document, vision, audio, etc.)"""
        pass
        
    @property
    @abstractmethod
    def input_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for valid inputs"""
        pass
        
    @property
    @abstractmethod
    def output_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for outputs"""
        pass
        
    @property
    @abstractmethod
    def config_schema(self) -> Dict[str, Any]:
        """Return the JSON schema for configuration"""
        pass
        
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> ComponentResult:
        """
        Execute the component with the given input data.
        
        Args:
            input_data: Input data for the component
            
        Returns:
            ComponentResult with the execution result
        """
        pass
        
    async def validate_input(self, input_data: Dict[str, Any]) -> bool:
        """
        Validate input data against the component's input schema.
        
        Args:
            input_data: Input data to validate
            
        Returns:
            True if valid, False otherwise
        """
        # TODO: Implement JSON schema validation
        return True
        
    async def validate_config(self, config: Dict[str, Any]) -> bool:
        """
        Validate configuration against the component's config schema.
        
        Args:
            config: Configuration to validate
            
        Returns:
            True if valid, False otherwise
        """
        # TODO: Implement JSON schema validation
        return True
        
    def update_settings(self, new_settings: Dict[str, Any]) -> None:
        """Update component settings"""
        self.settings.update(new_settings)
        
    def add_connection(self, target_component_id: str) -> None:
        """Add a connection to another component"""
        if target_component_id not in self.connections:
            self.connections.append(target_component_id)
            
    def remove_connection(self, target_component_id: str) -> None:
        """Remove a connection to another component"""
        if target_component_id in self.connections:
            self.connections.remove(target_component_id)
            
    def get_statistics(self) -> Dict[str, Any]:
        """Get component execution statistics"""
        return {
            "execution_count": self._execution_count,
            "last_execution": self._last_execution,
            "status": self.status.value,
            "connections": len(self.connections)
        }
        
    def _start_execution(self) -> str:
        """Mark the start of component execution"""
        execution_id = str(uuid.uuid4())
        self.status = ComponentStatus.PROCESSING
        self._last_execution = datetime.utcnow()
        logger.info(
            "component_execution_started",
            component_id=self.id,
            component_name=self.name,
            execution_id=execution_id
        )
        return execution_id
        
    def _complete_execution(self, execution_id: str, success: bool = True, error: str = None) -> None:
        """Mark the completion of component execution"""
        self.status = ComponentStatus.COMPLETED if success else ComponentStatus.ERROR
        self._execution_count += 1
        logger.info(
            "component_execution_completed",
            component_id=self.id,
            component_name=self.name,
            execution_id=execution_id,
            success=success,
            error=error,
            execution_count=self._execution_count
        )
        
    async def cleanup(self) -> None:
        """Cleanup component resources"""
        self.status = ComponentStatus.IDLE
        logger.info(
            "component_cleanup",
            component_id=self.id,
            component_name=self.name
        )
