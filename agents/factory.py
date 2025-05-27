from typing import Dict, Any, Optional, Type
from enum import Enum
import structlog
from dataclasses import dataclass
from .cache import AgentCache, AgentCacheType, CacheManager
from .base_agent import BaseAgent
from .document_agent import DocumentAgent
from .wellbeing_agent import WellbeingAgent
from .vision_agent import VisionAgent
from .meeting_agent import MeetingAgent
from .creative_agent import CreativeAgent
from .student_agent import StudentAgent

# Configure logging
logger = structlog.get_logger(__name__)

class AgentType(Enum):
    """Types of available agents."""
    DOCUMENT = "document"
    WELLBEING = "wellbeing"
    VISION = "vision"
    MEETING = "meeting"
    CREATIVE = "creative"
    STUDENT = "student"

@dataclass
class AgentConfig:
    """Configuration for agent initialization."""
    agent_type: AgentType
    model: str = "perplexity-sonar"  # Default to Perplexity Sonar
    cache_enabled: bool = True
    cache_ttl: Optional[int] = 3600  # Default 1 hour cache
    max_retries: int = 3
    timeout: int = 30
    api_key: Optional[str] = None
    additional_config: Dict[str, Any] = None

class AgentFactory:
    """Factory for creating and managing agents."""
    
    _agent_classes: Dict[AgentType, Type[BaseAgent]] = {
        AgentType.DOCUMENT: DocumentAgent,
        AgentType.WELLBEING: WellbeingAgent,
        AgentType.VISION: VisionAgent,
        AgentType.MEETING: MeetingAgent,
        AgentType.CREATIVE: CreativeAgent,
        AgentType.STUDENT: StudentAgent
    }
    
    def __init__(self, cache_manager: Optional[CacheManager] = None):
        """
        Initialize the agent factory.
        
        Args:
            cache_manager: Optional cache manager for agent responses
        """
        self.cache_manager = cache_manager or CacheManager("memory")
        self.agent_cache = AgentCache(self.cache_manager)
        self._active_agents: Dict[str, BaseAgent] = {}
    
    def create_agent(self, config: AgentConfig) -> BaseAgent:
        """
        Create a new agent instance.
        
        Args:
            config: Agent configuration
            
        Returns:
            Initialized agent instance
        """
        if config.agent_type not in self._agent_classes:
            raise ValueError(f"Unknown agent type: {config.agent_type}")
        
        agent_class = self._agent_classes[config.agent_type]
        
        # Create agent instance with configuration
        agent = agent_class(
            model=config.model,
            cache=self.agent_cache if config.cache_enabled else None,
            cache_ttl=config.cache_ttl,
            max_retries=config.max_retries,
            timeout=config.timeout,
            api_key=config.api_key,
            **(config.additional_config or {})
        )
        
        # Store active agent
        agent_id = f"{config.agent_type.value}:{id(agent)}"
        self._active_agents[agent_id] = agent
        
        logger.info(
            "agent_created",
            agent_type=config.agent_type.value,
            agent_id=agent_id,
            model=config.model,
            cache_enabled=config.cache_enabled
        )
        
        return agent
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """
        Get an active agent by ID.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            Agent instance if found, None otherwise
        """
        return self._active_agents.get(agent_id)
    
    async def destroy_agent(self, agent_id: str) -> bool:
        """
        Destroy an active agent.
        
        Args:
            agent_id: Agent identifier
            
        Returns:
            True if agent was destroyed, False if not found
        """
        if agent_id in self._active_agents:
            agent = self._active_agents[agent_id]
            await agent.cleanup()  # Cleanup agent resources
            del self._active_agents[agent_id]
            
            logger.info(
                "agent_destroyed",
                agent_id=agent_id
            )
            return True
        return False
    
    async def cleanup(self):
        """Cleanup all active agents."""
        for agent_id, agent in list(self._active_agents.items()):
            await self.destroy_agent(agent_id)
    
    @classmethod
    def register_agent(cls, agent_type: AgentType, agent_class: Type[BaseAgent]):
        """
        Register a new agent type.
        
        Args:
            agent_type: Type of agent
            agent_class: Agent class implementation
        """
        if not issubclass(agent_class, BaseAgent):
            raise ValueError(f"Agent class must inherit from BaseAgent")
        
        cls._agent_classes[agent_type] = agent_class
        logger.info(
            "agent_registered",
            agent_type=agent_type.value,
            agent_class=agent_class.__name__
        )

# Example usage:
"""
# Initialize factory
factory = AgentFactory(CacheManager("redis", redis_url="redis://localhost:6379"))

# Create document agent
doc_config = AgentConfig(
    agent_type=AgentType.DOCUMENT,
    model="perplexity-sonar",
    cache_enabled=True,
    cache_ttl=3600,
    api_key="your-api-key"
)
document_agent = factory.create_agent(doc_config)

# Create wellbeing agent
wellbeing_config = AgentConfig(
    agent_type=AgentType.WELLBEING,
    model="perplexity-sonar",
    cache_enabled=True,
    cache_ttl=1800,  # 30 minutes cache
    additional_config={
        "check_interval": 300,  # Check every 5 minutes
        "break_reminder": True
    }
)
wellbeing_agent = factory.create_agent(wellbeing_config)

# Use agents
async def process_document():
    result = await document_agent.process_async({
        "document_id": "123",
        "content": "test document"
    })
    return result

async def check_wellbeing():
    result = await wellbeing_agent.process_async({
        "user_id": "user123",
        "activity_data": {...}
    })
    return result

# Cleanup
await factory.cleanup()
"""
