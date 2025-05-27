import os
import uuid
from typing import Dict, Any, List, Optional, Union, Type
import structlog
import asyncio
from datetime import datetime

from .base_agent import BaseAgent
from .document_agent import DocumentAgent
from .wellbeing_agent import WellbeingAgent
try:
    from .vision_agent import VisionAgent
except ImportError:
    VisionAgent = None
from .meeting_agent import MeetingAgent
from .creative_agent import CreativeAgent
from .student_agent import StudentAgent
from .research_mentor_agent import ResearchMentorAgent
from .social_media_agent import SocialMediaAgent
from .legal_document_agent import LegalDocumentAgent
from .learning_coach_agent import LearningCoachAgent
from .knowledge_graph_agent import KnowledgeGraphAgent
from .ethics_bias_agent import EthicsBiasAgent
from .rag_agent import RAGAgent
from .focus_flow_agent import FocusFlowAgent
from .workspace_ergonomics_agent import WorkspaceErgonomicsAgent
from .meeting_fact_checker_agent import MeetingFactCheckerAgent

# Configure logging
logger = structlog.get_logger(__name__)

class AgentService:
    """
    Service for managing and orchestrating AI agents in the Lumicoria platform.
    Handles agent creation, execution, configuration and monitoring.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the AgentService with configuration.
        
        Args:
            config: Application configuration dictionary
        """
        self.config = config
        self.agents: Dict[str, BaseAgent] = {}
        self.agent_types = {
            "document": DocumentAgent,
            "wellbeing": WellbeingAgent,
            "meeting": MeetingAgent,
            "creative": CreativeAgent,
            "student": StudentAgent,
            "rag": RAGAgent,
            "meeting_fact_checker": MeetingFactCheckerAgent
        }
        
        # Add vision agent if available
        if VisionAgent:
            self.agent_types["vision"] = VisionAgent
        
        # Keep track of agent executions for monitoring
        self.recent_executions: List[Dict[str, Any]] = []
        self.max_execution_history = 100
        
        # Initialize agents from configuration
        self._load_agents()

    def _load_agents(self):
        """Initialize agent instances from configuration."""
        agent_configs = self.config.get("agents", {})
        
        for agent_name, agent_config in agent_configs.items():
            try:
                agent_type = agent_config.get("type")
                if not agent_type:
                    logger.warning(f"Missing agent type for {agent_name}, skipping")
                    continue
                
                # Check if we have a class for this agent type
                if agent_type not in self.agent_types:
                    logger.warning(f"Unknown agent type '{agent_type}' for {agent_name}, skipping")
                    continue
                
                # Get the agent class and instantiate it
                agent_class = self.agent_types[agent_type]
                self.agents[agent_name] = agent_class(agent_config)
                logger.info(f"Loaded agent {agent_name} of type {agent_type}")
            
            except Exception as e:
                logger.error(f"Failed to load agent {agent_name}: {str(e)}")

    def get_agent(self, agent_name: str) -> BaseAgent:
        """
        Get agent instance by name.
        
        Args:
            agent_name: Name of the agent to retrieve
            
        Returns:
            Agent instance
            
        Raises:
            ValueError: If agent not found
        """
        if agent_name not in self.agents:
            raise ValueError(f"Agent '{agent_name}' not found.")
        return self.agents[agent_name]
    
    def create_agent(self, 
                    agent_name: str, 
                    agent_type: str, 
                    config: Dict[str, Any]) -> BaseAgent:
        """
        Create a new agent instance.
        
        Args:
            agent_name: Name for the new agent
            agent_type: Type of agent to create
            config: Agent-specific configuration
            
        Returns:
            The created agent instance
            
        Raises:
            ValueError: If agent type is unknown or agent already exists
        """
        if agent_name in self.agents:
            raise ValueError(f"Agent '{agent_name}' already exists.")
            
        if agent_type not in self.agent_types:
            valid_types = ", ".join(self.agent_types.keys())
            raise ValueError(f"Unknown agent type '{agent_type}'. Valid types are: {valid_types}")
        
        # Create agent instance
        agent_class = self.agent_types[agent_type]
        
        # Add agent type to config
        config["type"] = agent_type
        
        # Create agent instance
        agent = agent_class(config)
        self.agents[agent_name] = agent
        
        # Update main config
        if "agents" not in self.config:
            self.config["agents"] = {}
        self.config["agents"][agent_name] = config
        
        logger.info(f"Created new agent {agent_name} of type {agent_type}")
        return agent
    
    def execute_agent(self, 
                     agent_name: str, 
                     input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an agent with input data.
        
        Args:
            agent_name: Name of the agent to execute
            input_data: Input data for the agent
            
        Returns:
            Agent execution results
            
        Raises:
            ValueError: If agent not found
        """
        agent = self.get_agent(agent_name)
        
        # Record execution start
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        logger.info(f"Executing agent {agent_name}", agent_name=agent_name, execution_id=execution_id)
        
        try:
            # Execute agent
            result = agent.process(input_data)
            
            # Record successful execution
            self._record_execution(
                execution_id=execution_id,
                agent_name=agent_name,
                agent_type=agent.config.get("type", "unknown"),
                start_time=start_time,
                end_time=datetime.utcnow(),
                success=True
            )
            
            return result
        
        except Exception as e:
            # Record failed execution
            self._record_execution(
                execution_id=execution_id,
                agent_name=agent_name,
                agent_type=agent.config.get("type", "unknown"), 
                start_time=start_time,
                end_time=datetime.utcnow(),
                success=False,
                error_message=str(e)
            )
            
            logger.error(f"Error executing agent {agent_name}: {str(e)}")
            raise
    
    async def execute_agent_async(self, 
                                agent_name: str, 
                                input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute an agent asynchronously with input data.
        
        Args:
            agent_name: Name of the agent to execute
            input_data: Input data for the agent
            
        Returns:
            Agent execution results
            
        Raises:
            ValueError: If agent not found or agent doesn't support async execution
        """
        agent = self.get_agent(agent_name)
        
        # Check if agent supports async processing
        if not hasattr(agent, "process_async"):
            raise ValueError(f"Agent '{agent_name}' does not support async execution.")
            
        # Record execution start
        execution_id = str(uuid.uuid4())
        start_time = datetime.utcnow()
        
        logger.info(f"Executing agent {agent_name} asynchronously", 
                   agent_name=agent_name, execution_id=execution_id)
        
        try:
            # Execute agent asynchronously
            result = await agent.process_async(input_data)
            
            # Record successful execution
            self._record_execution(
                execution_id=execution_id,
                agent_name=agent_name,
                agent_type=agent.config.get("type", "unknown"),
                start_time=start_time,
                end_time=datetime.utcnow(),
                success=True,
                async_execution=True
            )
            
            return result
        
        except Exception as e:
            # Record failed execution
            self._record_execution(
                execution_id=execution_id,
                agent_name=agent_name, 
                agent_type=agent.config.get("type", "unknown"),
                start_time=start_time,
                end_time=datetime.utcnow(),
                success=False,
                error_message=str(e),
                async_execution=True
            )
            
            logger.error(f"Error executing agent {agent_name} asynchronously: {str(e)}")
            raise
    
    def update_agent_config(self, 
                           agent_name: str, 
                           config_updates: Dict[str, Any]) -> BaseAgent:
        """
        Update agent configuration.
        
        Args:
            agent_name: Name of agent to update
            config_updates: Configuration updates to apply
            
        Returns:
            Updated agent instance
            
        Raises:
            ValueError: If agent not found
        """
        agent = self.get_agent(agent_name)
        
        # Apply configuration updates
        agent.config.update(config_updates)
        
        # Reinitialize agent models to apply new configuration
        agent.initialize_models()
        
        # Update main config
        if "agents" in self.config and agent_name in self.config["agents"]:
            self.config["agents"][agent_name].update(config_updates)
        
        logger.info(f"Updated configuration for agent {agent_name}")
        return agent
    
    def list_agents(self) -> List[Dict[str, Any]]:
        """
        List all available agents with basic information.
        
        Returns:
            List of agent information dictionaries
        """
        return [
            {
                "name": name,
                "type": agent.config.get("type", "unknown"),
                "description": agent.config.get("description", ""),
                "capabilities": agent.config.get("capabilities", []),
            }
            for name, agent in self.agents.items()
        ]
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """
        Get agent execution statistics.
        
        Returns:
            Statistics about agent executions
        """
        # Count total executions
        total_executions = len(self.recent_executions)
        
        # Count successful executions
        successful = sum(1 for exec in self.recent_executions if exec.get("success", False))
        
        # Count by agent type
        by_type = {}
        for exec in self.recent_executions:
            agent_type = exec.get("agent_type", "unknown")
            if agent_type not in by_type:
                by_type[agent_type] = 0
            by_type[agent_type] += 1
        
        return {
            "total_executions": total_executions,
            "successful_executions": successful,
            "failed_executions": total_executions - successful,
            "success_rate": (successful / total_executions) if total_executions > 0 else 0,
            "by_agent_type": by_type
        }
    
    def _record_execution(self, 
                         execution_id: str,
                         agent_name: str,
                         agent_type: str, 
                         start_time: datetime,
                         end_time: datetime,
                         success: bool,
                         error_message: Optional[str] = None,
                         async_execution: bool = False) -> None:
        """Record agent execution for monitoring."""
        # Calculate execution time in milliseconds
        execution_time_ms = int((end_time - start_time).total_seconds() * 1000)
        
        # Create execution record
        execution_record = {
            "execution_id": execution_id,
            "agent_name": agent_name,
            "agent_type": agent_type,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "execution_time_ms": execution_time_ms,
            "success": success,
            "async": async_execution
        }
        
        if error_message:
            execution_record["error"] = error_message
        
        # Add to recent executions
        self.recent_executions.append(execution_record)
        
        # Trim history if needed
        if len(self.recent_executions) > self.max_execution_history:
            self.recent_executions = self.recent_executions[-self.max_execution_history:]

    async def process_research_mentor_request(
        self,
        mode: str,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process a request for the Research Mentor Agent."""
        try:
            agent = self.agents.get("research_mentor")
            if not agent:
                raise ValueError("Research Mentor Agent not available")
            
            request = {
                "mode": mode,
                "data": data,
                "context": context,
                "parameters": parameters
            }
            
            result = await agent.process_async(request)
            return result
            
        except Exception as e:
            logger.error(f"Error processing research mentor request: {str(e)}")
            raise

    def configure(self, config: Dict[str, Any]) -> None:
        """Configure the agent service."""
        self.config = config
        
        # Initialize agents with their configurations
        agent_configs = config.get("agents", {})
        
        # Social Media Agent
        if "social_media" in agent_configs:
            self.agents["social_media"] = SocialMediaAgent(agent_configs["social_media"])
            logger.info("Social Media Agent initialized")
        
        # Legal Document Agent
        if "legal_document" in agent_configs:
            self.agents["legal_document"] = LegalDocumentAgent(agent_configs["legal_document"])
            logger.info("Legal Document Agent initialized")
        
        # Learning Coach Agent
        if "learning_coach" in agent_configs:
            self.agents["learning_coach"] = LearningCoachAgent(agent_configs["learning_coach"])
            logger.info("Learning Coach Agent initialized")
        
        # Research Mentor Agent
        if "research_mentor" in agent_configs:
            self.agents["research_mentor"] = ResearchMentorAgent(agent_configs["research_mentor"])
            logger.info("Research Mentor Agent initialized")
        
        # Knowledge Graph Agent
        if "knowledge_graph" in agent_configs:
            self.agents["knowledge_graph"] = KnowledgeGraphAgent(agent_configs["knowledge_graph"])
            logger.info("Knowledge Graph Agent initialized")
        
        # Ethics & Bias Detector Agent
        if "ethics_bias" in agent_configs:
            self.agents["ethics_bias"] = EthicsBiasAgent(agent_configs["ethics_bias"])
            logger.info("Ethics & Bias Detector Agent initialized")
        
        # Focus & Flow Guardian Agent
        if "focus_flow" in agent_configs:
            self.agents["focus_flow"] = FocusFlowAgent(agent_configs["focus_flow"])
            logger.info("Focus & Flow Guardian Agent initialized")

        # Workspace Ergonomics Agent
        if "workspace_ergonomics" in agent_configs:
            self.agents["workspace_ergonomics"] = WorkspaceErgonomicsAgent(agent_configs["workspace_ergonomics"])
            logger.info("Workspace Ergonomics Agent initialized")

# Global agent service instance
_agent_service: Optional[AgentService] = None

async def setup_agent_service() -> None:
    """Setup the global agent service."""
    global _agent_service
    
    if _agent_service is None:
        _agent_service = AgentService({})
        
        # Load configuration
        # In a real application, this would load from a config file or environment
        config = {
            "agents": {
                "social_media": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 2048
                },
                "legal_document": {
                    "model": "gpt-4",
                    "temperature": 0.3,
                    "max_tokens": 4096
                },
                "learning_coach": {
                    "model": "gpt-4",
                    "temperature": 0.7,
                    "max_tokens": 2048
                },
                "research_mentor": {
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "max_tokens": 4096
                },
                "knowledge_graph": {
                    "model": "gpt-4",
                    "temperature": 0.2,
                    "max_tokens": 4096,
                    "graph_storage_path": "data/knowledge_graph.pkl"
                },
                "ethics_bias": {
                    "model": "gpt-4",
                    "temperature": 0.3,
                    "max_tokens": 2000
                },
                "focus_flow": {
                    "model": "gpt-4",
                    "temperature": 0.5,
                    "max_tokens": 2000
                },
                "workspace_ergonomics": {
                    "model": "gpt-4",
                    "temperature": 0.3,
                    "max_tokens": 2000
                }
            }
        }
        
        _agent_service.configure(config)
        logger.info("Agent service configured successfully")

def get_agent_service() -> AgentService:
    """Get the global agent service instance."""
    if _agent_service is None:
        raise RuntimeError("Agent service not initialized")
    return _agent_service

