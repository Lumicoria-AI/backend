from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import os
# Import the provider-agnostic LLM interface
from backend.ai_models import get_llm_client, LLMClient, LLMConfig, LLMResponse
from backend.core.config import settings as app_settings
import structlog
import asyncio

# Configure logger
logger = structlog.get_logger(__name__)

class BaseAgent(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Prioritize the agent_model_config passed from AgentService
        if "agent_model_config" in config:
            self.model_config = config["agent_model_config"]
        elif config.get("model"):
             # Fallback to global model config if model name is provided
            self.model_config = self._get_global_model_config(config["model"])
        else:
            self.model_config = {}

        self.llm_client: Optional[LLMClient] = None
        # Keep backward-compat alias
        self.perplexity_client = None
        self.initialize_models()

    def initialize_models(self):
        """Initialize the LLM client via the provider-agnostic abstraction."""
        try:
            # Determine which provider to use for this agent
            # Priority: agent config > model name mapping > global default
            provider = self._resolve_provider()
            
            self.llm_client = get_llm_client(provider=provider)
            # Backward-compat alias so existing agent code that uses self.perplexity_client
            # continues to work via the abstraction layer
            self.perplexity_client = self.llm_client
            
        except Exception as e:
            # Don't re-raise — let the agent register without a client.
            # _call_model_async already handles None llm_client gracefully.
            # The error will surface on the first actual LLM call, not at startup.
            logger.warning(f"LLM client failed to initialize (will retry on first call): {str(e)}")
            self.llm_client = None
            self.perplexity_client = None

    def _resolve_provider(self) -> Optional[str]:
        """
        Determine the LLM provider for this agent.
        
        Resolution order:
        1. Explicit 'provider' key in agent config
        2. Model name mapping (e.g., 'perplexity' or 'sonar' → perplexity provider)
        3. DEFAULT_LLM_PROVIDER from settings
        """
        # 1. Explicit provider in config
        provider = self.config.get("provider") or self.model_config.get("provider")
        if provider:
            return provider
        
        # 2. Infer from model name
        model_name = (self.model_config.get("model") or self.config.get("model") or "").lower()
        if "perplexity" in model_name or "sonar" in model_name:
            return "perplexity"
        if "gemini" in model_name:
            return "gemini"
        if "gpt" in model_name or "openai" in model_name or model_name.startswith(("o1", "o3")):
            return "openai"
        if "claude" in model_name or "anthropic" in model_name:
            return "anthropic"
        if "mistral" in model_name or "codestral" in model_name or "pixtral" in model_name or "mixtral" in model_name:
            return "mistral"
        
        # 3. Global default
        return None  # get_llm_client() will use DEFAULT_LLM_PROVIDER

    def _get_global_model_config(self, model_name: str) -> Dict[str, Any]:
        # This is a placeholder to retrieve model configuration from a global settings object
        global_models_config = self.config.get("ai_models", {})
        return global_models_config.get(model_name, {})

    @abstractmethod
    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process data asynchronously."""
        pass

    @abstractmethod
    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Query the agent asynchronously."""
        pass

    def process(self, data: Any) -> Any:
        # This method should be overridden by subclasses
        raise NotImplementedError("Subclasses must implement the 'process' method")

    def get_model_name(self) -> str:
        return self.model_config.get("model")

    def _call_model(self, prompt: str, model_name: str = None, **kwargs) -> str:
        """Calls the LLM via the provider-agnostic interface.

        Args:
            prompt: The input prompt for the model.
            model_name: Optional. The specific model name to use.
            **kwargs: Additional keyword arguments to pass to the LLM.

        Returns:
            The response text from the LLM.
        """
        if not self.llm_client:
            logger.error("LLM client not initialized")
            return "Error: LLM client not initialized correctly."
        
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        # Build config
        config = LLMConfig(
            model=model_name or self.get_model_name(),
            temperature=kwargs.pop("temperature", 0.7),
            max_tokens=kwargs.pop("max_tokens", 1024),
            top_p=kwargs.pop("top_p", 0.9),
            extra=kwargs,
        )
        
        messages = [{"role": "user", "content": prompt}]
        
        response = loop.run_until_complete(
            self.llm_client.generate(messages, config=config)
        )
        
        return response.content
    
    async def _call_model_async(
        self,
        prompt: str,
        model_name: str = None,
        system_prompt: str = None,
        conversation_history: list = None,
        **kwargs,
    ) -> str:
        """Asynchronous version of _call_model with optional conversation history.
        
        Args:
            prompt: The input prompt for the model.
            model_name: Optional. The specific model name to use.
            system_prompt: Optional. System instructions for the model.
            conversation_history: Optional list of {"role": "user"|"assistant", "content": "..."} dicts.
                These are prepended to give the model conversational context.
            **kwargs: Additional keyword arguments to pass to the LLM.
            
        Returns:
            The response text from the LLM.
        """
        # Lazy re-initialization: if client failed at startup, try again now
        if not self.llm_client:
            try:
                provider = self._resolve_provider()
                self.llm_client = get_llm_client(provider=provider)
                self.perplexity_client = self.llm_client
                logger.info("LLM client lazily initialized on first call")
            except Exception as e:
                logger.error(f"LLM client re-initialization failed: {str(e)}")
                return f"I'm sorry, the AI service is temporarily unavailable ({str(e)[:120]}). Please try again shortly."
            
        try:
            # Build config
            config = LLMConfig(
                model=model_name or self.get_model_name(),
                temperature=kwargs.pop("temperature", 0.7),
                max_tokens=kwargs.pop("max_tokens", 1024),
                top_p=kwargs.pop("top_p", 0.9),
                extra=kwargs,
            )
            
            # Build messages with optional history
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            
            # Inject conversation history for context-aware responses
            if conversation_history:
                for msg in conversation_history[-8:]:  # Last 8 messages max
                    messages.append({
                        "role": msg.get("role", "user"),
                        "content": msg.get("content", ""),
                    })
            
            messages.append({"role": "user", "content": prompt})
            
            response = await self.llm_client.generate(messages, config=config)
            return response.content
                
        except Exception as e:
            logger.error(f"Error calling LLM: {str(e)}")
            return f"Error: {str(e)}"
