from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import os # Import os for accessing environment variables
# Import necessary libraries for AI models
from backend.ai_models.perplexity import create_perplexity_client, PerplexityClient
import structlog
import asyncio

# Configure logger
logger = structlog.get_logger(__name__)

class BaseAgent(ABC):
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        # Access the global AI models config from the main app settings if available
        # Otherwise, expect agent_model_config to be passed in the agent's config
        self.model_config = config.get("agent_model_config", {}) or self._get_global_model_config(config.get("model"))
        self.perplexity_client: Optional[PerplexityClient] = None
        self.initialize_models()

    def initialize_models(self):
        """Initialize AI models and clients."""
        try:
            # If we have a model name but no model config, try to get it from the global config
            if not self.model_config and self.config.get("model"):
                self.model_config = self._get_global_model_config(self.config["model"])
            
            # Ensure we have the API key
            if not self.model_config.get("api_key"):
                # Try to get it from environment variables
                model_name = self.model_config.get("model", "").lower()
                if "perplexity" in model_name or "sonar" in model_name:
                    api_key = os.environ.get("PERPLEXITY_API_KEY")
                    if api_key:
                        self.model_config["api_key"] = api_key
            
            if not self.model_config.get("api_key"):
                raise ValueError("Perplexity API key not found in configuration or environment variables")
            
            self.perplexity_client = create_perplexity_client(
                config=self.model_config
            )
        except Exception as e:
            logger.error(f"Error initializing Perplexity client: {str(e)}")
            raise

    def _get_global_model_config(self, model_name: str) -> Dict[str, Any]:
        # This is a placeholder to retrieve model configuration from a global settings object
        # In a real FastAPI app, you would likely access this from the app state or a settings module
        # For now, we'll assume the config passed to AgentService has the structure:
        # {"ai_models": {"model_name": {...}}}
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
        """Calls the appropriate AI model based on configuration.

        Args:
            prompt: The input prompt for the model.
            model_name: Optional. The specific model name to use. If not provided,
                        the model from the agent's config will be used.
            **kwargs: Additional keyword arguments to pass to the model API.

        Returns:
            The response from the AI model.
        """
        actual_model_name = model_name or self.get_model_name()
        model_config = self._get_global_model_config(actual_model_name)
        
        # It's recommended to load API keys from environment variables for production
        api_key = os.environ.get(f"{actual_model_name.upper()}_API_KEY") or model_config.get("api_key")

        if not api_key:
            logger.warning(f"API key not found for model {actual_model_name}")
            return f"Error: API key not configured for {actual_model_name}"

        logger.info(f"Calling model {actual_model_name} with prompt: {prompt[:100]}...")
        
        try:
            if actual_model_name and ("perplexity" in actual_model_name.lower() or "sonar" in actual_model_name.lower()):
                # Use Perplexity API integration
                if not self.perplexity_client:
                    self.initialize_models()
                    
                if not self.perplexity_client:
                    logger.error("Perplexity client not initialized")
                    return "Error: Perplexity client not initialized correctly."
                
                # Run this synchronously for compatibility with the existing method signature
                try:
                    loop = asyncio.get_event_loop()
                except RuntimeError:
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                
                # Format messages for Perplexity API
                messages = [{"role": "user", "content": prompt}]
                
                # Use any model-specific parameters from kwargs
                model_params = {k: v for k, v in kwargs.items() if k not in ["messages"]}
                
                # Make the async call synchronously
                response = loop.run_until_complete(
                    self.perplexity_client.chat_completion(messages, **model_params)
                )
                
                # Return the content from the response
                return response.content
            elif actual_model_name and "gemini" in actual_model_name.lower():
                 # Example Gemini API integration:
                # import google.generativeai as genai
                # genai.configure(api_key=api_key)
                # model = genai.GenerativeModel(actual_model_name)
                # response = model.generate_content(prompt, **kwargs)
                # return response.text
                 return f"Placeholder response from Gemini ({actual_model_name}) for prompt: {prompt[:50]}..."

            elif actual_model_name and "mistral" in actual_model_name.lower():
                 # Example Mistral AI API integration:
                # from mistralai.client import MistralClient
                # client = MistralClient(api_key=api_key)
                # response = client.chat(
                #    model=actual_model_name,
                #    messages=[{"role": "user", "content": prompt}],
                #    **kwargs
                # )
                # return response.choices[0].message.content
                 return f"Placeholder response from Mistral ({actual_model_name}) for prompt: {prompt[:50]}..."

            else:
                logger.warning(f"Unknown model {actual_model_name}")
                return f"Error: Unknown model {actual_model_name}"

        except Exception as e:
            logger.error(f"Error calling model {actual_model_name}: {e}")
            # Log the error and potentially re-raise or return a specific error response
            return f"Error calling model {actual_model_name}: {str(e)}"
    
    async def _call_model_async(self, prompt: str, model_name: str = None, system_prompt: str = None, **kwargs) -> str:
        """Asynchronous version of _call_model.
        
        Args:
            prompt: The input prompt for the model.
            model_name: Optional. The specific model name to use.
            system_prompt: Optional. System instructions for the model.
            **kwargs: Additional keyword arguments to pass to the model API.
            
        Returns:
            The response from the AI model.
        """
        actual_model_name = model_name or self.get_model_name()
        model_config = self._get_global_model_config(actual_model_name)
        
        # It's recommended to load API keys from environment variables for production
        api_key = os.environ.get(f"{actual_model_name.upper()}_API_KEY") or model_config.get("api_key")

        if not api_key:
            logger.warning(f"API key not found for model {actual_model_name}")
            return "Error: API key not configured correctly."
            
        try:
            if actual_model_name and ("perplexity" in actual_model_name.lower() or "sonar" in actual_model_name.lower()):
                # Use Perplexity's Sonar models
                if not self.perplexity_client:
                    self.initialize_models()
                    
                if not self.perplexity_client:
                    logger.error("Perplexity client not initialized")
                    return "Error: Perplexity client not initialized correctly."
                
                # Format messages for Perplexity API
                messages = []
                
                # Add system prompt if provided
                if system_prompt:
                    messages.append({"role": "system", "content": system_prompt})
                
                # Add user prompt
                messages.append({"role": "user", "content": prompt})
                
                # Use any model-specific parameters from kwargs
                model_params = {k: v for k, v in kwargs.items() if k not in ["messages"]}
                
                # Make the async call
                response = await self.perplexity_client.chat_completion(messages, **model_params)
                
                # Return the content from the response
                return response.content
            else:
                logger.warning(f"Unsupported model: {actual_model_name}")
                return f"Error: Unsupported model {actual_model_name}"
                
        except Exception as e:
            logger.error(f"Error calling model {actual_model_name}: {str(e)}")
            return f"Error: {str(e)}"
