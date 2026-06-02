from typing import Dict, Any, Optional
from abc import ABC, abstractmethod
import os
# Import the provider-agnostic LLM interface
from backend.ai_models import get_llm_client, LLMClient, LLMConfig, LLMResponse
from backend.ai_models.pricing import compute_cost
from backend.core.config import settings as app_settings
import structlog
import asyncio

# Configure logger
logger = structlog.get_logger(__name__)


# ── Usage accumulator helper ────────────────────────────────────────────
#
# Every BaseAgent carries a `last_usage` dict that auto-accumulates
# tokens + cost across LLM calls.  The task_executor / step_graph call
# `agent.reset_usage()` before invoking the agent, then
# `agent.consume_usage()` after to grab whatever ran in that single
# invocation and stamp it onto the AgentRun row.
def _empty_usage() -> Dict[str, Any]:
    return {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "model_used": None,
        "provider": None,
        "calls": 0,
    }


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
        # Per-invocation usage — populated automatically by the wrapped LLM
        # client.  Callers can `reset_usage()` and `consume_usage()` to
        # bracket exactly one agent invocation.
        self.last_usage: Dict[str, Any] = _empty_usage()
        self.initialize_models()

    # ── Usage tracking ────────────────────────────────────────────────
    def reset_usage(self) -> None:
        """Clear the usage accumulator.  Call before running the agent."""
        self.last_usage = _empty_usage()

    def consume_usage(self) -> Dict[str, Any]:
        """Return the accumulator and reset it in one step."""
        u = self.last_usage
        self.last_usage = _empty_usage()
        return u

    def _record_llm_response(self, response: Any) -> None:
        """Accumulate token + cost data from one LLMResponse.

        Auto-invoked by the wrapped llm_client.generate() — agents never
        have to call this directly.  Safe to call with anything: defensive
        against missing fields so older / mocked clients don't crash it.
        """
        try:
            usage = getattr(response, "usage", None)
            prompt = int(getattr(usage, "prompt_tokens", 0) or 0)
            completion = int(getattr(usage, "completion_tokens", 0) or 0)
            total = int(getattr(usage, "total_tokens", 0) or (prompt + completion))
            model = getattr(response, "model", None) or self.get_model_name()
            provider = getattr(response, "provider", None)
            cost = compute_cost(
                model=model,
                prompt_tokens=prompt,
                completion_tokens=completion,
            )
            self.last_usage["prompt_tokens"] += prompt
            self.last_usage["completion_tokens"] += completion
            self.last_usage["total_tokens"] += total
            self.last_usage["cost_usd"] = round(self.last_usage["cost_usd"] + cost, 6)
            self.last_usage["calls"] += 1
            # First-seen model/provider wins; subsequent same-agent calls
            # tend to be on the same SKU.
            if not self.last_usage.get("model_used") and model:
                self.last_usage["model_used"] = model
            if not self.last_usage.get("provider") and provider:
                self.last_usage["provider"] = provider
        except Exception as e:  # noqa: BLE001
            logger.debug("usage_record_failed", error=str(e))

    def _wrap_llm_client_with_usage(self, client: Any) -> Any:
        """Monkey-patch `client.generate()` so every call auto-records
        token + cost data into `self.last_usage`.  Streaming calls are
        wrapped the same way — usage is read from the final chunk when
        the provider includes it.

        Idempotent: if the client is already wrapped, returns it as-is.
        """
        if client is None or getattr(client, "_lumi_usage_wrapped", False):
            return client

        original_generate = client.generate

        async def _wrapped_generate(*args, **kwargs):
            response = await original_generate(*args, **kwargs)
            try:
                self._record_llm_response(response)
            except Exception:
                pass
            return response

        client.generate = _wrapped_generate

        # Wrap streaming too — last chunk's `.usage` (when present)
        # is fed through the accumulator.
        if hasattr(client, "stream"):
            original_stream = client.stream

            async def _wrapped_stream(*args, **kwargs):
                async for chunk in original_stream(*args, **kwargs):
                    if getattr(chunk, "usage", None) is not None:
                        try:
                            self._record_llm_response(chunk)
                        except Exception:
                            pass
                    yield chunk

            client.stream = _wrapped_stream

        client._lumi_usage_wrapped = True
        return client

    def initialize_models(self):
        """Initialize the LLM client via the provider-agnostic abstraction."""
        try:
            # Determine which provider to use for this agent
            # Priority: agent config > model name mapping > global default
            provider = self._resolve_provider()

            raw_client = get_llm_client(provider=provider)
            self.llm_client = self._wrap_llm_client_with_usage(raw_client)
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
            max_tokens=kwargs.pop("max_tokens", 8192),
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
                raw_client = get_llm_client(provider=provider)
                self.llm_client = self._wrap_llm_client_with_usage(raw_client)
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
                max_tokens=kwargs.pop("max_tokens", 8192),
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

    # ── Phase 6: Context summary for autonomous task execution ──
    async def context_summary(
        self,
        query: str,
        *,
        user_id: Optional[str] = None,
        organization_id: Optional[str] = None,
        task_id: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return the agent's best-effort context for a task query.

        The autonomous task executor calls this *before* `process_async`
        so the agent has a chance to assemble its own grounding (uploaded
        documents, prior meeting notes, knowledge graph hits, etc.).

        Default implementation returns an empty snippet set + a
        suggested prompt that simply restates the query.  Specialised
        agents (RAG, Legal, Meeting) override this to pull real context.

        Returns shape:
            {
              "context_snippets": List[str],
              "sources": List[{"label": str, "url": str?, "metadata": dict}],
              "suggested_prompt": str,
            }
        """
        return {
            "context_snippets": [],
            "sources": [],
            "suggested_prompt": query,
        }
