"""
General Agent — The catch-all for greetings, chitchat, and unclear intent.

Handles any message that doesn't clearly map to a specialized agent.
Keeps conversations flowing naturally.
"""

import structlog
from typing import Dict, Any, Optional
from .base_agent import BaseAgent

logger = structlog.get_logger(__name__)

SYSTEM_PROMPT = """You are Lumicoria, a friendly and intelligent AI assistant. 
You help users with a wide range of tasks. Be conversational, helpful, and concise.
If the user's request would be better served by a specialist (like research, data analysis, 
document processing, etc.), let them know you can help with that too.
Always be warm and approachable."""


class GeneralAgent(BaseAgent):
    """Catch-all agent for general conversation and unclear intent."""

    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name = "General Agent"
        self.description = "General-purpose conversational assistant"

    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        query = data.get("query", data.get("content", data.get("prompt", "")))
        if not query:
            return {"response": "Hi! How can I help you today?", "success": True}

        try:
            response = await self._call_model_async(
                prompt=query,
                system_prompt=SYSTEM_PROMPT,
                conversation_history=data.get("conversation_history"),
                max_tokens=1024,
            )
            return {"response": response, "success": True}
        except Exception as e:
            logger.error("general_agent_error", error=str(e))
            return {
                "response": "I'm sorry, I encountered an issue. Please try again.",
                "success": False,
                "error": str(e),
            }

    async def query_async(self, query: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        return await self.process_async({"query": query, **(context or {})})

    def process(self, data: Any) -> Any:
        import asyncio
        return asyncio.run(self.process_async(data))
