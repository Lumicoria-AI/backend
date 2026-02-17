"""
Intent Router — The Brain of Lumicoria

Takes a user message and conversation history, uses the LLM to classify
which agent should handle it. Returns the agent key.

This is the ONLY new file needed for routing. Everything else is wiring.
"""

import structlog
import json
from typing import Optional, List, Dict, Any

logger = structlog.get_logger(__name__)

# ── All agent types the router can choose from ──
AGENT_REGISTRY = {
    "research": "Deep web research, academic topics, literature reviews, fact-finding, trend analysis",
    "research_mentor": "Research methodology guidance, hypothesis framing, study design, academic writing",
    "document": "Document processing, OCR, text extraction, PDF/DOCX analysis, document classification",
    "meeting": "Meeting notes, scheduling, action items, follow-ups, calendar management",
    "meeting_fact_checker": "Fact-checking claims, verifying statistics, source validation",
    "creative": "Creative writing, brainstorming, content creation, storytelling, copywriting",
    "social_media": "Social media posts, engagement strategy, platform-specific content, hashtags",
    "student": "Study help, homework, exam prep, flashcards, academic Q&A",
    "learning_coach": "Personalized learning plans, skill development, curriculum design, progress tracking",
    "rag": "Questions about uploaded documents, knowledge base queries, context-specific answers",
    "data_analysis": "Data analysis, statistics, CSV processing, charts, anomaly detection, data insights",
    "knowledge_graph": "Entity relationships, knowledge mapping, connection discovery",
    "legal_document": "Legal documents, contract analysis, clause extraction, legal terminology",
    "translation": "Language translation, multi-language content, localization",
    "customer_service": "Customer support, FAQ generation, response templates, ticket handling",
    "ethics_bias": "Ethics review, bias detection, fairness auditing, responsible AI",
    "wellbeing": "Mental health, stress management, work-life balance, wellness tips, break reminders",
    "focus_flow": "Focus techniques, deep work, productivity, distraction management, flow state",
    "workspace_ergonomics": "Desk setup, posture, ergonomic equipment, physical workspace optimization",
    "vision": "Image analysis, photo description, visual content understanding, OCR from images",
    "general": "General conversation, greetings, unclear intent, chitchat, anything that doesn't fit above",
}

ROUTER_PROMPT = """You are an intent classifier for the Lumicoria AI platform. Your ONLY job is to decide which specialized agent should handle the user's message.

Available agents and their specialties:
{agent_list}

Rules:
1. Return ONLY a JSON object: {{"agent": "<agent_key>", "confidence": <0.0-1.0>, "reason": "<brief reason>"}}
2. If the user is asking about their uploaded documents or knowledge base, choose "rag".
3. If the intent is ambiguous or is general chat/greeting, choose "general".
4. Consider the conversation history for context — a follow-up question should usually go to the same agent.
5. Never explain your reasoning outside the JSON. Return ONLY the JSON object.

Conversation history (last {history_count} messages):
{history}

User's new message:
{message}

Return your classification as JSON:"""


class AgentRouter:
    """
    LLM-based intent classifier that routes user messages to the right agent.
    
    Usage:
        router = AgentRouter()
        result = await router.route("Help me analyze this CSV file")
        print(result)  # {"agent": "data_analysis", "confidence": 0.95, "reason": "..."}
    """
    
    def __init__(self, provider: str = None, model: str = None):
        """
        Args:
            provider: LLM provider to use for classification. Defaults to DEFAULT_LLM_PROVIDER.
            model: Specific model for routing. Smaller/faster models work well here.
        """
        from backend.ai_models import get_llm_client
        self.llm_client = get_llm_client(provider=provider)
        self.model = model  # None = use provider default
        self._default_agent = "general"
    
    async def route(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None,
        max_history: int = 6,
    ) -> Dict[str, Any]:
        """
        Classify a user message and return the target agent.
        
        Args:
            message: The user's message to classify.
            conversation_history: Previous messages for context. 
                Each dict has 'role' ('user'/'assistant') and 'content'.
            max_history: Max previous messages to include (keeps prompt small + cheap).
            
        Returns:
            Dict with keys:
                - agent: str — the agent key (e.g., "research", "data_analysis")
                - confidence: float — 0.0 to 1.0
                - reason: str — brief explanation
        """
        from backend.ai_models import LLMConfig

        # Build the agent list for the prompt
        agent_list = "\n".join(
            f'  - "{key}": {desc}' for key, desc in AGENT_REGISTRY.items()
        )
        
        # Format conversation history
        history = conversation_history or []
        recent = history[-max_history:] if len(history) > max_history else history
        
        if recent:
            history_text = "\n".join(
                f"  [{msg['role']}]: {msg['content'][:200]}"  # Truncate long messages
                for msg in recent
            )
        else:
            history_text = "  (No previous messages — this is the start of the conversation)"
        
        # Build the prompt
        prompt = ROUTER_PROMPT.format(
            agent_list=agent_list,
            history_count=len(recent),
            history=history_text,
            message=message,
        )
        
        try:
            config = LLMConfig(
                model=self.model,
                temperature=0.1,   # Low temp for consistent classification
                max_tokens=150,    # Short response — just JSON
            )
            
            response = await self.llm_client.generate(
                messages=[{"role": "user", "content": prompt}],
                config=config,
            )
            
            # Parse the JSON response
            result = self._parse_response(response.content)
            
            logger.info(
                "router_decision",
                message_preview=message[:80],
                agent=result["agent"],
                confidence=result["confidence"],
                reason=result["reason"],
            )
            
            return result
            
        except Exception as e:
            logger.error("router_failed", error=str(e), message_preview=message[:80])
            return {
                "agent": self._default_agent,
                "confidence": 0.0,
                "reason": f"Router error — defaulting to general: {str(e)}",
            }
    
    def _parse_response(self, raw: str) -> Dict[str, Any]:
        """Parse the LLM's JSON response, with fallbacks for malformed output."""
        import re

        # Strip markdown code fences if the LLM wraps the JSON
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]  # Remove first line (```json)
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]  # Remove trailing ```
        cleaned = cleaned.strip()
        
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            # Try to find JSON object in the response
            match = re.search(r'\{[^}]+\}', raw)
            if match:
                try:
                    parsed = json.loads(match.group())
                except json.JSONDecodeError:
                    return {"agent": self._default_agent, "confidence": 0.0, "reason": "Failed to parse router response"}
            else:
                return {"agent": self._default_agent, "confidence": 0.0, "reason": "No JSON in router response"}
        
        # Validate the agent key
        agent = parsed.get("agent", self._default_agent)
        if agent not in AGENT_REGISTRY:
            logger.warning("router_unknown_agent", agent=agent)
            agent = self._default_agent
        
        return {
            "agent": agent,
            "confidence": min(1.0, max(0.0, float(parsed.get("confidence", 0.5)))),
            "reason": parsed.get("reason", ""),
        }


# ── Module-level singleton for convenience ──
_router_instance: Optional[AgentRouter] = None

async def get_router() -> AgentRouter:
    """Get or create the singleton router instance."""
    global _router_instance
    if _router_instance is None:
        _router_instance = AgentRouter()
    return _router_instance
