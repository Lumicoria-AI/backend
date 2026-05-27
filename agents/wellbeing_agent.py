"""Well-being Coach agent.

Provider locked to Gemini per platform decision.  Four actions, each
backed by a strict-JSON prompt so the orchestrator + frontend never
have to parse free-form prose:

  - recommendations       → up to 8 personalised recommendations
  - break_recommendation  → single break suggestion (type / duration / activities)
  - chat                  → conversational turn (response + follow-ups)
  - weekly_reflection     → end-of-week summary (highlights / concerns / focus)

The dispatcher and JSON-salvage parser mirror the patterns we already
shipped for the Knowledge Graph, Legal Document, and Ethics & Bias
agents — defensive, never crashes on a malformed LLM response, falls
back to safe defaults instead.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional

import structlog

from .base_agent import BaseAgent

logger = structlog.get_logger(__name__)


class WellbeingAgent(BaseAgent):
    """Gemini-powered well-being coach.

    The orchestrator passes:
        { "action": "recommendations" | "break_recommendation" | "chat" | "weekly_reflection",
          "data":   {...},
          "context": {...},
          "parameters": {...} }
    """

    def __init__(self, config: Dict[str, Any]):
        # Force provider=gemini regardless of what the caller asked for —
        # the wellbeing experience is intentionally locked to Gemini for
        # cost/latency reasons.
        config = dict(config or {})
        config["provider"] = "gemini"
        super().__init__(config)

        self.capabilities = {
            "recommendations": True,
            "break_recommendation": True,
            "chat": True,
            "weekly_reflection": True,
        }

        self.model_config.update({
            "temperature": 0.4,
            "max_tokens": 8192,
            "top_p": 0.9,
        })

    # ── Dispatcher ─────────────────────────────────────────────

    async def process_async(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Action dispatcher.  Mirrors the shape used by every other
        agent on the platform."""
        try:
            action = (request or {}).get("action") or "recommendations"
            data = (request or {}).get("data") or {}
            context = (request or {}).get("context") or {}
            parameters = (request or {}).get("parameters") or {}

            if action == "recommendations":
                results = await self._generate_recommendations(data, context, parameters)
            elif action == "break_recommendation":
                results = await self._generate_break(data, context, parameters)
            elif action == "chat":
                results = await self._chat(data, context, parameters)
            elif action == "weekly_reflection":
                results = await self._weekly_reflection(data, context, parameters)
            else:
                # Legacy compatibility: if the caller passes the old
                # flat `user_data` shape (no `action` key), treat it
                # as a recommendations request.
                if "user_data" in (request or {}) or not action:
                    results = await self._generate_recommendations(
                        {"user_data": (request or {}).get("user_data") or request},
                        context,
                        parameters,
                    )
                else:
                    raise ValueError(f"Unsupported wellbeing action: {action}")

            return {
                "results": results,
                "metadata": {
                    "action": action,
                    "timestamp": datetime.utcnow().isoformat() + "Z",
                    "model": self.model_config.get("model"),
                },
            }
        except Exception as e:  # noqa: BLE001
            logger.error("wellbeing_agent_failed", error=str(e))
            return {"error": str(e), "results": {}, "metadata": {}}

    async def query_async(
        self, query: str, context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Conversational shortcut — treat any free-form query as a
        chat turn against the coach."""
        return await self.process_async({
            "action": "chat",
            "data": {"message": query},
            "context": context or {},
        })

    # ── Public legacy entry (sync) ─────────────────────────────

    def process(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sync fallback kept for backward-compatibility with the
        small set of callers that used the original `process(user_data)`
        signature.  Internally builds a recommendations prompt."""
        try:
            prompt_user = self._format_user_data(user_data or {})
            prompt = self._recommendations_prompt() + "\n\nUSER DATA:\n" + prompt_user
            raw = self._call_model(prompt)
            parsed = self._extract_json(raw)
            return {
                "wellbeing_advice": parsed or {"summary": raw[:1000]},
                "processed_at": datetime.utcnow().isoformat() + "Z",
                "model_used": self.model_config.get("model"),
            }
        except Exception as e:  # noqa: BLE001
            logger.error(f"Error processing wellbeing data: {e}")
            return {"error": f"Wellbeing analysis failed: {str(e)}"}

    # ── Action implementations ─────────────────────────────────

    async def _generate_recommendations(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        user_data = data.get("user_data") or data
        formatted = self._format_user_data(user_data)
        prompt = self._recommendations_prompt()
        raw = await self._call_llm(
            system_prompt=prompt,
            user_payload=formatted,
            parameters=parameters,
        )
        parsed = self._extract_json(raw)
        items = parsed.get("recommendations") if isinstance(parsed, dict) else None
        if not isinstance(items, list):
            items = []
        out: List[Dict[str, Any]] = []
        for raw_item in items[:8]:
            if not isinstance(raw_item, dict):
                continue
            out.append({
                "category": str(raw_item.get("category") or "general"),
                "title": str(raw_item.get("title") or "Recommendation"),
                "description": str(raw_item.get("description") or "").strip(),
                "priority": self._normalise_priority(raw_item.get("priority")),
                "suggested_activity": str(raw_item.get("suggested_activity") or ""),
                "duration_minutes": int(raw_item.get("duration_minutes") or 0) or None,
            })
        return out

    async def _generate_break(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        user_data = data.get("user_data") or data
        formatted = self._format_user_data(user_data)
        prompt = self._break_prompt()
        raw = await self._call_llm(
            system_prompt=prompt,
            user_payload=formatted,
            parameters=parameters,
        )
        parsed = self._extract_json(raw)
        if not isinstance(parsed, dict):
            parsed = {}
        return {
            "break_type": str(parsed.get("break_type") or "micro_break"),
            "duration_minutes": int(parsed.get("duration_minutes") or 5),
            "reason": str(parsed.get("reason") or "").strip(),
            "suggested_activities": [
                str(a) for a in (parsed.get("suggested_activities") or []) if a
            ][:5],
            "confidence": float(parsed.get("confidence", 0.85) or 0.85),
        }

    async def _chat(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        message = str(data.get("message") or "").strip()
        history = data.get("history") or []
        user_data = data.get("user_data") or {}

        history_str = ""
        for turn in history[-6:]:
            if not isinstance(turn, dict):
                continue
            role = str(turn.get("role") or "user").upper()
            content = str(turn.get("content") or "").strip()
            if content:
                history_str += f"{role}: {content}\n"

        user_payload = (
            f"USER MESSAGE:\n{message}\n\n"
            f"USER DATA:\n{self._format_user_data(user_data)}\n\n"
            f"RECENT CONVERSATION:\n{history_str or '(no prior turns)'}"
        )
        prompt = self._chat_prompt()
        raw = await self._call_llm(
            system_prompt=prompt,
            user_payload=user_payload,
            parameters=parameters,
        )
        parsed = self._extract_json(raw)
        if not isinstance(parsed, dict):
            parsed = {"response": raw[:2000]}
        return {
            "response": str(parsed.get("response") or "").strip(),
            "follow_up_suggestions": [
                str(s) for s in (parsed.get("follow_up_suggestions") or []) if s
            ][:4],
            "action_buttons": [
                {
                    "label": str(a.get("label") or ""),
                    "action": str(a.get("action") or ""),
                }
                for a in (parsed.get("action_buttons") or [])
                if isinstance(a, dict) and a.get("label")
            ][:3],
        }

    async def _weekly_reflection(
        self,
        data: Dict[str, Any],
        context: Dict[str, Any],
        parameters: Dict[str, Any],
    ) -> Dict[str, Any]:
        user_data = data.get("user_data") or data
        formatted = self._format_user_data(user_data)
        prompt = self._reflection_prompt()
        raw = await self._call_llm(
            system_prompt=prompt,
            user_payload=formatted,
            parameters=parameters,
        )
        parsed = self._extract_json(raw)
        if not isinstance(parsed, dict):
            parsed = {}
        return {
            "summary": str(parsed.get("summary") or "").strip(),
            "highlights": [str(x) for x in (parsed.get("highlights") or []) if x][:5],
            "concerns": [str(x) for x in (parsed.get("concerns") or []) if x][:5],
            "focus_for_next_week": [
                str(x) for x in (parsed.get("focus_for_next_week") or []) if x
            ][:5],
            "encouragement": str(parsed.get("encouragement") or "").strip(),
        }

    # ── Prompts (strict JSON) ──────────────────────────────────

    def _recommendations_prompt(self) -> str:
        return """You are a compassionate well-being coach for knowledge workers.  Given a user's recent metrics, activities, and productivity signals, produce up to 8 personalised recommendations.

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  No prose, no markdown, no fences.

{
  "recommendations": [
    {
      "category": "<break | physical | mental | focus | nutrition | sleep | social | other>",
      "title": "<short title, under 60 chars>",
      "description": "<one or two sentences, actionable>",
      "priority": "<critical | high | medium | low>",
      "suggested_activity": "<short label>",
      "duration_minutes": 5
    }
  ]
}

Rules:
- Up to 8 recommendations, ordered most useful first.
- Be specific to the data given.  Avoid generic advice ("drink water") unless it's clearly the most relevant.
- duration_minutes is the realistic time the action takes; use 0 if not applicable.
- Use only the allowed enum values, lowercase, exactly.
"""

    def _break_prompt(self) -> str:
        return """You decide whether a knowledge worker should take a break right now and what kind.

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  No prose, no markdown, no fences.

{
  "break_type": "<micro_break | short_break | lunch_break | long_break | rest_day>",
  "duration_minutes": 5,
  "reason": "<one sentence on why this break, grounded in the user's data>",
  "suggested_activities": ["<two to four short activity labels>"],
  "confidence": 0.0
}

Rules:
- Use only the allowed break_type values.
- duration_minutes: 1-5 for micro, 5-15 short, 30-60 lunch, 60+ long, 1440 rest_day.
- confidence between 0 and 1.
"""

    def _chat_prompt(self) -> str:
        return """You are a warm, evidence-based well-being coach in conversation with a knowledge worker.  Reply to their message with care, brevity, and concrete next steps.

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  No prose, no markdown, no fences.

{
  "response": "<your reply to the user, 1-4 short paragraphs>",
  "follow_up_suggestions": ["<short question the user might ask next>"],
  "action_buttons": [
    {"label": "<short label>", "action": "<one of: start_break, log_mood, view_metrics, view_recommendations, view_goals>"}
  ]
}

Rules:
- Keep `response` under 800 characters.  Prefer plain English over jargon.
- Up to 3 follow_up_suggestions, up to 3 action_buttons.
- Only use the action values listed above.  Drop any button you don't have a clean fit for.
"""

    def _reflection_prompt(self) -> str:
        return """You are a well-being coach writing a weekly reflection for a knowledge worker, based on their week's data.

OUTPUT FORMAT — STRICT
Reply with ONE JSON object and NOTHING else.  No prose, no markdown, no fences.

{
  "summary": "<two to four sentences summarising the week's wellbeing+productivity arc>",
  "highlights": ["<short positive observations grounded in the data>"],
  "concerns": ["<short concerns grounded in the data>"],
  "focus_for_next_week": ["<concrete habit or small change to try next week>"],
  "encouragement": "<one warm sentence to close>"
}

Rules:
- Up to 5 each for highlights / concerns / focus_for_next_week.
- Be specific to the metrics provided.  No filler.
"""

    # ── Helpers ────────────────────────────────────────────────

    async def _call_llm(
        self,
        *,
        system_prompt: str,
        user_payload: str,
        parameters: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Adapter onto BaseAgent._call_model_async."""
        parameters = parameters or {}
        temperature = parameters.get(
            "temperature", self.model_config.get("temperature", 0.4)
        )
        max_tokens = parameters.get(
            "max_tokens", self.model_config.get("max_tokens", 8192)
        )
        return await self._call_model_async(
            prompt=user_payload or "",
            system_prompt=system_prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _format_user_data(self, user_data: Dict[str, Any]) -> str:
        """Compact, prompt-friendly serialisation of the user payload."""
        if not user_data:
            return "(no data)"
        sections: List[str] = []

        metrics = user_data.get("metrics") or user_data.get("metrics_summary") or {}
        if isinstance(metrics, dict) and metrics:
            lines = []
            for key, val in metrics.items():
                if isinstance(val, dict):
                    avg = val.get("avg") or val.get("last") or val.get("value")
                    if avg is not None:
                        lines.append(f"- {key}: avg {avg}, n={val.get('count', 1)}")
                elif isinstance(val, (int, float, str)):
                    lines.append(f"- {key}: {val}")
            if lines:
                sections.append("Metrics:\n" + "\n".join(lines))

        prod = user_data.get("productivity") or {}
        if isinstance(prod, dict) and prod:
            lines = []
            for key in (
                "focus_minutes_today",
                "agent_runs_today",
                "tasks_completed_today",
                "tasks_completed_week",
                "tasks_in_progress",
                "tasks_not_started",
                "completion_ratio",
                "streak_days",
            ):
                if prod.get(key) is not None:
                    lines.append(f"- {key}: {prod[key]}")
            if lines:
                sections.append("Productivity:\n" + "\n".join(lines))

        activities = user_data.get("activity_log") or user_data.get("activities") or []
        if isinstance(activities, list) and activities:
            recent = activities[-8:]
            lines = []
            for a in recent:
                if isinstance(a, dict):
                    label = a.get("activity_type") or a.get("type") or "activity"
                    lines.append(f"- {label}")
                elif isinstance(a, str):
                    lines.append(f"- {a}")
            if lines:
                sections.append("Recent activities:\n" + "\n".join(lines))

        for key in (
            "screen_time",
            "breaks",
            "minutes_since_last_break",
            "current_time",
        ):
            if key in user_data and isinstance(user_data[key], (int, float, str)):
                sections.append(f"{key}: {user_data[key]}")

        goals = user_data.get("goals") or []
        if isinstance(goals, list) and goals:
            lines = []
            for g in goals[:5]:
                if isinstance(g, dict):
                    lines.append(
                        f"- {g.get('goal_type', 'Goal')}: {g.get('current_value', 'N/A')}/{g.get('target_value', 'N/A')}"
                    )
            if lines:
                sections.append("Goals:\n" + "\n".join(lines))

        return "\n\n".join(sections) if sections else "(no signals yet)"

    @staticmethod
    def _normalise_priority(value: Any) -> str:
        if not value:
            return "medium"
        s = str(value).strip().lower()
        if s in ("critical", "high", "medium", "low"):
            return s
        return "medium"

    # ── JSON extractor (same shape as KG / Legal / Ethics) ────

    @staticmethod
    def _extract_json(response: Any) -> Dict[str, Any]:
        """Best-effort JSON extraction.  Handles fenced output, prose
        prefixes, and partial truncation.  Returns {} on total failure."""
        if not response:
            return {}
        text = response.strip() if isinstance(response, str) else str(response)

        try:
            return json.loads(text)
        except Exception:
            pass

        fenced = re.search(
            r"```(?:json)?\s*(\{.*\}|\[.*\])\s*```",
            text,
            re.DOTALL | re.IGNORECASE,
        )
        if fenced:
            try:
                return json.loads(fenced.group(1))
            except Exception:
                pass

        stripped = re.sub(r"^```(?:json)?\s*", "", text).strip()
        try:
            return json.loads(stripped)
        except Exception:
            pass

        s = stripped.find("{")
        e = stripped.rfind("}")
        if s != -1 and e > s:
            try:
                return json.loads(stripped[s : e + 1])
            except Exception:
                pass

        logger.warning(
            "wellbeing_llm_non_json",
            len=len(text),
            head=text[:200],
        )
        return {}
