"""Brain Agent — the autonomous daily prioritisation agent.

The Brain Agent is the reasoning core of the autonomous morning +
evening digest pipeline (services/brain/). It is NOT routed to from
chat — the LangGraph state machine invokes it directly inside the
`prioritise` node.

Input  (``process_async(data)``):
    data = {
        "classified_emails":  List[ClassifiedEmail dicts],
        "calendar_events":    List[CalendarEventRef dicts],
        "open_tasks":         List[OpenTaskRef dicts],
        "huddle_recents":     List[HuddleSummaryRef dicts],
        "user_id":            str,
        "organization_id":    Optional[str],
        "mode":               "morning" | "evening",
    }

Output:
    {
        "ranked_actions":  List[RankedAction dicts],  # for create_tasks
        "summary_line":    str,                       # for compose
        "confidence":      float (0-1),
        "reasoning":       str,                       # short rationale
    }

Pipeline inside this agent:

  1. **Ground**: pull top-k RAG snippets for each classified email
     via the rebuilt context_service.get_context_for_query (hybrid
     search + diversity + recency, all transparent to us).
  2. **Reason**: one structured LLM call with the entire day's items
     + their grounding context. The LLM outputs a JSON array of
     ranked actions with title, priority, agent_key, evidence, and
     self-reported confidence.
  3. **Validate**: schema_eval against the RankedAction Pydantic
     model. Drops malformed items; if too many drop, the caller
     (prioritise node) retries us with a stricter prompt.
  4. **Score**: mean confidence across surviving items. The caller
     uses this to decide retry/fallback.

The agent is provider-agnostic — it talks to the existing
``self.llm_client.generate(messages)`` interface that every other
agent uses, so swapping Anthropic ↔ OpenAI ↔ Gemini for the brain is
a one-line provider switch.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog

from backend.agents.base_agent import BaseAgent
from backend.services.brain.evals import check_confidence, check_schema
from backend.services.brain.state import RankedAction

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Prompt scaffolding
# ─────────────────────────────────────────────────────────────────────


_SYSTEM_PROMPT = """You are the Brain Agent inside Lumicoria — a focused, calm,
production-grade reasoning agent that runs once at 06:00 and once at 22:00 to
shape the user's day.

Your job in MORNING mode:
  Decide what the user should do today. Output a ranked, deduplicated list of
  concrete tasks pulled from new emails, calendar events, Drive changes, and
  yesterday's huddle summaries. Each task carries: a title (verb-led), a
  description, a priority, optional due_date, an assigned specialist agent
  key, and pointer(s) to the evidence (message_ids / event_ids / file_ids).

Your job in EVENING mode:
  Recap what shipped, what slipped, and where energy should go tomorrow.
  Output the same ranked list shape — these become tomorrow's draft tasks
  the user can approve in the morning.

You must NEVER invent tasks that have no evidence in the provided inputs.
You must NEVER include sensitive content (passwords, full SSNs, full credit-
card numbers) in titles or descriptions. You must prefer fewer high-confidence
tasks over many noisy ones — quality beats quantity.

Available specialist agents you can assign each task to:
{agent_directory}

Priority rubric:
  critical: explicit deadline today/tomorrow with high stakes, or a
            named blocker for an active project.
  high:     deadline within a week, or a clear ask from a named human.
  medium:   noteworthy but not time-sensitive (FYI digests, scheduling).
  low:      reference-only, nice-to-have.

You will output ONLY a JSON object — no prose, no markdown fences — matching
the schema in the user message exactly.
"""


_USER_PROMPT_TEMPLATE = """Mode: {mode}
User timezone hint: {timezone}
Today (UTC): {now_iso}
Default due-date if none explicit: {default_due_iso}

# Classified emails ({n_emails})
{emails_block}

# Calendar (next 24h, {n_events})
{events_block}

# Open Lumicoria tasks ({n_open})
{open_tasks_block}

# Recent huddle summaries ({n_huddles})
{huddles_block}

# Grounding context (RAG snippets)
{context_block}

# Schema (output exactly this shape — JSON only)
{{
  "ranked_actions": [
    {{
      "title": "verb-led short title",
      "description": "one or two sentences",
      "priority": "critical|high|medium|low",
      "due_date": "ISO-8601 string or null",
      "assigned_to_agent": "<key from the agent directory above or null>",
      "confidence": 0.0,
      "evidence_message_ids": ["..."],
      "evidence_event_ids":   ["..."],
      "evidence_file_ids":    ["..."]
    }}
  ],
  "summary_line": "one sentence the user reads first thing",
  "overall_confidence": 0.0,
  "reasoning": "two-sentence rationale describing how you ranked"
}}

Rules:
  • Maximum {max_actions} actions.
  • Each action's evidence_* lists must reference IDs from the inputs above.
  • If you have no high-confidence actions, return an empty ranked_actions
    list and explain in `reasoning`.
"""


# ─────────────────────────────────────────────────────────────────────
# Agent
# ─────────────────────────────────────────────────────────────────────


class BrainAgent(BaseAgent):
    """Daily prioritisation + agent-assignment reasoning agent."""

    def get_model_name(self) -> str:
        # Inherits the global default model. Provider routing happens in
        # `initialize_models` from BaseAgent.
        return (self.model_config or {}).get("model", "")

    async def process_async(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.llm_client:
            return {
                "ranked_actions": [],
                "summary_line": "",
                "confidence": 0.0,
                "reasoning": "LLM client not initialised",
                "error": "llm_unavailable",
            }

        mode = (data.get("mode") or "morning").lower()
        if mode not in ("morning", "evening"):
            mode = "morning"

        classified = data.get("classified_emails") or []
        events = data.get("calendar_events") or []
        open_tasks = data.get("open_tasks") or []
        huddles = data.get("huddle_recents") or []
        user_id = str(data.get("user_id") or "")
        organization_id = data.get("organization_id")
        timezone = data.get("timezone") or "UTC"
        max_actions = int(data.get("max_actions") or 8)

        # 1. Ground each classified email with a small RAG snippet.
        context_block = await self._gather_grounding(
            classified=classified,
            user_id=user_id,
            organization_id=organization_id,
            mode=mode,
        )

        # 2. Build the prompt.
        now_utc = datetime.utcnow().replace(microsecond=0)
        default_due = (now_utc + timedelta(days=5)).isoformat() + "Z"

        agent_directory = self._render_agent_directory()
        user_prompt = _USER_PROMPT_TEMPLATE.format(
            mode=mode,
            timezone=timezone,
            now_iso=now_utc.isoformat() + "Z",
            default_due_iso=default_due,
            n_emails=len(classified),
            n_events=len(events),
            n_open=len(open_tasks),
            n_huddles=len(huddles),
            emails_block=self._render_emails(classified),
            events_block=self._render_events(events),
            open_tasks_block=self._render_open_tasks(open_tasks),
            huddles_block=self._render_huddles(huddles),
            context_block=context_block,
            max_actions=max_actions,
        )
        system_prompt = _SYSTEM_PROMPT.format(agent_directory=agent_directory)

        # 3. Call the LLM. Try twice — the second time we tighten the
        #    system prompt if the first produced an unparseable result.
        raw_text, parsed, eval_result = await self._call_with_retry(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # 4. Confidence eval.
        actions: List[RankedAction] = parsed
        conf_eval = check_confidence(actions, floor=0.5, attr="confidence")

        # Overall confidence — LLM self-reports `overall_confidence` in
        # the JSON; we extract it for the trace.
        overall_confidence = self._extract_overall_confidence(raw_text)

        return {
            "ranked_actions": [a.model_dump() for a in actions],
            "summary_line": self._extract_summary_line(raw_text),
            "confidence": overall_confidence,
            "mean_action_confidence": conf_eval.score,
            "reasoning": self._extract_reasoning(raw_text),
            "eval": {
                "schema": eval_result.model_dump(),
                "confidence": conf_eval.model_dump(),
            },
        }

    # ─────────────────────────────────────────────────────────────────
    # Internals
    # ─────────────────────────────────────────────────────────────────

    async def _call_with_retry(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ):
        """Two-shot: standard prompt, then stricter prompt on parse fail."""
        from ..services.brain.state import RankedAction as _RA

        attempts = [
            ("standard", system_prompt, user_prompt),
            (
                "strict",
                system_prompt + "\n\nCRITICAL: Output strict JSON only. "
                "No prose. No markdown fences. Validate against the schema "
                "before responding.",
                user_prompt,
            ),
        ]

        last_raw = ""
        last_actions: List[_RA] = []
        last_eval = None

        for label, sys_p, usr_p in attempts:
            try:
                response = await self.llm_client.generate(
                    messages=[
                        {"role": "system", "content": sys_p},
                        {"role": "user", "content": usr_p},
                    ],
                )
                raw = getattr(response, "content", None) or str(response)
            except Exception as e:  # noqa: BLE001
                logger.exception("brain_agent.llm_call_failed", attempt=label)
                continue

            payload = self._extract_actions_json(raw)
            actions, schema_eval_result = check_schema(payload, _RA, pass_floor=0.7)

            last_raw = raw
            last_actions = actions
            last_eval = schema_eval_result

            if schema_eval_result.passed:
                logger.info(
                    "brain_agent.parse_ok",
                    attempt=label,
                    count=len(actions),
                    score=schema_eval_result.score,
                )
                break

            logger.warning(
                "brain_agent.parse_failed",
                attempt=label,
                reason=schema_eval_result.reason,
            )

        return last_raw, last_actions, last_eval

    async def _gather_grounding(
        self,
        *,
        classified: List[Dict[str, Any]],
        user_id: str,
        organization_id: Optional[str],
        mode: str,
    ) -> str:
        """Pull a short RAG snippet for each (high-priority) classified
        email so the LLM has evidence to anchor its prioritisation on.

        We only ground the top 8 — pulling RAG context per email
        balloons cost otherwise. The brain's structured input already
        carries snippets from Gmail's preview, so we focus the
        retrieval on items most likely to need disambiguation.
        """
        if not classified or not user_id:
            return "(no grounding context — empty inputs)"

        from backend.services.context_service import context_service

        try:
            top = sorted(
                classified,
                key=lambda c: {"critical": 0, "high": 1, "medium": 2, "low": 3}.get(
                    (c.get("urgency") or "low").lower(), 3
                ),
            )[:8]
        except Exception:
            top = classified[:8]

        snippets: List[str] = []
        for c in top:
            query = c.get("summary") or c.get("message_id") or ""
            if not query:
                continue
            try:
                ctx = await context_service.get_context_for_query(
                    query=query,
                    user_id=user_id,
                    organization_id=organization_id,
                    k=3,
                    token_budget=400,
                    recency_half_life_days=7,
                )
                for chunk in (ctx.get("context") or [])[:3]:
                    text = (chunk.get("text") or "")[:300]
                    if text:
                        src = (chunk.get("source") or "unknown")
                        snippets.append(f"  [{src}] {text}")
            except Exception as e:  # noqa: BLE001
                logger.debug("brain_agent.grounding_skip", error=str(e))

        if not snippets:
            return "(no grounding snippets found)"
        return "\n".join(snippets[:24])  # cap to keep prompt under context window

    def _render_agent_directory(self) -> str:
        try:
            from backend.agents.router import AGENT_REGISTRY
        except Exception:
            return "  (agent registry unavailable)"
        return "\n".join(f'  - "{k}": {v}' for k, v in AGENT_REGISTRY.items())

    @staticmethod
    def _render_emails(emails: List[Dict[str, Any]]) -> str:
        if not emails:
            return "(none)"
        lines = []
        for i, e in enumerate(emails[:50]):
            lines.append(
                f"  [{i}] id={e.get('message_id')} "
                f"cat={e.get('category')} urg={e.get('urgency')} "
                f"conf={e.get('confidence', 0):.2f} "
                f"summary={(e.get('summary') or '')[:160]!r}"
            )
        if len(emails) > 50:
            lines.append(f"  … and {len(emails) - 50} more (truncated)")
        return "\n".join(lines)

    @staticmethod
    def _render_events(events: List[Dict[str, Any]]) -> str:
        if not events:
            return "(none)"
        lines = []
        for ev in events[:20]:
            lines.append(
                f"  - id={ev.get('event_id')} "
                f"{(ev.get('start') or '')} → {(ev.get('end') or '')}: "
                f"{(ev.get('summary') or '')[:100]!r}"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_open_tasks(tasks: List[Dict[str, Any]]) -> str:
        if not tasks:
            return "(none)"
        lines = []
        for t in tasks[:20]:
            lines.append(
                f"  - id={t.get('task_id')} "
                f"p={t.get('priority')} due={t.get('due_date')}: "
                f"{(t.get('title') or '')[:100]!r}"
            )
        return "\n".join(lines)

    @staticmethod
    def _render_huddles(huddles: List[Dict[str, Any]]) -> str:
        if not huddles:
            return "(none)"
        lines = []
        for h in huddles[:10]:
            lines.append(
                f"  - id={h.get('huddle_id')} ended={h.get('ended_at')}: "
                f"{(h.get('summary') or h.get('title') or '')[:200]!r}"
            )
        return "\n".join(lines)

    @staticmethod
    def _extract_actions_json(raw: str) -> Any:
        """Pull the ranked_actions array from the LLM response. Robust
        to: bare JSON, JSON wrapped in markdown fences, or a string
        that contains a JSON block somewhere inside it."""
        if not raw:
            return []
        text = raw.strip()
        # Strip code fences if present.
        if text.startswith("```"):
            lines = text.splitlines()
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and "ranked_actions" in obj:
                return obj["ranked_actions"]
            if isinstance(obj, list):
                return obj
        except Exception:
            pass
        # Last-ditch: find the first balanced [..] in the text.
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end + 1])
            except Exception:
                return []
        return []

    @staticmethod
    def _extract_overall_confidence(raw: str) -> float:
        try:
            obj = json.loads(raw.strip().strip("`"))
            if isinstance(obj, dict):
                return float(obj.get("overall_confidence") or 0.0)
        except Exception:
            pass
        return 0.0

    @staticmethod
    def _extract_summary_line(raw: str) -> str:
        try:
            obj = json.loads(raw.strip().strip("`"))
            if isinstance(obj, dict):
                return str(obj.get("summary_line") or "")
        except Exception:
            pass
        return ""

    @staticmethod
    def _extract_reasoning(raw: str) -> str:
        try:
            obj = json.loads(raw.strip().strip("`"))
            if isinstance(obj, dict):
                return str(obj.get("reasoning") or "")
        except Exception:
            pass
        return ""

    # ─────────────────────────────────────────────────────────────────
    # BaseAgent abstract surface — context_summary not strictly needed
    # for the brain (it's only called by task_executor when running a
    # task proposal, not for the daily digest flow), but BaseAgent
    # marks it @abstractmethod so we provide a sensible default.
    # ─────────────────────────────────────────────────────────────────

    async def context_summary(
        self,
        query: str,
        user_id: str,
        organization_id: Optional[str] = None,
        task_id: Optional[str] = None,
    ) -> str:
        """Return a one-line description of what this agent would do for
        the given query — surfaced in the task executor's pre-flight
        check. The brain doesn't take adhoc queries; it owns the daily
        loop. So we describe that here.
        """
        return (
            "Brain Agent reviews the day's inbox, calendar, and Drive "
            "deltas, then drafts a prioritised task list with agent "
            "assignments. Runs autonomously at 06:00 and 22:00 local — "
            "not directly invokable from chat."
        )
