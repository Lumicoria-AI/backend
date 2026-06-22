"""LLM classify each email — action_required / scheduling / fyi / promo / spam.

Phase 2 stub. Phase 3 will:
  - Batch 10 emails per LLM call to amortise overhead.
  - Use structured output (Anthropic `tool_use` or OpenAI strict JSON).
  - Schema = list[ClassifiedEmail] — Pydantic validates each return.
  - Retry x2 on schema-fail. Below 0.6 confidence → mark
    suggested_agent=None so prioritise treats it conservatively.
  - Fallback: regex heuristic ("Re: " + last_thread_reply_at < 24h →
    action_required; from "noreply@" → informational).
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState
from ..tracing import traced_node


@traced_node("classify")
async def classify(state: BrainState) -> Dict[str, Any]:
    return {
        "classified": [],
        "__payload_summary": {
            "input_emails": len(state.emails),
            "classified": 0,
        },
        "__eval_score": 1.0,
    }
