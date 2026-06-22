"""Brain Agent ranks the classified items into a prioritised task list.

Phase 2 stub. Phase 3 will:
  - Instantiate `BrainAgent` from `agents/brain_agent.py` (added in
    Phase 3b) — subclasses BaseAgent, registered in
    `task_executor._agent_class_map["brain"]`.
  - Feed: classified emails + open tasks + calendar today + huddle recents
    + RAG context for each (via the rebuilt `context_service.get_context_for_query`
    — hybrid + rerank + diversity + recency).
  - The Brain Agent outputs `RankedAction` list with title, description,
    priority, assigned_to_agent, confidence, and evidence pointers.
  - Eval: confidence floor 0.6, top-N validated against the schema.
  - Fallback: rule-based ranker — sort classified items by urgency,
    take top 10, infer agent from a static keyword map.
"""

from __future__ import annotations

from typing import Any, Dict

from ..state import BrainState
from ..tracing import traced_node


@traced_node("prioritise")
async def prioritise(state: BrainState) -> Dict[str, Any]:
    return {
        "ranked_actions": [],
        "__payload_summary": {
            "classified": len(state.classified),
            "open_tasks": len(state.open_tasks),
            "ranked": 0,
        },
        "__eval_score": 1.0,
    }
