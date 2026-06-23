"""LangGraph wiring for the autonomous brain.

The graph topology:

    START → gate
       ├─ skip_reason set → audit → END  (gate aborted)
       └─ continue → [fetch_gmail, fetch_calendar, fetch_drive,
                       fetch_huddle, fetch_open_tasks] (parallel)
                          ↓ (join)
                     download_attachments
                          ↓
                     ingest_to_rag
                          ↓
                     classify
                          ↓
                     prioritise_attempt_1   ─┐
                          │                  │
                     (judge passed?)         │
                       │       │             │
                      yes     no             │
                       │       │             │
                       │   prioritise_attempt_2   ─┐
                       │       │ (judge passed?)   │
                       │       │       │           │
                       │      yes     no           │
                       │       │       │           │
                       │       │   prioritise_rule_fallback
                       │       │       │
                       └───────┴───────┘
                              ↓
                       create_tasks
                          ↓
                       fire_agents
                          ↓
                       wait_proposals
                          ↓
                     compose_primary       ─┐
                          │                  │
                     (quality passed?)       │
                       │       │             │
                      yes     no             │
                       │       │             │
                       │   compose_prune_promote   ─┐
                       │       │ (quality passed?)  │
                       │       │       │            │
                       │      yes     no            │
                       │       │       │            │
                       │       │   compose_minimal_safe
                       │       │       │
                       └───────┴───────┘
                              ↓
                            send
                              ↓
                            audit → END

Why split prioritise + compose into multiple nodes:
  * Each retry attempt produces its own ``brain_traces`` row, so the
    admin timeline shows "attempt 1 failed judge (score 0.35) → retry
    cleared at 0.78" instead of one opaque "prioritise" row.
  * Conditional edges are declarative — the retry policy lives in the
    graph wiring, not inside a node, which makes it easy to reason
    about and change without re-reading 200 lines of node code.
  * Langfuse traces inherit the same shape — each attempt is a span.

State fields the routers read:
  * ``state.meta["prioritise_attempt_1_passed"]`` / ``_attempt_2_passed``
  * ``state.meta["digest_quality_passed"]``
"""

from __future__ import annotations

from typing import Literal

from langgraph.graph import END, START, StateGraph

from .nodes import (
    audit,
    classify,
    compose,
    create_tasks,
    download_attachments,
    fetch_calendar,
    fetch_drive,
    fetch_gmail,
    fetch_huddle,
    fetch_open_tasks,
    fire_agents,
    gate,
    ingest_to_rag,
    prioritise,
    send,
    wait_proposals,
)
from .state import BrainState


# ─────────────────────────────────────────────────────────────────────
# Conditional routers
# ─────────────────────────────────────────────────────────────────────


def _after_gate(state: BrainState) -> Literal["skip", "continue"]:
    """If the gate set a skip_reason, jump straight to audit so we
    still record the run + reason. Otherwise fan out to the parallel
    fetch nodes."""
    return "skip" if state.skip_reason else "continue"


# ─────────────────────────────────────────────────────────────────────
# Build the graph
# ─────────────────────────────────────────────────────────────────────


def _build_graph():
    builder = StateGraph(BrainState)

    # ── Nodes ───────────────────────────────────────────────────────
    builder.add_node("gate", gate.gate)
    builder.add_node("fetch_gmail", fetch_gmail.fetch_gmail)
    builder.add_node("fetch_calendar", fetch_calendar.fetch_calendar)
    builder.add_node("fetch_drive", fetch_drive.fetch_drive)
    builder.add_node("fetch_huddle", fetch_huddle.fetch_huddle)
    builder.add_node("fetch_open_tasks", fetch_open_tasks.fetch_open_tasks)
    builder.add_node("download_attachments", download_attachments.download_attachments)
    builder.add_node("ingest_to_rag", ingest_to_rag.ingest_to_rag)
    builder.add_node("classify", classify.classify)

    # Prioritise sub-graph — three nodes wired with conditional routing.
    builder.add_node("prioritise_attempt_1", prioritise.prioritise_attempt_1)
    builder.add_node("prioritise_attempt_2", prioritise.prioritise_attempt_2)
    builder.add_node("prioritise_rule_fallback", prioritise.prioritise_rule_fallback)

    builder.add_node("create_tasks", create_tasks.create_tasks)
    builder.add_node("fire_agents", fire_agents.fire_agents)
    builder.add_node("wait_proposals", wait_proposals.wait_proposals)

    # Compose sub-graph — three nodes wired with conditional routing.
    builder.add_node("compose_primary", compose.compose_primary)
    builder.add_node("compose_prune_promote", compose.compose_prune_promote)
    builder.add_node("compose_minimal_safe", compose.compose_minimal_safe)

    builder.add_node("send", send.send)
    builder.add_node("audit", audit.audit)

    # ── Edges ───────────────────────────────────────────────────────
    builder.add_edge(START, "gate")

    # Conditional after gate — skip to audit or fan out to fetchers.
    builder.add_conditional_edges(
        "gate",
        _after_gate,
        {
            "skip": "audit",
            "continue": "fetch_gmail",
        },
    )
    builder.add_edge("gate", "fetch_calendar")
    builder.add_edge("gate", "fetch_drive")
    builder.add_edge("gate", "fetch_huddle")
    builder.add_edge("gate", "fetch_open_tasks")

    # Join: every fetcher feeds the same successor.
    for fetcher in (
        "fetch_gmail", "fetch_calendar", "fetch_drive",
        "fetch_huddle", "fetch_open_tasks",
    ):
        builder.add_edge(fetcher, "download_attachments")

    builder.add_edge("download_attachments", "ingest_to_rag")
    builder.add_edge("ingest_to_rag", "classify")

    # ── Prioritise retry chain ──────────────────────────────────────
    builder.add_edge("classify", "prioritise_attempt_1")
    builder.add_conditional_edges(
        "prioritise_attempt_1",
        prioritise.after_attempt_1,
        {
            "ok": "create_tasks",
            "retry": "prioritise_attempt_2",
        },
    )
    builder.add_conditional_edges(
        "prioritise_attempt_2",
        prioritise.after_attempt_2,
        {
            "ok": "create_tasks",
            "fallback": "prioritise_rule_fallback",
        },
    )
    builder.add_edge("prioritise_rule_fallback", "create_tasks")

    builder.add_edge("create_tasks", "fire_agents")
    builder.add_edge("fire_agents", "wait_proposals")

    # ── Compose retry chain ─────────────────────────────────────────
    builder.add_edge("wait_proposals", "compose_primary")
    builder.add_conditional_edges(
        "compose_primary",
        compose.after_primary,
        {
            "ok": "send",
            "retry": "compose_prune_promote",
        },
    )
    builder.add_conditional_edges(
        "compose_prune_promote",
        compose.after_prune_promote,
        {
            "ok": "send",
            "retry": "compose_minimal_safe",
        },
    )
    builder.add_edge("compose_minimal_safe", "send")

    builder.add_edge("send", "audit")
    builder.add_edge("audit", END)

    return builder.compile()


# Compiled once at import. The compiled graph is stateless; state lives
# on each invocation.
brain_graph = _build_graph()


__all__ = ["brain_graph"]
