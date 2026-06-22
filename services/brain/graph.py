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
                     prioritise
                          ↓
                     create_tasks
                          ↓
                     fire_agents
                          ↓
                     wait_proposals
                          ↓
                     compose
                          ↓
                     send
                          ↓
                     audit → END

Notes:
  - The five fetch_* nodes run in parallel; LangGraph waits for all
    five to finish before advancing. State-list fields (emails, events,
    drive_changes, huddle_recents, open_tasks) are annotated
    ``Annotated[List[X], operator.add]`` in state.py so the parallel
    branches concatenate cleanly into the merged state.
  - Conditional after gate: if gate set `skip_reason`, we route
    directly to `audit` so a skipped run still gets a single trace row
    and a BrainRun summary. Everything in between is bypassed.
  - The graph is compiled once at module import and reused. LangGraph
    compiled graphs are stateless (state lives on the call); reusing
    one compile saves cold-start latency on every run.
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
    builder.add_node("prioritise", prioritise.prioritise)
    builder.add_node("create_tasks", create_tasks.create_tasks)
    builder.add_node("fire_agents", fire_agents.fire_agents)
    builder.add_node("wait_proposals", wait_proposals.wait_proposals)
    builder.add_node("compose", compose.compose)
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
            "continue": "fetch_gmail",   # all 5 fetchers fan from gate
        },
    )

    # Parallel fan-out: every fetch_* node runs concurrently.
    # LangGraph executes nodes that share a parent in parallel
    # automatically — we just connect each one to the same predecessor.
    # Because the "continue" route from the conditional edge above only
    # picks ONE next node (LangGraph 0.2 conditional API), we add the
    # remaining four fetchers as separate edges from gate.
    builder.add_edge("gate", "fetch_calendar")
    builder.add_edge("gate", "fetch_drive")
    builder.add_edge("gate", "fetch_huddle")
    builder.add_edge("gate", "fetch_open_tasks")

    # Join: every fetcher feeds the same successor. LangGraph waits for
    # all incoming edges to be satisfied before invoking the join node.
    for fetcher in (
        "fetch_gmail",
        "fetch_calendar",
        "fetch_drive",
        "fetch_huddle",
        "fetch_open_tasks",
    ):
        builder.add_edge(fetcher, "download_attachments")

    # Sequential spine from here through send.
    builder.add_edge("download_attachments", "ingest_to_rag")
    builder.add_edge("ingest_to_rag", "classify")
    builder.add_edge("classify", "prioritise")
    builder.add_edge("prioritise", "create_tasks")
    builder.add_edge("create_tasks", "fire_agents")
    builder.add_edge("fire_agents", "wait_proposals")
    builder.add_edge("wait_proposals", "compose")
    builder.add_edge("compose", "send")
    builder.add_edge("send", "audit")
    builder.add_edge("audit", END)

    return builder.compile()


# Compiled once at import. The compiled graph is stateless; state lives
# on each invocation.
brain_graph = _build_graph()


__all__ = ["brain_graph"]
