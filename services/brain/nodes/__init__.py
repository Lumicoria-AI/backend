"""Brain graph nodes.

One file per node — every node is decorated with @traced_node and
returns a LangGraph state-update dict. Nodes are stateless: they read
the BrainState, do their thing, return updates. Side effects (DB
writes, Google API calls, LLM calls) live inside the nodes; the graph
just orchestrates.

Phase 2 ships everything as a working stub — the graph compiles and
runs end-to-end, persisting 16 trace rows. Phase 3+ fills the actual
fetch / classify / prioritise / send logic in place.
"""

from . import (
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

__all__ = [
    "audit",
    "classify",
    "compose",
    "create_tasks",
    "download_attachments",
    "fetch_calendar",
    "fetch_drive",
    "fetch_gmail",
    "fetch_huddle",
    "fetch_open_tasks",
    "fire_agents",
    "gate",
    "ingest_to_rag",
    "prioritise",
    "send",
    "wait_proposals",
]
