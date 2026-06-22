"""Lumicoria Autonomous Brain.

The brain runs as a daily LangGraph state machine — one fan-out per
user, twice a day (06:00 + 22:00 local). It reads Gmail, Calendar,
Drive, and recent Huddles; ingests everything into RAG; uses the
specialised Brain Agent to prioritise; creates Lumicoria tasks with
the right specialist agent already assigned; fires those agents to
draft proposals; and sends a digest email with one-click review
buttons.

Layout:
    state.py      Pydantic state passed between nodes
    tracing.py    @traced_node decorator + Postgres BrainTrace writes
    _time.py      TZ-aware "is it 06:00 local for this user?" helper
    nodes/        One file per node — gate, fetch_*, ingest, classify,
                  prioritise, create_tasks, fire_agents, wait_proposals,
                  compose, send, audit
    graph.py      LangGraph wiring — entry points, edges, conditionals
    runner.py     Drives a single run end-to-end (writes BrainRun row,
                  invokes graph, finalises)

Phase 2 ships every node as a stub returning empty results — the graph
compiles, runs end-to-end, and persists 16 BrainTrace rows + 1 BrainRun
row. Phase 3+ fills in real LLM logic.
"""
