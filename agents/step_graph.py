"""
Phase 7 — Lightweight multi-agent orchestrator (custom step-graph).

This runner takes a plan produced by `agents.router.plan_steps` and walks
the steps in order, calling each agent's `context_summary` then
`process_async` (or streaming tokens directly for the final composer
step).  It emits structured events as it goes so the chat endpoint can
forward them onto the SSE stream and the frontend can render the live
"Agent flow" panel.

LangGraph is deferred — this is intentionally minimal: a sequential
walker with parent / child `agent_runs` rows, defensive timing, and a
clean async-generator surface.

Event shapes the runner yields:

    {"type": "plan",
     "multi_step": bool,
     "steps": [{"agent": str, "purpose": str}, ...],
     "reason": str}

    {"type": "step_start",
     "step_id": str,         # client-side step identifier (uuid4)
     "step_index": int,
     "agent": str,
     "purpose": str}

    # Streamed tokens, only emitted by the final composer step:
    {"type": "delta",
     "step_id": str,
     "text": str}

    {"type": "step_end",
     "step_id": str,
     "step_index": int,
     "agent": str,
     "status": "completed" | "error" | "skipped",
     "duration_ms": int,
     "output_preview": str,
     "sources_count": int,
     "error": Optional[str]}

    {"type": "all_done",
     "final_text": str,
     "all_sources": [...]}

The chat endpoint serialises each yield to an SSE `data:` frame.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from datetime import datetime
from typing import Any, AsyncGenerator, Dict, List, Optional

import structlog
from bson import ObjectId

logger = structlog.get_logger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────


def _preview(text: str, limit: int = 220) -> str:
    """Compact preview suitable for the Agent Flow panel."""
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: limit - 1].rstrip() + "…"


def _agent_label(agent_key: str) -> str:
    return agent_key.replace("_", " ").title()


# ── Runner ────────────────────────────────────────────────────────────


class StepGraph:
    """Sequential, instrumented multi-agent walker.

    Usage:
        sg = StepGraph(plan=plan, user_id=..., organization_id=..., ...)
        async for event in sg.run(query=msg, history=history, rag_context=ctx):
            ...
    """

    def __init__(
        self,
        *,
        plan: Dict[str, Any],
        user_id: str,
        organization_id: Optional[str] = None,
        conversation_id: Optional[str] = None,
        instantiate_agent,           # callable: agent_key -> agent instance
        composer_streamer=None,      # optional: (agent, messages, cfg) -> async iterator of (chunk, full)
    ) -> None:
        self.plan = plan or {"multi_step": False, "steps": [], "reason": ""}
        self.steps: List[Dict[str, str]] = list(self.plan.get("steps") or [])
        self.user_id = user_id
        self.organization_id = organization_id
        self.conversation_id = conversation_id
        self._instantiate_agent = instantiate_agent
        self._composer_streamer = composer_streamer

        # Will be filled during run() — chat endpoint reads these after the
        # generator exhausts so it can persist sources / final text.
        self.final_text: str = ""
        self.all_sources: List[Dict[str, Any]] = []
        self.context_used: int = 0
        self.parent_run_id: Optional[str] = None
        # Aggregate token / cost totals across all steps — stamped onto
        # the parent run when the plan closes.
        self._parent_tokens_in: int = 0
        self._parent_tokens_out: int = 0
        self._parent_cost: float = 0.0

    # ── public API ────────────────────────────────────────────────────

    async def run(
        self,
        *,
        query: str,
        history: Optional[List[Dict[str, str]]] = None,
        rag_context: str = "",
        rag_sources: Optional[List[Dict[str, Any]]] = None,
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Walk the plan, yielding events.  Always yields `all_done` last."""
        history = history or []
        if rag_sources:
            self.all_sources.extend(rag_sources)
            self.context_used = len(rag_sources)

        # Emit the plan upfront so the UI can pre-render the step list with
        # "pending" placeholders.
        yield {
            "type": "plan",
            "multi_step": bool(self.plan.get("multi_step")),
            "steps": [
                {"agent": s["agent"], "purpose": s["purpose"]}
                for s in self.steps
            ],
            "reason": str(self.plan.get("reason") or ""),
        }

        if not self.steps:
            # Shouldn't happen — plan_steps always returns at least one step
            # — but degrade gracefully.
            self.final_text = ""
            yield {
                "type": "all_done",
                "final_text": "",
                "all_sources": self.all_sources,
            }
            return

        # Open a parent AgentRun so analytics can join all child runs.
        await self._open_parent_run(query=query)

        # Carry over context produced by each step so the composer sees it.
        # We append each "context-gather" step's output as an additional
        # snippet block.
        accumulated_snippets: List[str] = []
        if rag_context:
            accumulated_snippets.append(rag_context)

        last_index = len(self.steps) - 1

        for step_index, step in enumerate(self.steps):
            agent_key = step["agent"]
            purpose = step["purpose"]
            is_composer = step_index == last_index

            step_id = str(uuid.uuid4())
            start_ts = time.perf_counter()

            yield {
                "type": "step_start",
                "step_id": step_id,
                "step_index": step_index,
                "agent": agent_key,
                "agent_label": _agent_label(agent_key),
                "purpose": purpose,
                "is_composer": is_composer,
            }

            agent = None
            error_msg: Optional[str] = None
            step_output: str = ""
            step_sources: List[Dict[str, Any]] = []

            try:
                agent = self._instantiate_agent(agent_key)
                if agent is None:
                    raise RuntimeError(f"Agent '{agent_key}' is not registered for orchestration.")

                # Reset usage so each step's tokens/cost are isolated.
                try:
                    agent.reset_usage()
                except Exception:
                    pass

                if is_composer:
                    # Final step — stream tokens directly to the client.  The
                    # composer reads everything accumulated so far + the original
                    # query, then writes the answer.  We DON'T re-run
                    # context_summary here — earlier steps have already gathered.
                    accumulated_context = "\n\n---\n\n".join(s for s in accumulated_snippets if s)

                    async for delta_text, full_so_far in self._stream_composer(
                        agent=agent,
                        query=query,
                        history=history,
                        accumulated_context=accumulated_context,
                        purpose=purpose,
                    ):
                        if delta_text:
                            yield {
                                "type": "delta",
                                "step_id": step_id,
                                "text": delta_text,
                            }
                            self.final_text = full_so_far
                    step_output = self.final_text
                else:
                    # Context-gather step — call context_summary then process_async.
                    ctx = {}
                    try:
                        ctx = await agent.context_summary(
                            query=query,
                            user_id=self.user_id,
                            organization_id=self.organization_id,
                            task_id=None,
                        )
                    except Exception as ctx_err:  # noqa: BLE001
                        logger.warning(
                            "step_context_summary_failed",
                            agent=agent_key, error=str(ctx_err),
                        )
                        ctx = {}

                    step_sources = list(ctx.get("sources") or [])
                    suggested_prompt = (
                        ctx.get("suggested_prompt")
                        or f"For context — {purpose}\n\nUser query: {query}"
                    )

                    agent_input = {
                        "query": suggested_prompt,
                        "text": suggested_prompt,
                        "prompt": suggested_prompt,
                        "task_id": None,
                        "user_id": self.user_id,
                        "organization_id": self.organization_id,
                        "context_snippets": ctx.get("context_snippets", []),
                        "metadata": {"phase7_step_purpose": purpose},
                    }
                    result = await agent.process_async(agent_input)
                    if isinstance(result, dict):
                        for k in ("response", "analysis", "summary", "content", "query_response"):
                            v = result.get(k)
                            if isinstance(v, str) and v.strip():
                                step_output = v
                                break
                        if not step_output:
                            for v in result.values():
                                if isinstance(v, str) and len(v.strip()) > 20:
                                    step_output = v
                                    break
                    else:
                        step_output = str(result)

                    # Stash this step's output for the composer.
                    block = (
                        f"[From {_agent_label(agent_key)} agent — {purpose}]\n{step_output}".strip()
                    )
                    accumulated_snippets.append(block)
                    self.all_sources.extend(step_sources)
                    if step_sources:
                        self.context_used += len(step_sources)

            except Exception as e:  # noqa: BLE001
                error_msg = str(e)[:400]
                logger.error(
                    "step_failed",
                    agent=agent_key, step_index=step_index, error=error_msg,
                )

            duration_ms = int((time.perf_counter() - start_ts) * 1000)
            status = "error" if error_msg else "completed"

            # Grab whatever tokens / cost the agent burned during this step.
            step_usage: Dict[str, Any] = {}
            if agent is not None:
                try:
                    step_usage = agent.consume_usage() or {}
                except Exception:
                    step_usage = {}

            # Persist the child run row.
            await self._record_child_run(
                step_index=step_index,
                agent_key=agent_key,
                purpose=purpose,
                input_payload={"query": query, "purpose": purpose},
                output_text=step_output,
                error=error_msg,
                duration_ms=duration_ms,
                sources_count=len(step_sources),
                usage=step_usage,
            )

            yield {
                "type": "step_end",
                "step_id": step_id,
                "step_index": step_index,
                "agent": agent_key,
                "agent_label": _agent_label(agent_key),
                "purpose": purpose,
                "status": status,
                "duration_ms": duration_ms,
                "output_preview": _preview(step_output),
                "sources_count": len(step_sources),
                "error": error_msg,
            }

        # Close the parent run.
        await self._close_parent_run(error=None if self.final_text else "no_composer_output")

        yield {
            "type": "all_done",
            "final_text": self.final_text,
            "all_sources": self.all_sources,
            "context_used": self.context_used,
        }

    # ── composer streaming ────────────────────────────────────────────

    async def _stream_composer(
        self,
        *,
        agent,
        query: str,
        history: List[Dict[str, str]],
        accumulated_context: str,
        purpose: str,
    ) -> AsyncGenerator[tuple, None]:
        """Yield (delta_chunk, running_full_text) tuples.

        Uses the agent's LLM client `.stream()` when available (the common
        case — every BaseAgent exposes `llm_client`).  Falls back to a
        single-shot `process_async` and emits the whole thing as one delta.
        """
        if self._composer_streamer:
            # Caller-provided streamer (chat endpoint passes its own that
            # builds the system prompt + RAG framing identical to today's
            # single-step flow).
            async for delta_text, full_text in self._composer_streamer(
                agent=agent,
                query=query,
                history=history,
                accumulated_context=accumulated_context,
                purpose=purpose,
            ):
                yield delta_text, full_text
            return

        # Default streamer — used when the chat endpoint doesn't override.
        from backend.ai_models import LLMConfig

        llm = getattr(agent, "llm_client", None) or getattr(agent, "perplexity_client", None)
        base_system = getattr(agent, "system_prompt", None) or (
            "You are Lumicoria.ai, a helpful AI assistant. Answer the user's question accurately and helpfully."
        )

        if accumulated_context:
            system_prompt = (
                f"{base_system}\n\n"
                "You have been given research and context from other Lumicoria agents below. "
                "Use it to compose a clear, helpful final answer to the user.  When you cite "
                "an upstream source by number, place [1], [2] inline.\n\n"
                f"{accumulated_context}"
            )
        else:
            system_prompt = base_system

        messages = [{"role": "system", "content": system_prompt}]
        if history:
            for m in history[-8:]:
                messages.append({"role": m.get("role", "user"), "content": m.get("content", "")})
        messages.append({"role": "user", "content": query})

        cfg = LLMConfig(temperature=0.7, max_tokens=None)
        full = ""

        if llm and hasattr(llm, "stream"):
            try:
                async for chunk in llm.stream(messages, config=cfg):
                    if getattr(chunk, "content", None):
                        full += chunk.content
                        yield chunk.content, full
                return
            except Exception as e:  # noqa: BLE001
                logger.warning("composer_stream_failed_falling_back", error=str(e))

        # Fallback — single delta.
        try:
            agent_input = {
                "query": query, "content": query, "prompt": query,
                "user_id": self.user_id, "conversation_id": self.conversation_id,
                "conversation_history": history,
                "context_snippets": [accumulated_context] if accumulated_context else [],
            }
            result = await agent.process_async(agent_input)
            text = ""
            if isinstance(result, dict):
                text = (
                    result.get("response")
                    or result.get("analysis")
                    or result.get("summary")
                    or result.get("content")
                    or result.get("query_response")
                    or ""
                )
            else:
                text = str(result)
            if text:
                full = text
                yield text, full
        except Exception as e:  # noqa: BLE001
            logger.error("composer_fallback_failed", error=str(e))
            err = f"Error composing answer: {str(e)[:200]}"
            full = err
            yield err, full

    # ── AgentRun persistence ──────────────────────────────────────────

    async def _open_parent_run(self, *, query: str) -> None:
        try:
            from backend.db.mongodb.repositories.agent_run_repository import agent_run_repository
            from backend.db.mongodb.models.agent_run import AgentRunCreate, AgentRunTrigger

            user_oid = ObjectId(self.user_id) if ObjectId.is_valid(str(self.user_id)) else ObjectId()
            org_oid = (
                ObjectId(self.organization_id)
                if self.organization_id and ObjectId.is_valid(str(self.organization_id))
                else None
            )
            payload = AgentRunCreate(
                agent_key="step_graph",
                agent_name="Step Graph Orchestrator",
                user_id=user_oid,
                organization_id=org_oid,
                conversation_id=self.conversation_id,
                trigger=AgentRunTrigger.STEP_GRAPH,
                input={
                    "query": query[:600],
                    "step_count": len(self.steps),
                    "agents": [s["agent"] for s in self.steps],
                },
            )
            run = await agent_run_repository.start_run(payload)
            self.parent_run_id = str(getattr(run, "id", "") or "")
        except Exception as e:  # noqa: BLE001
            logger.warning("step_graph_open_parent_failed", error=str(e))

    async def _close_parent_run(self, *, error: Optional[str]) -> None:
        if not self.parent_run_id:
            return
        try:
            from backend.db.mongodb.repositories.agent_run_repository import agent_run_repository
            tokens_in = self._parent_tokens_in or None
            tokens_out = self._parent_tokens_out or None
            cost = round(self._parent_cost, 6) or None
            if error:
                await agent_run_repository.fail_run(self.parent_run_id, error)
            else:
                await agent_run_repository.complete_run(
                    self.parent_run_id,
                    output={
                        "final_text_preview": _preview(self.final_text, limit=400),
                        "sources_count": len(self.all_sources),
                        "steps_run": len(self.steps),
                    },
                    tokens_input=tokens_in,
                    tokens_output=tokens_out,
                    cost_usd=cost,
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("step_graph_close_parent_failed", error=str(e))

    async def _record_child_run(
        self,
        *,
        step_index: int,
        agent_key: str,
        purpose: str,
        input_payload: Dict[str, Any],
        output_text: str,
        error: Optional[str],
        duration_ms: int,
        sources_count: int,
        usage: Optional[Dict[str, Any]] = None,
    ) -> None:
        try:
            from backend.db.mongodb.repositories.agent_run_repository import agent_run_repository
            from backend.db.mongodb.models.agent_run import AgentRunCreate, AgentRunTrigger

            user_oid = ObjectId(self.user_id) if ObjectId.is_valid(str(self.user_id)) else ObjectId()
            org_oid = (
                ObjectId(self.organization_id)
                if self.organization_id and ObjectId.is_valid(str(self.organization_id))
                else None
            )
            parent_oid = (
                ObjectId(self.parent_run_id)
                if self.parent_run_id and ObjectId.is_valid(self.parent_run_id)
                else None
            )
            payload = AgentRunCreate(
                agent_key=agent_key,
                agent_name=_agent_label(agent_key),
                user_id=user_oid,
                organization_id=org_oid,
                conversation_id=self.conversation_id,
                parent_run_id=parent_oid,
                step_index=step_index,
                trigger=AgentRunTrigger.STEP_GRAPH,
                input={**input_payload, "purpose": purpose},
                model_used=(usage or {}).get("model_used"),
                provider=(usage or {}).get("provider"),
            )
            run = await agent_run_repository.start_run(payload)
            run_id = str(getattr(run, "id", "") or "")
            if not run_id:
                return

            u = usage or {}
            # Accumulate this step's spend onto the parent so the parent
            # run carries the totals across the whole plan.
            self._parent_tokens_in += int(u.get("prompt_tokens") or 0)
            self._parent_tokens_out += int(u.get("completion_tokens") or 0)
            self._parent_cost += float(u.get("cost_usd") or 0.0)

            if error:
                await agent_run_repository.fail_run(
                    run_id,
                    error,
                    metadata_patch={
                        "model_used": u.get("model_used"),
                        "provider": u.get("provider"),
                    } if u else None,
                )
            else:
                await agent_run_repository.complete_run(
                    run_id,
                    output={
                        "preview": _preview(output_text, limit=300),
                        "sources_count": sources_count,
                    },
                    tokens_input=int(u.get("prompt_tokens") or 0) or None,
                    tokens_output=int(u.get("completion_tokens") or 0) or None,
                    cost_usd=float(u.get("cost_usd") or 0.0) or None,
                    metadata_patch={
                        "model_used": u.get("model_used"),
                        "provider": u.get("provider"),
                        "llm_calls": u.get("calls", 0),
                    },
                )
        except Exception as e:  # noqa: BLE001
            logger.warning("step_graph_child_record_failed", error=str(e))
