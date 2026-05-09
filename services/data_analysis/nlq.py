"""Natural language Q&A on a stored data analysis run.

Builds a compact prompt from the run's preview rows + summary stats +
column metadata, asks the existing CustomerServiceAgent style dispatcher
to draft a grounded answer, and returns text the operator can read.

We intentionally do NOT re run the heavy pandas pipeline on every
question — the run's preview + summary already capture enough for the
LLM to answer typical follow up questions ("which row is highest",
"compare A vs B", "is there a trend").  For deeper drilling, the
operator can use the Regenerate button to re run analysis in a
different mode.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Dict, Optional

import structlog

from ...core.config import settings
from . import runs as runs_svc
from .sanitize import clean_question

logger = structlog.get_logger(__name__)


def _build_prompt(question: str, run: Dict[str, Any]) -> str:
    """Compose the user-message prompt fed to the LLM."""
    columns = run.get("columns") or []
    preview = run.get("preview_rows") or []
    summary = run.get("summary_stats") or {}

    parts: list[str] = []
    parts.append(
        "You are a data analyst answering a question about an uploaded dataset. "
        "Use only the information provided below. If the answer cannot be "
        "determined from this data, say so plainly. Keep the answer concise."
    )
    parts.append("")
    parts.append(f"Filename: {run.get('original_filename') or run.get('filename')}")
    parts.append(f"Rows: {run.get('row_count')}, Columns: {run.get('column_count')}")
    if columns:
        col_summary = ", ".join(
            f"{c.get('name')} ({c.get('dtype')})" for c in columns[:30]
        )
        parts.append(f"Columns: {col_summary}")
    if summary:
        # Trim per_column to keep prompt short.
        slim = {k: v for k, v in summary.items() if k != "per_column"}
        parts.append("Summary statistics: " + json.dumps(slim, default=str)[:1500])
    if preview:
        parts.append("First rows of the data (truncated for prompt):")
        parts.append(json.dumps(preview[:20], default=str)[:3000])
    parts.append("")
    parts.append(f"Question: {question}")
    parts.append("")
    parts.append(
        "Answer in 1 to 4 short paragraphs. Cite specific values when relevant. "
        "If a chart would help, describe it briefly at the end as 'Suggested "
        "chart: ...'. Do not fabricate values that are not in the data above."
    )
    return "\n".join(parts)


async def ask(
    *,
    organization_id: str,
    run_id: str,
    question: str,
) -> Dict[str, Any]:
    """Run a single NL question against a stored run.  Returns
    `{question, answer, model_used, asked_at}`."""
    cleaned = clean_question(question, max_len=1000)
    if not cleaned:
        return {
            "question": question or "",
            "answer": "Question was empty.",
            "model_used": None,
            "asked_at": datetime.utcnow().isoformat(),
        }

    run = await runs_svc.get_run(organization_id, run_id)
    if not run:
        raise ValueError("run_not_found")
    if run.get("status") != "ready":
        return {
            "question": cleaned,
            "answer": "This dataset is still being analyzed. Please try again in a moment.",
            "model_used": None,
            "asked_at": datetime.utcnow().isoformat(),
        }

    prompt = _build_prompt(cleaned, run)

    # Use the existing customer service agent dispatch, which already
    # handles provider selection from settings.DEFAULT_LLM_PROVIDER.
    # The same agent class accepts `request_type='generate_response'`
    # and returns a flexible response shape we can extract.
    answer = ""
    model_used: Optional[str] = None
    try:
        from ...agents.customer_service_agent import CustomerServiceAgent

        provider = (settings.DEFAULT_LLM_PROVIDER or "gemini").lower()
        model_map = {
            "gemini": getattr(settings, "GEMINI_MODEL", None) or "gemini-2.5-flash",
            "openai": "gpt-4o-mini",
            "anthropic": "claude-haiku-4-5-20251001",
            "mistral": "mistral-small-latest",
            "perplexity": "sonar",
        }
        model_name = model_map.get(provider, "sonar")
        agent = CustomerServiceAgent({
            "provider": provider,
            "model": model_name,
            "agent_model_config": {
                "model": model_name,
                "temperature": 0.3,
                "max_tokens": 1200,
            },
        })
        result = await agent.process_async({
            "content": prompt,
            "request_type": "generate_response",
            "context": {"run_id": run_id},
        })
        if isinstance(result, dict):
            model_used = result.get("model_used")
            response_blob = result.get("response")
            if isinstance(response_blob, dict):
                answer = (
                    response_blob.get("response")
                    or response_blob.get("draft")
                    or response_blob.get("message")
                    or response_blob.get("text")
                    or ""
                )
            if not answer:
                answer = result.get("raw_response") or ""
    except Exception as e:  # noqa: BLE001
        logger.error("nlq_llm_call_failed", run_id=run_id, error=str(e))
        answer = "I couldn't generate an answer right now. Please try again."

    turn = {
        "question": cleaned,
        "answer": answer.strip() or "No answer produced.",
        "model_used": model_used,
        "asked_at": datetime.utcnow().isoformat(),
    }

    # Best effort persist into question_history.
    try:
        await runs_svc.append_question_turn(organization_id, run_id, turn)
    except Exception as e:  # noqa: BLE001
        logger.warning("nlq_persist_failed", run_id=run_id, error=str(e))

    return turn
