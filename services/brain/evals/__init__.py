"""Brain node evals — runtime quality checks.

Every node optionally runs through an eval. The decorator in
``services/brain/tracing.py`` consumes ``__eval_score`` (0–1) from the
node's return dict and persists it on the BrainTrace row.

Eval categories:

  schema     — Pydantic validation of structured LLM output.
  confidence — LLM self-reported confidence vs. a floor (default 0.6).
  coverage   — % of input items that produced valid output.
  freshness  — % of fetched items newer than the last brain run.

Each eval returns an ``EvalResult`` (defined in state.py) — score,
passed, reason, checked_fields. Failed evals don't raise; the graph
chooses to retry, fallback, or continue based on the eval status.
"""

from .confidence_eval import check_confidence
from .coverage_eval import check_coverage
from .freshness_eval import check_freshness
from .schema_eval import check_schema

__all__ = [
    "check_confidence",
    "check_coverage",
    "check_freshness",
    "check_schema",
]
