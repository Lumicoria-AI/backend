"""Coverage eval — what % of inputs survived a node.

The ingest_to_rag node calls this to check that ≥80% of fetched
emails / attachments / Drive files actually made it into Weaviate.
Below that → flag the run as degraded so the prioritise node knows
its RAG context is partial.
"""

from __future__ import annotations

from ..state import EvalResult


def check_coverage(
    inputs: int,
    outputs: int,
    *,
    floor: float = 0.8,
    field: str = "items",
) -> EvalResult:
    """passed iff (outputs / inputs) ≥ floor.

    ``inputs=0`` is treated as passing with score=1.0 — having nothing
    to process isn't a failure.
    """
    if inputs == 0:
        return EvalResult(
            score=1.0,
            passed=True,
            reason="no_inputs",
            checked_fields=[field],
        )
    ratio = outputs / inputs
    return EvalResult(
        score=round(ratio, 3),
        passed=ratio >= floor,
        reason=f"{outputs}/{inputs} = {ratio:.2%} (floor={floor:.0%})",
        checked_fields=[field],
    )
