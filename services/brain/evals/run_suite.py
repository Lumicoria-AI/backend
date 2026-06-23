"""Offline eval suite — runs the brain's reasoning nodes against the
hand-curated fixture set and grades them.

Three metrics:

  classification_f1 : macro F1 across the 6 email categories.
  ranking_ndcg      : NDCG@5 across the 5 priority-run fixtures.
  confidence_ece    : Expected Calibration Error on classify outputs.

Two modes:

  ``heuristic``  — exercises the deterministic fallbacks in classify.py
                   (regex heuristic) + prioritise_rule_fallback. Safe to
                   run in CI: no API keys, no network, ~1s end-to-end.
                   Catches regressions in the safety net.

  ``live``       — calls the real LLM via ``classify`` and
                   ``prioritise_attempt_1``. Requires API credentials.
                   This is the score we gate production deploys on.

Both modes return the same ``EvalSuiteReport`` shape so the CLI doesn't
care which one ran.

The runner persists each completed suite as a ``brain_evals`` Postgres
row so we can trend the metric over time (useful for spotting prompt-
or-model regressions early).
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Literal, Optional, Tuple

import structlog

from ..state import BrainState, ClassifiedEmail, GmailMessageRef
from . import dataset
from .eval_metrics import (
    expected_calibration_error,
    macro_f1,
    ndcg_at_k,
    per_class_f1,
)

logger = structlog.get_logger(__name__)

Mode = Literal["heuristic", "live"]


# ─────────────────────────────────────────────────────────────────────
# Report shape
# ─────────────────────────────────────────────────────────────────────


@dataclass
class EvalSuiteReport:
    suite_id: str
    ran_at: datetime
    dataset_version: str
    mode: Mode
    classification_f1: float
    ranking_ndcg: float
    confidence_ece: float
    n_emails_evaluated: int
    n_runs_evaluated: int
    per_class_f1: List[Tuple[str, float]] = field(default_factory=list)
    per_run_ndcg: List[Tuple[str, float]] = field(default_factory=list)
    failures: List[Dict[str, Any]] = field(default_factory=list)
    duration_ms: int = 0

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["ran_at"] = self.ran_at.isoformat()
        return d


# ─────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────


async def run_eval_suite(
    *,
    mode: Mode = "heuristic",
    persist: bool = True,
) -> EvalSuiteReport:
    """Run the full eval suite end-to-end. Returns a report dict."""
    started = datetime.utcnow()
    suite_id = str(uuid.uuid4())

    # ── Classification + calibration ────────────────────────────────
    y_true_cat: List[str] = []
    y_pred_cat: List[str] = []
    confidences: List[float] = []
    correctness: List[bool] = []
    failures: List[Dict[str, Any]] = []

    for fixture in dataset.GOLDEN_EMAILS:
        try:
            pred = await _classify_one(fixture["input"], mode=mode)
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "fixture_id": fixture["id"],
                "stage": "classify",
                "error": str(exc),
            })
            continue
        if not pred:
            failures.append({
                "fixture_id": fixture["id"],
                "stage": "classify",
                "error": "no_classification",
            })
            continue

        expected_cat = fixture["expected"]["category"]
        predicted_cat = pred.category
        y_true_cat.append(expected_cat)
        y_pred_cat.append(predicted_cat)
        confidences.append(float(pred.confidence or 0.0))
        correctness.append(predicted_cat == expected_cat)

    f1 = macro_f1(y_true_cat, y_pred_cat)
    per_cls = per_class_f1(y_true_cat, y_pred_cat)
    ece = expected_calibration_error(confidences, correctness)

    # ── Priority ranking ────────────────────────────────────────────
    per_run_ndcg: List[Tuple[str, float]] = []
    for fixture in dataset.GOLDEN_PRIORITY_RUNS:
        try:
            score = await _score_priority_run(fixture, mode=mode)
        except Exception as exc:  # noqa: BLE001
            failures.append({
                "fixture_id": fixture["id"],
                "stage": "prioritise",
                "error": str(exc),
            })
            per_run_ndcg.append((fixture["id"], 0.0))
            continue
        per_run_ndcg.append((fixture["id"], score))

    avg_ndcg = (
        sum(s for _, s in per_run_ndcg) / len(per_run_ndcg)
        if per_run_ndcg else 0.0
    )

    duration_ms = int((datetime.utcnow() - started).total_seconds() * 1000)
    report = EvalSuiteReport(
        suite_id=suite_id,
        ran_at=started,
        dataset_version=dataset.DATASET_VERSION,
        mode=mode,
        classification_f1=round(f1, 4),
        ranking_ndcg=round(avg_ndcg, 4),
        confidence_ece=round(ece, 4),
        n_emails_evaluated=len(y_true_cat),
        n_runs_evaluated=len(per_run_ndcg),
        per_class_f1=[(c, round(s, 4)) for c, s in per_cls],
        per_run_ndcg=[(rid, round(s, 4)) for rid, s in per_run_ndcg],
        failures=failures,
        duration_ms=duration_ms,
    )

    if persist:
        await _persist_brain_eval(report)

    logger.info(
        "brain.eval_suite.complete",
        suite_id=suite_id,
        mode=mode,
        f1=report.classification_f1,
        ndcg=report.ranking_ndcg,
        ece=report.confidence_ece,
        duration_ms=duration_ms,
    )

    return report


# ─────────────────────────────────────────────────────────────────────
# Per-fixture runners
# ─────────────────────────────────────────────────────────────────────


async def _classify_one(
    email_input: Dict[str, Any],
    *,
    mode: Mode,
) -> Optional[ClassifiedEmail]:
    """Run a single email through the classify pipeline.

    Heuristic mode: invoke the heuristic fallback directly — gives a
      cheap floor signal that the regex never regresses.
    Live mode: build a 1-element BrainState and call the classify node.
    """
    ref = GmailMessageRef.model_validate(email_input)

    if mode == "heuristic":
        # Pull the private heuristic without going through the LLM path.
        from ..nodes.classify import _heuristic_for_batch
        results = _heuristic_for_batch([ref])
        return results[0] if results else None

    # Live mode — exercise the full classify node.
    from ..nodes.classify import classify as classify_node

    state = BrainState(
        run_id=f"eval-{uuid.uuid4()}",
        user_id="eval-user",
        mode="morning",
        emails=[ref],
    )
    update = await classify_node(state)
    classified_dicts = update.get("classified") or []
    if not classified_dicts:
        return None

    # ``classified`` returned by the node is List[ClassifiedEmail] (typed)
    # in current code, but defensively handle dict form too.
    first = classified_dicts[0]
    if isinstance(first, ClassifiedEmail):
        return first
    return ClassifiedEmail.model_validate(first)


async def _score_priority_run(
    fixture: Dict[str, Any],
    *,
    mode: Mode,
) -> float:
    """Score one priority-run fixture via NDCG@5 of fuzzy-matched titles
    against the expected_top_k keyword sets."""
    classified = [ClassifiedEmail.model_validate(c) for c in fixture["input"]["classified_emails"]]
    events = fixture["input"].get("calendar_events") or []
    huddles = fixture["input"].get("huddle_recents") or []
    open_tasks = fixture["input"].get("open_tasks") or []

    expected_top_k = fixture.get("expected_top_k") or []
    max_actions = fixture.get("max_actions")

    # Build minimal state.
    from ..state import CalendarEventRef, HuddleSummaryRef, OpenTaskRef
    state = BrainState(
        run_id=f"eval-{uuid.uuid4()}",
        user_id="eval-user",
        mode=fixture.get("mode", "morning"),
        classified=classified,
        events=[CalendarEventRef.model_validate(_strip_underscores(e)) for e in events],
        huddle_recents=[HuddleSummaryRef.model_validate(h) for h in huddles],
        open_tasks=[OpenTaskRef.model_validate(t) for t in open_tasks],
    )

    if mode == "heuristic":
        from ..nodes.prioritise import prioritise_rule_fallback
        update = await prioritise_rule_fallback(state)
    else:
        from ..nodes.prioritise import prioritise_attempt_1
        update = await prioritise_attempt_1(state)

    predicted_actions = update.get("ranked_actions") or []

    # Empty-expectation fixtures (quiet morning, all-promo) — score 1.0
    # if the model produced zero actions; otherwise scale by overshoot.
    if not expected_top_k:
        if max_actions is not None and len(predicted_actions) > max_actions:
            return 0.0
        if len(predicted_actions) == 0:
            return 1.0
        # 1 spurious action → 0.5, 3+ → 0.
        return max(0.0, 1.0 - 0.4 * len(predicted_actions))

    # Map each predicted action to the index of the expected item whose
    # title_contains keywords it best matches. Use that index for NDCG.
    predicted_order: List[str] = []
    ideal_order: List[str] = [f"slot-{i}" for i in range(len(expected_top_k))]
    used_slots: set[int] = set()
    for action in predicted_actions[:5]:
        title = (action.title or "").lower()
        best_idx: Optional[int] = None
        best_overlap = 0
        for i, expected in enumerate(expected_top_k):
            if i in used_slots:
                continue
            kws = [k.lower() for k in expected.get("title_contains") or []]
            overlap = sum(1 for k in kws if k in title)
            if overlap > best_overlap:
                best_overlap = overlap
                best_idx = i
        if best_idx is not None and best_overlap > 0:
            predicted_order.append(f"slot-{best_idx}")
            used_slots.add(best_idx)
        else:
            predicted_order.append("miss")

    return ndcg_at_k(predicted_order, ideal_order, k=5)


def _strip_underscores(d: Dict[str, Any]) -> Dict[str, Any]:
    """Drop debug-only ``_time_label``-style keys from fixture dicts so
    they validate against the strict Pydantic models."""
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ─────────────────────────────────────────────────────────────────────
# Persistence — best-effort row in brain_evals
# ─────────────────────────────────────────────────────────────────────


async def _persist_brain_eval(report: EvalSuiteReport) -> None:
    """Best-effort write of the suite's summary metrics into Postgres."""
    try:
        from backend.db.postgres import get_async_sessionmaker
        from backend.db.postgres_models import BrainEval

        session_factory = get_async_sessionmaker()
        async with session_factory() as session:
            row = BrainEval(
                id=report.suite_id,
                ran_at=report.ran_at,
                dataset_version=report.dataset_version,
                classification_f1=report.classification_f1,
                ranking_ndcg=report.ranking_ndcg,
                confidence_ece=report.confidence_ece,
                extra={
                    "mode": report.mode,
                    "n_emails": report.n_emails_evaluated,
                    "n_runs": report.n_runs_evaluated,
                    "per_class_f1": dict(report.per_class_f1),
                    "per_run_ndcg": dict(report.per_run_ndcg),
                    "failures": report.failures,
                    "duration_ms": report.duration_ms,
                },
            )
            session.add(row)
            await session.commit()
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "brain.eval_persist_failed",
            suite_id=report.suite_id,
            error=str(exc),
        )


# ─────────────────────────────────────────────────────────────────────
# Sync wrapper for the CLI
# ─────────────────────────────────────────────────────────────────────


def run_eval_suite_sync(
    *,
    mode: Mode = "heuristic",
    persist: bool = True,
) -> EvalSuiteReport:
    """Run the suite in a fresh asyncio loop — for CLI / CI use."""
    return asyncio.run(run_eval_suite(mode=mode, persist=persist))
