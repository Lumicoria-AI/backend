"""Offline brain eval CLI — deploy gate + trend monitor.

Usage:

    # Quick CI run — no LLM, ~1s.
    python -m backend.scripts.eval_brain --mode heuristic

    # Full production grade — calls real LLM, ~30–60s.
    python -m backend.scripts.eval_brain --mode live

    # Custom thresholds (default: F1>=0.65, NDCG>=0.70, ECE<=0.20 for heuristic).
    python -m backend.scripts.eval_brain --mode live \\
        --f1-floor 0.75 --ndcg-floor 0.75 --ece-ceiling 0.15

    # Quiet (CI-friendly) output.
    python -m backend.scripts.eval_brain --json

Exit codes:
    0 — all thresholds met.
    1 — at least one threshold missed (deploy should be blocked).
    2 — suite errored before producing a report.

The suite writes a row to ``brain_evals`` Postgres unless ``--no-persist``
is passed. Trend the metric over time to catch slow regressions
(prompts drifting, models being silently upgraded by providers).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Tuple

# Auto-load .env so the suite can talk to Postgres + the LLM provider
# without the operator pre-exporting variables. We look in the standard
# places and stop at the first hit.
try:
    from dotenv import load_dotenv  # type: ignore

    _here = Path(__file__).resolve()
    for candidate in (
        _here.parent.parent / ".env",       # backend/.env
        _here.parent.parent.parent / ".env",  # project_root/.env
        Path.cwd() / ".env",
    ):
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            break
except ImportError:
    pass  # dotenv is optional — env may already be exported.

# Allow running both as `python -m backend.scripts.eval_brain` and as
# `python scripts/eval_brain.py` from the backend directory.
try:
    from backend.services.brain.evals.run_suite import (
        EvalSuiteReport,
        run_eval_suite_sync,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    from backend.services.brain.evals.run_suite import (  # type: ignore
        EvalSuiteReport,
        run_eval_suite_sync,
    )


# Default thresholds — separate per mode.
#
# Heuristic mode floors guard the deterministic fallback paths
# (regex classifier + urgency-rank ranker). They are intentionally
# loose: their job is to catch a regression in the safety net, not to
# gate production. Production gating is done with --mode live.
#
# Live-mode floors are the actual deploy gate.
_DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "heuristic": {"f1": 0.15, "ndcg": 0.40, "ece": 0.40},
    "live":      {"f1": 0.75, "ndcg": 0.70, "ece": 0.15},
}


# ─────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="eval_brain",
        description="Run the autonomous brain offline eval suite.",
    )
    p.add_argument(
        "--mode",
        choices=("heuristic", "live"),
        default="heuristic",
        help="heuristic (no LLM, CI-safe) or live (calls real LLM).",
    )
    p.add_argument(
        "--f1-floor",
        type=float,
        default=None,
        help="Minimum macro-F1 to pass. Defaults vary by mode.",
    )
    p.add_argument(
        "--ndcg-floor",
        type=float,
        default=None,
        help="Minimum mean NDCG@5 to pass. Defaults vary by mode.",
    )
    p.add_argument(
        "--ece-ceiling",
        type=float,
        default=None,
        help="Maximum ECE to pass (lower is better). Defaults vary by mode.",
    )
    p.add_argument(
        "--no-persist",
        action="store_true",
        help="Skip writing the suite row to brain_evals Postgres.",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit the report as JSON only (machine-readable).",
    )
    return p


def _resolve_thresholds(args: argparse.Namespace) -> Dict[str, float]:
    defaults = _DEFAULT_THRESHOLDS[args.mode]
    return {
        "f1": args.f1_floor if args.f1_floor is not None else defaults["f1"],
        "ndcg": args.ndcg_floor if args.ndcg_floor is not None else defaults["ndcg"],
        "ece": args.ece_ceiling if args.ece_ceiling is not None else defaults["ece"],
    }


# ─────────────────────────────────────────────────────────────────────
# Pretty-printing
# ─────────────────────────────────────────────────────────────────────


def _print_human(report: EvalSuiteReport, thresholds: Dict[str, float], passed: bool) -> None:
    """Plain-text summary for terminal use."""
    pass_marker = "✓" if passed else "✗"
    print(f"\nBrain Eval Suite — {report.mode} mode ({report.dataset_version})")
    print("─" * 60)
    print(f"Ran at      : {report.ran_at.isoformat()}")
    print(f"Suite ID    : {report.suite_id}")
    print(f"Duration    : {report.duration_ms}ms")
    print(f"Emails eval : {report.n_emails_evaluated}")
    print(f"Runs eval   : {report.n_runs_evaluated}")
    print(f"Failures    : {len(report.failures)}")
    print()
    print("Headline metrics:")
    print(f"  Macro F1         : {report.classification_f1:.4f}   (floor {thresholds['f1']})  "
          f"{_status_marker(report.classification_f1, thresholds['f1'], higher_is_better=True)}")
    print(f"  Mean NDCG@5      : {report.ranking_ndcg:.4f}   (floor {thresholds['ndcg']})  "
          f"{_status_marker(report.ranking_ndcg, thresholds['ndcg'], higher_is_better=True)}")
    print(f"  ECE              : {report.confidence_ece:.4f}   (ceil  {thresholds['ece']})  "
          f"{_status_marker(report.confidence_ece, thresholds['ece'], higher_is_better=False)}")
    print()
    if report.per_class_f1:
        print("Per-class F1:")
        for cls, score in report.per_class_f1:
            print(f"  {cls:<18s}  {score:.4f}")
        print()
    if report.per_run_ndcg:
        print("Per-run NDCG@5:")
        for rid, score in report.per_run_ndcg:
            print(f"  {rid:<18s}  {score:.4f}")
        print()
    if report.failures:
        print(f"Failures ({len(report.failures)}):")
        for f in report.failures[:10]:
            print(f"  - {f.get('fixture_id')} @ {f.get('stage')}: {f.get('error')}")
        if len(report.failures) > 10:
            print(f"  … and {len(report.failures) - 10} more")
        print()
    print("─" * 60)
    print(f"OVERALL: {pass_marker}  {'PASS' if passed else 'FAIL'}")
    print()


def _status_marker(value: float, threshold: float, *, higher_is_better: bool) -> str:
    ok = (value >= threshold) if higher_is_better else (value <= threshold)
    return "✓" if ok else "✗  ← MISSED"


# ─────────────────────────────────────────────────────────────────────
# Threshold gate
# ─────────────────────────────────────────────────────────────────────


def _evaluate_thresholds(
    report: EvalSuiteReport,
    thresholds: Dict[str, float],
) -> Tuple[bool, Dict[str, bool]]:
    """Return (overall_passed, per-threshold-passed dict)."""
    checks = {
        "f1": report.classification_f1 >= thresholds["f1"],
        "ndcg": report.ranking_ndcg >= thresholds["ndcg"],
        "ece": report.confidence_ece <= thresholds["ece"],
    }
    return all(checks.values()), checks


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────


def main(argv: list | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    thresholds = _resolve_thresholds(args)

    try:
        report = run_eval_suite_sync(
            mode=args.mode,
            persist=not args.no_persist,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"eval suite errored: {exc}", file=sys.stderr)
        return 2

    passed, _ = _evaluate_thresholds(report, thresholds)

    if args.json:
        out = {
            "report": report.to_dict(),
            "thresholds": thresholds,
            "passed": passed,
        }
        print(json.dumps(out, indent=2, default=str))
    else:
        _print_human(report, thresholds, passed)

    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
