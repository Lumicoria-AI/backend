"""Prometheus metrics for the autonomous brain.

These metrics expose the production health of the morning + evening
brain runs. They're registered on the default ``prometheus_client``
registry, so the existing ``/metrics`` endpoint in ``backend/main.py``
picks them up automatically.

The four spec metrics from the plan:

  brain_run_duration_ms              Histogram (mode, status)
  brain_node_duration_ms{node}       Histogram (node)
  brain_eval_score{eval}             Histogram (node)        — labelled by node
  brain_run_status{status}           Counter   (mode, status)

Plus a few production-useful extras:

  brain_node_status_total            Counter (node, status)  — for SLO alerts
  brain_tasks_created_total          Counter (mode)
  brain_emails_processed_total       Counter (mode)
  brain_judge_score                  Histogram (target)      — LLM-as-judge

All metric names are prefixed ``brain_`` and globally unique so they
won't collide with any other service in this process. Registration is
idempotent — re-importing this module (e.g. during a `uvicorn --reload`
cycle) won't raise.
"""

from __future__ import annotations

from typing import Any, Optional

import structlog

logger = structlog.get_logger(__name__)


# ─────────────────────────────────────────────────────────────────────
# Idempotent registration
# ─────────────────────────────────────────────────────────────────────


def _get_or_create(metric_cls: Any, name: str, *args: Any, **kwargs: Any) -> Any:
    """Return a metric, reusing the one already in the default REGISTRY.

    prometheus_client raises ValueError when the same name is registered
    twice. That's annoying during dev hot-reload and during tests where
    modules import this file repeatedly. We catch and look up.
    """
    from prometheus_client import REGISTRY
    try:
        return metric_cls(name, *args, **kwargs)
    except ValueError:
        # Already registered — find and return the existing collector.
        for collector in list(getattr(REGISTRY, "_collector_to_names", {}).keys()):
            collector_names = getattr(REGISTRY, "_collector_to_names", {}).get(collector, set())
            if name in collector_names:
                return collector
            if getattr(collector, "_name", None) == name:
                return collector
        # Fall through — shouldn't happen, but if it does just rebuild fresh.
        raise


# Histogram buckets tuned for the brain's expected latencies. A full
# morning run runs in 5–60s; individual nodes mostly land 20ms–2s.
_RUN_BUCKETS = (
    100, 250, 500, 1000, 2500, 5000, 10000, 20000, 30000,
    60000, 120000, 300000, 600000,
)
_NODE_BUCKETS = (
    5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000,
)
_SCORE_BUCKETS = (0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 1.0)


def _init_metrics() -> dict:
    from prometheus_client import Counter, Histogram

    return {
        "run_duration": _get_or_create(
            Histogram,
            "brain_run_duration_ms",
            "Wall-clock duration of a single end-to-end brain run, in ms.",
            labelnames=("mode", "status"),
            buckets=_RUN_BUCKETS,
        ),
        "node_duration": _get_or_create(
            Histogram,
            "brain_node_duration_ms",
            "Wall-clock duration of one node execution inside the brain graph.",
            labelnames=("node",),
            buckets=_NODE_BUCKETS,
        ),
        "eval_score": _get_or_create(
            Histogram,
            "brain_eval_score",
            "Per-node quality score (0..1) emitted by an in-graph eval.",
            labelnames=("node",),
            buckets=_SCORE_BUCKETS,
        ),
        "run_status_total": _get_or_create(
            Counter,
            "brain_run_status_total",
            "Count of completed brain runs by mode + final status.",
            labelnames=("mode", "status"),
        ),
        "node_status_total": _get_or_create(
            Counter,
            "brain_node_status_total",
            "Count of node executions by node + status (ok/retry/fallback/fail).",
            labelnames=("node", "status"),
        ),
        "tasks_created_total": _get_or_create(
            Counter,
            "brain_tasks_created_total",
            "Total tasks the brain has created, by mode.",
            labelnames=("mode",),
        ),
        "emails_processed_total": _get_or_create(
            Counter,
            "brain_emails_processed_total",
            "Total emails the brain has touched, by mode.",
            labelnames=("mode",),
        ),
        "judge_score": _get_or_create(
            Histogram,
            "brain_judge_score",
            "LLM-as-judge score on agent outputs (0..1).",
            labelnames=("target",),
            buckets=_SCORE_BUCKETS,
        ),
    }


_M: dict = _init_metrics()


# ─────────────────────────────────────────────────────────────────────
# Recording helpers — call sites use these, never the raw collectors.
# ─────────────────────────────────────────────────────────────────────


def record_node(
    node: str,
    duration_ms: int,
    status: str,
    eval_score: Optional[float] = None,
) -> None:
    """Emit per-node metrics. Best-effort — never raises."""
    try:
        _M["node_duration"].labels(node=node).observe(duration_ms)
        _M["node_status_total"].labels(node=node, status=status).inc()
        if eval_score is not None:
            _M["eval_score"].labels(node=node).observe(float(eval_score))
    except Exception as exc:  # noqa: BLE001
        logger.debug("brain.metrics.record_node_failed", node=node, error=str(exc))


def record_run(
    mode: str,
    status: str,
    duration_ms: int,
    *,
    tasks_created: int = 0,
    emails_processed: int = 0,
) -> None:
    """Emit per-run metrics. Best-effort — never raises."""
    try:
        _M["run_duration"].labels(mode=mode, status=status).observe(duration_ms)
        _M["run_status_total"].labels(mode=mode, status=status).inc()
        if tasks_created:
            _M["tasks_created_total"].labels(mode=mode).inc(tasks_created)
        if emails_processed:
            _M["emails_processed_total"].labels(mode=mode).inc(emails_processed)
    except Exception as exc:  # noqa: BLE001
        logger.debug("brain.metrics.record_run_failed", mode=mode, error=str(exc))


def record_judge(target: str, score: float) -> None:
    """Emit one LLM-as-judge observation. ``target`` is one of:
    ``ranked_action``, ``proposal``, ``digest``."""
    try:
        _M["judge_score"].labels(target=target).observe(float(score))
    except Exception as exc:  # noqa: BLE001
        logger.debug("brain.metrics.record_judge_failed", target=target, error=str(exc))
