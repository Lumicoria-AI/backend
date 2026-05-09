"""Data Analysis pipeline orchestrator.

Given a run row that already has a file in MinIO, this module:
  1. Downloads the file via storage_service.
  2. Parses it with pandas (CSV / XLSX / JSON).
  3. Computes the cheap summary stats + preview rows.
  4. Hands the DataFrame's serialized form to the existing
     DataAnalysisAgent for AI-driven analysis.
  5. Updates the run row with results (status=ready) or marks error.

All side effects on the DB go through `runs.update_run_results`, so
nothing in this module touches SQLAlchemy directly.
"""

from __future__ import annotations

import io
import json
import time
from typing import Any, Dict, List, Optional, Tuple

import structlog

from ..storage_service import storage_service
from . import runs as runs_svc
from .sanitize import coerce_jsonable

logger = structlog.get_logger(__name__)


PREVIEW_ROWS = 50
MAX_ROWS_FOR_AGENT = 5000


# ── Pandas parsing ──────────────────────────────────────────────────────


def _parse_dataframe(content: bytes, content_type: str, filename: str):
    """Parse uploaded bytes into a pandas DataFrame.  Raises ValueError
    on unsupported / malformed inputs."""
    import pandas as pd  # local import keeps cold start fast

    name = (filename or "").lower()
    ct = (content_type or "").lower()

    if "json" in ct or name.endswith(".json"):
        try:
            return pd.read_json(io.BytesIO(content))
        except ValueError:
            # Some JSON exports are line-delimited; retry.
            return pd.read_json(io.BytesIO(content), lines=True)
    if "csv" in ct or name.endswith(".csv") or "plain" in ct or "text" in ct:
        return pd.read_csv(io.BytesIO(content))
    if (
        "spreadsheet" in ct
        or "excel" in ct
        or name.endswith(".xlsx")
        or name.endswith(".xls")
    ):
        return pd.read_excel(io.BytesIO(content))
    raise ValueError(f"Unsupported content type {content_type!r} for {filename!r}")


def _columns_summary(df) -> List[Dict[str, Any]]:
    return [
        {
            "name": str(col),
            "dtype": str(df[col].dtype),
            "null_count": int(df[col].isna().sum()),
            "unique_count": int(df[col].nunique(dropna=True)),
        }
        for col in df.columns
    ]


def _preview_rows(df, *, limit: int = PREVIEW_ROWS) -> List[Dict[str, Any]]:
    sample = df.head(limit)
    return [
        {str(k): coerce_jsonable(v) for k, v in row.items()}
        for row in sample.to_dict(orient="records")
    ]


def _summary_stats(df) -> Dict[str, Any]:
    """A flat summary stat block compatible with the frontend's six
    card layout (Mean, Median, Std, Min, Max, Count) plus a `per_column`
    map for richer panels."""
    try:
        import numpy as np
    except ImportError:
        np = None  # type: ignore

    numeric = df.select_dtypes(include="number") if np is not None else df.select_dtypes(include=["int64", "float64"])

    flat: Dict[str, Any] = {
        "count": int(len(df)),
    }
    per_column: Dict[str, Any] = {}
    if numeric.shape[1] > 0:
        # Aggregate "Mean / Median / Std / Min / Max" over the FIRST
        # numeric column so the headline cards always have a value.
        first = numeric.columns[0]
        s = numeric[first]
        flat.update({
            "primary_column": str(first),
            "mean": coerce_jsonable(s.mean()),
            "median": coerce_jsonable(s.median()),
            "std": coerce_jsonable(s.std()),
            "min": coerce_jsonable(s.min()),
            "max": coerce_jsonable(s.max()),
        })

        for col in numeric.columns:
            s = numeric[col]
            per_column[str(col)] = {
                "mean": coerce_jsonable(s.mean()),
                "median": coerce_jsonable(s.median()),
                "std": coerce_jsonable(s.std()),
                "min": coerce_jsonable(s.min()),
                "max": coerce_jsonable(s.max()),
                "count": int(s.count()),
            }
    flat["per_column"] = per_column
    return flat


# ── Visualization specs ────────────────────────────────────────────────


def _viz_bar_categorical(df) -> Optional[Dict[str, Any]]:
    """Bar chart of mean numeric value grouped by first categorical."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    object_cols = df.select_dtypes(include="object").columns.tolist()
    if not (object_cols and numeric_cols):
        return None
    x_col = object_cols[0]
    y_col = numeric_cols[0]
    grouped = df.groupby(x_col)[y_col].mean().sort_values(ascending=False).head(15)
    return {
        "type": "bar",
        "title": f"Average {y_col} by {x_col}",
        "x_axis": str(x_col),
        "y_axis": f"avg {y_col}",
        "data": [
            {"x": coerce_jsonable(idx), "y": coerce_jsonable(val)}
            for idx, val in grouped.items()
        ],
    }


def _viz_line(df) -> Optional[Dict[str, Any]]:
    """Line chart of first numeric column over its row index."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None
    y_col = numeric_cols[0]
    sample = df[[y_col]].head(200).reset_index(drop=True)
    return {
        "type": "line",
        "title": f"{y_col} over rows",
        "x_axis": "row",
        "y_axis": str(y_col),
        "data": [
            {"x": int(i), "y": coerce_jsonable(v)}
            for i, v in enumerate(sample[y_col].tolist())
        ],
    }


def _viz_pie_distribution(df) -> Optional[Dict[str, Any]]:
    """Pie of value counts in the first categorical column."""
    object_cols = df.select_dtypes(include="object").columns.tolist()
    if not object_cols:
        return None
    cat = object_cols[0]
    counts = df[cat].value_counts(dropna=True).head(6)
    return {
        "type": "pie",
        "title": f"Distribution of {cat}",
        "x_axis": str(cat),
        "y_axis": "count",
        "data": [
            {"name": coerce_jsonable(name), "value": int(value)}
            for name, value in counts.items()
        ],
    }


def _viz_histogram(df) -> Optional[Dict[str, Any]]:
    """Histogram (binned bar) of the first numeric column."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None
    col = numeric_cols[0]
    series = df[col].dropna()
    if series.empty:
        return None
    try:
        import numpy as np
        counts, edges = np.histogram(series, bins=12)
    except Exception:  # noqa: BLE001
        return None
    return {
        "type": "bar",
        "title": f"Histogram of {col}",
        "x_axis": str(col),
        "y_axis": "count",
        "data": [
            {
                "x": f"{coerce_jsonable(edges[i])}–{coerce_jsonable(edges[i + 1])}",
                "y": int(counts[i]),
            }
            for i in range(len(counts))
        ],
    }


def _viz_scatter_outliers(df, anomalies: Dict[str, Any] | None) -> Optional[Dict[str, Any]]:
    """Scatter that places every row in (index, value) space, useful for
    seeing where flagged outliers sit relative to the bulk."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None
    col = numeric_cols[0]
    sample = df[[col]].head(500).reset_index(drop=True)
    return {
        "type": "scatter",
        "title": f"{col} values across rows",
        "x_axis": "row",
        "y_axis": str(col),
        "data": [
            {"x": int(i), "y": coerce_jsonable(v)}
            for i, v in enumerate(sample[col].tolist())
        ],
    }


def _viz_rolling_average(df) -> Optional[Dict[str, Any]]:
    """Line chart of rolling average of the first numeric column."""
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    if not numeric_cols:
        return None
    col = numeric_cols[0]
    series = df[col].head(400)
    window = max(2, min(20, len(series) // 10 or 2))
    rolling = series.rolling(window=window, min_periods=1).mean().tolist()
    return {
        "type": "line",
        "title": f"{col} rolling average (window {window})",
        "x_axis": "row",
        "y_axis": str(col),
        "data": [
            {"x": int(i), "y": coerce_jsonable(v)}
            for i, v in enumerate(rolling)
        ],
    }


def _build_visualizations(df, mode: str = "exploratory") -> List[Dict[str, Any]]:
    """Mode aware chart spec builder.  Each mode produces a tailored
    set so users see something meaningfully different when they switch.
    Generic and never crashes on empty / malformed columns."""
    out: List[Dict[str, Any]] = []
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        return out

    if mode == "statistical":
        for build in (_viz_histogram, _viz_pie_distribution, _viz_bar_categorical):
            spec = build(df)
            if spec:
                out.append(spec)
    elif mode == "anomaly":
        for build in (_viz_scatter_outliers, _viz_histogram, _viz_pie_distribution):
            spec = build(df, None) if build is _viz_scatter_outliers else build(df)
            if spec:
                out.append(spec)
    elif mode == "trend":
        for build in (_viz_rolling_average, _viz_line, _viz_bar_categorical):
            spec = build(df)
            if spec:
                out.append(spec)
    elif mode == "visualization":
        for build in (
            _viz_bar_categorical,
            _viz_line,
            _viz_pie_distribution,
            _viz_histogram,
            _viz_scatter_outliers,
        ):
            spec = build(df, None) if build is _viz_scatter_outliers else build(df)
            if spec:
                out.append(spec)
    elif mode == "report":
        # Report shows the same broad picture as exploratory; the LLM
        # narrative on top is what makes the report different.
        for build in (_viz_bar_categorical, _viz_line, _viz_pie_distribution, _viz_histogram):
            spec = build(df)
            if spec:
                out.append(spec)
    else:  # exploratory + default
        for build in (_viz_bar_categorical, _viz_line, _viz_pie_distribution):
            spec = build(df)
            if spec:
                out.append(spec)
    return out


# ── Mode specific deterministic compute (no LLM) ────────────────────────


def _compute_anomalies(df) -> Dict[str, Any]:
    """Z-score + IQR based outlier detection on every numeric column.
    Returns a structured dict the frontend can render directly."""
    out: Dict[str, Any] = {"by_column": {}, "summary": {}}
    try:
        import numpy as np
    except ImportError:
        return out
    numeric = df.select_dtypes(include="number")
    total_outliers = 0

    for col in numeric.columns:
        series = numeric[col].dropna()
        if len(series) < 4:
            continue
        mean = float(series.mean())
        std = float(series.std() or 0.0)
        flagged_indices: List[int] = []
        flagged_values: List[Any] = []
        if std > 0:
            zscores = (series - mean) / std
            mask = zscores.abs() > 3.0
            for idx in series.index[mask].tolist()[:25]:
                flagged_indices.append(int(idx))
                flagged_values.append(coerce_jsonable(series.loc[idx]))
        # IQR based fallback flags.
        q1 = float(series.quantile(0.25))
        q3 = float(series.quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        iqr_count = int(((series < lower) | (series > upper)).sum())
        out["by_column"][str(col)] = {
            "mean": coerce_jsonable(mean),
            "std": coerce_jsonable(std),
            "z_outlier_count": len(flagged_indices),
            "iqr_outlier_count": iqr_count,
            "z_threshold": 3.0,
            "iqr_lower": coerce_jsonable(lower),
            "iqr_upper": coerce_jsonable(upper),
            "samples": [
                {"row": ri, "value": rv}
                for ri, rv in zip(flagged_indices[:10], flagged_values[:10])
            ],
        }
        total_outliers += len(flagged_indices)

    out["summary"] = {
        "total_z_outliers": total_outliers,
        "columns_checked": int(numeric.shape[1]),
    }
    return out


def _compute_trends(df) -> Dict[str, Any]:
    """Rolling averages and a coarse direction signal per numeric column."""
    out: Dict[str, Any] = {}
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return out
    numeric = df.select_dtypes(include="number")
    for col in numeric.columns[:6]:
        series = numeric[col].dropna()
        if len(series) < 4:
            continue
        window = max(3, min(30, len(series) // 10))
        rolling = series.rolling(window=window, min_periods=1).mean().tolist()
        first = float(series.head(max(1, len(series) // 4)).mean())
        last = float(series.tail(max(1, len(series) // 4)).mean())
        delta = last - first
        if abs(delta) < 1e-9:
            direction = "flat"
        elif delta > 0:
            direction = "increasing"
        else:
            direction = "decreasing"
        out[str(col)] = {
            "rolling_window": window,
            "rolling_average": [coerce_jsonable(v) for v in rolling[:200]],
            "first_quarter_mean": coerce_jsonable(first),
            "last_quarter_mean": coerce_jsonable(last),
            "delta": coerce_jsonable(delta),
            "direction": direction,
        }
    return out


def _compute_statistical(df) -> Dict[str, Any]:
    """Per-column descriptive stats plus optional hypothesis tests
    when scipy is available."""
    out: Dict[str, Any] = {"per_column": {}, "tests": []}
    try:
        from scipy import stats as scipy_stats  # type: ignore
        has_scipy = True
    except ImportError:
        scipy_stats = None  # type: ignore
        has_scipy = False

    numeric = df.select_dtypes(include="number")
    for col in numeric.columns:
        series = numeric[col].dropna()
        if len(series) == 0:
            continue
        out["per_column"][str(col)] = {
            "mean": coerce_jsonable(series.mean()),
            "median": coerce_jsonable(series.median()),
            "std": coerce_jsonable(series.std()),
            "min": coerce_jsonable(series.min()),
            "max": coerce_jsonable(series.max()),
            "skew": coerce_jsonable(series.skew()) if hasattr(series, "skew") else None,
            "kurtosis": coerce_jsonable(series.kurtosis()) if hasattr(series, "kurtosis") else None,
            "count": int(series.count()),
        }

    if has_scipy:
        # One sample t-test against zero on each numeric column.
        for col in numeric.columns[:6]:
            series = numeric[col].dropna()
            if len(series) < 2:
                continue
            try:
                t_stat, p_value = scipy_stats.ttest_1samp(series, 0.0)
                out["tests"].append({
                    "test": "ttest_1samp_vs_zero",
                    "column": str(col),
                    "t_statistic": coerce_jsonable(t_stat),
                    "p_value": coerce_jsonable(p_value),
                    "significant_at_0_05": bool(float(p_value) < 0.05),
                })
            except Exception:  # noqa: BLE001
                continue
    else:
        out["tests"].append({
            "warning": "scipy not installed; hypothesis tests skipped",
        })
    return out


def _build_mode_insights(
    df,
    mode: str,
    anomalies: Optional[Dict[str, Any]],
    trends: Optional[Dict[str, Any]],
    stats_results: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Mode specific insights so each tab feels meaningfully different
    even before the LLM call."""
    items: List[Dict[str, Any]] = []
    if mode == "anomaly" and anomalies:
        total = (anomalies.get("summary") or {}).get("total_z_outliers", 0)
        if total:
            items.append({
                "text": f"Found {total:,} outliers across numeric columns using a 3 sigma rule.",
                "type": "negative" if total > 50 else "neutral",
            })
        for col, info in (anomalies.get("by_column") or {}).items():
            z = info.get("z_outlier_count", 0)
            if z:
                items.append({
                    "text": f"{col!r} has {z:,} extreme values beyond 3 standard deviations.",
                    "type": "negative" if z > 10 else "neutral",
                })
        if not items:
            items.append({
                "text": "No statistically significant outliers detected in any numeric column.",
                "type": "positive",
            })
    elif mode == "trend" and trends:
        directions = {"increasing": [], "decreasing": [], "flat": []}
        for col, info in trends.items():
            directions.setdefault(info.get("direction", "flat"), []).append(col)
        if directions["increasing"]:
            items.append({
                "text": "Increasing trend in: " + ", ".join(directions["increasing"][:5]) + ".",
                "type": "positive",
            })
        if directions["decreasing"]:
            items.append({
                "text": "Decreasing trend in: " + ", ".join(directions["decreasing"][:5]) + ".",
                "type": "negative",
            })
        if directions["flat"]:
            items.append({
                "text": "No significant movement in: " + ", ".join(directions["flat"][:5]) + ".",
                "type": "neutral",
            })
    elif mode == "statistical" and stats_results:
        sig = [
            t for t in (stats_results.get("tests") or [])
            if isinstance(t, dict) and t.get("significant_at_0_05")
        ]
        if sig:
            cols = [t.get("column") for t in sig[:5]]
            items.append({
                "text": f"Mean of {', '.join(filter(None, cols))} significantly differs from zero (p < 0.05).",
                "type": "neutral",
            })
        per_col = (stats_results.get("per_column") or {})
        if per_col:
            items.append({
                "text": f"Computed descriptive stats for {len(per_col)} numeric columns.",
                "type": "neutral",
            })
    return items


# ── Insight extraction (heuristic, no LLM required) ────────────────────


def _build_heuristic_insights(df, summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Cheap, deterministic insights generated without an LLM call.
    The agent's LLM-driven insights come on top of these via the
    existing per-mode methods; these guarantee something useful even
    when the LLM is unavailable."""
    out: List[Dict[str, Any]] = []
    try:
        import numpy as np  # noqa: F401
    except ImportError:
        return out

    rows = int(len(df))
    if rows == 0:
        out.append({"text": "Uploaded file contains no rows.", "type": "negative"})
        return out

    out.append({
        "text": f"Dataset contains {rows:,} rows and {df.shape[1]} columns.",
        "type": "neutral",
    })

    # Highlight the column with the highest null count.
    null_counts = df.isna().sum()
    if null_counts.max() > 0:
        worst_col = null_counts.idxmax()
        worst = int(null_counts.max())
        pct = round((worst / rows) * 100, 1) if rows else 0
        out.append({
            "text": f"{worst_col!r} has {worst:,} missing values ({pct}% of rows).",
            "type": "negative" if pct > 20 else "neutral",
        })

    # Highlight the strongest numeric column.
    numeric = df.select_dtypes(include="number")
    if numeric.shape[1] > 0:
        col = numeric.columns[0]
        s = numeric[col]
        try:
            out.append({
                "text": (
                    f"Average {col!r} is {coerce_jsonable(s.mean())}, "
                    f"with values between {coerce_jsonable(s.min())} and {coerce_jsonable(s.max())}."
                ),
                "type": "positive",
            })
        except Exception:  # noqa: BLE001
            pass

    return out


# ── Public entry point ────────────────────────────────────────────────


async def run_pipeline(
    *,
    run_id: str,
    organization_id: str,
    user_id: str,
    s3_key: str,
    content_type: str,
    filename: str,
    mode: str = "exploratory",
) -> Dict[str, Any]:
    """End to end: download, parse, analyze, persist.

    Returns the updated run dict.  On any failure, marks the run as
    `status=error` with the exception message, then re-raises so the
    caller can surface a 5xx (the row stays in the DB for debugging).
    """
    started = time.perf_counter()

    # Mark processing.
    await runs_svc.update_run_results(
        organization_id, run_id, status="processing",
    )

    try:
        # 1. Download.
        content = await storage_service.download_file(s3_key)

        # 2. Parse.
        df = _parse_dataframe(content, content_type, filename)
        if df is None or len(df.columns) == 0:
            raise ValueError("Parsed file produced an empty DataFrame")

        columns = _columns_summary(df)
        preview = _preview_rows(df, limit=PREVIEW_ROWS)
        summary = _summary_stats(df)
        heuristic_insights = _build_heuristic_insights(df, summary)

        # 3. Mode-specific deterministic compute. Pandas + numpy + scipy
        # only — no LLM call here, so this stays fast.
        anomalies = _compute_anomalies(df) if mode in ("anomaly", "report") else None
        trends = _compute_trends(df) if mode in ("trend", "report") else None
        statistical_results = _compute_statistical(df) if mode in ("statistical", "report") else None

        # 4. Mode-aware visualizations (different chart sets per mode).
        viz_specs = _build_visualizations(df, mode)

        # 5. Mode-specific insights on top of the heuristic ones.
        insights: List[Dict[str, Any]] = list(heuristic_insights)
        insights.extend(_build_mode_insights(df, mode, anomalies, trends, statistical_results))

        # 6. Optional LLM step. Only mode=report calls the LLM, so all
        # other modes return in milliseconds. The LLM produces the
        # narrative `ai_summary` field that the report panel renders.
        ai_summary: Optional[str] = None
        if mode == "report":
            try:
                from ...core.config import settings as _settings
                provider = (_settings.DEFAULT_LLM_PROVIDER or "gemini").lower()
                model_name = {
                    "gemini": getattr(_settings, "GEMINI_MODEL", None) or "gemini-2.5-flash",
                    "openai": "gpt-4o-mini",
                    "anthropic": "claude-haiku-4-5-20251001",
                    "mistral": "mistral-small-latest",
                    "perplexity": "sonar",
                }.get(provider, "sonar")

                from ...agents.customer_service_agent import CustomerServiceAgent
                agent = CustomerServiceAgent({
                    "provider": provider,
                    "model": model_name,
                    "agent_model_config": {
                        "model": model_name,
                        "temperature": 0.4,
                        "max_tokens": 1800,
                    },
                })

                # Compose a compact prompt the LLM can summarise. We
                # include columns, summary stats, and the deterministic
                # mode outputs so the narrative is grounded.
                prompt_parts = [
                    "Write a clear, professional analytical report on the dataset described below. "
                    "Cover the most important patterns, any data quality issues, and the most "
                    "actionable findings. Use 3 to 5 short paragraphs. Do not invent values.",
                    "",
                    f"Filename: {filename}",
                    f"Rows: {int(len(df))}, Columns: {int(df.shape[1])}",
                    f"Columns: {', '.join(c['name'] for c in columns[:25])}",
                    "",
                    f"Summary stats: {json.dumps({k: v for k, v in summary.items() if k != 'per_column'}, default=str)[:1500]}",
                ]
                if anomalies:
                    prompt_parts.append(f"Anomalies summary: {json.dumps(anomalies.get('summary'), default=str)[:600]}")
                if trends:
                    direction_lines = [
                        f"{col}: {info.get('direction')}"
                        for col, info in list(trends.items())[:8]
                    ]
                    prompt_parts.append("Trend directions: " + "; ".join(direction_lines))
                if preview:
                    prompt_parts.append("First rows: " + json.dumps(preview[:8], default=str)[:1200])

                grounded = "\n".join(prompt_parts)
                result = await agent.process_async({
                    "content": grounded,
                    "request_type": "generate_response",
                    "context": {"run_id": run_id},
                })
                if isinstance(result, dict):
                    response_blob = result.get("response")
                    if isinstance(response_blob, dict):
                        ai_summary = (
                            response_blob.get("response")
                            or response_blob.get("draft")
                            or response_blob.get("message")
                            or response_blob.get("text")
                        )
                    if not ai_summary:
                        ai_summary = result.get("raw_response")
                if ai_summary:
                    ai_summary = ai_summary.strip()
            except Exception as e:  # noqa: BLE001
                logger.warning("data_analysis_report_llm_failed", run_id=run_id, error=str(e))
                ai_summary = None

        # 7. Persist. `coerce_jsonable` scrubs NaN / Infinity from every
        # JSONB-bound field.
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        await runs_svc.update_run_results(
            organization_id,
            run_id,
            status="ready",
            row_count=int(len(df)),
            column_count=int(df.shape[1]),
            columns=coerce_jsonable(columns),
            preview_rows=coerce_jsonable(preview),
            summary_stats=coerce_jsonable(summary),
            visualizations=coerce_jsonable(viz_specs),
            anomalies=coerce_jsonable(anomalies),
            trends=coerce_jsonable(trends),
            statistical_results=coerce_jsonable(statistical_results),
            insights=coerce_jsonable(insights),
            ai_summary=ai_summary,
            processing_time_ms=elapsed_ms,
            mode=mode,
        )
        return await runs_svc.get_run(organization_id, run_id) or {}

    except Exception as e:  # noqa: BLE001
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        logger.error("data_analysis_pipeline_failed", run_id=run_id, error=str(e))
        await runs_svc.update_run_results(
            organization_id,
            run_id,
            status="error",
            processing_time_ms=elapsed_ms,
            error_message=str(e),
        )
        raise
