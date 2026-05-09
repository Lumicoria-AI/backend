"""Real Data Analysis analytics — replaces the mock dict in
`backend/api/v1/endpoints/data_analysis.py`.

All aggregations run against `data_analysis_runs`.  Same response shape
the frontend already expects.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict

from sqlalchemy import case, func, select

from ...db.postgres import get_async_sessionmaker
from ...db.postgres_models import DataAnalysisRunSQL


_TIME_RANGE_TO_DAYS = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}


def _cutoff(time_range: str) -> datetime:
    days = _TIME_RANGE_TO_DAYS.get(time_range, 7)
    return datetime.utcnow() - timedelta(days=days)


def _empty_payload(time_range: str) -> Dict[str, Any]:
    return {
        "time_range": time_range,
        "total_analyses": 0,
        "average_processing_time": 0.0,  # seconds
        "mode_usage": {},
        "file_types": {},
        "error_rate": 0.0,
        "quality_metrics": {
            "ready": 0,
            "error": 0,
            "processing": 0,
            "pending": 0,
        },
    }


async def get_analytics(
    organization_id: str,
    time_range: str = "7d",
) -> Dict[str, Any]:
    if not organization_id:
        return _empty_payload(time_range)
    if time_range not in _TIME_RANGE_TO_DAYS:
        time_range = "7d"

    cutoff = _cutoff(time_range)
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:

        # ── total + status breakdown ──────────────────────────────────
        total = await session.scalar(
            select(func.count())
            .select_from(DataAnalysisRunSQL)
            .where(
                DataAnalysisRunSQL.organization_id == organization_id,
                DataAnalysisRunSQL.deleted_at.is_(None),
                DataAnalysisRunSQL.created_at >= cutoff,
            )
        ) or 0

        status_rows = (await session.execute(
            select(DataAnalysisRunSQL.status, func.count())
            .where(
                DataAnalysisRunSQL.organization_id == organization_id,
                DataAnalysisRunSQL.deleted_at.is_(None),
                DataAnalysisRunSQL.created_at >= cutoff,
            )
            .group_by(DataAnalysisRunSQL.status)
        )).all()
        quality_metrics = {"ready": 0, "error": 0, "processing": 0, "pending": 0}
        for status_value, count in status_rows:
            if status_value in quality_metrics:
                quality_metrics[status_value] = int(count)

        error_rate = (quality_metrics["error"] / total) if total else 0.0

        # ── average processing time ──────────────────────────────────
        avg_ms = await session.scalar(
            select(func.avg(DataAnalysisRunSQL.processing_time_ms))
            .where(
                DataAnalysisRunSQL.organization_id == organization_id,
                DataAnalysisRunSQL.deleted_at.is_(None),
                DataAnalysisRunSQL.created_at >= cutoff,
                DataAnalysisRunSQL.processing_time_ms.is_not(None),
            )
        )
        average_processing_time = round((float(avg_ms) / 1000.0), 3) if avg_ms is not None else 0.0

        # ── mode usage ────────────────────────────────────────────────
        mode_rows = (await session.execute(
            select(DataAnalysisRunSQL.mode, func.count())
            .where(
                DataAnalysisRunSQL.organization_id == organization_id,
                DataAnalysisRunSQL.deleted_at.is_(None),
                DataAnalysisRunSQL.created_at >= cutoff,
            )
            .group_by(DataAnalysisRunSQL.mode)
        )).all()
        mode_usage = {mode: int(count) for mode, count in mode_rows if mode}

        # ── file type usage ──────────────────────────────────────────
        file_type_rows = (await session.execute(
            select(DataAnalysisRunSQL.content_type, func.count())
            .where(
                DataAnalysisRunSQL.organization_id == organization_id,
                DataAnalysisRunSQL.deleted_at.is_(None),
                DataAnalysisRunSQL.created_at >= cutoff,
            )
            .group_by(DataAnalysisRunSQL.content_type)
        )).all()
        file_types: Dict[str, int] = {}
        for ct, count in file_type_rows:
            label = _label_for_content_type(ct)
            file_types[label] = file_types.get(label, 0) + int(count)

    return {
        "time_range": time_range,
        "total_analyses": int(total),
        "average_processing_time": average_processing_time,
        "mode_usage": mode_usage,
        "file_types": file_types,
        "error_rate": round(error_rate, 3),
        "quality_metrics": quality_metrics,
    }


def _label_for_content_type(ct: str | None) -> str:
    if not ct:
        return "other"
    ct = ct.lower()
    if "csv" in ct:
        return "csv"
    if "spreadsheet" in ct or "excel" in ct or "xlsx" in ct or "xls" in ct:
        return "xlsx"
    if "json" in ct:
        return "json"
    if "plain" in ct or "text" in ct:
        return "csv"  # most plaintext uploads are CSV in this product
    return "other"
