"""CRUD over `data_analysis_runs`.

Tenant scoped helpers: every read / write requires an `organization_id`
and applies it as a filter before any other clause.  Soft delete via
`deleted_at`.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import and_, asc, desc, func, select, update

from ...db.postgres import get_async_sessionmaker
from ...db.postgres_models import DataAnalysisRunSQL

logger = structlog.get_logger(__name__)


VALID_MODES = (
    "exploratory",
    "statistical",
    "visualization",
    "anomaly",
    "trend",
    "report",
)
VALID_STATUSES = ("pending", "processing", "ready", "error")
_TIME_RANGE_TO_DAYS = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}


def to_dict(row: DataAnalysisRunSQL, *, include_full: bool = True) -> Dict[str, Any]:
    """Serialize a row.  When `include_full=False`, omit the heavy
    JSONB fields so the list endpoint stays responsive."""
    base: Dict[str, Any] = {
        "id": row.id,
        "organization_id": row.organization_id,
        "user_id": row.user_id,
        "mode": row.mode,
        "status": row.status,
        "s3_key": row.s3_key,
        "filename": row.filename,
        "original_filename": row.original_filename,
        "content_type": row.content_type,
        "size_bytes": row.size_bytes,
        "row_count": row.row_count,
        "column_count": row.column_count,
        "processing_time_ms": row.processing_time_ms,
        "error_message": row.error_message,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }
    if include_full:
        base.update({
            "columns": list(row.columns or []),
            "preview_rows": list(row.preview_rows or []),
            "summary_stats": row.summary_stats,
            "visualizations": list(row.visualizations or []),
            "anomalies": row.anomalies,
            "trends": row.trends,
            "statistical_results": getattr(row, "statistical_results", None),
            "insights": list(row.insights or []),
            "ai_summary": row.ai_summary,
            "question_history": list(row.question_history or []),
        })
    return base


# ── Create ──────────────────────────────────────────────────────────────


async def create_run(
    *,
    organization_id: str,
    user_id: str,
    mode: str,
    s3_key: str,
    filename: str,
    original_filename: Optional[str] = None,
    content_type: Optional[str] = None,
    size_bytes: int = 0,
    status: str = "pending",
) -> Dict[str, Any]:
    if mode not in VALID_MODES:
        mode = "exploratory"
    if status not in VALID_STATUSES:
        status = "pending"

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        row = DataAnalysisRunSQL(
            organization_id=organization_id,
            user_id=user_id,
            mode=mode,
            status=status,
            s3_key=s3_key,
            filename=filename,
            original_filename=original_filename,
            content_type=content_type,
            size_bytes=size_bytes,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return to_dict(row)


# ── Read ────────────────────────────────────────────────────────────────


async def list_runs(
    organization_id: str,
    *,
    status: Optional[str] = None,
    mode: Optional[str] = None,
    time_range: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> Dict[str, Any]:
    limit = max(1, min(int(limit or 50), 200))
    offset = max(0, int(offset or 0))

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        where = [
            DataAnalysisRunSQL.organization_id == organization_id,
            DataAnalysisRunSQL.deleted_at.is_(None),
        ]
        if status and status in VALID_STATUSES:
            where.append(DataAnalysisRunSQL.status == status)
        if mode and mode in VALID_MODES:
            where.append(DataAnalysisRunSQL.mode == mode)
        if time_range and time_range in _TIME_RANGE_TO_DAYS:
            cutoff = datetime.utcnow() - timedelta(days=_TIME_RANGE_TO_DAYS[time_range])
            where.append(DataAnalysisRunSQL.created_at >= cutoff)

        rows_stmt = (
            select(DataAnalysisRunSQL)
            .where(*where)
            .order_by(desc(DataAnalysisRunSQL.created_at))
            .limit(limit)
            .offset(offset)
        )
        count_stmt = (
            select(func.count())
            .select_from(DataAnalysisRunSQL)
            .where(*where)
        )

        rows = (await session.execute(rows_stmt)).scalars().all()
        total = await session.scalar(count_stmt) or 0

    return {
        "runs": [to_dict(r, include_full=False) for r in rows],
        "total": int(total),
        "limit": limit,
        "offset": offset,
    }


async def get_run(
    organization_id: str, run_id: str
) -> Optional[Dict[str, Any]]:
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        row = (
            await session.execute(
                select(DataAnalysisRunSQL).where(
                    DataAnalysisRunSQL.id == run_id,
                    DataAnalysisRunSQL.organization_id == organization_id,
                    DataAnalysisRunSQL.deleted_at.is_(None),
                )
            )
        ).scalar_one_or_none()
        return to_dict(row) if row else None


# ── Update ──────────────────────────────────────────────────────────────


async def update_run_results(
    organization_id: str,
    run_id: str,
    *,
    status: Optional[str] = None,
    row_count: Optional[int] = None,
    column_count: Optional[int] = None,
    columns: Optional[List[Dict[str, Any]]] = None,
    preview_rows: Optional[List[Dict[str, Any]]] = None,
    summary_stats: Optional[Dict[str, Any]] = None,
    visualizations: Optional[List[Dict[str, Any]]] = None,
    anomalies: Optional[Dict[str, Any]] = None,
    trends: Optional[Dict[str, Any]] = None,
    statistical_results: Optional[Dict[str, Any]] = None,
    insights: Optional[List[Dict[str, Any]]] = None,
    ai_summary: Optional[str] = None,
    processing_time_ms: Optional[int] = None,
    error_message: Optional[str] = None,
    mode: Optional[str] = None,
) -> None:
    """Patch a subset of result fields after the pipeline runs."""
    fields: Dict[str, Any] = {}
    if status is not None and status in VALID_STATUSES:
        fields["status"] = status
    if row_count is not None:
        fields["row_count"] = int(row_count)
    if column_count is not None:
        fields["column_count"] = int(column_count)
    if columns is not None:
        fields["columns"] = columns
    if preview_rows is not None:
        fields["preview_rows"] = preview_rows
    if summary_stats is not None:
        fields["summary_stats"] = summary_stats
    if visualizations is not None:
        fields["visualizations"] = visualizations
    if anomalies is not None:
        fields["anomalies"] = anomalies
    if trends is not None:
        fields["trends"] = trends
    if statistical_results is not None:
        fields["statistical_results"] = statistical_results
    if insights is not None:
        fields["insights"] = insights
    if ai_summary is not None:
        fields["ai_summary"] = ai_summary
    if processing_time_ms is not None:
        fields["processing_time_ms"] = int(processing_time_ms)
    if error_message is not None:
        fields["error_message"] = error_message
    if mode is not None and mode in VALID_MODES:
        fields["mode"] = mode

    if not fields:
        return
    fields["updated_at"] = datetime.utcnow()

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        await session.execute(
            update(DataAnalysisRunSQL)
            .where(
                DataAnalysisRunSQL.id == run_id,
                DataAnalysisRunSQL.organization_id == organization_id,
            )
            .values(**fields)
        )
        await session.commit()


async def append_question_turn(
    organization_id: str,
    run_id: str,
    turn: Dict[str, Any],
) -> None:
    """Push a new {question, answer, model_used, asked_at} entry onto
    the run's question_history JSONB array.  Best effort."""
    SessionLocal = get_async_sessionmaker()
    try:
        async with SessionLocal() as session:
            row = (await session.execute(
                select(DataAnalysisRunSQL).where(
                    DataAnalysisRunSQL.id == run_id,
                    DataAnalysisRunSQL.organization_id == organization_id,
                    DataAnalysisRunSQL.deleted_at.is_(None),
                )
            )).scalar_one_or_none()
            if not row:
                return
            history = list(row.question_history or [])
            history.append(turn)
            row.question_history = history
            row.updated_at = datetime.utcnow()
            await session.commit()
    except Exception as e:  # noqa: BLE001
        logger.warning("question_history_append_failed", error=str(e), run_id=run_id)


async def soft_delete_run(organization_id: str, run_id: str) -> bool:
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        result = await session.execute(
            update(DataAnalysisRunSQL)
            .where(
                DataAnalysisRunSQL.id == run_id,
                DataAnalysisRunSQL.organization_id == organization_id,
                DataAnalysisRunSQL.deleted_at.is_(None),
            )
            .values(deleted_at=datetime.utcnow(), updated_at=datetime.utcnow())
        )
        await session.commit()
        return (result.rowcount or 0) > 0
