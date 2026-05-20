"""Extraction audit log.

Every call to /extract, /discover-relations, or /fill-gaps produces a
row here so the operator UI can show recent activity, error rates per
action, and how many nodes / edges each run produced.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional

import structlog
from sqlalchemy import desc, func, select, update

from ...db.postgres import get_async_sessionmaker
from ...db.postgres_models import KGEdgeSQL, KGExtractionSQL, KGNodeSQL
from .sanitize import clean_label, clean_text

logger = structlog.get_logger(__name__)


VALID_ACTIONS = ("extract", "discover", "fill_gaps")
VALID_STATUSES = ("ready", "error")
_TIME_RANGE_TO_DAYS = {"1d": 1, "7d": 7, "30d": 30, "90d": 90, "1y": 365}


def _iso_utc(dt: Optional[datetime]) -> Optional[str]:
    """Serialize a naive UTC datetime as an ISO string with explicit
    `+00:00` so the browser does not reinterpret it as local time."""
    if dt is None:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.isoformat()


def to_dict(row: KGExtractionSQL) -> Dict[str, Any]:
    return {
        "id": row.id,
        "organization_id": row.organization_id,
        "user_id": row.user_id,
        "action": row.action,
        "status": row.status,
        "title": row.title,
        "source_kind": row.source_kind,
        "source_ref": row.source_ref,
        "content_preview": row.content_preview,
        "node_ids": list(row.node_ids or []),
        "edge_ids": list(row.edge_ids or []),
        "node_count": int(row.node_count or 0),
        "edge_count": int(row.edge_count or 0),
        "processing_time_ms": row.processing_time_ms,
        "error_message": row.error_message,
        "created_at": _iso_utc(row.created_at),
    }


async def create_extraction(
    *,
    organization_id: str,
    user_id: str,
    action: str,
    title: Optional[str] = None,
    source_kind: Optional[str] = None,
    source_ref: Optional[str] = None,
    content_preview: Optional[str] = None,
) -> Dict[str, Any]:
    if action not in VALID_ACTIONS:
        action = "extract"
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        row = KGExtractionSQL(
            organization_id=organization_id,
            user_id=user_id,
            action=action,
            status="ready",
            title=clean_label(title or "", max_len=500) or None,
            source_kind=(source_kind or None),
            source_ref=clean_label(source_ref or "", max_len=500) or None,
            content_preview=clean_label(content_preview or "", max_len=500) or None,
        )
        session.add(row)
        await session.commit()
        await session.refresh(row)
        return to_dict(row)


async def finalize_extraction(
    organization_id: str,
    extraction_id: str,
    *,
    status: str = "ready",
    node_ids: Optional[List[str]] = None,
    edge_ids: Optional[List[str]] = None,
    processing_time_ms: Optional[int] = None,
    error_message: Optional[str] = None,
) -> None:
    if status not in VALID_STATUSES:
        status = "ready"
    fields: Dict[str, Any] = {"status": status}
    if node_ids is not None:
        fields["node_ids"] = node_ids
        fields["node_count"] = len(node_ids)
    if edge_ids is not None:
        fields["edge_ids"] = edge_ids
        fields["edge_count"] = len(edge_ids)
    if processing_time_ms is not None:
        fields["processing_time_ms"] = int(processing_time_ms)
    if error_message is not None:
        fields["error_message"] = clean_text(error_message, max_len=2000)

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        await session.execute(
            update(KGExtractionSQL)
            .where(
                KGExtractionSQL.id == extraction_id,
                KGExtractionSQL.organization_id == organization_id,
            )
            .values(**fields)
        )
        await session.commit()


async def list_extractions(
    organization_id: str,
    *,
    action: Optional[str] = None,
    time_range: Optional[str] = None,
    limit: int = 30,
    offset: int = 0,
) -> Dict[str, Any]:
    limit = max(1, min(int(limit or 30), 200))
    offset = max(0, int(offset or 0))

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        where = [
            KGExtractionSQL.organization_id == organization_id,
            KGExtractionSQL.deleted_at.is_(None),
        ]
        if action and action in VALID_ACTIONS:
            where.append(KGExtractionSQL.action == action)
        if time_range and time_range in _TIME_RANGE_TO_DAYS:
            cutoff = datetime.utcnow() - timedelta(days=_TIME_RANGE_TO_DAYS[time_range])
            where.append(KGExtractionSQL.created_at >= cutoff)

        rows_stmt = (
            select(KGExtractionSQL)
            .where(*where)
            .order_by(desc(KGExtractionSQL.created_at))
            .limit(limit)
            .offset(offset)
        )
        count_stmt = select(func.count()).select_from(KGExtractionSQL).where(*where)

        rows = (await session.execute(rows_stmt)).scalars().all()
        total = await session.scalar(count_stmt) or 0

    return {
        "extractions": [to_dict(r) for r in rows],
        "total": int(total),
        "limit": limit,
        "offset": offset,
    }


async def soft_delete_extraction(organization_id: str, extraction_id: str) -> bool:
    """Soft-delete an extraction and the nodes / edges it produced.

    The user expects "delete extraction" to remove the work it created
    from the graph, not just hide an audit row.  We mark every row
    that has `source_extraction_id == extraction_id` as deleted in the
    same transaction so the visualization, stats, and search all stop
    seeing them.
    """
    now = datetime.utcnow()
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        result = await session.execute(
            update(KGExtractionSQL)
            .where(
                KGExtractionSQL.id == extraction_id,
                KGExtractionSQL.organization_id == organization_id,
                KGExtractionSQL.deleted_at.is_(None),
            )
            .values(deleted_at=now)
        )
        if (result.rowcount or 0) == 0:
            await session.rollback()
            return False

        # Cascade: drop the edges first (FK-by-convention to nodes), then
        # the nodes themselves.  Scope strictly by org + extraction.
        await session.execute(
            update(KGEdgeSQL)
            .where(
                KGEdgeSQL.organization_id == organization_id,
                KGEdgeSQL.source_extraction_id == extraction_id,
                KGEdgeSQL.deleted_at.is_(None),
            )
            .values(deleted_at=now)
        )
        await session.execute(
            update(KGNodeSQL)
            .where(
                KGNodeSQL.organization_id == organization_id,
                KGNodeSQL.source_extraction_id == extraction_id,
                KGNodeSQL.deleted_at.is_(None),
            )
            .values(deleted_at=now)
        )
        await session.commit()
        return True
