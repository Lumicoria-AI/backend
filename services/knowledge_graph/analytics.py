"""Real Knowledge Graph stats backing /knowledge-graph/stats.

Counts come straight from `kg_nodes` and `kg_edges` filtered by
`organization_id`.  No mock dict.
"""

from __future__ import annotations

from datetime import timezone
from typing import Any, Dict

from sqlalchemy import func, select

from ...db.postgres import get_async_sessionmaker
from ...db.postgres_models import KGEdgeSQL, KGNodeSQL


def _empty_stats() -> Dict[str, Any]:
    return {
        "node_count": 0,
        "edge_count": 0,
        "node_types": {},
        "relation_types": {},
        "last_updated_at": None,
    }


async def get_stats(organization_id: str) -> Dict[str, Any]:
    """Aggregate counts for this org's graph."""
    if not organization_id:
        return _empty_stats()

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:

        node_count = await session.scalar(
            select(func.count())
            .select_from(KGNodeSQL)
            .where(
                KGNodeSQL.organization_id == organization_id,
                KGNodeSQL.deleted_at.is_(None),
            )
        ) or 0

        edge_count = await session.scalar(
            select(func.count())
            .select_from(KGEdgeSQL)
            .where(
                KGEdgeSQL.organization_id == organization_id,
                KGEdgeSQL.deleted_at.is_(None),
            )
        ) or 0

        # Per-type counts.
        node_type_rows = (await session.execute(
            select(KGNodeSQL.type, func.count())
            .where(
                KGNodeSQL.organization_id == organization_id,
                KGNodeSQL.deleted_at.is_(None),
            )
            .group_by(KGNodeSQL.type)
        )).all()
        node_types = {t: int(c) for t, c in node_type_rows if t}

        edge_type_rows = (await session.execute(
            select(KGEdgeSQL.type, func.count())
            .where(
                KGEdgeSQL.organization_id == organization_id,
                KGEdgeSQL.deleted_at.is_(None),
            )
            .group_by(KGEdgeSQL.type)
        )).all()
        relation_types = {t: int(c) for t, c in edge_type_rows if t}

        # Most recent update across either table.
        last_node = await session.scalar(
            select(func.max(KGNodeSQL.updated_at)).where(
                KGNodeSQL.organization_id == organization_id,
                KGNodeSQL.deleted_at.is_(None),
            )
        )
        last_edge = await session.scalar(
            select(func.max(KGEdgeSQL.updated_at)).where(
                KGEdgeSQL.organization_id == organization_id,
                KGEdgeSQL.deleted_at.is_(None),
            )
        )
        candidates = [t for t in (last_node, last_edge) if t is not None]
        last_updated = max(candidates) if candidates else None

    if last_updated is not None and last_updated.tzinfo is None:
        last_updated = last_updated.replace(tzinfo=timezone.utc)

    return {
        "node_count": int(node_count),
        "edge_count": int(edge_count),
        "node_types": node_types,
        "relation_types": relation_types,
        "last_updated_at": last_updated.isoformat() if last_updated else None,
    }
