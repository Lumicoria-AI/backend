"""Per-organization knowledge graph repository.

The router gives us an `organization_id`. We build a fresh `nx.DiGraph`
from that org's rows in `kg_nodes` / `kg_edges`, hand it to the
KnowledgeGraphAgent for in-memory work, then flush any new nodes / edges
back through this module.

The agent itself is NEVER allowed to touch SQLAlchemy.  All durability
lives here so org isolation cannot be bypassed by future agent edits.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Tuple

import structlog
from sqlalchemy import and_, asc, desc, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from ...db.postgres import get_async_sessionmaker
from ...db.postgres_models import KGEdgeSQL, KGNodeSQL
from .sanitize import clean_label, clean_text, coerce_jsonable

logger = structlog.get_logger(__name__)


# Hard cap on how many nodes we will materialise into the in-memory
# graph for a single request.  Protects against runaway memory if a
# tenant somehow accumulates a graph that does not fit comfortably.
MAX_NODES_PER_REQUEST = 50_000


# ── Build an in-memory DiGraph for the agent ────────────────────────


def _node_row_to_attrs(row: KGNodeSQL) -> Dict[str, Any]:
    return {
        "id": row.id,
        "type": row.type,
        "label": row.label,
        "properties": dict(row.properties or {}),
        "confidence": float(row.confidence or 1.0),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def _edge_row_to_attrs(row: KGEdgeSQL) -> Dict[str, Any]:
    return {
        "id": row.id,
        "type": row.type,
        "properties": dict(row.properties or {}),
        "confidence": float(row.confidence or 1.0),
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


async def load_org_graph(organization_id: str):
    """Build a fresh networkx DiGraph from this org's persisted rows.

    Importing networkx here (rather than at module load) keeps cold
    imports cheap for non-KG paths.
    """
    import networkx as nx

    g = nx.DiGraph()
    if not organization_id:
        return g

    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        # Bound the load.  Any tenant that hits this cap should look at
        # node merging / pruning rather than expecting the full graph
        # to come back on every request.
        nodes_stmt = (
            select(KGNodeSQL)
            .where(
                KGNodeSQL.organization_id == organization_id,
                KGNodeSQL.deleted_at.is_(None),
            )
            .order_by(KGNodeSQL.created_at.asc())
            .limit(MAX_NODES_PER_REQUEST)
        )
        nodes = (await session.execute(nodes_stmt)).scalars().all()
        for n in nodes:
            g.add_node(n.id, **_node_row_to_attrs(n))

        if not nodes:
            return g

        node_ids = {n.id for n in nodes}
        edges_stmt = (
            select(KGEdgeSQL)
            .where(
                KGEdgeSQL.organization_id == organization_id,
                KGEdgeSQL.deleted_at.is_(None),
                KGEdgeSQL.source_id.in_(node_ids),
                KGEdgeSQL.target_id.in_(node_ids),
            )
        )
        edges = (await session.execute(edges_stmt)).scalars().all()
        for e in edges:
            g.add_edge(e.source_id, e.target_id, **_edge_row_to_attrs(e))

    return g


# ── Persist new nodes / edges from agent output ─────────────────────


async def upsert_nodes(
    organization_id: str,
    nodes: Iterable[Dict[str, Any]],
    *,
    extraction_id: Optional[str] = None,
) -> List[str]:
    """Insert nodes that don't already exist for this org (matched by
    case-insensitive label + type).  Returns the inserted ids.

    Implementation: SELECT existing (lower(label), type) tuples in one
    query, then INSERT only the missing ones, each row wrapped in its
    own SAVEPOINT so a unique-constraint race on one row never poisons
    the whole batch.  The partial unique index on (org, lower(label),
    type) WHERE deleted_at IS NULL is still the source of truth — this
    just avoids the ON-CONFLICT-against-partial-index dance.
    """
    rows: List[Dict[str, Any]] = []
    for n in nodes or []:
        label = clean_label((n or {}).get("label"))
        if not label:
            continue
        node_type = clean_label((n or {}).get("type") or "concept", max_len=32) or "concept"
        properties = coerce_jsonable((n or {}).get("properties") or {})
        try:
            confidence = float((n or {}).get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        confidence = max(0.0, min(1.0, confidence))
        rows.append({
            "id": (n or {}).get("id") or None,
            "organization_id": organization_id,
            "type": node_type,
            "label": label,
            "properties": properties if isinstance(properties, dict) else {},
            "confidence": confidence,
            "source_extraction_id": extraction_id,
        })

    if not rows:
        return []

    inserted_ids: List[str] = []
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        # Pre-check: which (lower(label), type) tuples already exist?
        candidates = {(r["label"].lower(), r["type"]) for r in rows}
        existing_rows = (await session.execute(
            select(func.lower(KGNodeSQL.label), KGNodeSQL.type).where(
                KGNodeSQL.organization_id == organization_id,
                KGNodeSQL.deleted_at.is_(None),
                func.lower(KGNodeSQL.label).in_([c[0] for c in candidates]),
                KGNodeSQL.type.in_([c[1] for c in candidates]),
            )
        )).all()
        existing_set = {(lbl, typ) for lbl, typ in existing_rows}

        for row in rows:
            key = (row["label"].lower(), row["type"])
            if key in existing_set:
                continue
            if row.get("id") is None:
                row.pop("id")
            # SAVEPOINT per row so a race on the unique index does not
            # abort the surrounding transaction.
            try:
                async with session.begin_nested():
                    inserted = await session.scalar(
                        pg_insert(KGNodeSQL)
                        .values(**row)
                        .returning(KGNodeSQL.id)
                    )
                if inserted:
                    inserted_ids.append(inserted)
                    existing_set.add(key)  # avoid duplicates within batch
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "kg_node_insert_failed",
                    error=str(e),
                    label=row.get("label"),
                )
        await session.commit()
    return inserted_ids


async def upsert_edges(
    organization_id: str,
    edges: Iterable[Dict[str, Any]],
    *,
    extraction_id: Optional[str] = None,
) -> List[str]:
    """Insert edges.  No dedupe constraint at the DB level for edges
    (the same two nodes can legitimately have multiple relations), but
    we drop edges whose endpoints don't exist for this org."""
    rows: List[Dict[str, Any]] = []
    for e in edges or []:
        source_id = (e or {}).get("source_id") or (e or {}).get("source")
        target_id = (e or {}).get("target_id") or (e or {}).get("target")
        if not source_id or not target_id:
            continue
        edge_type = clean_label((e or {}).get("type") or "related_to", max_len=32) or "related_to"
        properties = coerce_jsonable((e or {}).get("properties") or {})
        try:
            confidence = float((e or {}).get("confidence", 1.0))
        except (TypeError, ValueError):
            confidence = 1.0
        confidence = max(0.0, min(1.0, confidence))
        rows.append({
            "organization_id": organization_id,
            "source_id": str(source_id),
            "target_id": str(target_id),
            "type": edge_type,
            "properties": properties if isinstance(properties, dict) else {},
            "confidence": confidence,
            "source_extraction_id": extraction_id,
        })

    if not rows:
        return []

    SessionLocal = get_async_sessionmaker()
    inserted_ids: List[str] = []
    async with SessionLocal() as session:
        # Validate that source / target nodes exist for this org before
        # we insert.  Tenant isolation enforced at the SQL layer.
        node_ids_in_rows = {r["source_id"] for r in rows} | {r["target_id"] for r in rows}
        existing = await session.execute(
            select(KGNodeSQL.id).where(
                KGNodeSQL.organization_id == organization_id,
                KGNodeSQL.deleted_at.is_(None),
                KGNodeSQL.id.in_(node_ids_in_rows),
            )
        )
        existing_ids = {r[0] for r in existing}
        valid_rows = [
            r for r in rows
            if r["source_id"] in existing_ids and r["target_id"] in existing_ids
        ]
        if not valid_rows:
            return []
        for r in valid_rows:
            try:
                async with session.begin_nested():
                    inserted = await session.scalar(
                        pg_insert(KGEdgeSQL).values(**r).returning(KGEdgeSQL.id)
                    )
                if inserted:
                    inserted_ids.append(inserted)
            except Exception as e:  # noqa: BLE001
                logger.warning("kg_edge_insert_failed", error=str(e))
        await session.commit()
    return inserted_ids


# ── Lookups for the agent's existing helpers ─────────────────────────


async def find_node_id_by_label_type(
    organization_id: str,
    label: str,
    node_type: Optional[str] = None,
) -> Optional[str]:
    """Look up a node id by its label.  If `node_type` is provided we
    try an exact (label, type) match first; if that misses (the LLM
    routinely guesses the wrong type when emitting edges) we fall
    back to any node with that label in this org.  Case-insensitive.
    """
    if not label:
        return None
    SessionLocal = get_async_sessionmaker()
    async with SessionLocal() as session:
        base = select(KGNodeSQL.id).where(
            KGNodeSQL.organization_id == organization_id,
            func.lower(KGNodeSQL.label) == label.lower(),
            KGNodeSQL.deleted_at.is_(None),
        )
        if node_type:
            typed = await session.execute(
                base.where(KGNodeSQL.type == node_type).limit(1)
            )
            hit = typed.scalar_one_or_none()
            if hit:
                return hit
        # Fallback: any node in this org with this label.
        return (await session.execute(base.limit(1))).scalar_one_or_none()
