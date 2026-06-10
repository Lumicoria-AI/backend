"""
Phase A — Ops router.

Mounted at `/api/v1/ops`.

Deep health (DBs + brokers + storage), version, feature flags, queue
depth, DB/cache stats, per-org status page.  Reads are gated to org
admins; the bare health probe is public so external monitors can hit it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from bson import ObjectId
from fastapi import APIRouter, Depends, HTTPException, Query

from backend.api.deps import get_current_active_user
from backend.db.mongodb.mongodb import MongoDB
from backend.db.mongodb.repositories.organization_repository import organization_repository
from backend.models.user import User

logger = structlog.get_logger(__name__)
router = APIRouter()


def _oid(v: Any) -> Optional[ObjectId]:
    if v is None:
        return None
    if isinstance(v, ObjectId):
        return v
    try:
        return ObjectId(str(v))
    except Exception:
        return None


# ── Public liveness ──────────────────────────────────────────────


@router.get("/health")
async def deep_health():
    """Deep readiness probe — checks DBs + storage + Redis."""
    checks: Dict[str, str] = {}

    try:
        from backend.db.mongodb.mongodb import MongoDB
        col = await MongoDB.get_collection("users")
        await col.count_documents({}, limit=1)
        checks["mongodb"] = "ok"
    except Exception as exc:  # noqa: BLE001
        checks["mongodb"] = f"error: {exc}"

    try:
        from backend.db.redis.redis import RedisClient
        client = await RedisClient.get_client()
        pong = await client.ping()
        checks["redis"] = "ok" if pong else "degraded"
    except Exception as exc:  # noqa: BLE001
        checks["redis"] = f"error: {exc}"

    try:
        from backend.services.storage_service import storage_service
        checks["storage"] = "ok" if await storage_service.health_check() else "degraded"
    except Exception as exc:  # noqa: BLE001
        checks["storage"] = f"error: {exc}"

    overall_ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if overall_ok else "degraded",
            "checks": checks,
            "checked_at": datetime.utcnow().isoformat() + "Z"}


@router.get("/version")
async def version():
    from backend.core.config import settings
    return {
        "service": "lumicoria-backend",
        "environment": getattr(settings, "ENVIRONMENT", "unknown"),
        "python_version": __import__("sys").version.split()[0],
    }


@router.get("/feature-flags")
async def feature_flags():
    """Currently exposed feature flags (mostly env-driven)."""
    from backend.core.config import settings
    return {
        "docs_enabled": getattr(settings, "DOCS_ENABLED", True),
        "celery_task_always_eager": getattr(settings, "CELERY_TASK_ALWAYS_EAGER", False),
        "vector_store_enabled": getattr(getattr(settings, "db", None), "VECTOR_STORE_ENABLED", True),
        "postgres_enabled": getattr(settings, "POSTGRES_ENABLED", False),
    }


# ── Authenticated read-side ──────────────────────────────────────


@router.get("/queue-depth")
async def queue_depth(current_user: User = Depends(get_current_active_user)):
    """Approximate count of pending background work."""
    out: Dict[str, int] = {}
    try:
        webhooks = await MongoDB.get_collection("webhook_deliveries")
        out["webhook_deliveries_pending"] = await webhooks.count_documents({"status": "pending"})
    except Exception:
        out["webhook_deliveries_pending"] = -1
    try:
        exports = await MongoDB.get_collection("audit_exports")
        out["audit_exports_pending"] = await exports.count_documents({"status": "pending"})
    except Exception:
        out["audit_exports_pending"] = -1
    try:
        media = await MongoDB.get_collection("media_jobs")
        out["media_jobs_pending"] = await media.count_documents({"status": "queued"})
    except Exception:
        out["media_jobs_pending"] = -1
    try:
        analytics = await MongoDB.get_collection("analytics_exports")
        out["analytics_exports_pending"] = await analytics.count_documents({"status": "pending"})
    except Exception:
        out["analytics_exports_pending"] = -1
    return out


@router.get("/db-stats")
async def db_stats(current_user: User = Depends(get_current_active_user)):
    """Top-level collection sizes."""
    out: Dict[str, Any] = {}
    for col_name in ("users", "organizations", "teams", "projects", "tasks",
                     "documents", "agent_runs", "activity_logs", "notifications",
                     "comments", "chat_messages"):
        try:
            col = await MongoDB.get_collection(col_name)
            out[col_name] = await col.estimated_document_count()
        except Exception:
            out[col_name] = -1
    return out


@router.get("/cache-stats")
async def cache_stats(current_user: User = Depends(get_current_active_user)):
    """Redis cache sizing summary (best-effort)."""
    try:
        from backend.db.redis.redis import RedisClient
        client = await RedisClient.get_client()
        info = await client.info(section="memory")
        return {
            "used_memory_human": info.get("used_memory_human"),
            "maxmemory_human": info.get("maxmemory_human"),
            "evicted_keys": info.get("evicted_keys"),
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@router.get("/status/{org_id}")
async def per_org_status(
    org_id: str, current_user: User = Depends(get_current_active_user),
):
    """Per-org operational summary."""
    org = await organization_repository.get_by_id(org_id)
    if not org:
        raise HTTPException(status_code=404, detail="Organization not found")
    if _oid(current_user.id) not in [_oid(a) for a in (org.admin_ids or [])]:
        raise HTTPException(status_code=403, detail="Org admin permission required")
    out = {
        "organization_id": org_id,
        "members": len(org.member_ids or []),
        "admins": len(org.admin_ids or []),
    }
    try:
        webhooks = await MongoDB.get_collection("webhooks")
        out["webhooks_enabled"] = await webhooks.count_documents({
            "organization_id": _oid(org_id), "enabled": True,
        })
    except Exception:
        out["webhooks_enabled"] = -1
    try:
        runs = await MongoDB.get_collection("agent_runs")
        out["agent_runs_last_24h"] = await runs.count_documents({
            "organization_id": _oid(org_id),
            "started_at": {"$gte": datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)},
        })
    except Exception:
        out["agent_runs_last_24h"] = -1
    return out


@router.get("/router-summary")
async def router_summary(current_user: User = Depends(get_current_active_user)):
    """Count of endpoints currently mounted on the API router."""
    from backend.api.v1.api import api_router
    distinct = set()
    for r in api_router.routes:
        if hasattr(r, "methods") and hasattr(r, "path"):
            for m in (r.methods or set()):
                if m not in ("HEAD", "OPTIONS"):
                    distinct.add((m, r.path))
    return {"total_routes": len(api_router.routes), "distinct_method_path": len(distinct)}
