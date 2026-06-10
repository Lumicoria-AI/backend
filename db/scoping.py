"""
Lumicoria AI — Tenant scoping helper

Every new repository method must accept and enforce an `organization_id`.
This module gives that requirement teeth:

    from backend.db.scoping import require_org, scoped_filter

    @require_org
    async def list_projects(self, organization_id: str, **kw):
        col = await self.collection
        return await col.find(scoped_filter({"status": "active"}, organization_id)).to_list(None)

The decorator raises an explicit `TenantScopingError` if `organization_id` is
missing or empty, which is far easier to debug than `find()` silently
returning everyone's rows.
"""

from __future__ import annotations

import functools
from typing import Any, Awaitable, Callable, Dict, Optional, TypeVar

from bson import ObjectId


class TenantScopingError(RuntimeError):
    """Raised when a repository call is made without an `organization_id`."""


F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def require_org(fn: F) -> F:
    """Decorator: assert that `organization_id` is present and non-empty."""

    @functools.wraps(fn)
    async def wrapper(*args, **kwargs):  # type: ignore[no-untyped-def]
        org_id = kwargs.get("organization_id")
        if not org_id:
            raise TenantScopingError(
                f"{fn.__qualname__} requires organization_id (got {org_id!r})"
            )
        return await fn(*args, **kwargs)

    return wrapper  # type: ignore[return-value]


def scoped_filter(
    extra: Optional[Dict[str, Any]],
    organization_id: str,
) -> Dict[str, Any]:
    """Build a Mongo find() filter that always includes `organization_id`.

    Security: the canonical `organization_id` cannot be overridden by
    caller-supplied extras.  Even if the caller passes their own
    `organization_id` (e.g. accidentally inheriting it from a generic
    filter dict), the scoped value wins.
    """
    org_oid = ObjectId(organization_id)
    base: Dict[str, Any] = {}
    if extra:
        base.update(extra)
    # Apply org scoping LAST so it always wins.
    base["organization_id"] = org_oid
    return base


def to_oid(value: Any) -> Optional[ObjectId]:
    """Best-effort ObjectId coercion.  Returns None on invalid input rather
    than raising — used by request validators."""
    if value is None:
        return None
    if isinstance(value, ObjectId):
        return value
    try:
        return ObjectId(str(value))
    except Exception:
        return None
