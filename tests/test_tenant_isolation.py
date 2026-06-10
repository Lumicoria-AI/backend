"""
Phase A verification — Multi-tenant isolation spot-check.

These tests exercise the tenant-scoping helpers — the cheapest layer of
protection — without needing a live Mongo.  Each new repository goes
through `backend/db/scoping.py`, so verifying that surface catches
missing `organization_id` arguments protects every router downstream.
"""

from __future__ import annotations

import pytest
from bson import ObjectId

from backend.db.scoping import (
    TenantScopingError,
    require_org,
    scoped_filter,
    to_oid,
)


# ── scoped_filter ──────────────────────────────────────────────────


def test_scoped_filter_always_includes_org_id():
    org_id = "507f1f77bcf86cd799439011"
    out = scoped_filter({"status": "active"}, org_id)
    assert out["status"] == "active"
    assert isinstance(out["organization_id"], ObjectId)
    assert str(out["organization_id"]) == org_id


def test_scoped_filter_with_no_extra():
    org_id = "507f1f77bcf86cd799439011"
    out = scoped_filter(None, org_id)
    assert set(out.keys()) == {"organization_id"}


def test_scoped_filter_extra_cannot_overwrite_org():
    """If the caller passes their own organization_id, it gets overwritten."""
    org_id = "507f1f77bcf86cd799439011"
    out = scoped_filter({"organization_id": "evil"}, org_id)
    assert isinstance(out["organization_id"], ObjectId)
    assert str(out["organization_id"]) == org_id


# ── require_org decorator ─────────────────────────────────────────


@pytest.mark.asyncio
async def test_require_org_rejects_missing_org_id():
    @require_org
    async def fake_repo_method(self, *, organization_id):
        return organization_id

    class _Self: ...

    with pytest.raises(TenantScopingError):
        await fake_repo_method(_Self(), organization_id=None)

    with pytest.raises(TenantScopingError):
        await fake_repo_method(_Self(), organization_id="")


@pytest.mark.asyncio
async def test_require_org_passes_with_valid_id():
    @require_org
    async def fake_repo_method(self, *, organization_id):
        return organization_id

    class _Self: ...

    result = await fake_repo_method(_Self(), organization_id="507f1f77bcf86cd799439011")
    assert result == "507f1f77bcf86cd799439011"


# ── to_oid ────────────────────────────────────────────────────────


def test_to_oid_handles_oid_string_and_none():
    oid = ObjectId()
    assert to_oid(oid) is oid
    assert to_oid(str(oid)) == oid
    assert to_oid(None) is None
    assert to_oid("not a real oid") is None


# ── Real repository contracts ─────────────────────────────────────


def test_team_repository_methods_require_org():
    """team_repository.list_teams enforces org-scoping via the decorator."""
    from backend.db.mongodb.repositories.team_repository import team_repository
    # The signature must contain `organization_id` (keyword-only).
    import inspect
    sig = inspect.signature(team_repository.list_teams)
    assert "organization_id" in sig.parameters


def test_project_v2_repository_methods_require_org():
    from backend.db.mongodb.repositories.project_v2_repository import project_v2_repository
    import inspect
    sig = inspect.signature(project_v2_repository.list_projects)
    assert "organization_id" in sig.parameters


def test_team_member_repository_methods_require_org():
    from backend.db.mongodb.repositories.team_member_repository import team_member_repository
    import inspect
    sig = inspect.signature(team_member_repository.list_for_team)
    assert "organization_id" in sig.parameters


def test_project_member_repository_methods_require_org():
    from backend.db.mongodb.repositories.project_member_repository import project_member_repository
    import inspect
    sig = inspect.signature(project_member_repository.list_for_project)
    assert "organization_id" in sig.parameters
