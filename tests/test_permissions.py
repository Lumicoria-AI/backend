"""
Phase A verification — Permissions matrix.

Exercises `backend/services/permissions.py:can()` against the action
catalogue with a representative role × plan grid.  Designed to catch
regressions when new actions or roles are added.
"""

from __future__ import annotations

import pytest

from backend.services.permissions import (
    ACTIONS,
    OrgRole,
    PermissionContext,
    ProjectRole,
    TeamRole,
    can,
)

# ── Helpers ────────────────────────────────────────────────────────


def ctx(
    *,
    org_role: str | None = None,
    team_role: str | None = None,
    project_role: str | None = None,
    plan_caps: dict | None = None,
    is_org_owner: bool = False,
    is_superuser: bool = False,
) -> PermissionContext:
    return PermissionContext(
        user_id="u1",
        organization_id="o1",
        org_role=org_role,
        team_role=team_role,
        project_role=project_role,
        plan="team" if plan_caps else "free",
        plan_caps=plan_caps or {},
        is_org_owner=is_org_owner,
        is_superuser=is_superuser,
    )


def _team_caps() -> dict:
    return {"teams_enabled": True, "advanced_features": True, "api_access": True,
            "custom_agent_templates": True, "strict_mode_enabled": False,
            "sso_enabled": False, "scim_enabled": False}


def _enterprise_caps() -> dict:
    return {**_team_caps(), "sso_enabled": True, "scim_enabled": True,
            "strict_mode_enabled": True}


# ── Superuser & owner bypass ───────────────────────────────────────


def test_superuser_bypasses_every_check():
    d = can("org.delete", ctx(is_superuser=True))
    assert d.allowed
    assert d.reason == "superuser"


def test_org_owner_bypasses_every_check():
    d = can("org.delete", ctx(is_org_owner=True))
    assert d.allowed
    assert d.reason == "org_owner"


# ── Org role escalation ────────────────────────────────────────────


@pytest.mark.parametrize(
    "action,role,expect",
    [
        ("org.read", OrgRole.VIEWER.value, True),
        ("org.read", OrgRole.MEMBER.value, True),
        ("org.read", OrgRole.ADMIN.value, True),
        ("org.update", OrgRole.VIEWER.value, False),
        ("org.update", OrgRole.MEMBER.value, False),
        ("org.update", OrgRole.ADMIN.value, True),
        ("org.delete", OrgRole.ADMIN.value, False),  # owner-only — admin not enough
        ("org.member.add", OrgRole.MEMBER.value, False),
        ("org.member.add", OrgRole.ADMIN.value, True),
        ("org.member.promote", OrgRole.ADMIN.value, False),  # owner-only
    ],
)
def test_org_role_gates(action, role, expect):
    d = can(action, ctx(org_role=role))
    assert d.allowed is expect, f"{action} for {role} expected {expect}, got {d}"


# ── Team role escalation ───────────────────────────────────────────


@pytest.mark.parametrize(
    "action,team_role,expect",
    [
        ("team.update", TeamRole.VIEWER.value, False),
        ("team.update", TeamRole.OPERATOR.value, False),
        ("team.update", TeamRole.EDITOR.value, False),
        ("team.update", TeamRole.TEAM_ADMIN.value, True),
        ("team.member.add", TeamRole.EDITOR.value, False),
        ("team.member.add", TeamRole.TEAM_ADMIN.value, True),
        ("team.analytics.read", TeamRole.OPERATOR.value, True),
        ("team.analytics.read", TeamRole.VIEWER.value, False),
    ],
)
def test_team_role_gates(action, team_role, expect):
    d = can(action, ctx(team_role=team_role, plan_caps=_team_caps()))
    assert d.allowed is expect, f"{action} for team_role={team_role} expected {expect}, got {d}"


# ── Project role escalation ────────────────────────────────────────


@pytest.mark.parametrize(
    "action,project_role,expect",
    [
        ("project.read", ProjectRole.VIEWER.value, True),
        ("project.update", ProjectRole.VIEWER.value, False),
        ("project.update", ProjectRole.EDITOR.value, True),
        ("project.delete", ProjectRole.EDITOR.value, False),
        ("project.delete", ProjectRole.LEAD.value, True),
        ("project.member.add", ProjectRole.EDITOR.value, False),
        ("project.member.add", ProjectRole.LEAD.value, True),
        ("project.task.create", ProjectRole.REVIEWER.value, False),
        ("project.task.create", ProjectRole.EDITOR.value, True),
        ("project.task.delete", ProjectRole.EDITOR.value, False),
        ("project.task.delete", ProjectRole.LEAD.value, True),
        ("project.chat.write", ProjectRole.VIEWER.value, True),
        ("project.automation.write", ProjectRole.LEAD.value, True),
    ],
)
def test_project_role_gates(action, project_role, expect):
    d = can(action, ctx(project_role=project_role, plan_caps=_team_caps()))
    assert d.allowed is expect, f"{action} for project_role={project_role} expected {expect}, got {d}"


# ── Scope additivity (team role lifts project, etc.) ──────────────


def test_team_admin_can_update_project_even_without_project_role():
    """team_admin lifts users above the project-level requirement."""
    d = can("project.update", ctx(team_role=TeamRole.EDITOR.value, plan_caps=_team_caps()))
    assert d.allowed


def test_org_member_can_create_project():
    d = can("project.create", ctx(org_role=OrgRole.MEMBER.value, plan_caps=_team_caps()))
    assert d.allowed


def test_unknown_action_denied():
    d = can("totally.bogus.action", ctx(org_role=OrgRole.ADMIN.value))
    assert not d.allowed
    assert d.reason.startswith("unknown_action")


# ── Plan capability gating ────────────────────────────────────────


def test_custom_agent_create_requires_plan_cap():
    """An editor on the Free plan can't create custom agents."""
    d = can(
        "agent.custom.create",
        ctx(project_role=ProjectRole.EDITOR.value, plan_caps={"custom_agent_templates": False}),
    )
    assert not d.allowed
    assert "plan_cap_missing" in d.reason
    assert d.upgrade_required in ("professional", "business", "enterprise")


def test_custom_agent_create_succeeds_on_team_plan():
    d = can("agent.custom.create", ctx(project_role=ProjectRole.EDITOR.value, plan_caps=_team_caps()))
    assert d.allowed


def test_sso_configure_requires_enterprise_cap():
    d = can("org.sso.configure", ctx(is_org_owner=True, plan_caps=_team_caps()))
    # Owner bypass — passes
    assert d.allowed

    d2 = can("org.sso.configure", ctx(org_role=OrgRole.ADMIN.value, plan_caps=_team_caps()))
    # Owner required + plan cap; admin lacks owner level
    assert not d2.allowed

    d3 = can("org.sso.configure", ctx(is_org_owner=True, plan_caps={"sso_enabled": False}))
    # Owner bypass wins before plan-cap check
    assert d3.allowed


def test_scim_configure_requires_enterprise_cap():
    d = can("org.scim.configure", ctx(org_role=OrgRole.ADMIN.value, plan_caps=_enterprise_caps()))
    assert not d.allowed  # admin not enough — owner only

    d2 = can("org.scim.configure", ctx(is_org_owner=True, plan_caps={"scim_enabled": False}))
    assert d2.allowed  # owner bypass


def test_audit_export_requires_advanced_features():
    d = can("org.audit.export", ctx(org_role=OrgRole.ADMIN.value, plan_caps={"advanced_features": False}))
    assert not d.allowed
    assert d.upgrade_required in ("professional", "business", "enterprise")


# ── Explicit allow / deny ────────────────────────────────────────


def test_explicit_deny_overrides_role():
    c = ctx(org_role=OrgRole.ADMIN.value)
    c.explicit_deny.add("org.update")
    d = can("org.update", c)
    assert not d.allowed
    assert d.reason == "explicit_deny"


def test_explicit_allow_short_circuits_role_check():
    c = ctx(org_role=OrgRole.VIEWER.value)
    c.explicit_allow.add("org.update")
    d = can("org.update", c)
    assert d.allowed


# ── Catalogue sanity ─────────────────────────────────────────────


def test_action_catalogue_not_empty():
    assert len(ACTIONS) >= 40, f"Permission catalogue shrank: {len(ACTIONS)}"


def test_every_action_resolves_for_owner():
    """Every action must resolve cleanly for the org owner — sanity check."""
    for action in ACTIONS:
        d = can(action, ctx(is_org_owner=True))
        assert d.allowed, f"Owner blocked from {action!r}: {d}"


# ── Composite role × plan grid ───────────────────────────────────


@pytest.mark.parametrize(
    "action",
    [
        "org.read", "team.read", "project.read", "agent.platform.run",
    ],
)
def test_baseline_member_can_read(action):
    d = can(action, ctx(
        org_role=OrgRole.MEMBER.value,
        team_role=TeamRole.VIEWER.value,
        project_role=ProjectRole.VIEWER.value,
        plan_caps=_team_caps(),
    ))
    assert d.allowed, f"member denied {action}: {d}"


@pytest.mark.parametrize(
    "action",
    [
        "org.delete", "org.member.promote", "org.billing.write",
        "org.sso.configure", "org.scim.configure",
    ],
)
def test_owner_only_actions_block_admin(action):
    d = can(action, ctx(org_role=OrgRole.ADMIN.value, plan_caps=_enterprise_caps()))
    assert not d.allowed, f"admin should be blocked from {action!r}: {d}"
