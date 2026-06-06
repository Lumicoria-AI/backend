"""
Lumicoria AI — Permissions resolver

Single source of truth for `can(user, action, resource)`.

Roles stack additively:

    Org role        owner   > admin   > member  > viewer
    Team role       admin   > editor  > operator > viewer
    Project role    lead    > editor  > reviewer > viewer

Resolution order for any decision:
    1. Owner override            — org owner always wins.
    2. Explicit deny             — a per-resource deny grant beats any allow.
    3. Project role              — most specific scope wins.
    4. Team role                 — escalates above org-member baseline.
    5. Org role                  — baseline.
    6. Plan capability           — gates premium features regardless of role.

The resolver is pure: given a User snapshot + the membership snapshot for the
resource it's evaluating, it returns a `PermissionDecision`.  Callers are
responsible for assembling the snapshot (cheap, since Phase A adds an
in-request cache).

This module deliberately knows nothing about FastAPI; it can be unit-tested
without spinning up the stack.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set


# ─────────────────────────────────────────────────────────── role hierarchies


class OrgRole(str, Enum):
    OWNER = "owner"
    ADMIN = "admin"
    MEMBER = "member"
    VIEWER = "viewer"


class TeamRole(str, Enum):
    TEAM_ADMIN = "team_admin"
    EDITOR = "editor"
    OPERATOR = "operator"
    VIEWER = "viewer"


class ProjectRole(str, Enum):
    LEAD = "lead"
    EDITOR = "editor"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


ORG_ROLE_RANK: Dict[str, int] = {
    OrgRole.OWNER.value: 4,
    OrgRole.ADMIN.value: 3,
    OrgRole.MEMBER.value: 2,
    OrgRole.VIEWER.value: 1,
}

TEAM_ROLE_RANK: Dict[str, int] = {
    TeamRole.TEAM_ADMIN.value: 4,
    TeamRole.EDITOR.value: 3,
    TeamRole.OPERATOR.value: 2,
    TeamRole.VIEWER.value: 1,
}

PROJECT_ROLE_RANK: Dict[str, int] = {
    ProjectRole.LEAD.value: 4,
    ProjectRole.EDITOR.value: 3,
    ProjectRole.REVIEWER.value: 2,
    ProjectRole.VIEWER.value: 1,
}


# ─────────────────────────────────────────────────────────── snapshot inputs


@dataclass
class PermissionContext:
    """All the data the resolver needs.  Assembled per-request by the API
    dependency layer."""
    user_id: str
    organization_id: Optional[str] = None
    team_id: Optional[str] = None
    project_id: Optional[str] = None

    # Roles for THIS context
    org_role: Optional[str] = None
    team_role: Optional[str] = None
    project_role: Optional[str] = None

    # Plan capability flags (resolved from PLAN_LIMITS at request time)
    plan: str = "free"
    plan_caps: Dict[str, object] = field(default_factory=dict)

    # Explicit deny / allow grants on the specific resource (rare)
    explicit_allow: Set[str] = field(default_factory=set)
    explicit_deny: Set[str] = field(default_factory=set)

    # True if this is the org owner (bypasses every check)
    is_org_owner: bool = False
    # True if the user is a platform superuser (Lumicoria staff)
    is_superuser: bool = False


@dataclass
class PermissionDecision:
    allowed: bool
    reason: str
    upgrade_required: Optional[str] = None  # plan name if denial is plan-based


# ─────────────────────────────────────────────────────────── action catalogue

# Each action declares the minimum role at each scope.  None means "this
# action does not exist at that scope" — fall back to the next.
#
# Plan capability key is checked in addition when set.
#
ACTIONS: Dict[str, Dict[str, Optional[str]]] = {
    # ── Organisation actions ──
    "org.read":               {"org": OrgRole.VIEWER.value},
    "org.update":             {"org": OrgRole.ADMIN.value},
    "org.delete":             {"org": OrgRole.OWNER.value},
    "org.invite":             {"org": OrgRole.ADMIN.value},
    "org.member.add":         {"org": OrgRole.ADMIN.value},
    "org.member.remove":      {"org": OrgRole.ADMIN.value},
    "org.member.promote":     {"org": OrgRole.OWNER.value},
    "org.billing.read":       {"org": OrgRole.ADMIN.value},
    "org.billing.write":      {"org": OrgRole.OWNER.value},
    "org.audit.read":         {"org": OrgRole.ADMIN.value},
    "org.audit.export":       {"org": OrgRole.ADMIN.value, "plan_cap": "advanced_features"},
    "org.sso.configure":      {"org": OrgRole.OWNER.value, "plan_cap": "sso_enabled"},
    "org.scim.configure":     {"org": OrgRole.OWNER.value, "plan_cap": "scim_enabled"},
    "org.webhook.manage":     {"org": OrgRole.ADMIN.value, "plan_cap": "api_access"},
    "org.api_token.manage":   {"org": OrgRole.ADMIN.value, "plan_cap": "api_access"},

    # ── Team actions ──
    "team.read":              {"org": OrgRole.MEMBER.value,  "team": TeamRole.VIEWER.value},
    "team.create":            {"org": OrgRole.MEMBER.value,  "plan_cap": "teams_enabled"},
    "team.update":            {"team": TeamRole.TEAM_ADMIN.value},
    "team.delete":            {"team": TeamRole.TEAM_ADMIN.value},
    "team.member.add":        {"team": TeamRole.TEAM_ADMIN.value},
    "team.member.remove":     {"team": TeamRole.TEAM_ADMIN.value},
    "team.invite":            {"team": TeamRole.TEAM_ADMIN.value},
    "team.analytics.read":    {"team": TeamRole.OPERATOR.value},
    "team.settings.write":    {"team": TeamRole.TEAM_ADMIN.value},

    # ── Project actions ──
    "project.read":           {"project": ProjectRole.VIEWER.value, "team": TeamRole.VIEWER.value, "org": OrgRole.MEMBER.value},
    "project.create":         {"org": OrgRole.MEMBER.value},
    "project.update":         {"project": ProjectRole.EDITOR.value, "team": TeamRole.EDITOR.value},
    "project.delete":         {"project": ProjectRole.LEAD.value,   "team": TeamRole.TEAM_ADMIN.value},
    "project.member.add":     {"project": ProjectRole.LEAD.value,   "team": TeamRole.EDITOR.value},
    "project.member.remove":  {"project": ProjectRole.LEAD.value,   "team": TeamRole.EDITOR.value},
    "project.agent.enable":   {"project": ProjectRole.EDITOR.value},
    "project.agent.run":      {"project": ProjectRole.EDITOR.value},
    "project.task.create":    {"project": ProjectRole.EDITOR.value},
    "project.task.update":    {"project": ProjectRole.EDITOR.value},
    "project.task.delete":    {"project": ProjectRole.LEAD.value},
    "project.doc.upload":     {"project": ProjectRole.EDITOR.value},
    "project.chat.write":     {"project": ProjectRole.VIEWER.value},
    "project.automation.write":  {"project": ProjectRole.LEAD.value, "plan_cap": "advanced_features"},
    "project.analytics.read": {"project": ProjectRole.REVIEWER.value, "team": TeamRole.OPERATOR.value},
    "project.strict_mode.set":{"project": ProjectRole.LEAD.value, "plan_cap": "strict_mode_enabled"},

    # ── Agent (platform) actions ──
    "agent.platform.run":     {"org": OrgRole.MEMBER.value},
    "agent.custom.create":    {"project": ProjectRole.EDITOR.value, "plan_cap": "custom_agent_templates"},
    "agent.custom.update":    {"project": ProjectRole.EDITOR.value, "plan_cap": "custom_agent_templates"},
    "agent.custom.delete":    {"project": ProjectRole.LEAD.value,   "plan_cap": "custom_agent_templates"},
}


# ─────────────────────────────────────────────────────────── resolver


def can(action: str, ctx: PermissionContext) -> PermissionDecision:
    """Pure-function permission resolver.  Returns a decision with a reason."""

    # 0) Platform superuser bypass — Lumicoria staff use this.
    if ctx.is_superuser:
        return PermissionDecision(True, "superuser")

    # 1) Org owner always wins.
    if ctx.is_org_owner:
        return PermissionDecision(True, "org_owner")

    # 2) Explicit deny first.
    if action in ctx.explicit_deny:
        return PermissionDecision(False, "explicit_deny")

    # 3) Explicit allow short-circuits role checks (still subject to plan caps below).
    if action in ctx.explicit_allow:
        return _check_plan_cap(action, ctx, baseline_allowed=True)

    spec = ACTIONS.get(action)
    if spec is None:
        return PermissionDecision(False, f"unknown_action:{action}")

    # 4) Try most specific scope first.  Higher rank role implies lower roles.
    if (req := spec.get("project")) and ctx.project_role:
        if PROJECT_ROLE_RANK.get(ctx.project_role, 0) >= PROJECT_ROLE_RANK[req]:
            return _check_plan_cap(action, ctx, baseline_allowed=True)

    if (req := spec.get("team")) and ctx.team_role:
        if TEAM_ROLE_RANK.get(ctx.team_role, 0) >= TEAM_ROLE_RANK[req]:
            return _check_plan_cap(action, ctx, baseline_allowed=True)

    if (req := spec.get("org")) and ctx.org_role:
        if ORG_ROLE_RANK.get(ctx.org_role, 0) >= ORG_ROLE_RANK[req]:
            return _check_plan_cap(action, ctx, baseline_allowed=True)

    return PermissionDecision(False, "insufficient_role")


def _check_plan_cap(action: str, ctx: PermissionContext, baseline_allowed: bool) -> PermissionDecision:
    spec = ACTIONS.get(action, {})
    plan_cap = spec.get("plan_cap")
    if not plan_cap:
        return PermissionDecision(baseline_allowed, "role_ok")
    # plan_cap may live in plan_caps dict (e.g. "advanced_features": True)
    if bool(ctx.plan_caps.get(plan_cap, False)):
        return PermissionDecision(True, "role_ok+plan_ok")
    # Suggest an upgrade plan that DOES include this cap.
    upgrade = _suggest_upgrade_for(plan_cap, ctx.plan)
    return PermissionDecision(False, f"plan_cap_missing:{plan_cap}", upgrade_required=upgrade)


def _suggest_upgrade_for(plan_cap: str, current_plan: str) -> Optional[str]:
    """Crude suggestion ladder.  Replace with PLAN_LIMITS introspection later."""
    if plan_cap in ("custom_agent_templates", "advanced_features", "api_access"):
        return "professional"
    if plan_cap in ("teams_enabled", "sso_enabled"):
        return "business"
    if plan_cap in ("scim_enabled", "strict_mode_enabled"):
        return "enterprise"
    return None


# Convenience matchers used by FastAPI dependencies

def assert_can(action: str, ctx: PermissionContext) -> None:
    """Raise PermissionError if denied.  Routers translate this to HTTPException."""
    decision = can(action, ctx)
    if not decision.allowed:
        err = PermissionError(decision.reason)
        # Attach upgrade hint so the API layer can craft a 402 vs 403.
        setattr(err, "upgrade_required", decision.upgrade_required)
        raise err
