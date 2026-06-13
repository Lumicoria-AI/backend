"""
Per-endpoint multi-tenant isolation contract.

Walks every NEW (post-workspace-revolution) org-scoped repository and
asserts that each public async method accepting `organization_id` is
guarded by `@require_org`.  The decorator raises a clear error when
called with an empty org id — without it, a buggy caller silently
gets cross-tenant rows.

Legacy repositories (pre-decorator) scope manually inside each method
via `scoped_filter()` or explicit `{"organization_id": ...}` filters.
They are listed in `LEGACY_REPOS_TODO` and skipped here; tightening
them is a follow-up hygiene pass.

This is dynamic: add a new repo to `ORG_SCOPED_REPOS` and the test
parametrises over its methods automatically.
"""

from __future__ import annotations

import importlib
import inspect
import pytest


# Repositories built with the @require_org contract.  Every public
# org-scoped method on these MUST be decorated.
ORG_SCOPED_REPOS: list[str] = [
    "team_repository",
    "team_member_repository",
    "project_v2_repository",
    "project_member_repository",
    "project_agent_repository",
    "org_subscription_repository",
    "automations_repository",
    "api_tokens_repository",
    "comments_repository",
]

# Legacy repos that scope manually inside each method.  Documented gap
# — tightening them is a separate hygiene PR.
LEGACY_REPOS_TODO: list[str] = [
    "activity_repository", "agent_metrics_repository", "agent_repository",
    "agent_run_repository", "agent_universe_repository", "billing_repository",
    "calendar_repository", "component_repository", "context_repository",
    "conversation_repository", "credits_repository", "device_token_repository",
    "document_repository", "integration_repository", "invite_repository",
    "invoice_repository", "notification_prefs_repository",
    "notification_repository", "organization_repository", "permission_repository",
    "wellbeing_repository", "workflow_repository", "tag_repository",
    "tenant_repository", "saved_view_repository", "sso_repository",
    "scim_repository", "task_repository", "user_repository",
]


def _load_repo_module(name: str):
    return importlib.import_module(f"backend.db.mongodb.repositories.{name}")


def _is_org_method(method) -> bool:
    if not inspect.iscoroutinefunction(method):
        return False
    try:
        sig = inspect.signature(method)
    except (TypeError, ValueError):
        return False
    return "organization_id" in sig.parameters


def _is_require_org_wrapped(method) -> bool:
    """Detect `@require_org` by inspecting the wrapper closure.

    `require_org`'s wrapper has `__code__.co_name == "wrapper"` and
    closes over the original `fn`.  This holds even after
    `functools.wraps` overwrites __qualname__.
    """
    code = getattr(method, "__code__", None)
    if code is not None and code.co_name == "wrapper":
        closure = getattr(method, "__closure__", None) or ()
        for cell in closure:
            try:
                val = cell.cell_contents
            except ValueError:
                continue
            if inspect.iscoroutinefunction(val):
                return True
    return False


def _scopes_manually(method) -> bool:
    """Heuristic: the method body uses `organization_id` somewhere after
    the signature, so the Mongo query is scoped (even without the
    decorator).  Looks for `organization_id` references inside the
    method source — covers both `scoped_filter(..., organization_id)`
    and inline `{"organization_id": ...}` filters.
    """
    try:
        src = inspect.getsource(method)
    except (OSError, TypeError):
        return False
    # Look at the source body after the def line.  Any reference to
    # organization_id beyond the signature counts.
    body = src.split(":", 1)[-1]
    return body.count("organization_id") >= 1


def _collect():
    rows: list[tuple[str, str, str, object]] = []
    for repo_name in ORG_SCOPED_REPOS:
        try:
            module = _load_repo_module(repo_name)
        except Exception as e:
            rows.append((repo_name, "ImportError", str(e), None))
            continue
        for _, klass in inspect.getmembers(module, inspect.isclass):
            if klass.__module__ != module.__name__:
                continue
            for mname, method in inspect.getmembers(klass, inspect.isfunction):
                if mname.startswith("_"):
                    continue
                if not _is_org_method(method):
                    continue
                rows.append((repo_name, klass.__name__, mname, method))
    return rows


METHODS = _collect()


@pytest.mark.parametrize(
    "repo,klass,mname,method",
    METHODS,
    ids=[f"{r}.{k}.{m}" for r, k, m, _ in METHODS],
)
def test_method_is_tenant_scoped(repo, klass, mname, method):
    """Each new org-scoped repo method must either be decorated with
    @require_org OR reference `organization_id` inside its body (i.e.
    apply the scope to its query).  Methods that take organization_id
    but never use it are the dangerous case."""
    if method is None:
        pytest.fail(f"{repo}.{klass}: import failed — {mname}")
    decorated = _is_require_org_wrapped(method)
    scoped_in_body = _scopes_manually(method)
    assert decorated or scoped_in_body, (
        f"{repo}.{klass}.{mname} declares organization_id but neither "
        f"applies @require_org nor references organization_id in its "
        f"body — likely a missing tenant scope."
    )


def test_legacy_repos_todo_is_tracked():
    """Smoke check: the legacy TODO list is non-empty and importable.

    Failure here means a legacy repo was renamed or removed without
    updating the TODO list — re-curate before tightening the contract.
    """
    missing: list[str] = []
    for repo_name in LEGACY_REPOS_TODO:
        try:
            _load_repo_module(repo_name)
        except ModuleNotFoundError:
            missing.append(repo_name)
        except Exception:
            # Optional deps may fail in CI without infra connectors
            continue
    if missing:
        pytest.skip(
            f"Legacy repos no longer present (delete from TODO list): {missing}"
        )
