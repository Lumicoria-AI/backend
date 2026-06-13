"""
SCIM 2.0 filter-parser test rig.

Exercises `backend.api.v1.endpoints.scim._parse_filter` against the
canonical filter shapes Okta and Azure AD emit during SCIM provisioning
discovery.  These calls happen on every group / user sync, so
regressions silently break provisioning — the rig is fast and runs
without Mongo.

The function under test is pure: filter string in, dict out, no I/O.
"""

from __future__ import annotations

import re
import pytest

from backend.api.v1.endpoints.scim import _parse_filter


# ── Okta canonical filters ────────────────────────────────────────


def test_okta_user_lookup_by_userName():
    # Okta probes for an existing user via:
    #   GET /Users?filter=userName eq "alice@example.com"
    q = _parse_filter('userName eq "alice@example.com"')
    assert q == {"email": "alice@example.com"}


def test_okta_user_lookup_by_externalId():
    q = _parse_filter('externalId eq "okta_user_12345"')
    assert q == {"external_id": "okta_user_12345"}


def test_okta_user_search_emails_eq():
    q = _parse_filter('emails eq "bob@example.com"')
    assert q == {"email": "bob@example.com"}


# ── Azure AD canonical filters ────────────────────────────────────


def test_azure_user_lookup_by_userName_dot_value():
    # Azure sends userName.value for some PATCH-based discovery probes.
    q = _parse_filter('userName.value eq "carol@contoso.com"')
    assert q == {"email": "carol@contoso.com"}


def test_azure_emails_value_eq():
    q = _parse_filter('emails.value eq "dave@contoso.com"')
    assert q == {"email": "dave@contoso.com"}


# ── Substring / prefix matching (browsing the directory) ──────────


def test_starts_with_email_prefix():
    q = _parse_filter('userName sw "team-"')
    # Regex with case-insensitive prefix
    assert "email" in q
    assert isinstance(q["email"], dict)
    assert q["email"]["$regex"].startswith("^team\\-") or q["email"]["$regex"] == "^team\\-"
    assert q["email"]["$options"] == "i"


def test_contains_email_substring():
    q = _parse_filter('emails co "@acme."')
    assert "email" in q
    assert q["email"]["$regex"] == r"@acme\."
    assert q["email"]["$options"] == "i"


# ── Negative cases ────────────────────────────────────────────────


def test_empty_filter_returns_empty_dict():
    assert _parse_filter("") == {}
    assert _parse_filter(None) == {}


def test_unsupported_operator_returns_empty_dict():
    # `ne` is part of SCIM but we deliberately don't support it.  The
    # parser should fall through to an empty dict rather than raise so
    # the endpoint can return a full result set as a soft-fallback.
    assert _parse_filter('userName ne "alice@example.com"') == {}


def test_garbled_filter_returns_empty_dict():
    assert _parse_filter("this is not a SCIM filter") == {}


# ── Regex escape safety ───────────────────────────────────────────


def test_special_characters_in_value_are_escaped():
    # SCIM value contains regex metacharacters.  We must NOT pass them
    # through to MongoDB unescaped.
    q = _parse_filter('userName sw "user+tag.foo"')
    rx = q["email"]["$regex"]
    # The escaped form must contain backslashes before the metacharacters
    # (or render them literally — re.escape behavior).
    assert re.escape("user+tag.foo") in rx or rx.endswith(re.escape("user+tag.foo"))
