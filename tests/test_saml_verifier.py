"""
SAML response verifier test rig.

Exercises `verify_saml_response` against synthetic Okta-style and
Azure AD-style SAMLResponses.  Uses `skip_signature=True` so the rig
runs without a real IdP cert / keypair — the goal is to verify the
parser handles each vendor's quirks (NameID format, attribute claim
URIs, audience nesting).

We construct minimal-but-valid SAML2 XML inline, base64-encode it, and
hand it to the verifier.  Each test asserts the resulting
`SamlAssertion` carries the right NameID, email, and audience.
"""

from __future__ import annotations

import base64
import pytest

from backend.services.saml_verifier import (
    SamlVerificationError,
    verify_saml_response,
)


def _b64(xml: str) -> str:
    return base64.b64encode(xml.encode("utf-8")).decode("ascii")


# ── Okta-shaped response ──────────────────────────────────────────


OKTA_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
                xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
                ID="okta_resp_1" Version="2.0" IssueInstant="2026-06-13T10:00:00Z">
  <saml:Issuer>http://www.okta.com/exk1okta</saml:Issuer>
  <saml:Assertion ID="okta_a1" Version="2.0" IssueInstant="2026-06-13T10:00:00Z">
    <saml:Issuer>http://www.okta.com/exk1okta</saml:Issuer>
    <saml:Subject>
      <saml:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress">alice@okta.example</saml:NameID>
    </saml:Subject>
    <saml:Conditions NotBefore="2026-06-13T10:00:00Z" NotOnOrAfter="2099-12-31T23:59:59Z">
      <saml:AudienceRestriction>
        <saml:Audience>https://lumicoria.ai/sso/acs</saml:Audience>
      </saml:AudienceRestriction>
    </saml:Conditions>
    <saml:AttributeStatement>
      <saml:Attribute Name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress">
        <saml:AttributeValue>alice@okta.example</saml:AttributeValue>
      </saml:Attribute>
      <saml:Attribute Name="firstName">
        <saml:AttributeValue>Alice</saml:AttributeValue>
      </saml:Attribute>
      <saml:Attribute Name="lastName">
        <saml:AttributeValue>O'Connor</saml:AttributeValue>
      </saml:Attribute>
    </saml:AttributeStatement>
  </saml:Assertion>
</samlp:Response>
"""


def test_okta_response_parses_email_and_name():
    pytest.importorskip("lxml")
    assertion = verify_saml_response(
        saml_response_b64=_b64(OKTA_RESPONSE),
        idp_certificate_pem="dummy",
        expected_issuer="http://www.okta.com/exk1okta",
        expected_audience="https://lumicoria.ai/sso/acs",
        skip_signature=True,
    )
    assert assertion.name_id == "alice@okta.example"
    assert assertion.email == "alice@okta.example"
    assert assertion.given_name == "Alice"
    assert assertion.family_name == "O'Connor"
    assert assertion.issuer == "http://www.okta.com/exk1okta"
    assert assertion.audience == "https://lumicoria.ai/sso/acs"


# ── Azure AD-shaped response ──────────────────────────────────────


AZURE_RESPONSE = """<?xml version="1.0" encoding="UTF-8"?>
<samlp:Response xmlns:samlp="urn:oasis:names:tc:SAML:2.0:protocol"
                xmlns:saml="urn:oasis:names:tc:SAML:2.0:assertion"
                ID="azure_resp_1" Version="2.0" IssueInstant="2026-06-13T10:00:00Z">
  <saml:Issuer>https://sts.windows.net/contoso/</saml:Issuer>
  <saml:Assertion ID="azure_a1" Version="2.0" IssueInstant="2026-06-13T10:00:00Z">
    <saml:Issuer>https://sts.windows.net/contoso/</saml:Issuer>
    <saml:Subject>
      <saml:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:unspecified">bob@contoso.com</saml:NameID>
    </saml:Subject>
    <saml:Conditions NotBefore="2026-06-13T10:00:00Z" NotOnOrAfter="2099-12-31T23:59:59Z">
      <saml:AudienceRestriction>
        <saml:Audience>https://lumicoria.ai/sso/acs</saml:Audience>
      </saml:AudienceRestriction>
    </saml:Conditions>
    <saml:AttributeStatement>
      <saml:Attribute Name="http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress">
        <saml:AttributeValue>bob@contoso.com</saml:AttributeValue>
      </saml:Attribute>
      <saml:Attribute Name="http://schemas.microsoft.com/identity/claims/displayname">
        <saml:AttributeValue>Bob Carlsson</saml:AttributeValue>
      </saml:Attribute>
    </saml:AttributeStatement>
  </saml:Assertion>
</samlp:Response>
"""


def test_azure_response_parses_email_via_microsoft_claim_uri():
    pytest.importorskip("lxml")
    assertion = verify_saml_response(
        saml_response_b64=_b64(AZURE_RESPONSE),
        idp_certificate_pem="dummy",
        expected_issuer="https://sts.windows.net/contoso/",
        expected_audience="https://lumicoria.ai/sso/acs",
        skip_signature=True,
    )
    assert assertion.email == "bob@contoso.com"
    assert assertion.name_id == "bob@contoso.com"
    assert assertion.audience == "https://lumicoria.ai/sso/acs"


# ── Negative paths ────────────────────────────────────────────────


def test_audience_mismatch_raises():
    pytest.importorskip("lxml")
    with pytest.raises(SamlVerificationError):
        verify_saml_response(
            saml_response_b64=_b64(OKTA_RESPONSE),
            idp_certificate_pem="dummy",
            expected_audience="https://wrong.example/acs",
            skip_signature=True,
        )


def test_issuer_mismatch_raises():
    pytest.importorskip("lxml")
    with pytest.raises(SamlVerificationError):
        verify_saml_response(
            saml_response_b64=_b64(OKTA_RESPONSE),
            idp_certificate_pem="dummy",
            expected_issuer="https://impostor.example/",
            skip_signature=True,
        )


def test_invalid_base64_payload_raises():
    pytest.importorskip("lxml")
    with pytest.raises(SamlVerificationError):
        verify_saml_response(
            saml_response_b64="!!!not valid base64!!!",
            idp_certificate_pem="dummy",
            skip_signature=True,
        )


def test_garbage_xml_raises():
    pytest.importorskip("lxml")
    with pytest.raises(SamlVerificationError):
        verify_saml_response(
            saml_response_b64=_b64("<not><valid></valid"),
            idp_certificate_pem="dummy",
            skip_signature=True,
        )


def test_missing_nameid_raises():
    pytest.importorskip("lxml")
    no_nameid = OKTA_RESPONSE.replace(
        '<saml:NameID Format="urn:oasis:names:tc:SAML:1.1:nameid-format:emailAddress">alice@okta.example</saml:NameID>',
        ""
    )
    with pytest.raises(SamlVerificationError):
        verify_saml_response(
            saml_response_b64=_b64(no_nameid),
            idp_certificate_pem="dummy",
            skip_signature=True,
        )
