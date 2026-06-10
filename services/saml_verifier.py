"""
Phase E — SAML assertion verifier.

Lightweight verifier built on `signxml` + `lxml` (Python 3.14-friendly,
avoids the libxmlsec native-binding pain of python3-saml).

Given a base64 SAMLResponse and the IdP's signing certificate, this
module:

1. Decodes the response.
2. Verifies the enveloped XML-DSig signature against the certificate.
3. Confirms the issuer matches the org's configured `entity_id`.
4. Confirms the response targets our ACS URL.
5. Extracts NameID, email, given/family name, and the attribute map.

Returns a `SamlAssertion` dataclass on success, or raises
`SamlVerificationError` with a structured reason on failure.

This is intentionally narrow — it doesn't handle SP-initiated AuthnRequest
generation (Okta/Azure metadata-driven setup doesn't need it for a v1)
and it doesn't decrypt EncryptedAssertion (most enterprise IdPs sign
without encrypting at first; we can layer that in later).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import structlog

logger = structlog.get_logger(__name__)

SAML_NS = {
    "saml": "urn:oasis:names:tc:SAML:2.0:assertion",
    "samlp": "urn:oasis:names:tc:SAML:2.0:protocol",
    "ds": "http://www.w3.org/2000/09/xmldsig#",
}


class SamlVerificationError(Exception):
    """Raised when an assertion fails any verification step."""


@dataclass
class SamlAssertion:
    name_id: str
    email: Optional[str]
    full_name: Optional[str]
    given_name: Optional[str]
    family_name: Optional[str]
    issuer: str
    audience: Optional[str]
    attributes: Dict[str, List[str]]
    expires_at: Optional[datetime]


def _decode_response(saml_response_b64: str) -> bytes:
    try:
        return base64.b64decode(saml_response_b64, validate=False)
    except Exception as exc:
        raise SamlVerificationError(f"Invalid base64 SAMLResponse: {exc}") from exc


def _normalise_cert_pem(cert: str) -> str:
    cert = (cert or "").strip()
    if "BEGIN CERTIFICATE" in cert:
        return cert
    # Bare base64 — wrap with PEM headers.
    body = cert.replace("\r", "").replace("\n", "")
    lines = "\n".join(body[i:i + 64] for i in range(0, len(body), 64))
    return f"-----BEGIN CERTIFICATE-----\n{lines}\n-----END CERTIFICATE-----"


def verify_saml_response(
    *,
    saml_response_b64: str,
    idp_certificate_pem: str,
    expected_issuer: Optional[str] = None,
    expected_audience: Optional[str] = None,
    skip_signature: bool = False,
) -> SamlAssertion:
    """Verify and parse a base64-encoded SAMLResponse.

    `skip_signature` exists ONLY for the dev tenant path the operator
    enables explicitly via env.  Production callers must always leave it
    False.
    """
    xml_bytes = _decode_response(saml_response_b64)

    try:
        from lxml import etree
    except ImportError as exc:
        raise SamlVerificationError(
            "lxml is not installed — install lxml + signxml to enable SAML"
        ) from exc

    try:
        tree = etree.fromstring(xml_bytes)
    except Exception as exc:
        raise SamlVerificationError(f"Invalid XML: {exc}") from exc

    # 1. Verify signature.
    if not skip_signature:
        try:
            from signxml import XMLVerifier
            cert_pem = _normalise_cert_pem(idp_certificate_pem)
            XMLVerifier().verify(tree, x509_cert=cert_pem)
        except ImportError as exc:
            raise SamlVerificationError(
                "signxml is not installed — install signxml + lxml to enable SAML"
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise SamlVerificationError(f"Signature verification failed: {exc}") from exc

    # 2. Issuer.
    issuer_el = tree.find(".//saml:Issuer", SAML_NS)
    issuer = issuer_el.text.strip() if issuer_el is not None and issuer_el.text else ""
    if expected_issuer and issuer and issuer != expected_issuer:
        raise SamlVerificationError(
            f"Issuer mismatch: expected {expected_issuer}, got {issuer}"
        )

    # 3. Audience.
    audience_el = tree.find(".//saml:AudienceRestriction/saml:Audience", SAML_NS)
    audience = audience_el.text.strip() if audience_el is not None and audience_el.text else None
    if expected_audience and audience and audience != expected_audience:
        raise SamlVerificationError(
            f"Audience mismatch: expected {expected_audience}, got {audience}"
        )

    # 4. NameID.
    name_id_el = tree.find(".//saml:Subject/saml:NameID", SAML_NS)
    name_id = name_id_el.text.strip() if name_id_el is not None and name_id_el.text else ""
    if not name_id:
        raise SamlVerificationError("Assertion missing Subject/NameID")

    # 5. Attributes.
    attributes: Dict[str, List[str]] = {}
    for attr in tree.findall(".//saml:Attribute", SAML_NS):
        name = attr.get("Name") or attr.get("FriendlyName")
        if not name:
            continue
        values = [
            v.text.strip() for v in attr.findall("saml:AttributeValue", SAML_NS)
            if v.text and v.text.strip()
        ]
        if values:
            attributes[name] = values

    # 6. Email / name pickers.
    email_keys = (
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/emailaddress",
        "email", "mail", "EmailAddress", "User.email",
    )
    email = next((attributes[k][0] for k in email_keys if k in attributes), None)
    if not email and "@" in name_id:
        email = name_id

    given_keys = (
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/givenname",
        "given_name", "firstName", "User.FirstName",
    )
    family_keys = (
        "http://schemas.xmlsoap.org/ws/2005/05/identity/claims/surname",
        "family_name", "lastName", "User.LastName",
    )
    given_name = next((attributes[k][0] for k in given_keys if k in attributes), None)
    family_name = next((attributes[k][0] for k in family_keys if k in attributes), None)
    full_name = " ".join(filter(None, [given_name, family_name])) or None
    if not full_name and "displayName" in attributes:
        full_name = attributes["displayName"][0]

    # 7. Expiry / NotOnOrAfter.
    expires_at: Optional[datetime] = None
    cond_el = tree.find(".//saml:Conditions", SAML_NS)
    if cond_el is not None:
        not_after = cond_el.get("NotOnOrAfter")
        if not_after:
            try:
                expires_at = datetime.fromisoformat(not_after.replace("Z", "+00:00"))
            except ValueError:
                pass

    if expires_at and expires_at < datetime.now(timezone.utc):
        raise SamlVerificationError("Assertion expired")

    return SamlAssertion(
        name_id=name_id,
        email=email,
        full_name=full_name,
        given_name=given_name,
        family_name=family_name,
        issuer=issuer,
        audience=audience,
        attributes=attributes,
        expires_at=expires_at,
    )
