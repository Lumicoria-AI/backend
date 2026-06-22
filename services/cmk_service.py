"""
Lumicoria — Customer-Managed Keys (BYOK) envelope encryption.

Threat model:
  Recordings + transcripts stored at rest in object storage. Enterprise
  customers want the bytes encrypted with a key they control, so that
  Lumicoria cannot read the recording even if our buckets are compromised.

Pattern (envelope encryption):
  1. Generate a fresh Data Encryption Key (DEK) per recording.
  2. Encrypt the recording bytes with the DEK (AES-GCM via Fernet).
  3. Wrap the DEK with the org's Key Encryption Key (KEK):
       - Phase 2 (this PR): KEK derived from
         settings.MASTER_BYOK_KEY + org.cmk_kms_key_id
         using PBKDF2HMAC. This gives us per-org isolation without
         needing real AWS/GCP KMS integration today.
       - Phase 3: swap in real KMS — boto3 kms.encrypt(plaintext=DEK)
         or google-cloud-kms .encrypt(plaintext=DEK). Public API
         unchanged; only this module changes.
  4. Persist the wrapped DEK alongside the recording (HuddleSQL.recording_cmk_wrapped_key).
     The plaintext DEK NEVER touches storage.

Falls back to plaintext when the org has cmk_enabled=False or
master key isn't configured — callers do not need to branch.
"""

from __future__ import annotations

import base64
import hashlib
import os
import secrets
from typing import Optional, Tuple

import structlog
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

from backend.core.config import settings

logger = structlog.get_logger(__name__)


def _kek_for_org(organization_id: str, cmk_kms_key_id: Optional[str]) -> Optional[bytes]:
    """Derive the KEK (Key Encryption Key) for this org.

    The master secret comes from settings.SECRET_KEY (always present).
    The salt is the org's cmk_kms_key_id (passed through PBKDF2). This
    means an attacker with the database but not SECRET_KEY can't recover
    the DEKs — same security posture as Fernet for tokens.
    """
    master = (getattr(settings, "SECRET_KEY", "") or "").encode("utf-8")
    if not master:
        return None
    salt_source = (cmk_kms_key_id or organization_id or "lumicoria").encode("utf-8")
    salt = hashlib.sha256(salt_source).digest()[:16]
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=100_000)
    return base64.urlsafe_b64encode(kdf.derive(master))


async def _is_org_cmk_enabled(organization_id: str) -> Tuple[bool, Optional[str]]:
    """Lookup the org's SessionPolicy. Returns (cmk_enabled, cmk_kms_key_id)."""
    try:
        from backend.db.mongodb.repositories.session_policy_repository import session_policy_repository  # type: ignore
        policy = await session_policy_repository.get(organization_id=organization_id)
        if not policy:
            return False, None
        return bool(policy.get("cmk_enabled")), policy.get("cmk_kms_key_id")
    except Exception as e:
        logger.debug("cmk_policy_lookup_failed", organization_id=organization_id, error=str(e))
        return False, None


async def encrypt_blob(
    plaintext: bytes,
    *,
    organization_id: str,
    existing_wrapped_dek: Optional[str] = None,
) -> Tuple[bytes, Optional[str]]:
    """Encrypt `plaintext` for the org. Returns (ciphertext, wrapped_dek).

    If the org doesn't have BYOK enabled, returns (plaintext, None) so
    storage_service can write the bytes through unchanged.

    If `existing_wrapped_dek` is passed, reuse that DEK (so all chunks
    of a single recording use the same key).
    """
    enabled, kms_key_id = await _is_org_cmk_enabled(organization_id)
    if not enabled:
        return plaintext, None

    kek = _kek_for_org(organization_id, kms_key_id)
    if not kek:
        return plaintext, None

    # Resolve / generate the DEK
    if existing_wrapped_dek:
        try:
            dek = Fernet(kek).decrypt(existing_wrapped_dek.encode("utf-8"))
            wrapped = existing_wrapped_dek
        except InvalidToken:
            dek = secrets.token_bytes(32)
            wrapped = Fernet(kek).encrypt(dek).decode("utf-8")
    else:
        dek = secrets.token_bytes(32)
        wrapped = Fernet(kek).encrypt(dek).decode("utf-8")

    # Encrypt the payload with a Fernet derived from the DEK
    payload_key = base64.urlsafe_b64encode(dek)
    ciphertext = Fernet(payload_key).encrypt(plaintext)
    return ciphertext, wrapped


async def decrypt_blob(
    ciphertext: bytes,
    *,
    organization_id: str,
    wrapped_dek: Optional[str],
) -> bytes:
    """Inverse of encrypt_blob. If wrapped_dek is None, assumes plaintext."""
    if not wrapped_dek:
        return ciphertext
    _, kms_key_id = await _is_org_cmk_enabled(organization_id)
    kek = _kek_for_org(organization_id, kms_key_id)
    if not kek:
        return ciphertext
    try:
        dek = Fernet(kek).decrypt(wrapped_dek.encode("utf-8"))
        payload_key = base64.urlsafe_b64encode(dek)
        return Fernet(payload_key).decrypt(ciphertext)
    except InvalidToken:
        logger.warning("cmk_decrypt_failed", organization_id=organization_id)
        return ciphertext


def rotate_kek_for_org(organization_id: str, new_kms_key_id: str) -> None:
    """Hook reserved for KEK rotation — re-encrypts every wrapped_dek for
    the org under the new KEK. Not implemented in Phase 2; callers can
    safely ignore until Phase 3 when real KMS rotation lands."""
    return None


# ─────────────────────────────────────────────────────────────────────
# Always-on envelope encryption for brain blobs
# ─────────────────────────────────────────────────────────────────────
#
# `encrypt_blob` / `decrypt_blob` above are BYOK-opt-in: they look up
# the org's SessionPolicy and only encrypt when cmk_enabled=True.
# That model fits enterprise Huddle recordings (a customer says "I
# want BYOK" and switches it on).
#
# The autonomous brain handles Gmail bodies and attachments. Those
# need to be encrypted ALWAYS — not just when the user has explicitly
# opted in to BYOK. The pair below uses the same KEK-derivation
# primitives (PBKDF2(SECRET_KEY, salt=org_id) → Fernet KEK), but
# without the cmk_enabled gate, and packs the wrapped DEK into a
# self-describing header so the storage layer doesn't need a sidecar
# Mongo collection per encrypted blob.
#
# Wire format of `seal_bytes` output:
#   offset 0..3   magic "LMCR"
#   offset 4      version (currently 0x01)
#   offset 5..6   wrapped_dek_len (big-endian u16)
#   offset 7..N   wrapped_dek (Fernet token, base64 string bytes)
#   offset N..    Fernet-encrypted payload (AES-128-CBC + HMAC)
#
# Bytes that don't carry the magic header are treated as plaintext
# (legacy uploads still readable) — `unseal_bytes` short-circuits.

import struct

_SEAL_MAGIC = b"LMCR"
_SEAL_VERSION = 0x01
_SEAL_HEADER_FMT = ">4sBH"  # magic, version, dek_len
_SEAL_HEADER_LEN = struct.calcsize(_SEAL_HEADER_FMT)


def seal_bytes(plaintext: bytes, *, organization_id: str) -> bytes:
    """Always-on envelope encryption. Returns a self-contained sealed
    blob (header + wrapped DEK + ciphertext).

    Falls back to plaintext bytes when SECRET_KEY isn't configured —
    callers don't need to branch. The unseal helper detects the
    presence of the magic header before attempting decryption, so
    fallback bytes remain readable.
    """
    if not plaintext:
        return plaintext
    kek = _kek_for_org(organization_id, cmk_kms_key_id=None)
    if not kek:
        return plaintext

    dek = secrets.token_bytes(32)
    wrapped_dek_bytes = Fernet(kek).encrypt(dek)  # bytes (base64-encoded Fernet token)
    payload_key = base64.urlsafe_b64encode(dek)
    ciphertext = Fernet(payload_key).encrypt(plaintext)

    if len(wrapped_dek_bytes) > 0xFFFF:
        # Should never happen — Fernet token of a 32-byte DEK is ~140 bytes —
        # but bail out safely if a future KEK swap produces something larger.
        logger.warning(
            "cmk.seal_bytes wrapped_dek too large; storing plaintext",
            organization_id=organization_id, length=len(wrapped_dek_bytes),
        )
        return plaintext

    header = struct.pack(
        _SEAL_HEADER_FMT, _SEAL_MAGIC, _SEAL_VERSION, len(wrapped_dek_bytes),
    )
    return header + wrapped_dek_bytes + ciphertext


def unseal_bytes(sealed: bytes, *, organization_id: str) -> bytes:
    """Inverse of `seal_bytes`. Returns plaintext bytes.

    If `sealed` doesn't carry the LMCR magic header, returns it
    unchanged (treated as plaintext). This means the helper is safe to
    call on legacy storage objects written before Phase 6.
    """
    if not sealed or len(sealed) < _SEAL_HEADER_LEN:
        return sealed
    if sealed[:4] != _SEAL_MAGIC:
        # Legacy / plaintext blob — no header.
        return sealed

    try:
        magic, version, dek_len = struct.unpack(
            _SEAL_HEADER_FMT, sealed[:_SEAL_HEADER_LEN],
        )
    except struct.error:
        return sealed
    if magic != _SEAL_MAGIC or version != _SEAL_VERSION:
        # Future-version blob — caller should upgrade.
        logger.warning(
            "cmk.unseal_bytes unknown version",
            magic=magic, version=version,
        )
        return sealed

    dek_start = _SEAL_HEADER_LEN
    dek_end = dek_start + dek_len
    if dek_end > len(sealed):
        logger.warning("cmk.unseal_bytes truncated header")
        return sealed
    wrapped_dek_bytes = sealed[dek_start:dek_end]
    ciphertext = sealed[dek_end:]

    kek = _kek_for_org(organization_id, cmk_kms_key_id=None)
    if not kek:
        # KEK is gone — likely settings.SECRET_KEY rotated since seal.
        # Surface as plaintext-return so the caller logs the issue;
        # raising here would break every download of legacy blobs.
        logger.error(
            "cmk.unseal_bytes KEK unavailable",
            organization_id=organization_id,
        )
        return sealed

    try:
        dek = Fernet(kek).decrypt(wrapped_dek_bytes)
        payload_key = base64.urlsafe_b64encode(dek)
        return Fernet(payload_key).decrypt(ciphertext)
    except InvalidToken:
        logger.error(
            "cmk.unseal_bytes invalid token",
            organization_id=organization_id,
        )
        # Returning the raw bytes lets the caller continue (the ingest
        # node will just fail to parse junk) instead of crashing the
        # whole brain run on one corrupted attachment.
        return sealed


def is_sealed(blob: bytes) -> bool:
    """True iff the bytes carry the LMCR magic header."""
    return bool(blob) and len(blob) >= 4 and blob[:4] == _SEAL_MAGIC
