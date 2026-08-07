"""ECIES artifact sealing (ADR 0002 §3).

Seals the ANALYZE result to the client's per-device *encryption* public key so the
transcript/translation travel end-to-end encrypted and the Gateway only ever handles
ciphertext (zero-knowledge). The scheme is standard ephemeral-static ECDH on P-256 →
HKDF-SHA256 → AES-256-GCM — the exact primitives WebCrypto exposes, so the client can
decrypt with `crypto.subtle` (see tests/test_artifact_crypto.py for the reference path).

The encryption key is DISTINCT from the ECDSA signing key: sealing to a device and
authenticating a device are different jobs, and reusing one key for both is a footgun.
"""
import base64
import json
import os

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

# Mirrors ARTIFACT_SCHEMA_VERSION / ARTIFACT_ALG in packages/shared-types. These are the
# only two constants that cross the language boundary, so they are pinned here explicitly
# rather than imported; a change on either side must be made on both (checked by the
# round-trip interop test).
ARTIFACT_SCHEMA_VERSION = 2
ARTIFACT_ALG = "ECIES-P256-HKDF-SHA256-AES256GCM"
ANALYZE_RESULT_SCHEMA_VERSION = 1

# HKDF domain-separates the derived AES key to this exact algorithm identifier. Empty salt
# matches WebCrypto's default (an all-zero salt is HMAC-block-padded to the same PRK on
# both sides), so the client derives the identical key.
_HKDF_INFO = ARTIFACT_ALG.encode("ascii")
_HKDF_SALT = b""
_IV_LEN = 12  # 96-bit GCM nonce (the size WebCrypto's AES-GCM expects).


def _b64url_encode(raw: bytes) -> str:
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


def load_recipient_public_key(pub_b64url: str) -> ec.EllipticCurvePublicKey:
    """Parse a base64url-encoded raw P-256 point (X9.62 uncompressed, 0x04 || X || Y) into
    a public key. Raises on anything that is not a valid point on the curve — the caller
    maps that deterministic client fault to a terminal 422."""
    raw = base64.urlsafe_b64decode(pub_b64url + "=" * (-len(pub_b64url) % 4))
    return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), raw)


def artifact_aad(context: dict) -> bytes:
    """Canonical UTF-8 AAD shared with the WebCrypto client."""
    return json.dumps(
        context, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def seal_artifact(
    plaintext: bytes,
    recipient_pub: ec.EllipticCurvePublicKey,
    *,
    analyze_job_id: str,
    analyze_revision: int,
    payload_schema_version: int = ANALYZE_RESULT_SCHEMA_VERSION,
) -> dict:
    """Encrypt ``plaintext`` to ``recipient_pub`` and return an EncryptedArtifact envelope.

    A fresh ephemeral keypair and IV are generated per call. The device key is static, so
    compromising it can expose historical envelopes; rotation and retention remain required.
    AES-GCM AAD authenticates lineage, revision, and both schema versions before plaintext
    is released, preventing a valid same-device envelope from being moved to another context.
    """
    if not analyze_job_id or not isinstance(analyze_revision, int) or analyze_revision <= 0:
        raise ValueError("invalid artifact context")
    context = {
        "alg": ARTIFACT_ALG,
        "analyzeJobId": analyze_job_id,
        "analyzeRevision": analyze_revision,
        "artifactSchemaVersion": ARTIFACT_SCHEMA_VERSION,
        "payloadSchemaVersion": payload_schema_version,
    }
    ephemeral = ec.generate_private_key(ec.SECP256R1())
    shared = ephemeral.exchange(ec.ECDH(), recipient_pub)
    key = HKDF(
        algorithm=hashes.SHA256(), length=32, salt=_HKDF_SALT, info=_HKDF_INFO
    ).derive(shared)

    iv = os.urandom(_IV_LEN)
    ciphertext = AESGCM(key).encrypt(iv, plaintext, artifact_aad(context))

    ephemeral_raw = ephemeral.public_key().public_bytes(
        serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint
    )
    return {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "alg": ARTIFACT_ALG,
        "context": context,
        "ephemeralPublicKey": _b64url_encode(ephemeral_raw),
        "iv": _b64url_encode(iv),
        "ciphertext": _b64url_encode(ciphertext),
    }
