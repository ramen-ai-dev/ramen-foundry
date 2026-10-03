"""Independent verification of signed ALLOW receipts before tool dispatch.

``RamenClient.evaluate_compliance`` already verifies receipts, but the tool
nodes accept any client object and previously trusted its ``receipt_verified``
flag. A wrapper, a custom client, or a tampered result could report
``allowed=True`` and ``receipt_verified=True`` around a genuine, validly signed
BLOCK receipt, and the tool would run.

:func:`signed_allow_problem` re-derives the decision from the signed bytes
alone, so dispatch no longer depends on what the client claims. It is the same
gate ``@ramen-ai/mcp-shield-proxy`` 0.1.4 enforces.
"""

from __future__ import annotations

import base64
import hashlib
import json
from collections.abc import Mapping
from typing import Any

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import load_der_public_key
from ramen_ai.verifier import AUDIT_PUBLIC_KEYS

SCHEMA_VERSION = "5.0"
VERDICT_ALLOW = 1


def _b64decode(value: str) -> bytes:
    """Decode standard or URL-safe base64, with or without padding."""
    normalised = value.replace("-", "+").replace("_", "/")
    return base64.b64decode(normalised + "=" * (-len(normalised) % 4), validate=True)


def extract_receipt(verdict: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return the Schema V5 receipt from an SDK verdict.

    ``RamenClient.evaluate_compliance`` exposes the receipt inside the full
    ``data`` payload; a top-level ``receipt`` key takes precedence if present.
    """
    receipt = verdict.get("receipt")
    if not isinstance(receipt, Mapping):
        data = verdict.get("data")
        receipt = data.get("receipt") if isinstance(data, Mapping) else None
    return dict(receipt) if isinstance(receipt, Mapping) else None


def signed_allow_problem(
    verdict: Mapping[str, Any],
    input_text: str,
    public_keys: Mapping[str, str] | None = None,
) -> str | None:
    """Return why an ALLOW verdict lacks verifiable evidence, or ``None`` if it is sound.

    All of these must hold, read from the signed bytes rather than the
    response's unsigned fields:

    1. A receipt with ``kid``, ``signature``, and ``canonical_payload`` is present.
    2. The Ed25519 signature over ``canonical_payload`` verifies under the
       pinned key for ``kid`` (``ramen_pk_v1`` in production).
    3. The signed payload has ``schema_version`` "5.0", the same ``kid`` and
       ``id`` as the receipt, and ``payload_hash == SHA-256(input_text)``.
    4. The signed ``verdict`` is 1 (ALLOW).

    Never raises.
    """
    keys = AUDIT_PUBLIC_KEYS if public_keys is None else public_keys
    receipt = extract_receipt(verdict)
    if receipt is None:
        alert = verdict.get("receipt_alert")
        return f"no Schema V5 receipt was returned ({alert})" if alert else "no Schema V5 receipt was returned"
    kid, signature, canonical = receipt.get("kid"), receipt.get("signature"), receipt.get("canonical_payload")
    if not all(isinstance(value, str) and value for value in (kid, signature, canonical)):
        return "receipt is missing kid, signature, or canonical_payload"
    key_b64 = keys.get(kid)
    if not key_b64:
        return f"receipt is signed with an unknown key: {kid!r}"
    try:
        public_key = load_der_public_key(_b64decode(key_b64))
        if not isinstance(public_key, Ed25519PublicKey):
            return f"key {kid!r} is not an Ed25519 public key"
        public_key.verify(_b64decode(signature), canonical.encode("utf-8"))
    except InvalidSignature:
        return "receipt signature does not verify against canonical_payload"
    except (ValueError, TypeError) as error:
        return f"receipt signature could not be checked: {error}"

    try:
        signed = json.loads(canonical)
    except ValueError:
        return "canonical_payload is not valid JSON"
    if not isinstance(signed, dict):
        return "canonical_payload is not a JSON object"
    if signed.get("schema_version") != SCHEMA_VERSION:
        return f"signed schema_version is {signed.get('schema_version')!r}, not {SCHEMA_VERSION!r}"
    if signed.get("kid") != kid:
        return "signed kid does not match the receipt kid"
    if receipt.get("id") is not None and signed.get("id") != receipt.get("id"):
        return "signed id does not match the receipt id"
    if signed.get("payload_hash") != hashlib.sha256(input_text.encode("utf-8")).hexdigest():
        return "signed payload_hash does not match the evaluated tool call"
    if signed.get("verdict") != VERDICT_ALLOW:
        return f"signed verdict is {signed.get('verdict')!r}, not {VERDICT_ALLOW} (the evaluated call was blocked)"
    return None
