"""Genuinely signed Schema V5 receipts for tests.

A generated Ed25519 key stands in for ``ramen_pk_v1``, whose private half is
not available. ``tests/conftest.py`` points the tool nodes' default key map at
:data:`TEST_PUBLIC_KEYS`, so fake clients must return receipts made with
:func:`sign_receipt` over the exact evaluated input, just as the real service
does. Tests that need the production keys pass them explicitly.
"""

from __future__ import annotations

import base64
import hashlib
import json
import uuid
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

_SIGNING_KEY = Ed25519PrivateKey.generate()
_SPKI_B64 = base64.b64encode(
    _SIGNING_KEY.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
).decode()

# The kids the existing fake clients use all map to the one test key.
TEST_PUBLIC_KEYS: dict[str, str] = {
    "ramen_pk_v1": _SPKI_B64,
    "test-kid": _SPKI_B64,
    "test-ed25519-kid": _SPKI_B64,
}


def tool_payload(tool: str, arguments: dict[str, Any]) -> str:
    """Return the exact string RamenToolNode evaluates for a tool call."""
    return json.dumps(
        {"tool": tool, "arguments": arguments}, sort_keys=True, separators=(",", ":"), default=str
    )


def sign_receipt(
    input_text: str,
    *,
    verdict: int,
    receipt_id: str | None = None,
    kid: str = "ramen_pk_v1",
    statutory_anchors: list[str] | None = None,
) -> dict[str, Any]:
    """Return a receipt whose signature and payload_hash bind ``input_text``."""
    receipt_id = receipt_id or str(uuid.uuid4())
    anchors = list(statutory_anchors or [])
    canonical = json.dumps(
        {
            "id": receipt_id,
            "schema_version": "5.0",
            "kid": kid,
            "verdict": verdict,
            "payload_hash": hashlib.sha256(input_text.encode("utf-8")).hexdigest(),
            "statutory_anchors": anchors,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    signature = base64.urlsafe_b64encode(_SIGNING_KEY.sign(canonical.encode("utf-8"))).decode().rstrip("=")
    return {
        "id": receipt_id,
        "schema_version": "5.0",
        "kid": kid,
        "signature": signature,
        "canonical_payload": canonical,
        "statutory_anchors": anchors,
        "attestation": None,
    }


def signed_data(input_text: str, *, allowed: bool, kid: str = "ramen_pk_v1") -> dict[str, Any]:
    """Return an SDK ``data`` block carrying a receipt for this verdict."""
    return {"receipt": sign_receipt(input_text, verdict=1 if allowed else 0, kid=kid)}


def sign_canonical(signed: dict[str, Any], *, receipt_kid: str | None = None, receipt_id: str | None = None) -> dict[str, Any]:
    """Sign an arbitrary payload dict, optionally with different unsigned kid/id fields."""
    canonical = json.dumps(signed, sort_keys=True, separators=(",", ":"))
    signature = base64.urlsafe_b64encode(_SIGNING_KEY.sign(canonical.encode("utf-8"))).decode().rstrip("=")
    return {
        "id": receipt_id if receipt_id is not None else signed.get("id"),
        "kid": receipt_kid if receipt_kid is not None else signed.get("kid"),
        "signature": signature,
        "canonical_payload": canonical,
    }
