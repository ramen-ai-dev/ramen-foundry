"""Local verification of evidence presented alongside a tool call.

Templates call an :class:`EvidenceVerifier` before the ramen-ai backend is
contacted, so evidence that fails a deterministic check blocks the call locally
(fail closed). Verification is an additional gate: it never replaces the signed
policy verdict.
"""

from __future__ import annotations

import base64
import binascii
import json
import math
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

TOKEN_VERSION = "v1"
_SIGNING_DOMAIN = b"ramen-evidence-v1\n"


@dataclass(frozen=True)
class EvidenceResult:
    """Outcome of one evidence check. ``reason`` is set whenever ``valid`` is false."""

    valid: bool
    reason: str | None = None

    @classmethod
    def ok(cls) -> EvidenceResult:
        return cls(valid=True)

    @classmethod
    def rejected(cls, reason: str) -> EvidenceResult:
        return cls(valid=False, reason=reason)


@runtime_checkable
class EvidenceVerifier(Protocol):
    """Checks one piece of evidence against the claims it must assert.

    ``expected`` holds the claims the evidence has to bind (for example a request
    id and an amount). Implementations must return a rejected result for any
    evidence they cannot positively verify, and must not raise.
    """

    def verify(self, evidence: Any, *, expected: Mapping[str, Any]) -> EvidenceResult: ...


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def claims_equal(actual: Any, expected: Any) -> bool:
    """Return whether two claim values match without bool/number or str/number confusion."""

    if isinstance(actual, bool) or isinstance(expected, bool):
        return isinstance(actual, bool) and isinstance(expected, bool) and actual is expected
    if _is_number(actual) and _is_number(expected):
        return math.isfinite(actual) and math.isfinite(expected) and actual == expected
    if isinstance(actual, str) and isinstance(expected, str):
        return actual == expected
    return False


class Ed25519TokenVerifier:
    """Verify compact Ed25519-signed evidence tokens against pinned public keys.

    Token format: ``v1.<kid>.<claims>.<signature>`` where ``claims`` is the
    base64url (padding optional) encoding of a UTF-8 JSON object and
    ``signature`` is the base64url Ed25519 signature over
    ``b"ramen-evidence-v1\\n" + <claims bytes>``. The domain prefix keeps these
    signatures from being valid in any other signing context.

    A token is accepted only if its ``kid`` is pinned, the signature verifies,
    every ``expected`` claim is present with an identical value, and (when
    ``require_expiry`` is true, the default) it carries a numeric ``exp`` (Unix
    seconds) that has not passed.

    ``public_keys`` maps ``kid`` to a raw 32-byte Ed25519 public key in hex,
    optionally prefixed with ``ed25519:``. Key custody and token issuance stay
    with the host.
    """

    def __init__(
        self,
        public_keys: Mapping[str, str],
        *,
        require_expiry: bool = True,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if not public_keys:
            raise ValueError("at least one pinned public key is required")
        self._keys: dict[str, Ed25519PublicKey] = {}
        for kid, encoded in public_keys.items():
            if not isinstance(kid, str) or not kid or "." in kid:
                raise ValueError("key ids must be non-empty strings without '.'")
            try:
                raw = bytes.fromhex(encoded.removeprefix("ed25519:"))
                self._keys[kid] = Ed25519PublicKey.from_public_bytes(raw)
            except (ValueError, TypeError, AttributeError) as error:
                raise ValueError(f"public key {kid!r} is not a valid Ed25519 key") from error
        self._require_expiry = require_expiry
        self._clock = clock

    def verify(self, evidence: Any, *, expected: Mapping[str, Any]) -> EvidenceResult:
        try:
            return self._verify(evidence, expected)
        except Exception as error:  # noqa: BLE001
            return EvidenceResult.rejected(f"evidence token could not be checked: {error}")

    def _verify(self, evidence: Any, expected: Mapping[str, Any]) -> EvidenceResult:
        if not isinstance(evidence, str) or not evidence:
            return EvidenceResult.rejected("evidence token is missing")
        parts = evidence.split(".")
        if len(parts) != 4 or parts[0] != TOKEN_VERSION:
            return EvidenceResult.rejected("evidence token is malformed")
        _, kid, claims_b64, signature_b64 = parts
        key = self._keys.get(kid)
        if key is None:
            return EvidenceResult.rejected(f"evidence token is signed with an unknown key: {kid!r}")
        try:
            claims_bytes = _b64url_decode(claims_b64)
            signature = _b64url_decode(signature_b64)
        except (binascii.Error, ValueError):
            return EvidenceResult.rejected("evidence token is not valid base64url")
        try:
            key.verify(signature, _SIGNING_DOMAIN + claims_bytes)
        except InvalidSignature:
            return EvidenceResult.rejected("evidence token signature does not verify")
        claims = json.loads(claims_bytes)
        if not isinstance(claims, dict):
            return EvidenceResult.rejected("evidence token claims are not a JSON object")

        expires = claims.get("exp")
        if expires is None:
            if self._require_expiry:
                return EvidenceResult.rejected("evidence token has no expiry")
        elif not _is_number(expires) or not math.isfinite(expires):
            return EvidenceResult.rejected("evidence token expiry is not a number")
        elif expires <= self._clock():
            return EvidenceResult.rejected("evidence token has expired")

        for name, wanted in expected.items():
            if name not in claims:
                return EvidenceResult.rejected(f"evidence token does not assert {name!r}")
            if not claims_equal(claims[name], wanted):
                return EvidenceResult.rejected(f"evidence token claim {name!r} does not match the request")
        return EvidenceResult.ok()
