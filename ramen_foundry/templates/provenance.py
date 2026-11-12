"""Governed compliance and provenance gates backed by Schema V5 Ed25519 receipts.

Provenance is established only from signed Schema V5 receipts that the caller
presents together with the exact input each receipt was issued for. The
``verifier_report`` the backend evaluates is derived here from those verified
receipts. A caller-written report is rejected. This module reads no stored
lessons, remote records, or lookup URIs.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from langchain_core.tools import BaseTool
from ramen_ai import RamenClient

from ramen_foundry.core import ToolInvocation
from ramen_foundry.core.evidence import EvidenceResult, claims_equal
from ramen_foundry.core.receipts import VERDICT_ALLOW, signed_verdict_problem

from ._base import GovernedToolAgent

PROVENANCE_GATE_BUNDLE_ID = "ramen__compliance_provenance_invariance"
PROVENANCE_GATE_TOOL_NAMES = frozenset({"submit_filing", "release_report"})
DEFAULT_MIN_EVIDENCE = 1
MAX_EVIDENCE_ITEMS = 50
DEFAULT_MAX_RECEIPT_AGE_SECONDS = 86_400.0
_FUTURE_SKEW_SECONDS = 300.0

# Keys only this template may write. A caller that supplies one is attempting self-attestation.
DERIVED_KEYS = frozenset({"verifier_report", "record_type_token_required", "tokens_submitted_unmodified"})

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class VerifiedReceipt:
    """Facts read from the signed bytes of a receipt that passed verification."""

    receipt_id: str
    kid: str
    issued_at: str
    payload_sha256: str


class ReceiptEvidenceVerifier:
    """Verify one item of evidence as a signed, fresh, bound Schema V5 ALLOW receipt.

    An item is ``{"payload": <exact evaluated input text>, "receipt": <receipt>}``.
    It is accepted only when:

    * the receipt's Ed25519 signature verifies under a pinned key, the signed
      ``payload_hash`` equals SHA-256 of ``payload``, and the signed verdict is ALLOW;
    * the signed ``timestamp`` is present and no older than ``max_age_seconds``
      (nor more than five minutes in the future);
    * every ``expected`` claim appears, with an identical value, in the
      ``arguments`` of the evaluated call that the payload records. That is how a
      receipt is bound to a subject, a record type and an artifact.

    ``public_keys`` maps key ids to base64 DER Ed25519 keys. ``None`` uses the
    production keys shipped with ``ramen-ai-core``; override only in tests.
    """

    def __init__(
        self,
        public_keys: Mapping[str, str] | None = None,
        *,
        max_age_seconds: float = DEFAULT_MAX_RECEIPT_AGE_SECONDS,
        clock: Callable[[], float] = time.time,
    ) -> None:
        if (
            isinstance(max_age_seconds, bool)
            or not isinstance(max_age_seconds, (int, float))
            or not math.isfinite(max_age_seconds)
            or max_age_seconds <= 0
        ):
            raise ValueError("max_age_seconds must be a positive, finite number")
        self._public_keys = dict(public_keys) if public_keys is not None else None
        self._max_age = float(max_age_seconds)
        self._clock = clock

    def verify(self, evidence: Any, *, expected: Mapping[str, Any]) -> EvidenceResult:
        return self.inspect(evidence, expected=expected)[0]

    def inspect(
        self, evidence: Any, *, expected: Mapping[str, Any]
    ) -> tuple[EvidenceResult, VerifiedReceipt | None]:
        """Verify ``evidence`` and return the facts that were read from its signed bytes."""

        if not isinstance(evidence, Mapping):
            return EvidenceResult.rejected("evidence item is not an object"), None
        payload = evidence.get("payload")
        receipt = evidence.get("receipt")
        if not isinstance(payload, str) or not payload:
            return EvidenceResult.rejected("evidence item has no payload text"), None
        if not isinstance(receipt, Mapping):
            return EvidenceResult.rejected("evidence item has no receipt"), None
        problem = signed_verdict_problem(
            {"receipt": dict(receipt)}, payload, VERDICT_ALLOW, self._public_keys
        )
        if problem is not None:
            return EvidenceResult.rejected(problem), None

        signed = json.loads(receipt["canonical_payload"])  # verified above, so well-formed
        receipt_id, kid, stamp = signed.get("id"), signed.get("kid"), signed.get("timestamp")
        if not isinstance(receipt_id, str) or not receipt_id:
            return EvidenceResult.rejected("the signed receipt has no id"), None
        issued = _parse_timestamp(stamp)
        if issued is None:
            return EvidenceResult.rejected("the signed receipt has no usable timestamp"), None
        age = self._clock() - issued
        if age > self._max_age:
            return EvidenceResult.rejected("the receipt is older than the permitted age"), None
        if age < -_FUTURE_SKEW_SECONDS:
            return EvidenceResult.rejected("the receipt timestamp is in the future"), None

        try:
            call = json.loads(payload)
        except ValueError:
            return EvidenceResult.rejected("the receipt payload is not a recorded tool call"), None
        arguments = call.get("arguments") if isinstance(call, dict) else None
        if not isinstance(arguments, dict):
            return EvidenceResult.rejected("the receipt payload is not a recorded tool call"), None
        for name, wanted in expected.items():
            if name not in arguments or not claims_equal(arguments[name], wanted):
                return EvidenceResult.rejected(f"the receipt payload is not bound to the submitted {name}"), None

        return EvidenceResult.ok(), VerifiedReceipt(
            receipt_id=receipt_id,
            kid=str(kid),
            issued_at=str(stamp),
            payload_sha256=str(signed.get("payload_hash")),
        )


def _parse_timestamp(value: Any) -> float | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.timestamp()


class ProvenanceGateAgent(GovernedToolAgent):
    """Gate filings and report releases on verifiable receipt-based provenance.

    ``submit_filing`` and ``release_report`` must carry:

    * ``record_type``: one of the platform's ``token_required_record_types``;
    * ``subject_id``: the subject the record concerns;
    * ``artifact_sha256``: lowercase hex SHA-256 of the artifact being released;
    * ``evidence``: a list of ``{"payload", "receipt"}`` items, each a Schema V5
      receipt issued with an ALLOW verdict for the exact ``payload`` text.

    Before the ramen-ai backend is contacted, the call is blocked locally (fail
    closed) unless there are between ``min_evidence`` and 50 items, every item
    verifies, no receipt id repeats, every receipt is fresh, and every receipt
    payload records the same ``subject_id``, ``record_type`` and
    ``artifact_sha256`` as the submission.

    The ``verifier_report`` that the backend evaluates is written by this
    template from the receipts it verified. A call that carries its own
    ``verifier_report``, ``record_type_token_required`` or
    ``tokens_submitted_unmodified`` is blocked as self-attestation. Schema V5
    receipts are immutable signed statements with no revocation mechanism, so the
    report states them as not revoked and not superseded, and it states them as
    current only while they are within ``max_receipt_age_seconds`` (default 24 h).
    The report is added to the arguments the tool receives, so tools may declare
    a ``verifier_report`` parameter or ignore it.

    Only Schema V5 Ed25519 receipts count as evidence. There is deliberately no
    way to substitute another verifier, and no stored or remote lesson or lookup
    is consulted.
    """

    bundle_id = PROVENANCE_GATE_BUNDLE_ID
    allowed_tool_names = PROVENANCE_GATE_TOOL_NAMES
    node_name = "governed_provenance_action"

    def __init__(
        self,
        *,
        client: RamenClient,
        tools: Mapping[str, BaseTool],
        token_required_record_types: Collection[str],
        min_evidence: int = DEFAULT_MIN_EVIDENCE,
        max_receipt_age_seconds: float = DEFAULT_MAX_RECEIPT_AGE_SECONDS,
        public_keys: Mapping[str, str] | None = None,
        provider_key: str | None = None,
        provider_name: str | None = None,
    ) -> None:
        if (
            isinstance(min_evidence, bool)
            or not isinstance(min_evidence, int)
            or not 1 <= min_evidence <= MAX_EVIDENCE_ITEMS
        ):
            raise ValueError(f"min_evidence must be an integer from 1 to {MAX_EVIDENCE_ITEMS}")
        if isinstance(token_required_record_types, str) or not token_required_record_types:
            raise ValueError("token_required_record_types must be a non-empty collection of record types")
        record_types = frozenset(token_required_record_types)
        if any(not isinstance(item, str) or not item.strip() for item in record_types):
            raise ValueError("token_required_record_types must contain only non-blank strings")
        self._record_types = record_types
        self._min_evidence = min_evidence
        self._verifier = ReceiptEvidenceVerifier(public_keys, max_age_seconds=max_receipt_age_seconds)
        self._max_age = float(max_receipt_age_seconds)
        super().__init__(
            client=client,
            tools=tools,
            provider_key=provider_key,
            provider_name=provider_name,
        )

    @property
    def token_required_record_types(self) -> frozenset[str]:
        """Return the record types this gate accepts, as declared by the platform."""

        return self._record_types

    def _verify_submission(self, arguments: Mapping[str, Any]) -> tuple[str | None, list[VerifiedReceipt]]:
        attempted = sorted(DERIVED_KEYS & set(arguments))
        if attempted:
            return (
                f"Self-attestation is not accepted: {', '.join(attempted)} is derived from the "
                "verified receipts and must not be supplied by the caller.",
                [],
            )
        record_type, subject_id = arguments.get("record_type"), arguments.get("subject_id")
        if not isinstance(record_type, str) or record_type not in self._record_types:
            return "record_type must be one of the platform's token-required record types.", []
        if not isinstance(subject_id, str) or not subject_id.strip():
            return "subject_id must be a non-blank string.", []
        digest = arguments.get("artifact_sha256")
        if not isinstance(digest, str) or not _SHA256_HEX.fullmatch(digest):
            return "artifact_sha256 must be a lowercase hex SHA-256 digest of the artifact.", []
        evidence = arguments.get("evidence")
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            return "evidence must be a list of signed receipt items.", []
        if len(evidence) < self._min_evidence:
            return (
                f"Provenance requires at least {self._min_evidence} verified receipt(s); "
                f"{len(evidence)} supplied.",
                [],
            )
        if len(evidence) > MAX_EVIDENCE_ITEMS:
            return f"Provenance accepts at most {MAX_EVIDENCE_ITEMS} receipts per call.", []

        expected = {"artifact_sha256": digest, "subject_id": subject_id, "record_type": record_type}
        verified: list[VerifiedReceipt] = []
        seen: set[str] = set()
        for index, item in enumerate(evidence):
            result, facts = self._verifier.inspect(item, expected=expected)
            if not result.valid or facts is None:
                return f"Evidence item {index} rejected: {result.reason}", []
            if facts.receipt_id in seen:
                return (
                    f"Evidence item {index} repeats receipt {facts.receipt_id}; each receipt may count once.",
                    [],
                )
            seen.add(facts.receipt_id)
            verified.append(facts)
        return None, verified

    def _preflight(self, invocation: ToolInvocation) -> str | None:
        return self._verify_submission(invocation.arguments)[0]

    def _bind_invocation(self, invocation: ToolInvocation) -> ToolInvocation:
        problem, verified = self._verify_submission(invocation.arguments)
        if problem is not None:
            raise ValueError(problem)
        arguments = dict(invocation.arguments)
        kids = sorted({item.kid for item in verified})
        arguments["verifier_report"] = {
            "issuer": f"ramen-ai Schema V5 receipt verifier (kid {', '.join(kids)})",
            "record_type": arguments["record_type"],
            "bound_subject": arguments["subject_id"],
            "bound_artifact_sha256": arguments["artifact_sha256"],
            "status": "valid",
            "revoked": False,
            "superseded": False,
            "review_cycle": "current",
            "verified_out_of_band": True,
            "verification": (
                "Ed25519 signature verified against the pinned key; the signed payload hash "
                "binds the exact subject, record type and artifact digest; the signed verdict is "
                f"ALLOW; the signed timestamp is within {self._max_age:g} seconds."
            ),
            "receipts": [
                {"id": item.receipt_id, "kid": item.kid, "issued_at": item.issued_at}
                for item in verified
            ],
        }
        arguments["record_type_token_required"] = True
        arguments["tokens_submitted_unmodified"] = True
        return invocation.model_copy(update={"arguments": arguments})
