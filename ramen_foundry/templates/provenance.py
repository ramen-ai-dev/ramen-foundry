"""Governed compliance and provenance gates backed by Schema V5 Ed25519 receipts.

Provenance is established only from signed Schema V5 receipts that the caller
presents together with the exact input each receipt was issued for. This module
reads no stored lessons, remote records, or lookup URIs.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from typing import Any

from langchain_core.tools import BaseTool
from ramen_ai import RamenClient

from ramen_foundry.core import ToolInvocation
from ramen_foundry.core.evidence import EvidenceResult
from ramen_foundry.core.receipts import VERDICT_ALLOW, signed_verdict_problem

from ._base import GovernedToolAgent

PROVENANCE_GATE_BUNDLE_ID = "ramen__compliance_provenance_invariance"
PROVENANCE_GATE_TOOL_NAMES = frozenset({"submit_filing", "release_report"})
DEFAULT_MIN_EVIDENCE = 1
MAX_EVIDENCE_ITEMS = 50

_SHA256_HEX = re.compile(r"[0-9a-f]{64}")


class ReceiptEvidenceVerifier:
    """Verify one item of evidence as a signed Schema V5 ALLOW receipt.

    An item is ``{"payload": <exact evaluated input text>, "receipt": <receipt>}``.
    It is accepted only when the receipt's Ed25519 signature verifies under a
    pinned key, the signed ``payload_hash`` equals SHA-256 of ``payload``, and the
    signed verdict is ALLOW. If ``expected`` contains ``artifact_sha256``, that
    digest must also appear in ``payload``, binding the receipt to the artifact.

    ``public_keys`` maps key ids to base64 DER Ed25519 keys. ``None`` uses the
    production keys shipped with ``ramen-ai-core``; override only in tests.
    """

    def __init__(self, public_keys: Mapping[str, str] | None = None) -> None:
        self._public_keys = dict(public_keys) if public_keys is not None else None

    def verify(self, evidence: Any, *, expected: Mapping[str, Any]) -> EvidenceResult:
        if not isinstance(evidence, Mapping):
            return EvidenceResult.rejected("evidence item is not an object")
        payload = evidence.get("payload")
        receipt = evidence.get("receipt")
        if not isinstance(payload, str) or not payload:
            return EvidenceResult.rejected("evidence item has no payload text")
        if not isinstance(receipt, Mapping):
            return EvidenceResult.rejected("evidence item has no receipt")
        problem = signed_verdict_problem(
            {"receipt": dict(receipt)}, payload, VERDICT_ALLOW, self._public_keys
        )
        if problem is not None:
            return EvidenceResult.rejected(problem)
        digest = expected.get("artifact_sha256")
        if digest is not None and (not isinstance(digest, str) or digest not in payload):
            return EvidenceResult.rejected("receipt payload does not reference the artifact digest")
        return EvidenceResult.ok()


class ProvenanceGateAgent(GovernedToolAgent):
    """Gate filings and report releases on verifiable receipt-based provenance.

    ``submit_filing`` and ``release_report`` must carry:

    * ``artifact_sha256``: lowercase hex SHA-256 of the artifact being released;
    * ``evidence``: a list of ``{"payload", "receipt"}`` items, each a Schema V5
      receipt issued with an ALLOW verdict for the exact ``payload`` text.

    Before the ramen-ai backend is contacted, the call is blocked locally (fail
    closed) unless there are at least ``min_evidence`` items (at most 50), every
    item verifies, no receipt id repeats, and, with ``bind_artifact_digest``
    (the default), every payload references ``artifact_sha256``. The evidence is
    part of the evaluated payload, so the new receipt binds it.

    Only Schema V5 Ed25519 receipts count as evidence. There is deliberately no
    way to substitute another verifier, and no stored or remote lesson or
    lookup is consulted.
    """

    bundle_id = PROVENANCE_GATE_BUNDLE_ID
    allowed_tool_names = PROVENANCE_GATE_TOOL_NAMES
    node_name = "governed_provenance_action"

    def __init__(
        self,
        *,
        client: RamenClient,
        tools: Mapping[str, BaseTool],
        min_evidence: int = DEFAULT_MIN_EVIDENCE,
        bind_artifact_digest: bool = True,
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
        self._min_evidence = min_evidence
        self._bind_artifact_digest = bool(bind_artifact_digest)
        self._verifier = ReceiptEvidenceVerifier(public_keys)
        super().__init__(
            client=client,
            tools=tools,
            provider_key=provider_key,
            provider_name=provider_name,
        )

    def _preflight(self, invocation: ToolInvocation) -> str | None:
        arguments = invocation.arguments
        digest = arguments.get("artifact_sha256")
        if not isinstance(digest, str) or not _SHA256_HEX.fullmatch(digest):
            return "artifact_sha256 must be a lowercase hex SHA-256 digest of the artifact."
        evidence = arguments.get("evidence")
        if not isinstance(evidence, Sequence) or isinstance(evidence, (str, bytes)):
            return "evidence must be a list of signed receipt items."
        if len(evidence) < self._min_evidence:
            return (
                f"Provenance requires at least {self._min_evidence} verified receipt(s); "
                f"{len(evidence)} supplied."
            )
        if len(evidence) > MAX_EVIDENCE_ITEMS:
            return f"Provenance accepts at most {MAX_EVIDENCE_ITEMS} receipts per call."

        expected: dict[str, Any] = {"artifact_sha256": digest} if self._bind_artifact_digest else {}
        seen: set[str] = set()
        for index, item in enumerate(evidence):
            result = self._verifier.verify(item, expected=expected)
            if not result.valid:
                return f"Evidence item {index} rejected: {result.reason}"
            receipt_id = _signed_receipt_id(item["receipt"])
            if receipt_id is None:
                return f"Evidence item {index} rejected: the signed receipt has no id."
            if receipt_id in seen:
                return f"Evidence item {index} repeats receipt {receipt_id}; each receipt may count once."
            seen.add(receipt_id)
        return None


def _signed_receipt_id(receipt: Mapping[str, Any]) -> str | None:
    """Return the receipt id from the signed bytes (already verified by the caller)."""

    try:
        signed = json.loads(receipt["canonical_payload"])
    except (KeyError, TypeError, ValueError):
        return None
    receipt_id = signed.get("id") if isinstance(signed, dict) else None
    return receipt_id if isinstance(receipt_id, str) and receipt_id else None
