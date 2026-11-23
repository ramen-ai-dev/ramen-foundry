"""An in-process stand-in for ``POST /api/v1/paas/try``. No test here contacts the network."""
from __future__ import annotations

import base64
import hashlib
import json
import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import httpx
from ramen_ai import RamenClient

from ramen_foundry.trial import TrialComplianceClient
from tests.receipt_fixtures import _SIGNING_KEY, TEST_PUBLIC_KEYS

BLOCK_MARKERS = ("rm -rf", "active burner", "(the requesting agent)", "unauthorised structural alteration", "credentials")


def block_on_markers(text: str) -> bool:
    """Return True when ``text`` should be allowed (none of the BLOCK markers appears)."""
    return not any(marker in text for marker in BLOCK_MARKERS)


class TrialServer:
    """Answers trial requests with genuinely signed ``trial_<uuid>`` Schema V5 receipts.

    ``limit_after`` is the number of requests answered before every later one gets a 429.
    """

    def __init__(
        self,
        decide: Callable[[str], bool] = block_on_markers,
        *,
        limit_after: int | None = None,
        limit_scope: str = "network",
        remaining: int = 9,
        limit: int = 10,
    ) -> None:
        self.decide = decide
        self.limit_after = limit_after
        self.limit_scope = limit_scope
        self.remaining = remaining
        self.limit = limit
        self.requests: list[httpx.Request] = []
        self.bodies: list[bytes] = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self._handle)

    def client(self, **options: Any) -> TrialComplianceClient:
        return TrialComplianceClient(public_keys=TEST_PUBLIC_KEYS, transport=self.transport, **options)

    def install(self, monkeypatch: Any) -> None:
        """Make every ``TrialComplianceClient`` built without a client talk to this server."""
        server = self

        class _SdkStub:
            @staticmethod
            def trial(**options: Any) -> Any:
                return RamenClient.trial(public_keys=TEST_PUBLIC_KEYS, transport=server.transport, **options)

        monkeypatch.setattr("ramen_foundry.trial.RamenClient", _SdkStub)

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.bodies.append(request.content)
        if len(request.content) > 2048:
            return _error(413, "TRIAL_PAYLOAD_TOO_LARGE", "request body too large")
        if self.limit_after is not None and len(self.requests) > self.limit_after:
            return httpx.Response(
                429,
                headers={
                    "X-Ramen-Trial-Scope": self.limit_scope,
                    "X-Ramen-Trial-Limit": str(self.limit),
                    "X-Ramen-Trial-Remaining": "0",
                    "X-Ramen-Trial-Reset": "1893456000",
                    "Retry-After": "3600",
                },
                json={
                    "success": False,
                    "error": {
                        "code": "TRIAL_LIMIT",
                        "message": "Daily trial limit reached",
                        "details": {"signup_url": "https://ramenai.dev/pricing"},
                    },
                },
            )
        body = json.loads(request.content)
        text = body["input"]
        allowed = self.decide(text)
        stamp = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        receipt_id = f"trial_{uuid.uuid4()}"
        policy_ids = ["trial-policy-1"]
        canonical = json.dumps(
            {
                "id": receipt_id,
                "schema_version": "5.0",
                "kid": "ramen_pk_v1",
                "verdict": 1 if allowed else 0,
                "payload_hash": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                "policy_ids": policy_ids,
                "statutory_anchors": [],
                "timestamp": stamp,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        signature = base64.urlsafe_b64encode(_SIGNING_KEY.sign(canonical.encode("utf-8"))).decode().rstrip("=")
        data = {
            "allowed": allowed,
            "policy_ids": policy_ids,
            "executed_at": stamp,
            "statutory_anchors": [],
            "total_violations": [] if allowed else [{"recovery_instruction": "Route this call to human review."}],
            "results": [],
            "receipt": {
                "id": receipt_id,
                "kid": "ramen_pk_v1",
                "signature": signature,
                "canonical_payload": canonical,
            },
        }
        return httpx.Response(
            200,
            json={"success": True, "data": data},
            headers={
                "X-Ramen-Trial-Limit": str(self.limit),
                "X-Ramen-Trial-Remaining": str(self.remaining),
                "X-Ramen-Trial-Reset": "1893456000",
            },
        )


def _error(status: int, code: str, message: str) -> httpx.Response:
    return httpx.Response(status, json={"success": False, "error": {"code": code, "message": message}})
