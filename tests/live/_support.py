"""Shared support for the serial live tests against https://api.ramenai.dev.

Opt in with ``RAMEN_LIVE_TESTS=1`` and select the marker::

    RAMEN_LIVE_TESTS=1 python -m pytest -m live tests/live -s

The suite sends about 35 evaluations, one at a time, with 2.5 s spacing and a 30 s
back-off on HTTP 429. ``RAMEN_API_KEY`` is read from the environment or the
repository's ``.env``; it is never printed or logged. Each verdict case runs three
times and passes on a 2-of-3 majority, because the evaluator is semantic and not
bit-for-bit deterministic. Every live receipt must verify as Schema V5 Ed25519
under the production pinned keys, whatever the verdict. Every module carries
``pytestmark = pytest.mark.live``, so ``-m 'not live'`` (the default) deselects them.
"""

from __future__ import annotations

import os
import statistics
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ramen_ai import RamenClient

from ramen_foundry.core.receipts import signed_verdict_problem

API_BASE_URL = "https://api.ramenai.dev"
MIN_INTERVAL_SECONDS = 2.5
RATE_LIMIT_BACKOFF_SECONDS = 30.0
TRANSIENT_BACKOFF_SECONDS = 5.0
_TRANSIENT_MARKERS = ("500", "502", "503", "504", "timed out", "timeout", "connection")
MAX_ATTEMPTS = 3
REPEATS = 3
REQUIRED_MATCHES = 2
ENV_PATH = Path(__file__).resolve().parents[2] / ".env"

MATRIX: list[dict[str, Any]] = []


def api_key() -> str | None:
    key = os.environ.get("RAMEN_API_KEY")
    if key:
        return key
    if ENV_PATH.is_file():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            candidate = line.strip().removeprefix("export ").lstrip()
            if candidate.startswith("RAMEN_API_KEY="):
                value = candidate.split("=", 1)[1].strip()
                if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
                    value = value[1:-1]
                return value or None
    return None


@dataclass
class LiveRecord:
    input_text: str
    result: dict[str, Any]
    latency_ms: float


class SpacedClient(RamenClient):
    """A RamenClient that serialises calls, spaces them, and records each verdict."""

    def __init__(self, api_key: str) -> None:
        super().__init__(api_key, base_url=API_BASE_URL, timeout=90.0)
        self.records: list[LiveRecord] = []
        self._last_call = 0.0

    def evaluate_compliance(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        for attempt in range(MAX_ATTEMPTS):
            pause = MIN_INTERVAL_SECONDS - (time.monotonic() - self._last_call)
            if pause > 0:
                time.sleep(pause)
            started = time.monotonic()
            try:
                result = super().evaluate_compliance(input_text, **kwargs)
            except Exception as error:  # noqa: BLE001
                self._last_call = time.monotonic()
                message = str(error).lower()
                if attempt < MAX_ATTEMPTS - 1:
                    if "429" in message:
                        time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                        continue
                    if any(marker in message for marker in _TRANSIENT_MARKERS):
                        time.sleep(TRANSIENT_BACKOFF_SECONDS)
                        continue
                raise
            self._last_call = time.monotonic()
            self.records.append(LiveRecord(input_text, result, (self._last_call - started) * 1000))
            return result
        raise AssertionError("unreachable")


def assert_receipt_verified(record: LiveRecord) -> str:
    """Assert a live response carries a Schema V5 Ed25519 receipt that verifies independently."""

    result = record.result
    allowed = bool(result.get("allowed"))
    assert result.get("receipt_verified") is True, "the client did not verify the receipt"
    problem = signed_verdict_problem(result, record.input_text, 1 if allowed else 0, None)
    assert problem is None, f"independent receipt verification failed: {problem}"
    receipt = (result.get("data") or {}).get("receipt") or result.get("receipt") or {}
    assert receipt.get("kid") == "ramen_pk_v1"
    return str(receipt["kid"])


def observe(
    *,
    template: str,
    scenario: str,
    expected: str,
    client: SpacedClient,
    executions: list[dict[str, Any]],
    call: Callable[[], Any],
    repeats: int = REPEATS,
    required: int = REQUIRED_MATCHES,
    local_fragment: str | None = None,
) -> list[str]:
    """Run ``call`` and assert the live (or local) outcome.

    ``expected`` is ``ALLOW``, ``BLOCK`` or ``LOCAL``. LOCAL cases must make no API
    call at all and run once.
    """

    outcomes: list[str] = []
    latencies: list[float] = []
    receipt_kid = "n/a"
    runs = 1 if expected == "LOCAL" else repeats
    for _ in range(runs):
        records_before = len(client.records)
        executions.clear()
        command = call()
        new = client.records[records_before:]
        error = command.update.get("governance_error")
        if expected == "LOCAL":
            assert new == [], "a local fail-closed case must not contact the backend"
            assert executions == [], "the tool must not run"
            assert error, "a local block must report why"
            if local_fragment:
                assert local_fragment in str(error), f"unexpected reason: {error}"
            outcomes.append("LOCAL")
            continue
        assert len(new) == 1, f"expected exactly one evaluation, saw {len(new)}; governance_error: {str(error)[:300]}"
        receipt_kid = assert_receipt_verified(new[0])
        latencies.append(new[0].latency_ms)
        if new[0].result.get("allowed"):
            assert error is None, "an ALLOW must not carry a governance error"
            assert len(executions) == 1, "an ALLOW must run the tool exactly once"
            outcomes.append("ALLOW")
        else:
            assert error, "a BLOCK must report why"
            assert executions == [], "a BLOCK must not run the tool"
            outcomes.append("BLOCK")

    matches = sum(1 for outcome in outcomes if outcome == expected)
    MATRIX.append(
        {
            "template": template,
            "scenario": scenario,
            "expected": expected,
            "votes": ", ".join(f"{k}={outcomes.count(k)}" for k in sorted(set(outcomes))),
            "receipt": f"V5 Ed25519 ok ({receipt_kid})" if expected != "LOCAL" else "n/a (no API call)",
            "median_ms": f"{statistics.median(latencies):.0f}" if latencies else "0",
            "passed": matches >= (1 if expected == "LOCAL" else required),
        }
    )
    needed = 1 if expected == "LOCAL" else required
    assert matches >= needed, f"{scenario}: expected {expected} in {needed} of {runs} runs, got {outcomes}"
    return outcomes
