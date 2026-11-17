"""Shared helpers for the Core Architecture gate demos.

Each demo runs up to three stages against one governed template:

1. A local fail-closed pre-check. It needs no API key and makes no network call.
2. A live BLOCK from the ramen-ai bundle, with a verified Schema V5 Ed25519 receipt.
3. A live ALLOW from the bundle, with a verified receipt and exactly one tool execution.

Without ``RAMEN_API_KEY`` a demo runs stage 1 only and exits 0. With a key it runs all
three stages and exits 1 if any stage does not end as expected.

``RAMEN_API_KEY`` is read from the environment, then from the checkout's ``.env`` or the
file named by ``RAMEN_ENV_FILE``. It is never printed. When ``OPENAI_API_KEY`` is set the
evaluations use OpenAI BYOK; otherwise ramen-ai uses managed-provider inference.
"""

from __future__ import annotations

import copy
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from ramen_ai import RamenClient

import ramen_foundry
from ramen_foundry.core.receipts import signed_verdict_problem

try:  # Supports both direct execution and imports from repository tests.
    from examples.benchmark_asimov_robotics import API_BASE_URL, load_environment_credentials
except ModuleNotFoundError:  # pragma: no cover - direct-script import layout
    from benchmark_asimov_robotics import (  # type: ignore[no-redef]
        API_BASE_URL,
        load_environment_credentials,
    )

LOCAL = "LOCAL"
ALLOW = "ALLOW"
BLOCK = "BLOCK"

MIN_INTERVAL_SECONDS = 2.0
RATE_LIMIT_BACKOFF_SECONDS = 30.0
MAX_ATTEMPTS = 3


class ExpectationError(AssertionError):
    """A stage did not end the way the demo expects."""


@dataclass(frozen=True)
class Evaluation:
    """One live evaluation: the exact text sent, the verdict returned, and the latency."""

    input_text: str
    result: dict[str, Any]
    latency_ms: float


@dataclass(frozen=True)
class Outcome:
    """The result of one demo scenario, as shown in the summary table."""

    stage: str
    scenario: str
    expected: str
    decision: str
    receipt: str
    latency_ms: float
    note: str = ""


class GateRecordingClient(RamenClient):
    """A RamenClient that spaces its calls, retries rate limits, and records each verdict."""

    def __init__(
        self,
        api_key: str,
        *,
        base_url: str = API_BASE_URL,
        min_interval: float = MIN_INTERVAL_SECONDS,
    ) -> None:
        super().__init__(api_key, base_url=base_url, timeout=90.0)
        self.evaluations: list[Evaluation] = []
        self._min_interval = min_interval
        self._last_call = 0.0

    def _evaluate(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        return RamenClient.evaluate_compliance(self, input_text, **kwargs)

    def evaluate_compliance(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        for attempt in range(MAX_ATTEMPTS):
            pause = self._min_interval - (time.monotonic() - self._last_call)
            if pause > 0:
                time.sleep(pause)
            started = time.monotonic()
            try:
                result = self._evaluate(input_text, **kwargs)
            except Exception as error:  # noqa: BLE001
                self._last_call = time.monotonic()
                if "429" in str(error) and attempt < MAX_ATTEMPTS - 1:
                    time.sleep(RATE_LIMIT_BACKOFF_SECONDS)
                    continue
                raise
            self._last_call = time.monotonic()
            self.evaluations.append(Evaluation(input_text, result, (self._last_call - started) * 1000))
            return result
        raise AssertionError("unreachable")


class OfflineGateClient(GateRecordingClient):
    """Stands in when no API key is set. Any attempt to evaluate is a bug in the demo."""

    def __init__(self) -> None:
        super().__init__("offline-placeholder", min_interval=0.0)

    def _evaluate(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("an evaluation was attempted without RAMEN_API_KEY")


def provider_options(environment: Mapping[str, str]) -> dict[str, str]:
    """Return BYOK options when ``OPENAI_API_KEY`` is set, else an empty dict (managed mode)."""

    key = environment.get("OPENAI_API_KEY")
    return {"provider_key": key, "provider_name": "openai"} if key else {}


def _scrub(text: str, env: Mapping[str, str]) -> str:
    """Remove credential values from text that is about to be printed."""

    for name in ("RAMEN_API_KEY", "OPENAI_API_KEY"):
        value = env.get(name)
        if value:
            text = text.replace(value, "***")
    return text


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ExpectationError(message)


def _verdict_label(result: Mapping[str, Any]) -> str:
    return ALLOW if result.get("allowed") is True else BLOCK


def run_scenario(
    agent: Any,
    client: GateRecordingClient,
    executions: Sequence[Any],
    *,
    stage: str,
    scenario: str,
    expected: str,
    tool_name: str,
    arguments: Mapping[str, Any],
    local_fragment: str | None = None,
) -> Outcome:
    """Run one call through ``agent`` and require the expected outcome.

    ``expected`` is ``LOCAL`` (blocked before any API call), ``BLOCK`` or ``ALLOW``.
    Live outcomes also require a Schema V5 Ed25519 receipt that verifies independently
    under the production pinned keys and matches the evaluated text.
    """

    evaluations_before, executions_before = len(client.evaluations), len(executions)
    command = agent.execute(tool_name, copy.deepcopy(dict(arguments)))
    new = client.evaluations[evaluations_before:]
    ran = len(executions) - executions_before
    error = command.update.get("governance_error")

    if expected == LOCAL:
        _require(not new, f"{scenario}: a local pre-check must not contact the backend")
        _require(ran == 0, f"{scenario}: the tool must not run")
        _require(bool(error), f"{scenario}: the local pre-check did not block the call")
        if local_fragment:
            _require(local_fragment in str(error), f"{scenario}: unexpected reason: {error}")
        return Outcome(stage, scenario, LOCAL, "BLOCKED locally", "n/a (no API call)", 0.0, str(error))

    _require(len(new) == 1, f"{scenario}: expected one evaluation, saw {len(new)} ({error})")
    evaluation = new[0]
    verdict = _verdict_label(evaluation.result)
    _require(
        verdict == expected,
        f"{scenario}: expected {expected} but the bundle returned {verdict}"
        + (f" ({evaluation.result.get('steering')})" if evaluation.result.get("steering") else ""),
    )
    _require(evaluation.result.get("receipt_verified") is True, f"{scenario}: the client did not verify the receipt")
    problem = signed_verdict_problem(
        evaluation.result, evaluation.input_text, 1 if verdict == ALLOW else 0, None
    )
    _require(problem is None, f"{scenario}: receipt verification failed: {problem}")
    receipt = (evaluation.result.get("data") or {}).get("receipt") or {}
    if verdict == ALLOW:
        _require(error is None and ran == 1, f"{scenario}: an ALLOW must run the tool exactly once")
        note = "tool ran once"
    else:
        _require(bool(error) and ran == 0, f"{scenario}: a BLOCK must not run the tool")
        note = " ".join(str(evaluation.result.get("steering") or error).split())
    return Outcome(
        stage,
        scenario,
        expected,
        f"{verdict} (live)",
        f"Schema V5 Ed25519 verified, kid={receipt.get('kid')}",
        evaluation.latency_ms,
        note,
    )


def format_table(outcomes: Sequence[Outcome]) -> str:
    """Render the stages as a compact text table."""

    headers = ("Stage", "Scenario", "Decision", "Receipt", "Latency")
    rows = [
        (o.stage, o.scenario, o.decision, o.receipt, f"{o.latency_ms:.0f} ms" if o.latency_ms else "-")
        for o in outcomes
    ]
    widths = [max(len(row[i]) for row in (headers, *rows)) for i in range(len(headers))]
    rule = "+".join("-" * (width + 2) for width in widths)

    def render(row: Sequence[str]) -> str:
        return "|".join(f" {value.ljust(width)} " for value, width in zip(row, widths))

    return "\n".join((rule, render(headers), rule, *(render(row) for row in rows), rule))


RunFunction = Callable[..., list[Outcome]]
ClientFactory = Callable[[str], GateRecordingClient]


def gate_main(
    title: str,
    run: RunFunction,
    *,
    environment: Mapping[str, str] | None = None,
    client_factory: ClientFactory | None = None,
) -> int:
    """Run a demo: all three stages with ``RAMEN_API_KEY``, stage 1 alone without it."""

    env = load_environment_credentials() if environment is None else dict(environment)
    api_key = env.get("RAMEN_API_KEY") or None
    options = provider_options(env)
    print(f"{title} (ramen-foundry {ramen_foundry.__version__})")

    if api_key is None:
        print("RAMEN_API_KEY is not set: running Stage 1 only (local fail-closed pre-checks, no network).")
        with OfflineGateClient() as offline:
            offline_outcomes = run(offline, live=False, provider=options)
        print(format_table(offline_outcomes))
        print("Set RAMEN_API_KEY to run Stages 2 and 3 against the live bundle.")
        print("  export RAMEN_API_KEY=...   # https://ramenai.dev/pricing")
        return 0

    print(f"Provider mode: {'OpenAI BYOK' if options else 'managed'} | endpoint: {API_BASE_URL}")
    factory = client_factory or (lambda key: GateRecordingClient(key))
    outcomes: list[Outcome] = []
    try:
        with factory(api_key) as client:
            outcomes = run(client, live=True, provider=options)
    except ExpectationError as error:
        print(f"UNEXPECTED RESULT: {_scrub(str(error), env)[:500]}")
        return 1
    except Exception as error:  # noqa: BLE001
        print(f"ERROR: {type(error).__name__}: {_scrub(str(error), env)[:300]}")
        return 1

    print(format_table(outcomes))
    for outcome in outcomes:
        if outcome.stage.startswith("2"):
            print(f"Bundle steering: {outcome.note}")
        elif outcome.stage.startswith("3") and outcome.note:
            print(f"Allowed execution: {outcome.note}")
    return 0
