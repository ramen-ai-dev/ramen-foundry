"""Offline tests for ``--trial`` mode in the Core Architecture demos and the example scripts.

Every request goes to an in-process server (``tests/trial_fixtures.py``). Nothing here
contacts ``api.ramenai.dev``, and the autouse socket guard in ``conftest.py`` enforces it.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from langchain_core.tools import tool
from ramen_ai import RamenClient
from ramen_ai.constants import TRIAL_MAX_BODY_BYTES

from examples import _gate_showcase as showcase
from examples import (
    authorization_gate_demo,
    benchmark_asimov_robotics,
    eligibility_gate_demo,
    filter_dataset,
    kinematic_gate_demo,
    provenance_gate_demo,
)
from ramen_foundry import IndustrialAutomationAgent
from tests.receipt_fixtures import TEST_PUBLIC_KEYS
from tests.test_core_gate_examples import DEMOS, ScriptedClient
from tests.trial_fixtures import TrialServer

REPO = Path(__file__).resolve().parents[1]
SECRET = "test-key-SECRET-0123456789"
TRIAL_DEMOS = {
    "authorization": (authorization_gate_demo, "ramen__financial_authorization_invariance"),
    "eligibility": (eligibility_gate_demo, "ramen__contractual_eligibility_invariance"),
    "kinematic": (kinematic_gate_demo, "ramen__industrial_iot_actuation_invariance"),
}


def _factory(server: TrialServer) -> Any:
    return lambda: showcase.GateTrialClient(
        min_interval=0.0, public_keys=TEST_PUBLIC_KEYS, transport=server.transport
    )


def _run_trial(module: ModuleType, server: TrialServer, **kwargs: Any) -> int:
    return showcase.gate_main(
        module.TITLE,
        module.run,
        environment={},
        argv=["--trial"],
        script=module.SCRIPT,
        trial_client_factory=_factory(server),
        **kwargs,
    )


@pytest.mark.parametrize("name", TRIAL_DEMOS)
def test_trial_runs_one_block_and_one_allow_within_the_body_limit(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    module, bundle = TRIAL_DEMOS[name]
    server = TrialServer()
    assert _run_trial(module, server) == 0
    out = capsys.readouterr().out
    assert "Keyless public trial" in out and "no API key is read or sent" in out
    assert "BLOCK (live)" in out and "ALLOW (live)" in out and "BLOCKED locally" in out
    assert "Trial allowance left today: 9 of 10" in out
    assert len(server.requests) == 2
    for request, content in zip(server.requests, server.bodies, strict=True):
        body = json.loads(content)
        assert body["bundle_ids"] == [bundle]
        assert len(content) <= TRIAL_MAX_BODY_BYTES
        assert "authorization" not in request.headers


@pytest.mark.parametrize("name", TRIAL_DEMOS)
def test_an_exhausted_allowance_prints_a_partial_run_and_exits_zero(name: str, capsys: pytest.CaptureFixture[str]) -> None:
    module, _ = TRIAL_DEMOS[name]
    server = TrialServer(limit_after=1)
    assert _run_trial(module, server) == 0
    out = capsys.readouterr().out
    assert "Partial run:" in out and "scope: network" in out
    assert "2030-01-01 00:00 UTC" in out and "https://ramenai.dev" in out
    assert "BLOCKED locally" in out and "BLOCK (live)" in out  # what completed is still shown
    assert len(server.requests) == 2  # no retry of a daily limit


def test_a_burst_limit_with_a_long_wait_is_not_retried(capsys: pytest.CaptureFixture[str]) -> None:
    server = TrialServer(limit_after=0, limit_scope="burst")
    assert _run_trial(authorization_gate_demo, server) == 0
    assert "scope: burst" in capsys.readouterr().out
    assert len(server.requests) == 1


def test_a_key_in_the_environment_wins_over_trial(capsys: pytest.CaptureFixture[str]) -> None:
    module, decide, _ = DEMOS["kinematic"]

    def boom() -> Any:
        raise AssertionError("the trial client was built although a key is set")

    code = showcase.gate_main(
        module.TITLE,
        module.run,
        environment={"RAMEN_API_KEY": SECRET},
        argv=["--trial"],
        client_factory=lambda key: ScriptedClient(decide),
        trial_client_factory=boom,
    )
    out = capsys.readouterr().out
    assert code == 0 and "--trial is ignored" in out and SECRET not in out


def test_a_keyed_run_still_exits_one_on_an_unexpected_result(capsys: pytest.CaptureFixture[str]) -> None:
    module, _, _ = DEMOS["kinematic"]
    code = showcase.gate_main(
        module.TITLE,
        module.run,
        environment={"RAMEN_API_KEY": SECRET},
        client_factory=lambda key: ScriptedClient(lambda text: True),
    )
    assert code == 1 and "UNEXPECTED RESULT" in capsys.readouterr().out


def test_provenance_trial_runs_stage_one_only_and_explains_the_limit(capsys: pytest.CaptureFixture[str]) -> None:
    def boom() -> Any:
        raise AssertionError("the provenance demo must not call the trial")

    code = showcase.gate_main(
        provenance_gate_demo.TITLE,
        provenance_gate_demo.run,
        environment={},
        argv=["--trial"],
        script=provenance_gate_demo.SCRIPT,
        trial_client_factory=boom,
        trial_stage_one_only_reason=provenance_gate_demo.TRIAL_STAGE_ONE_ONLY_REASON,
    )
    out = capsys.readouterr().out
    assert code == 0 and "Stage 1 only in --trial mode" in out and "2,048-byte" in out and "BLOCKED locally" in out


def test_without_a_key_or_flag_the_demo_prints_the_exact_trial_command(capsys: pytest.CaptureFixture[str]) -> None:
    code = showcase.gate_main(
        kinematic_gate_demo.TITLE, kinematic_gate_demo.run, environment={}, script=kinematic_gate_demo.SCRIPT
    )
    out = capsys.readouterr().out
    assert code == 0 and "python examples/kinematic_gate_demo.py --trial" in out
    assert "Set RAMEN_API_KEY to run Stages 2 and 3" in out


def test_an_unknown_flag_is_rejected() -> None:
    with pytest.raises(SystemExit) as caught:
        showcase.parse_arguments(["--tial"])
    assert caught.value.code == 2


# --- the payload budget the demos were designed against ---


def test_provenance_stages_two_and_three_exceed_the_trial_body_limit() -> None:
    """Pins the reason the provenance demo stays on Stage 1 in ``--trial`` mode."""
    client = ScriptedClient(DEMOS["provenance"][1])
    provenance_gate_demo.run(client, live=True)
    sizes = []
    for evaluation, call in zip(client.evaluations, client.calls, strict=True):
        body = json.dumps(
            {"input": evaluation.input_text, "bundle_ids": call["bundle_ids"]},
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        sizes.append(len(body))
    mints, stages = sizes[:2], sizes[2:]
    assert all(size <= TRIAL_MAX_BODY_BYTES for size in mints)
    assert all(size > TRIAL_MAX_BODY_BYTES + 800 for size in stages)  # the demo text says "about 3,000 bytes"


def test_every_asimov_case_fits_the_trial_body_limit() -> None:
    server, ran = TrialServer(), []

    @tool
    def dispatch_manipulation(
        robot_id: str, action_type: str, target_object: str, destination_target: str,
        commanded_velocity_mps: float, commanded_force_nm: float, human_proximity_meters: float,
        active_hazard_flags: list[str], scene_context_id: str,
    ) -> str:
        """Canary."""
        ran.append(scene_context_id)
        return "ok"

    cases = benchmark_asimov_robotics.build_benchmark_cases()
    with server.client() as client:
        agent = IndustrialAutomationAgent(client=client, tools={"dispatch_manipulation": dispatch_manipulation})
        for case in cases:
            agent.execute("dispatch_manipulation", case.payload)
    assert len(server.bodies) == len(cases) == 5
    assert max(len(body) for body in server.bodies) <= TRIAL_MAX_BODY_BYTES - 1000


# --- examples ---


def test_asimov_table_wraps_steering_and_shows_latency() -> None:
    long_steering = "Re-plan the placement with a thermal containment tray and an isolated heat source. " * 4
    outcomes = [
        benchmark_asimov_robotics.BenchmarkOutcome("Case 2 — Hazardous thermal placement / RoboHarm", "[BLOCKED]", "ramen_pk_v1", True, long_steering, 842.4),
        benchmark_asimov_robotics.BenchmarkOutcome("Case 1", "[ALLOWED]", "ramen_pk_v1", True, "—"),
    ]
    table = benchmark_asimov_robotics.format_execution_table(outcomes)
    lines = table.splitlines()
    assert len({len(line) for line in lines}) == 1  # every line has the same width
    assert max(len(line) for line in lines) < 170
    assert "842 ms" in table and "Latency" in table
    assert "| - " in table


def test_the_recording_client_measures_latency(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RamenClient, "evaluate_compliance", lambda self, text, **kw: {"allowed": True})
    client = benchmark_asimov_robotics.RecordingRamenClient("test-key")
    client.evaluate_compliance("x")
    assert len(client.latencies_ms) == 1 and client.latencies_ms[0] >= 0


def test_importing_the_roboharm_benchmark_never_replaces_the_process(monkeypatch: pytest.MonkeyPatch) -> None:
    """A re-exec at import time ended a pytest run under another interpreter silently."""
    from examples import benchmark_roboharm_full_suite as roboharm

    replaced: list[Any] = []
    monkeypatch.setattr(os, "execv", lambda *args: replaced.append(args))
    monkeypatch.setattr(sys, "prefix", "/nonexistent/other/interpreter")
    importlib.reload(roboharm)
    assert replaced == []
    assert "execv" not in Path(roboharm.__file__).read_text(encoding="utf-8")


def test_the_sample_csv_ships_with_the_example_and_fits_the_trial() -> None:
    import pandas as pd

    frame = pd.read_csv(filter_dataset.SAMPLE_CSV)
    assert list(frame.columns) == ["document_id", "source", "content"] and len(frame) == 5
    server = TrialServer()
    with server.client() as client:
        for row in frame.to_dict("records"):
            client.evaluate_compliance(json.dumps(row, sort_keys=True), bundle_ids=[filter_dataset.DEFAULT_BUNDLE])
    assert max(len(body) for body in server.bodies) < 400


def test_filter_dataset_runs_on_the_trial_and_writes_the_allowed_rows(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import pandas as pd

    server = TrialServer()
    server.install(monkeypatch)
    monkeypatch.delenv("RAMEN_API_KEY", raising=False)
    destination = tmp_path / "out.csv"
    assert filter_dataset.main(["--trial", str(filter_dataset.SAMPLE_CSV), str(destination)]) == 0
    assert pd.read_csv(destination)["document_id"].tolist() == ["doc-001", "doc-002", "doc-005"]
    assert "Wrote 3 allowed rows" in capsys.readouterr().out
    assert len(server.requests) == 5


def test_filter_dataset_reports_an_exhausted_allowance_and_writes_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    server = TrialServer(limit_after=2)
    server.install(monkeypatch)
    destination = tmp_path / "out.csv"
    assert filter_dataset.main(["--trial", str(filter_dataset.SAMPLE_CSV), str(destination)]) == 0
    assert "Partial run, nothing written" in capsys.readouterr().out and not destination.exists()


def test_filter_dataset_without_a_key_or_flag_fails_with_a_tip(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("RAMEN_API_KEY", raising=False)
    assert filter_dataset.main([str(filter_dataset.SAMPLE_CSV), str(tmp_path / "out.csv")]) == 1
    assert "--trial" in capsys.readouterr().err
