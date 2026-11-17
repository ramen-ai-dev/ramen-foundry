"""Live end-to-end run of the Core Architecture gate demos (one pass each, serial)."""

from __future__ import annotations

from types import ModuleType

import pytest

from examples import (
    authorization_gate_demo,
    eligibility_gate_demo,
    kinematic_gate_demo,
    provenance_gate_demo,
)
from examples._gate_showcase import GateRecordingClient
from tests.live._support import api_key

pytestmark = pytest.mark.live

DEMOS = [authorization_gate_demo, eligibility_gate_demo, provenance_gate_demo, kinematic_gate_demo]


@pytest.mark.parametrize("demo", DEMOS, ids=lambda m: m.__name__.rsplit(".", 1)[-1])
def test_demo_ends_in_a_verified_block_and_a_verified_allow(demo: ModuleType) -> None:
    key = api_key()
    assert key, "RAMEN_API_KEY is not available"
    client = GateRecordingClient(key, min_interval=2.5)
    try:
        outcomes = demo.run(client, live=True)
    finally:
        client.close()

    live_outcomes = [o for o in outcomes if o.decision.endswith("(live)")]
    assert [o.decision for o in live_outcomes] == ["BLOCK (live)", "ALLOW (live)"]
    assert all("Schema V5 Ed25519 verified, kid=ramen_pk_v1" in o.receipt for o in live_outcomes)
    assert any(o.decision == "BLOCKED locally" for o in outcomes)
