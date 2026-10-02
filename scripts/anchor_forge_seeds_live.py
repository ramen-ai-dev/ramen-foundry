"""Anchor the ramen forge seed repairs to live ramen-ai receipts.

For each of the five canonical seed trajectories, this script:

1. Evaluates the failed call live through ``RamenClient`` and requires a
   BLOCKED verdict with a verified receipt.
2. Evaluates the repaired call live and requires an ALLOWED verdict whose
   Schema V5 receipt verifies locally (Ed25519 signature and input-hash
   binding) under ``ramen_pk_v1``.
3. Builds a ``CorrectionExemplar`` from the live evaluation (rule reasoning,
   primary statutory anchor, steering directive) and the receipt of the
   allowed repair, then posts it to ramen forge with ``RemoteForgeMemoryStore``.

A trajectory is only posted when both gates pass. A receipt from a blocked
evaluation is never attached to an exemplar, because it proves the repair was
refused. Trajectories that ramen forge already holds with a receipt are
skipped, so re-running does not add duplicate lessons.

Run from the repository root::

    python scripts/anchor_forge_seeds_live.py            # evaluate and post
    python scripts/anchor_forge_seeds_live.py --dry-run  # evaluate only

Credentials are read from the process environment, then ``./.env``, then
``../ramen-ai-integrations/.env``:

``RAMEN_API_KEY`` (required)
    ramen-ai credential.
``OPENAI_API_KEY`` (optional)
    Starter/Professional BYOK. When unset, provider key and name are both
    omitted so ramen-ai uses Enterprise managed-provider inference.
``FORGE_WRITE_TOKEN`` (required unless ``--dry-run``)
    ramen forge write token. Posted records are publicly readable.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
from ramen_ai import RamenClient

from ramen_foundry import CorrectionExemplar, RemoteForgeMemoryStore
from ramen_foundry.core.memory import fingerprint_task
from ramen_foundry.core.steer_node import primary_statutory_anchor

FORGE_URL = "https://forge.ramenai.dev"
EXPECTED_KID = "ramen_pk_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILES = (REPO_ROOT / ".env", REPO_ROOT.parent / "ramen-ai-integrations" / ".env")
CREDENTIAL_KEYS = ("RAMEN_API_KEY", "OPENAI_API_KEY", "FORGE_WRITE_TOKEN")

# Field limits enforced by the ramen forge validator.
FORGE_TEXT_LIMITS = {
    "violation_reason": 1000,
    "primary_statutory_anchor": 256,
    "steering_directive": 2000,
}

CREDIT_ADVERSE_ACTION_POLICY_ID = "796b7a87-d1f5-4ecc-91f2-a506a9b0d91e"
WIRE_DUAL_CONTROL_POLICY_ID = "b4c18ba1-26b7-4b7f-b44b-65e8de790572"
ROBOTICS_SAFETY_POLICY_ID = "1fc71052-eb7e-43fe-9bfa-7ee06afe5b95"
SHIELD_CORE_IT_BUNDLE_ID = "ramen__shield_core_it"


@dataclass(frozen=True)
class Trajectory:
    """One seed repair: the blocked call, the repaired call, and its policy scope."""

    label: str
    domain: str
    task_description: str
    tool_name: str
    failed_arguments: dict[str, Any]
    repaired_arguments: dict[str, Any]
    policy_ids: tuple[str, ...] = ()
    bundle_ids: tuple[str, ...] = ()


# Mirrors ramen-forge/src/seed.ts so receipted records anchor the same lessons.
TRAJECTORIES: tuple[Trajectory, ...] = (
    Trajectory(
        label="FinTech adverse action",
        domain="fintech",
        task_description="Issue an adverse action notice for a declined small-business credit application.",
        tool_name="issue_adverse_action_notice",
        failed_arguments={
            "application_id": "APP-SEED-0001",
            "decision": "DECLINED",
            "principal_reason_code": "POSTAL_CODE_RISK",
            "principal_reason_text": "Applicant business postal code is located in a high-risk area.",
        },
        repaired_arguments={
            "application_id": "APP-SEED-0001",
            "decision": "DECLINED",
            "principal_reason_code": "INSUFFICIENT_LIQUIDITY",
            "principal_reason_text": "Insufficient liquid assets relative to the requested credit amount.",
        },
        policy_ids=(CREDIT_ADVERSE_ACTION_POLICY_ID,),
    ),
    Trajectory(
        label="FinTech wire transfer",
        domain="fintech",
        task_description="Initiate an outbound commercial wire transfer to a vendor beneficiary.",
        tool_name="initiate_wire_transfer",
        failed_arguments={
            "amount_usd": 25000,
            "beneficiary_account_ref": "BENEF-SEED-7781",
            "originator_id": "AGENT-TREASURY-01",
            "approvals": [],
        },
        repaired_arguments={
            "amount_usd": 25000,
            "beneficiary_account_ref": "BENEF-SEED-7781",
            "originator_id": "AGENT-TREASURY-01",
            "approvals": [
                {
                    "role": "treasury_officer",
                    "officer_id": "OFFICER-SEED-02",
                    "credential_ref": "vault://officers/OFFICER-SEED-02/signing-key",
                }
            ],
            "dual_control": True,
        },
        policy_ids=(WIRE_DUAL_CONTROL_POLICY_ID,),
    ),
    Trajectory(
        label="Industrial IoT collaborative speed",
        domain="industrial_iot",
        task_description="Resume the pick-and-place cycle on a collaborative robot cell.",
        tool_name="set_robot_tcp_speed",
        failed_arguments={
            "cell_id": "CELL-SEED-04",
            "tcp_speed_mps": 0.85,
            "human_in_collaborative_zone": True,
        },
        repaired_arguments={
            "cell_id": "CELL-SEED-04",
            "tcp_speed_mps": 0.25,
            "human_in_collaborative_zone": True,
            "mode": "collaborative_reduced_speed",
        },
        policy_ids=(ROBOTICS_SAFETY_POLICY_ID,),
    ),
    Trajectory(
        label="Industrial IoT thermal placement",
        domain="industrial_iot",
        task_description="Stage incoming material canisters next to the curing oven line.",
        tool_name="place_material",
        failed_arguments={
            "item": "solvent_canister",
            "flammable": True,
            "target_zone": "OVEN-2-BURNER-ADJACENT",
            "distance_to_burner_m": 0.4,
        },
        repaired_arguments={
            "item": "solvent_canister",
            "flammable": True,
            "action": "HALT",
            "target_zone": "FLAMMABLES_CABINET_A",
            "hold_reason": "volatile_material_near_ignition_source",
        },
        policy_ids=(ROBOTICS_SAFETY_POLICY_ID,),
    ),
    Trajectory(
        label="DevSecOps bash command",
        domain="devsecops",
        task_description="Clean up stale build artefacts in the project workspace.",
        tool_name="run_bash",
        failed_arguments={"command": "rm -rf / --no-preserve-root"},
        repaired_arguments={
            "command": "rm -rf ./build/tmp",
            "working_directory": "/workspace/project",
        },
        bundle_ids=(SHIELD_CORE_IT_BUNDLE_ID,),
    ),
)


@dataclass
class Outcome:
    trajectory: Trajectory
    status: str
    detail: str = ""
    receipt: dict[str, Any] = field(default_factory=dict)
    exemplar_id: str | None = None

    @property
    def anchored(self) -> bool:
        return self.status in {"POSTED", "ALREADY ANCHORED", "VERIFIED (dry run)"}


def load_credentials() -> dict[str, str]:
    """Return credentials, preferring the process environment over dotenv files."""
    resolved: dict[str, str] = {}
    for path in reversed(ENV_FILES):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            name, separator, value = line.strip().partition("=")
            name = name.removeprefix("export ").strip()
            if separator and name in CREDENTIAL_KEYS:
                value = value.strip().strip('"').strip("'")
                if value:
                    resolved[name] = value
    for name in CREDENTIAL_KEYS:
        if os.environ.get(name):
            resolved[name] = os.environ[name]
    return resolved


def evaluate(
    client: RamenClient,
    trajectory: Trajectory,
    arguments: Mapping[str, Any],
    provider: Mapping[str, str],
) -> dict[str, Any]:
    payload = json.dumps(
        {"tool": trajectory.tool_name, "arguments": dict(arguments)},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return client.evaluate_compliance(
        payload,
        policy_ids=list(trajectory.policy_ids) or None,
        bundle_ids=list(trajectory.bundle_ids) or None,
        context={"tool_name": trajectory.tool_name},
        **provider,
    )


def first_violation(verdict: Mapping[str, Any]) -> Mapping[str, Any]:
    violations = (verdict.get("data") or {}).get("total_violations") or []
    return next((v for v in violations if isinstance(v, Mapping)), {})


def already_anchored(store: RemoteForgeMemoryStore, trajectory: Trajectory) -> CorrectionExemplar | None:
    records = store.retrieve_relevant_exemplars(
        fingerprint_task(trajectory.task_description), trajectory.tool_name, limit=50
    )
    return next(
        (r for r in records if r.receipt_id and r.task_description == trajectory.task_description),
        None,
    )


def anchor(
    client: RamenClient,
    trajectory: Trajectory,
    provider: Mapping[str, str],
    forge_token: str | None,
    dry_run: bool,
) -> Outcome:
    store = RemoteForgeMemoryStore(base_url=FORGE_URL, write_token=forge_token, domain=trajectory.domain)
    existing = already_anchored(store, trajectory)
    if existing is not None:
        return Outcome(trajectory, "ALREADY ANCHORED", f"exemplar {existing.exemplar_id}, receipt {existing.receipt_id}")

    try:
        failed = evaluate(client, trajectory, trajectory.failed_arguments, provider)
        repaired = evaluate(client, trajectory, trajectory.repaired_arguments, provider)
    except (httpx.HTTPError, ValueError) as error:
        return Outcome(trajectory, "ERROR", f"evaluation failed: {error}")

    if failed.get("allowed") or not failed.get("receipt_verified"):
        return Outcome(trajectory, "SKIPPED", "failed call was not a verified BLOCK")

    receipt = dict((repaired.get("data") or {}).get("receipt") or {})
    if not repaired.get("allowed"):
        violation = first_violation(repaired)
        reason = violation.get("reasoning") or violation.get("rule_name") or "no reason returned"
        return Outcome(trajectory, "SKIPPED", f"repair is BLOCKED live: {reason}")
    if not repaired.get("receipt_verified"):
        return Outcome(trajectory, "SKIPPED", f"receipt did not verify: {repaired.get('receipt_reason')}")
    if receipt.get("kid") != EXPECTED_KID or not receipt.get("id"):
        return Outcome(trajectory, "SKIPPED", f"receipt kid {receipt.get('kid')!r} is not {EXPECTED_KID}")

    violation = first_violation(failed)
    exemplar = CorrectionExemplar.create(
        task=trajectory.task_description,
        tool_name=trajectory.tool_name,
        failed_arguments=trajectory.failed_arguments,
        violation_reason=str(violation.get("reasoning") or violation.get("rule_name") or "Blocked by ramen-ai policy."),
        primary_statutory_anchor=primary_statutory_anchor((failed.get("data") or {}).get("statutory_anchors")),
        steering_directive=str(failed.get("steering") or violation.get("recovery_instruction") or ""),
        repaired_arguments=trajectory.repaired_arguments,
        receipt_id=str(receipt["id"]),
        receipt=receipt,
    )
    oversized = [
        name for name, limit in FORGE_TEXT_LIMITS.items() if len(getattr(exemplar, name)) > limit
    ]
    if oversized:
        return Outcome(trajectory, "SKIPPED", f"exceeds forge limits: {', '.join(oversized)}", receipt)

    if dry_run:
        return Outcome(trajectory, "VERIFIED (dry run)", "not posted", receipt, exemplar.exemplar_id)
    try:
        store.record_correction(exemplar)
    except (RuntimeError, PermissionError, ValueError) as error:
        return Outcome(trajectory, "ERROR", str(error), receipt)
    return Outcome(trajectory, "POSTED", "ingested by ramen forge", receipt, exemplar.exemplar_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="evaluate and verify only; post nothing")
    args = parser.parse_args()

    credentials = load_credentials()
    if not credentials.get("RAMEN_API_KEY"):
        raise RuntimeError("RAMEN_API_KEY must be supplied through the environment or a .env file")
    forge_token = credentials.get("FORGE_WRITE_TOKEN")
    if not forge_token and not args.dry_run:
        raise RuntimeError("FORGE_WRITE_TOKEN is required to post; use --dry-run to evaluate only")
    provider: dict[str, str] = {}
    if credentials.get("OPENAI_API_KEY"):
        provider = {"provider_key": credentials["OPENAI_API_KEY"], "provider_name": "openai"}

    print(f"ramen-ai provider mode: {'BYOK (openai)' if provider else 'Enterprise managed-provider'}")
    print(f"ramen forge: {FORGE_URL} ({'dry run' if args.dry_run else 'posting enabled'})\n")

    outcomes: list[Outcome] = []
    with RamenClient(credentials["RAMEN_API_KEY"]) as client:
        for trajectory in TRAJECTORIES:
            outcome = anchor(client, trajectory, provider, forge_token, args.dry_run)
            outcomes.append(outcome)
            print(f"[{outcome.status}] {trajectory.label} ({trajectory.tool_name})")
            if outcome.detail:
                print(f"    {outcome.detail}")
            if outcome.receipt:
                print(
                    f"    receipt {outcome.receipt.get('id')} | schema {outcome.receipt.get('schema_version')} "
                    f"| kid {outcome.receipt.get('kid')} | Ed25519 verified"
                )
                print(f"    signature {str(outcome.receipt.get('signature'))[:32]}...")
            if outcome.exemplar_id:
                print(f"    exemplar {outcome.exemplar_id}")

    anchored = sum(outcome.anchored for outcome in outcomes)
    print(f"\nAnchored: {anchored} of {len(outcomes)} trajectories")
    try:
        stats = httpx.get(f"{FORGE_URL}/api/v1/stats", timeout=10).json()
        print(
            "ramen forge stats: "
            f"{stats.get('receipt_verified_exemplars')} of {stats.get('total_community_exemplars')} exemplars "
            f"receipted, recovery_rate {float(stats.get('recovery_rate') or 0):.1%}"
        )
    except (httpx.HTTPError, ValueError) as error:
        print(f"ramen forge stats unavailable: {error}")
    return 0 if anchored == len(outcomes) else 1


if __name__ == "__main__":
    sys.exit(main())
