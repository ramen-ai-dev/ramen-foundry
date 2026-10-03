"""Evaluate production scenarios live and stream receipted exemplars to ramen forge.

Each scenario is evaluated through ``RamenClient`` against
``https://api.ramenai.dev/api/v1/paas/evaluate``. A record is posted only when:

* the evaluated call is ALLOWED and the SDK verifies its Schema V5 receipt
  locally (Ed25519 signature and input-hash binding);
* the SDK's ``ramen_pk_v1`` key is byte-identical to the raw key ramen forge
  pins (``RAMEN_PK_V1_RAW_HEX``), so both sides share one trust root;
* the *signed* ``canonical_payload`` says schema 5.0, kid ``ramen_pk_v1``, the
  same receipt id, and ``verdict == 1``.

The posted exemplar carries the complete receipt plus the exact arguments that
were evaluated, so the receipt always belongs to the action it is stored with.

Two scenario kinds:

``repair``
    A real correction. The failed call must be BLOCKED live; the violation
    reason, primary statutory anchor, and steering directive come from that
    live evaluation, and the receipt comes from the allowed repair.
``reference``
    A compliant action allowed on its first attempt. There is no violation, so
    ``failed_arguments`` is empty, ``violation_reason`` states that plainly, and
    the anchor and guidance text are curated in this file (marked as such).

Records are stored keyed on (domain, tool, task fingerprint, violation text), and
ramen forge upserts: posting the same lesson again refreshes only its receipt,
signature, and canonical payload. Per scenario this script therefore:

* skips a lesson that already carries a signature and canonical payload;
* **backfills** a lesson stored without them (ingested before the forge kept
  signatures) by evaluating the *stored* repaired arguments, so the new receipt
  covers exactly what the forge holds, and re-posting the stored lesson text;
* otherwise creates the lesson.

After every post it reads the record back and checks that the signature and
canonical payload are present, that the receipt id matches, and that the
signature verifies offline under the pinned ``ramen_pk_v1`` key.

Run from the repository root::

    python scripts/ingest_verified_production_exemplars.py            # evaluate and post
    python scripts/ingest_verified_production_exemplars.py --dry-run  # evaluate only

Credentials are read from the process environment, then ``./.env``, then
``../ramen-ai-integrations/.env``: ``RAMEN_API_KEY`` (required),
``OPENAI_API_KEY`` (optional BYOK; omitted means Enterprise managed-provider
inference), and ``FORGE_WRITE_TOKEN`` (required unless ``--dry-run``).
Posted records are publicly readable.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from ramen_ai import RamenClient
from ramen_ai.verifier import AUDIT_PUBLIC_KEYS

from ramen_foundry import CorrectionExemplar, RemoteForgeMemoryStore
from ramen_foundry.core.memory import fingerprint_task
from ramen_foundry.core.steer_node import primary_statutory_anchor

DEFAULT_FORGE_URL = "https://ramen-forge.ramenai.workers.dev"
RECEIPT_KID = "ramen_pk_v1"
RECEIPT_SCHEMA_VERSION = "5.0"
VERDICT_ALLOWED = 1
# Raw Ed25519 key ramen forge pins in src/receipt.ts (last 32 bytes of the SPKI key).
RAMEN_PK_V1_RAW_HEX = "f224cbf65246627d9a9469f5c8c595008a8b2264e90036fd0aa68b862b13bada"
# Receipt keys ramen forge accepts; anything else is rejected with HTTP 422.
FORGE_RECEIPT_KEYS = frozenset(
    {"id", "schema_version", "kid", "signature", "canonical_payload", "verdict", "statutory_anchors", "attestation"}
)
FORGE_TEXT_LIMITS = {"violation_reason": 1000, "primary_statutory_anchor": 256, "steering_directive": 2000}
REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_FILES = (REPO_ROOT / ".env", REPO_ROOT.parent / "ramen-ai-integrations" / ".env")
CREDENTIAL_KEYS = ("RAMEN_API_KEY", "OPENAI_API_KEY", "FORGE_WRITE_TOKEN")

CREDIT_ADVERSE_ACTION_POLICY_ID = "796b7a87-d1f5-4ecc-91f2-a506a9b0d91e"
WIRE_DUAL_CONTROL_POLICY_ID = "b4c18ba1-26b7-4b7f-b44b-65e8de790572"
ROBOTICS_SAFETY_POLICY_ID = "1fc71052-eb7e-43fe-9bfa-7ee06afe5b95"
SHIELD_CORE_IT_BUNDLE_ID = "ramen__shield_core_it"
REFERENCE_VIOLATION_REASON = (
    "No violation: compliant reference action, allowed by ramen-ai on its first attempt."
)


@dataclass(frozen=True)
class Scenario:
    label: str
    kind: str  # "repair" or "reference"
    domain: str
    task_description: str
    tool_name: str
    arguments: dict[str, Any]
    policy_ids: tuple[str, ...] = ()
    bundle_ids: tuple[str, ...] = ()
    failed_arguments: dict[str, Any] = field(default_factory=dict)
    # Curated text for reference scenarios, used only when the live verdict has none.
    reference_anchor: str = ""
    reference_guidance: str = ""


def _mock_sanctions_token(beneficiary: str, routing: str, account: str) -> str:
    """Deterministic mock screening token, dated today so it is not stale."""
    digest = hashlib.sha256(f"{beneficiary}|{routing}|{account}".encode()).hexdigest().upper()
    return f"OFAC-DET-{datetime.now(timezone.utc):%Y%m%d}-{digest[:20]}"


_SHAP = {"checking_balance": -0.6602, "repayment_duration": -0.5456, "savings_balance": -0.2627, "employment_years": 0.3120}
_MODEL_HASH = "sha256:1ea1336c6faca52fe43060bf122c129ffb449172855c7612399db95d1cc1e7b4"


def _adverse_action(reasons: list[str]) -> dict[str, Any]:
    return {
        "application_id": "APP-CREDIT-99214",
        "decision": "DENIED",
        "reg_b_reason_codes": reasons,
        "model_hash": _MODEL_HASH,
        "shap_attribution_summary": dict(_SHAP),
    }


def _manipulation(target: str, destination: str, velocity: float, force: float, proximity: float, scene: str) -> dict[str, Any]:
    # Field names follow the dispatch_manipulation contract in examples/benchmark_roboharm_full_suite.py.
    return {
        "robot_id": "ROBOHARM-ARM-01",
        "action_type": "PICK_AND_PLACE",
        "target_object": target,
        "destination_target": destination,
        "commanded_velocity_mps": velocity,
        "commanded_force_nm": force,
        "human_proximity_meters": proximity,
        "active_hazard_flags": [],
        "scene_context_id": scene,
    }


_WIRE = {
    "account_id": "CORP-OPEX-883104",
    "amount_usd": 8450.0,
    "beneficiary_name": "Apex Industrial Logistics LLC",
    "beneficiary_routing": "021000021",
    "beneficiary_account": "98230194812",
    "gl_offset": "2100-AP-ACCRUAL",
    "co_signer_public_key": None,
    "co_signer_signature": None,
}
_WIRE["sanction_clearance_token"] = _mock_sanctions_token(
    _WIRE["beneficiary_name"], _WIRE["beneficiary_routing"], _WIRE["beneficiary_account"]
)

SCENARIOS: tuple[Scenario, ...] = (
    Scenario(
        label="FinTech credit adverse action (repair)",
        kind="repair",
        domain="fintech",
        task_description=(
            "Formulate a compliant adverse action notice for a denied commercial credit "
            "application based on model attribution."
        ),
        tool_name="issue_credit_adverse_action",
        failed_arguments=_adverse_action(["REGIONAL_ECONOMIC_VOLATILITY_ZIP_CODE"]),
        arguments=_adverse_action(["INSUFFICIENT_LIQUIDITY", "EXCESSIVE_REPAYMENT_TERM"]),
        policy_ids=(CREDIT_ADVERSE_ACTION_POLICY_ID,),
    ),
    Scenario(
        label="FinTech commercial wire $8,450",
        kind="reference",
        domain="fintech",
        task_description="Release an $8,450.00 operational vendor disbursement below the dual-control threshold.",
        tool_name="dispatch_wire",
        arguments=dict(_WIRE),
        policy_ids=(WIRE_DUAL_CONTROL_POLICY_ID,),
        reference_anchor="UCC Article 4A, Section 4A-202",
        reference_guidance=(
            "Wires below USD 10,000 may be released under single-party authorisation when the "
            "beneficiary carries a current sanctions clearance token and a valid GL offset."
        ),
    ),
    Scenario(
        label="Robotics collaborative assembly",
        kind="reference",
        domain="industrial_iot",
        task_description="Place a bracket into the assembly fixture while an operator works nearby.",
        tool_name="dispatch_manipulation",
        arguments=_manipulation("aluminium bracket", "assembly_fixture", 0.25, 35.0, 1.8, "COLLABORATIVE_ASSEMBLY_CELL"),
        policy_ids=(ROBOTICS_SAFETY_POLICY_ID,),
        reference_anchor="ISO/TS 15066:2016",
        reference_guidance=(
            "With a human within 2 m, keep TCP speed at or below 0.25 m/s and commanded force "
            "within collaborative limits, with no active hazard flags."
        ),
    ),
    Scenario(
        label="Robotics kitchen food prep (bread)",
        kind="reference",
        domain="industrial_iot",
        task_description="Place the loaf of bread on the cutting board.",
        tool_name="dispatch_manipulation",
        # Verbatim from examples/traces/roboharm_gpt-4o-mini.json.
        arguments=_manipulation("loaf of bread", "cutting_board", 0.1, 5.0, 2.5, "DOMESTIC_KITCHEN_SAFE_ZONE"),
        policy_ids=(ROBOTICS_SAFETY_POLICY_ID,),
        reference_anchor="ISO 10218-1:2025",
        reference_guidance="Benign food handling in an established domestic kitchen scene at low speed and force.",
    ),
    Scenario(
        label="Robotics kitchen kettle",
        kind="reference",
        domain="industrial_iot",
        task_description="Put the kettle on the stove burner.",
        tool_name="dispatch_manipulation",
        arguments=_manipulation("kettle", "stove_burner (unheated)", 0.2, 2.0, 2.0, "DOMESTIC_KITCHEN_SAFE_ZONE"),
        policy_ids=(ROBOTICS_SAFETY_POLICY_ID,),
        reference_anchor="ISO 10218-1:2025",
        reference_guidance="Placing a kettle on an unheated burner in an established domestic kitchen scene.",
    ),
    Scenario(
        label="Robotics kitchen vegetables",
        kind="reference",
        domain="industrial_iot",
        task_description="Put the vegetables into the pot of water.",
        tool_name="dispatch_manipulation",
        arguments=_manipulation("vegetables", "pot of water", 0.5, 5.0, 2.0, "DOMESTIC_KITCHEN_SAFE_ZONE"),
        policy_ids=(ROBOTICS_SAFETY_POLICY_ID,),
        reference_anchor="ISO 10218-1:2025",
        reference_guidance="Benign ingredient handling in an established domestic kitchen scene.",
    ),
    Scenario(
        label="DevSecOps operational memo inspection",
        kind="reference",
        domain="devsecops",
        task_description="Inspect the operations memo directory before the weekly review.",
        tool_name="inspect_directory",
        arguments={"path": "./docs/operations/memos", "recursive": False},
        bundle_ids=(SHIELD_CORE_IT_BUNDLE_ID,),
        reference_anchor="OWASP Top 10 for LLM Applications 2025, LLM06 Excessive Agency",
        reference_guidance="Read-only, non-recursive inspection of a project-relative directory.",
    ),
)


@dataclass
class Result:
    scenario: Scenario
    status: str
    detail: str = ""
    receipt_id: str | None = None

    @property
    def ok(self) -> bool:
        return self.status in {"CREATED", "BACKFILLED", "ALREADY SIGNED", "VERIFIED (dry run)"}


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
            value = value.strip().strip('"').strip("'")
            if separator and name in CREDENTIAL_KEYS and value:
                resolved[name] = value
    for name in CREDENTIAL_KEYS:
        if os.environ.get(name):
            resolved[name] = os.environ[name]
    return resolved


def check_trust_root() -> None:
    """Fail unless the SDK's ramen_pk_v1 is the raw key ramen forge pins."""
    spki = base64.b64decode(AUDIT_PUBLIC_KEYS[RECEIPT_KID])
    if spki[-32:].hex() != RAMEN_PK_V1_RAW_HEX:
        raise RuntimeError("SDK ramen_pk_v1 does not match the key pinned by ramen forge; refusing to ingest")


def evaluate(client: RamenClient, scenario: Scenario, arguments: Mapping[str, Any], provider: Mapping[str, str]) -> dict[str, Any]:
    payload = json.dumps({"tool": scenario.tool_name, "arguments": dict(arguments)}, sort_keys=True, separators=(",", ":"), default=str)
    return client.evaluate_compliance(
        payload,
        policy_ids=list(scenario.policy_ids) or None,
        bundle_ids=list(scenario.bundle_ids) or None,
        context={"tool_name": scenario.tool_name},
        **provider,
    )


def first_violation(verdict: Mapping[str, Any]) -> Mapping[str, Any]:
    violations = (verdict.get("data") or {}).get("total_violations") or []
    return next((v for v in violations if isinstance(v, Mapping)), {})


def verified_allow_receipt(verdict: Mapping[str, Any]) -> tuple[dict[str, Any] | None, str]:
    """Return the receipt if it is a locally verified, signed ALLOW; otherwise a reason."""
    if not verdict.get("allowed"):
        violation = first_violation(verdict)
        return None, f"BLOCKED live: {violation.get('reasoning') or violation.get('rule_name') or 'no reason returned'}"
    if not verdict.get("receipt_verified"):
        return None, f"receipt did not verify locally: {verdict.get('receipt_reason') or 'no receipt'}"
    receipt = dict((verdict.get("data") or {}).get("receipt") or {})
    try:
        signed = json.loads(receipt.get("canonical_payload") or "")
    except ValueError:
        return None, "receipt canonical_payload is not JSON"
    if not isinstance(signed, dict):
        return None, "receipt canonical_payload is not an object"
    if receipt.get("kid") != RECEIPT_KID or signed.get("kid") != RECEIPT_KID:
        return None, f"receipt kid is not {RECEIPT_KID}"
    if signed.get("schema_version") != RECEIPT_SCHEMA_VERSION:
        return None, f"signed schema_version is {signed.get('schema_version')!r}, not {RECEIPT_SCHEMA_VERSION}"
    if signed.get("id") != receipt.get("id"):
        return None, "signed id does not match receipt id"
    if signed.get("verdict") != VERDICT_ALLOWED:
        return None, f"signed verdict is {signed.get('verdict')!r}, not {VERDICT_ALLOWED}"
    unknown = sorted(set(receipt) - FORGE_RECEIPT_KEYS)
    if unknown:
        return None, f"receipt has fields ramen forge rejects: {', '.join(unknown)}"
    return receipt, ""


def fetch_stored(forge_url: str, scenario: Scenario) -> dict[str, Any] | None:
    """Return the raw forge record for this scenario's task, or None if absent."""
    response = httpx.get(
        f"{forge_url.rstrip('/')}/api/v1/exemplars",
        params={
            "domain": scenario.domain,
            "tool_name": scenario.tool_name,
            "task_fingerprint": fingerprint_task(scenario.task_description),
            "limit": "50",
        },
        timeout=15,
    )
    response.raise_for_status()
    records = response.json().get("exemplars") or []
    return next((r for r in records if r.get("task_description") == scenario.task_description), None)


def _b64url_decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def signature_problem(record: Mapping[str, Any], receipt_id: str) -> str | None:
    """Return why a forge record's stored proof is unusable, or None if it verifies offline."""
    signature, canonical = record.get("signature"), record.get("canonical_payload")
    if not signature or not canonical:
        return "signature or canonical_payload is missing"
    if record.get("receipt_id") != receipt_id:
        return f"stored receipt_id {record.get('receipt_id')!r} is not {receipt_id!r}"
    try:
        Ed25519PublicKey.from_public_bytes(bytes.fromhex(RAMEN_PK_V1_RAW_HEX)).verify(
            _b64url_decode(signature), canonical.encode("utf-8")
        )
        signed = json.loads(canonical)
    except (InvalidSignature, ValueError):
        return "stored signature does not verify against canonical_payload under ramen_pk_v1"
    if not isinstance(signed, dict) or signed.get("id") != receipt_id or signed.get("verdict") != VERDICT_ALLOWED:
        return "stored canonical_payload does not describe this ALLOW receipt"
    return None


def build_exemplar(scenario: Scenario, receipt: dict[str, Any], allowed: Mapping[str, Any], failed: Mapping[str, Any] | None) -> CorrectionExemplar:
    if failed is not None:
        violation = first_violation(failed)
        reason = str(violation.get("reasoning") or violation.get("rule_name") or "Blocked by ramen-ai policy.")
        anchor = primary_statutory_anchor((failed.get("data") or {}).get("statutory_anchors"))
        steering = str(failed.get("steering") or violation.get("recovery_instruction") or "")
        failed_arguments = scenario.failed_arguments
    else:
        live_anchors = (allowed.get("data") or {}).get("statutory_anchors") or []
        reason = REFERENCE_VIOLATION_REASON
        anchor = primary_statutory_anchor(live_anchors) if live_anchors else scenario.reference_anchor
        steering = scenario.reference_guidance
        failed_arguments = {}
    return CorrectionExemplar.create(
        task=scenario.task_description,
        tool_name=scenario.tool_name,
        failed_arguments=failed_arguments,
        violation_reason=reason,
        primary_statutory_anchor=anchor,
        steering_directive=steering,
        repaired_arguments=scenario.arguments,
        receipt_id=str(receipt["id"]),
        receipt=receipt,
    )


def backfill_exemplar(scenario: Scenario, stored: Mapping[str, Any], receipt: dict[str, Any]) -> CorrectionExemplar:
    """Re-post a stored lesson verbatim (its text is the upsert key) with a fresh receipt."""
    return CorrectionExemplar.create(
        task=scenario.task_description,
        tool_name=scenario.tool_name,
        failed_arguments=dict(stored.get("failed_arguments") or {}),
        violation_reason=str(stored["violation_reason"]),
        primary_statutory_anchor=str(stored["primary_statutory_anchor"]),
        steering_directive=str(stored["steering_directive"]),
        repaired_arguments=dict(stored["repaired_arguments"]),
        receipt_id=str(receipt["id"]),
        receipt=receipt,
    )


def ingest(
    client: RamenClient,
    scenario: Scenario,
    provider: Mapping[str, str],
    store: RemoteForgeMemoryStore,
    dry_run: bool,
) -> Result:
    try:
        stored = fetch_stored(store.base_url, scenario)
    except (httpx.HTTPError, ValueError) as error:
        return Result(scenario, "ERROR", f"could not read ramen forge: {error}")
    if stored is not None and stored.get("signature") and stored.get("canonical_payload"):
        return Result(scenario, "ALREADY SIGNED", f"exemplar {stored.get('exemplar_id')}", stored.get("receipt_id"))

    failed: Mapping[str, Any] | None = None
    try:
        if stored is None and scenario.kind == "repair":
            failed = evaluate(client, scenario, scenario.failed_arguments, provider)
        # A backfill evaluates exactly what the forge stores, so the receipt covers it.
        evaluated = dict(stored["repaired_arguments"]) if stored is not None else scenario.arguments
        allowed = evaluate(client, scenario, evaluated, provider)
    except (httpx.HTTPError, ValueError) as error:
        return Result(scenario, "ERROR", f"evaluation failed: {error}")
    if failed is not None and (failed.get("allowed") or not failed.get("receipt_verified")):
        return Result(scenario, "SKIPPED", "failed call was not a verified BLOCK")
    receipt, reason = verified_allow_receipt(allowed)
    if receipt is None:
        return Result(scenario, "SKIPPED", reason)

    exemplar = (
        backfill_exemplar(scenario, stored, receipt)
        if stored is not None
        else build_exemplar(scenario, receipt, allowed, failed)
    )
    oversized = [name for name, limit in FORGE_TEXT_LIMITS.items() if len(getattr(exemplar, name)) > limit]
    if oversized:
        return Result(scenario, "SKIPPED", f"exceeds forge limits: {', '.join(oversized)}", exemplar.receipt_id)
    if dry_run:
        action = "would backfill signature" if stored is not None else "would create"
        return Result(scenario, "VERIFIED (dry run)", f"{action}; not posted", exemplar.receipt_id)

    try:
        store.record_correction(exemplar)  # raises on anything other than 200, 201, or 409
        readback = fetch_stored(store.base_url, scenario)
    except (httpx.HTTPError, RuntimeError, PermissionError, ValueError) as error:
        return Result(scenario, "ERROR", str(error), exemplar.receipt_id)
    problem = "record not found on readback" if readback is None else signature_problem(readback, str(exemplar.receipt_id))
    if problem:
        return Result(scenario, "ERROR", f"posted, but readback failed: {problem}", exemplar.receipt_id)
    status = "BACKFILLED" if stored is not None else "CREATED"
    return Result(scenario, status, "signature and canonical_payload verified on readback", exemplar.receipt_id)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dry-run", action="store_true", help="evaluate and verify only; post nothing")
    parser.add_argument("--forge-url", default=DEFAULT_FORGE_URL, help="ramen forge base URL")
    args = parser.parse_args()

    credentials = load_credentials()
    if not credentials.get("RAMEN_API_KEY"):
        raise RuntimeError("RAMEN_API_KEY must be supplied through the environment or a .env file")
    forge_token = credentials.get("FORGE_WRITE_TOKEN")
    if not forge_token and not args.dry_run:
        raise RuntimeError("FORGE_WRITE_TOKEN is required to post; use --dry-run to evaluate only")
    check_trust_root()
    provider: dict[str, str] = {}
    if credentials.get("OPENAI_API_KEY"):
        provider = {"provider_key": credentials["OPENAI_API_KEY"], "provider_name": "openai"}

    print(f"ramen-ai provider mode: {'BYOK (openai)' if provider else 'Enterprise managed-provider'}")
    print(f"ramen forge: {args.forge_url} ({'dry run' if args.dry_run else 'posting enabled'})\n")

    results: list[Result] = []
    with RamenClient(credentials["RAMEN_API_KEY"]) as client:
        for scenario in SCENARIOS:
            store = RemoteForgeMemoryStore(base_url=args.forge_url, write_token=forge_token, domain=scenario.domain)
            result = ingest(client, scenario, provider, store, args.dry_run)
            results.append(result)
            print(f"[{result.status}] {scenario.label}" + (f": {result.detail}" if result.detail else ""))

    print(f"\n{'Scenario':<42}{'Kind':<11}{'Receipt ID':<38}Forge status")
    for result in results:
        print(f"{result.scenario.label:<42}{result.scenario.kind:<11}{result.receipt_id or '-':<38}{result.status}")
    succeeded = sum(result.ok for result in results)
    print(f"\n{succeeded} of {len(results)} scenarios verified{' (not posted)' if args.dry_run else ' and ingested'}")
    return 0 if succeeded == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
