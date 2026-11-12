"""Context-complete payload builders for the live suite (synthetic, fictional data only)."""

from __future__ import annotations

import copy
import hashlib
import json
from typing import Any

# ------------------------------------------------------------------ authorization


def authorization(amount: float = 12_000.0, **overrides: Any) -> dict[str, Any]:
    """A compliant spend with complete platform context (no dual control needed)."""

    payload: dict[str, Any] = {
        "request_id": "REQ-LIVE-0001",
        "requested_by": "lab-agent-01",
        "amount_usd": amount,
        "payee": "Tenbury Scientific",
        "payee_account": "ACCT-TEN-4401",
        "cost_center": "Lab Operations",
        "fiscal_period": "FY2026-Q4",
        "spend_category": "supplies",
        "platform_limits": {
            "source": "finance platform",
            "per_order_usd": 30_000,
            "per_vendor_cumulative_usd": 90_000,
            "dual_approval_threshold_usd": 50_000,
        },
        "erp_clearance": {
            "id": "BC-7701",
            "status": "OPEN",
            "cost_center": "Lab Operations",
            "fiscal_period": "FY2026-Q4",
            "spend_category": "supplies",
            "balance_usd": 14_000,
            "confirmed_by": "ERP system of record",
        },
        "vendor_exposure": {
            "source": "finance platform",
            "vendor_group": "Tenbury Scientific",
            "open_pending_paid_usd": 21_000,
        },
        "approvals": [],
    }
    payload.update(overrides)
    return payload


def approval(approver: str, amount: float, *, role: str = "Finance Controller") -> dict[str, Any]:
    return {
        "approver_id": approver,
        "role": role,
        "approval_token": f"APR-{approver}",
        "approved_out_of_band": True,
        "bound_to": {"amount_usd": amount, "payee": "Tenbury Scientific", "payee_account": "ACCT-TEN-4401"},
    }


def authorization_dual(amount: float = 40_000.0, **overrides: Any) -> dict[str, Any]:
    """A spend at or above the dual-approval threshold, with two bound out-of-band approvals."""

    payload = authorization(amount)
    payload["platform_limits"] = {
        **payload["platform_limits"],
        "per_order_usd": 100_000,
        "per_vendor_cumulative_usd": 500_000,
        "dual_approval_threshold_usd": 25_000,
    }
    payload["erp_clearance"] = {**payload["erp_clearance"], "balance_usd": 80_000}
    payload["approvals"] = [approval("fin-ctrl-1", amount), approval("fin-dir-2", amount, role="CFO")]
    payload.update(overrides)
    return payload


# ------------------------------------------------------------------- eligibility


def eligibility_facts(**overrides: Any) -> dict[str, Any]:
    """Host-rule facts plus the complete platform-confirmed claim evidence."""

    facts: dict[str, Any] = {
        "claim_id": "CLM-9042",
        "loss_in_term": True,
        "policy_status": "ACTIVE",
        "exclusion_flag": False,
        "policy_system_record": {
            "effective": "2026-01-01",
            "expiry": "2026-12-31",
            "status": "ACTIVE",
            "date_of_loss": "2026-08-04",
            "confirmed_by": "policy system of record",
        },
        "inspection_evidence": {
            "token": "SS-9042",
            "issued_out_of_band_by": "independent inspection network",
            "verified_by_platform": True,
            "inspection_date": "2026-08-10",
            "scope": "units 12 to 20",
            "assessed_usd": 63_000,
        },
        "platform_schedule": {
            "source": "policy platform",
            "unit_block_limit_usd": 150_000,
            "remaining_aggregate_usd": 150_000,
            "deductible_usd": 0,
            "drawn_usd": 0,
        },
        "verified_file_findings": "File notes no excluded cause.",
    }
    facts.update(overrides)
    return facts


def claim(facts: dict[str, Any] | None = None, amount: float = 63_000.0) -> dict[str, Any]:
    return {
        "subject_id": "CLM-9042",
        "approved_amount_usd": amount,
        "eligibility_facts": copy.deepcopy(facts if facts is not None else eligibility_facts()),
    }


EXCLUDED_FILE_FINDINGS = (
    "Verified file shows an unauthorised structural alteration of the sprinkler riser "
    "before the loss, a triggered exclusion."
)

# ------------------------------------------------------------------- provenance

ARTIFACT_SHA256 = hashlib.sha256(b"kyc-onboarding-record-IND-5120-v1").hexdigest()
RECORD_TYPE = "KYC_CLEARANCE"
SUBJECT_ID = "IND-5120"


def evidence_call(step: str) -> str:
    """The exact text evaluated to mint one evidence receipt for the submission."""

    return json.dumps(
        {
            "tool": "inspect_directory",
            "arguments": {
                "path": "./build",
                "step": step,
                "subject_id": SUBJECT_ID,
                "record_type": RECORD_TYPE,
                "artifact_sha256": ARTIFACT_SHA256,
            },
        },
        sort_keys=True,
        separators=(",", ":"),
    )
