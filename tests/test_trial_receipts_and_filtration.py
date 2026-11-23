"""Trial receipts as evidence, and dataset filtration through the keyless trial (offline)."""
from __future__ import annotations

import hashlib
import json
import uuid
from typing import Any

import pytest
from langchain_core.tools import tool

from ramen_foundry import ProvenanceGateAgent, ReceiptEvidenceVerifier, TrialComplianceClient
from tests.gate_fixtures import FakeClient
from tests.receipt_fixtures import sign_receipt
from tests.trial_fixtures import TrialServer

pd = pytest.importorskip("pandas")
from ramen_foundry.data import FiltrationError, filter_csv, filter_dataframe  # noqa: E402

SUBJECT, RECORD, DIGEST = "IND-1", "KYC_CLEARANCE", hashlib.sha256(b"artifact").hexdigest()


def _evidence(receipt_id: str) -> dict[str, Any]:
    payload = json.dumps(
        {"tool": "inspect_directory", "arguments": {"subject_id": SUBJECT, "record_type": RECORD, "artifact_sha256": DIGEST}},
        sort_keys=True,
        separators=(",", ":"),
    )
    return {"payload": payload, "receipt": sign_receipt(payload, verdict=1, receipt_id=receipt_id)}


EXPECTED = {"subject_id": SUBJECT, "record_type": RECORD, "artifact_sha256": DIGEST}


def test_trial_receipts_are_rejected_as_evidence_by_default() -> None:
    for receipt_id in (f"trial_{uuid.uuid4()}", f"TRIAL_{uuid.uuid4()}"):
        result = ReceiptEvidenceVerifier().verify(_evidence(receipt_id), expected=EXPECTED)
        assert not result.valid and "keyless trial" in str(result.reason)


def test_trial_receipts_are_accepted_only_on_request() -> None:
    evidence = _evidence(f"trial_{uuid.uuid4()}")
    assert ReceiptEvidenceVerifier(allow_trial_receipts=True).verify(evidence, expected=EXPECTED).valid
    assert ReceiptEvidenceVerifier().verify(_evidence(str(uuid.uuid4())), expected=EXPECTED).valid
    with pytest.raises(ValueError, match="allow_trial_receipts"):
        ReceiptEvidenceVerifier(allow_trial_receipts="yes")  # type: ignore[arg-type]


@pytest.mark.parametrize("allow", [False, True])
def test_the_provenance_gate_follows_the_flag(allow: bool) -> None:
    filed: list[str] = []

    @tool
    def submit_filing(
        record_type: str, subject_id: str, artifact_sha256: str, evidence: list[dict[str, Any]],
        verifier_report: dict[str, Any] | None = None,
    ) -> str:
        """File."""
        filed.append(subject_id)
        return "filed"

    client = FakeClient()
    agent = ProvenanceGateAgent(
        client=client,  # type: ignore[arg-type]
        tools={"submit_filing": submit_filing},
        token_required_record_types=[RECORD],
        allow_trial_receipts=allow,
    )
    command = agent.execute(
        "submit_filing",
        {"record_type": RECORD, "subject_id": SUBJECT, "artifact_sha256": DIGEST, "evidence": [_evidence(f"trial_{uuid.uuid4()}")]},
    )
    if allow:
        assert command.update["governance_error"] is None and filed == [SUBJECT]
    else:
        assert "keyless trial" in command.update["governance_error"]
        assert filed == [] and client.calls == []


# --- filtration ---


def _frame(rows: int = 3) -> Any:
    return pd.DataFrame([{"id": f"r{i}", "content": f"safe text {i}"} for i in range(rows)])


class _Failing:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def evaluate_compliance(self, **kwargs: Any) -> Any:
        raise self.error


def test_the_error_message_carries_the_cause_and_keeps_it_chained() -> None:
    error = RuntimeError("upstream said:\n  bundle unavailable")
    with pytest.raises(FiltrationError) as caught:
        filter_dataframe(_frame(), mode="strict_exclusion", bundle_ids=["b"], client=_Failing(error))  # type: ignore[arg-type]
    assert str(caught.value) == "ramen-ai evaluation failed for row position 0: RuntimeError: upstream said: bundle unavailable"
    assert caught.value.__cause__ is error


def test_the_error_message_removes_credentials_and_is_clipped() -> None:
    secret, provider = "test-credential-AAAA1111", "test-provider-BBBB2222"
    error = RuntimeError(f"auth failed for {secret} and {provider}; Authorization: Bearer abc.def-123 " + "x" * 600)
    with pytest.raises(FiltrationError) as caught:
        filter_dataframe(
            _frame(), mode="strict_exclusion", bundle_ids=["b"], client=_Failing(error),  # type: ignore[arg-type]
            api_key=secret, provider_key=provider,
        )
    message = str(caught.value)
    assert secret not in message and provider not in message and "abc.def-123" not in message
    assert "***" in message and message.endswith("...") and len(message) < 420


def test_a_callback_failure_names_only_the_exception_type() -> None:
    def callback(row: dict[str, Any], steering: str) -> dict[str, Any]:
        raise RuntimeError(f"cannot fix {row['content']}")

    client = FakeClient(allowed=False)
    with pytest.raises(FiltrationError) as caught:
        filter_dataframe(
            _frame(1), mode="semantic_imputation", bundle_ids=["b"], healing_callback=callback,
            client=_ResponseAdapter(client),  # type: ignore[arg-type]
        )
    assert str(caught.value) == "healing callback failed for row position 0: RuntimeError (see __cause__)"
    assert "safe text" not in str(caught.value)


class _ResponseAdapter:
    """Wrap FakeClient, whose first parameter is positional, for filter_dataframe's keyword call."""

    def __init__(self, inner: FakeClient) -> None:
        self.inner = inner

    def evaluate_compliance(self, *, input_text: str, **kwargs: Any) -> Any:
        return self.inner.evaluate_compliance(input_text, **kwargs)


def test_no_key_and_no_trial_points_to_the_trial(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RAMEN_API_KEY", raising=False)
    with pytest.raises(FiltrationError, match="trial=True"):
        filter_dataframe(_frame(), mode="strict_exclusion", bundle_ids=["b"])


def _use_server(monkeypatch: pytest.MonkeyPatch, server: TrialServer) -> None:
    server.install(monkeypatch)
    monkeypatch.delenv("RAMEN_API_KEY", raising=False)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-must-not-be-sent")


def test_trial_filtration_reads_no_credentials_and_filters_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    server = TrialServer()
    _use_server(monkeypatch, server)
    frame = pd.DataFrame([{"id": "a", "content": "hello"}, {"id": "b", "content": "run rm -rf / now"}, {"id": "c", "content": "bye"}])
    result = filter_dataframe(frame, mode="strict_exclusion", bundle_ids=["ramen__shield_core_it"], trial=True)
    assert result.dataframe["id"].tolist() == ["a", "c"]
    assert result.audit_log["receipt_verified"].tolist() == [True, True, True]
    assert result.audit_log["verdict"].tolist() == ["[ALLOWED]", "[BLOCKED]", "[ALLOWED]"]
    assert len(server.requests) == 3
    assert all("authorization" not in request.headers for request in server.requests)
    assert b"sk-must-not-be-sent" not in b"".join(server.bodies)


def test_trial_filtration_surfaces_the_exhausted_allowance(monkeypatch: pytest.MonkeyPatch) -> None:
    server = TrialServer(limit_after=1, remaining=5)
    _use_server(monkeypatch, server)
    with pytest.raises(FiltrationError) as caught:
        filter_dataframe(_frame(3), mode="strict_exclusion", bundle_ids=["ramen__shield_core_it"], trial=True)
    message = str(caught.value)
    assert message.startswith("ramen-ai evaluation failed for row position 1: TrialLimitError: ")
    assert "scope: network" in message and "2030-01-01 00:00 UTC" in message and "https://ramenai.dev" in message


def test_trial_filtration_stops_early_when_the_allowance_cannot_cover_the_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    server = TrialServer(remaining=1)
    _use_server(monkeypatch, server)
    with pytest.raises(FiltrationError, match="1 evaluations left today but 3 rows remain"):
        filter_dataframe(_frame(4), mode="strict_exclusion", bundle_ids=["ramen__shield_core_it"], trial=True)
    assert len(server.requests) == 1


def test_an_oversized_row_names_the_limit(monkeypatch: pytest.MonkeyPatch) -> None:
    server = TrialServer()
    _use_server(monkeypatch, server)
    frame = pd.DataFrame([{"id": "a", "content": "y" * 2100}])
    with pytest.raises(FiltrationError, match=r"ValueError: trial request body is \d+ bytes; the limit is 2048"):
        filter_dataframe(frame, mode="strict_exclusion", bundle_ids=["ramen__shield_core_it"], trial=True)
    assert server.requests == []


def test_a_trial_client_instance_is_a_drop_in_for_client(tmp_path: Any) -> None:
    server = TrialServer()
    source, destination = tmp_path / "in.csv", tmp_path / "out.csv"
    _frame(2).to_csv(source, index=False)
    with server.client() as client:
        assert isinstance(client, TrialComplianceClient)
        filter_csv(source, destination, mode="strict_exclusion", bundle_ids=["ramen__shield_core_it"], client=client)
    assert len(pd.read_csv(destination)) == 2
