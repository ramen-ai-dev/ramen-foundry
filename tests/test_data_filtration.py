"""Isolated tests for ramen-ai dataset filtration."""

from unittest.mock import Mock

import pytest

pd = pytest.importorskip("pandas")

from ramen_ai import RamenClient
from ramen_foundry.data import (
    FiltrationError,
    FiltrationMode,
    filter_csv,
    filter_dataframe,
)


def _dataset() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"record_id": "safe-1", "email": "clean@example.com", "risk_score": 10},
            {"record_id": "blocked", "email": "export@attacker.test", "risk_score": 99},
            {"record_id": "safe-2", "email": "review@example.com", "risk_score": 30},
        ]
    )


def _mock_client() -> Mock:
    client = Mock(spec=RamenClient)
    client.evaluate_compliance.side_effect = [
        {
            "allowed": True,
            "steering": None,
            "receipt_verified": True,
            "policy_ids": ["policy-1"],
            "data": {},
        },
        {
            "allowed": False,
            "steering": "Replace email and risk_score before ingestion.",
            "receipt_verified": True,
            "policy_ids": ["policy-1"],
            "data": {},
        },
        {
            "allowed": True,
            "steering": None,
            "receipt_verified": True,
            "policy_ids": ["policy-1"],
            "data": {},
        },
    ]
    return client


def test_strict_exclusion_removes_blocked_rows() -> None:
    client = _mock_client()

    result = filter_dataframe(
        _dataset(),
        mode=FiltrationMode.STRICT_EXCLUSION,
        policy_ids=["policy-1"],
        client=client,
    )

    assert result.dataframe["record_id"].tolist() == ["safe-1", "safe-2"]
    assert result.dataframe.index.tolist() == [0, 1]
    assert result.audit_log["verdict"].tolist() == [
        "[ALLOWED]",
        "[BLOCKED]",
        "[ALLOWED]",
    ]
    assert result.imputation_log.empty
    assert client.evaluate_compliance.call_count == 3


def test_semantic_imputation_uses_callback_for_blocked_row() -> None:
    client = _mock_client()
    source = _dataset()

    def heal(row: dict, steering: str) -> dict:
        assert steering == "Replace email and risk_score before ingestion."
        return {
            **row,
            "email": "redacted@example.invalid",
            "risk_score": (10 + 30) // 2,
        }

    healing_callback = Mock(side_effect=heal)
    result = filter_dataframe(
        source,
        mode=FiltrationMode.SEMANTIC_IMPUTATION,
        policy_ids=["policy-1"],
        remediable_columns=["email", "risk_score"],
        healing_callback=healing_callback,
        client=client,
    )

    assert len(result.dataframe) == len(source)
    assert result.dataframe.loc[1, "email"] == "redacted@example.invalid"
    assert result.dataframe.loc[1, "risk_score"] == 20
    assert result.dataframe.loc[1, "record_id"] == "blocked"
    pd.testing.assert_series_equal(result.dataframe.loc[0], source.loc[0])
    pd.testing.assert_series_equal(result.dataframe.loc[2], source.loc[2])
    assert result.imputation_log.loc[0, "columns_changed"] == (
        "email",
        "risk_score",
    )
    healing_callback.assert_called_once_with(
        {
            "email": "export@attacker.test",
            "record_id": "blocked",
            "risk_score": 99,
        },
        "Replace email and risk_score before ingestion.",
    )
    assert client.evaluate_compliance.call_count == 3


def test_duplicate_columns_are_rejected_before_evaluation() -> None:
    client = Mock(spec=RamenClient)
    source = pd.DataFrame(
        [["visible", "hidden"]], columns=["content", "content"]
    )

    with pytest.raises(ValueError, match="columns must be unique"):
        filter_dataframe(
            source,
            mode=FiltrationMode.STRICT_EXCLUSION,
            policy_ids=["policy-1"],
            client=client,
        )

    client.evaluate_compliance.assert_not_called()


def test_semantic_imputation_without_callback_falls_back_to_exclusion() -> None:
    client = _mock_client()

    result = filter_dataframe(
        _dataset(),
        mode=FiltrationMode.SEMANTIC_IMPUTATION,
        policy_ids=["policy-1"],
        client=client,
    )

    assert result.dataframe["record_id"].tolist() == ["safe-1", "safe-2"]
    assert result.imputation_log.empty


def test_healing_callback_failure_is_wrapped_and_fails_closed() -> None:
    client = _mock_client()

    def fail_to_heal(row: dict, steering: str) -> dict:
        raise RuntimeError("healing service unavailable")

    with pytest.raises(
        FiltrationError, match="healing callback failed for row position 1"
    ) as error:
        filter_dataframe(
            _dataset(),
            mode=FiltrationMode.SEMANTIC_IMPUTATION,
            policy_ids=["policy-1"],
            healing_callback=fail_to_heal,
            client=client,
        )

    assert isinstance(error.value.__cause__, RuntimeError)
    assert str(error.value.__cause__) == "healing service unavailable"


def test_malformed_audit_metadata_is_rejected() -> None:
    client = Mock(spec=RamenClient)
    client.evaluate_compliance.return_value = {
        "allowed": True,
        "steering": None,
        "receipt_verified": "false",
        "policy_ids": ["policy-1"],
    }

    with pytest.raises(FiltrationError, match="verification state"):
        filter_dataframe(
            pd.DataFrame([{"content": "safe"}]),
            mode=FiltrationMode.STRICT_EXCLUSION,
            policy_ids=["policy-1"],
            client=client,
        )


def test_csv_write_failure_preserves_existing_destination(
    tmp_path, monkeypatch
) -> None:
    source_path = tmp_path / "source.csv"
    destination_path = tmp_path / "destination.csv"
    pd.DataFrame([{"content": "safe"}]).to_csv(source_path, index=False)
    destination_path.write_text("existing-output\n", encoding="utf-8")

    client = Mock(spec=RamenClient)
    client.evaluate_compliance.return_value = {
        "allowed": True,
        "steering": None,
        "receipt_verified": True,
        "policy_ids": ["policy-1"],
    }

    def fail_to_csv(*args, **kwargs) -> None:
        raise OSError("disk full")

    monkeypatch.setattr(pd.DataFrame, "to_csv", fail_to_csv)

    with pytest.raises(OSError, match="disk full"):
        filter_csv(
            source_path,
            destination_path,
            mode=FiltrationMode.STRICT_EXCLUSION,
            policy_ids=["policy-1"],
            client=client,
        )

    assert destination_path.read_text(encoding="utf-8") == "existing-output\n"
    assert not list(tmp_path.glob(f".{destination_path.name}.*.tmp"))


def test_healing_callback_can_mutate_nested_copy() -> None:
    client = Mock(spec=RamenClient)
    client.evaluate_compliance.return_value = {
        "allowed": False,
        "steering": "Redact metadata before ingestion.",
        "receipt_verified": True,
        "policy_ids": ["policy-1"],
    }
    source = pd.DataFrame(
        [{"record_id": "blocked", "metadata": {"secret": "value"}}]
    )

    def heal(row: dict, steering: str) -> dict:
        row["metadata"]["secret"] = "[REDACTED]"
        return row

    result = filter_dataframe(
        source,
        mode=FiltrationMode.SEMANTIC_IMPUTATION,
        policy_ids=["policy-1"],
        remediable_columns=["metadata"],
        healing_callback=heal,
        client=client,
    )

    assert result.dataframe.loc[0, "metadata"] == {"secret": "[REDACTED]"}
    assert source.loc[0, "metadata"] == {"secret": "value"}
    assert result.imputation_log.loc[0, "columns_changed"] == ("metadata",)


def test_healing_callback_rejects_non_json_output() -> None:
    client = Mock(spec=RamenClient)
    client.evaluate_compliance.return_value = {
        "allowed": False,
        "steering": "Replace content before ingestion.",
        "receipt_verified": True,
        "policy_ids": ["policy-1"],
    }

    def heal(row: dict, steering: str) -> dict:
        return {**row, "content": {"not-json"}}

    with pytest.raises(FiltrationError, match="JSON-compatible row"):
        filter_dataframe(
            pd.DataFrame([{"content": "blocked"}]),
            mode=FiltrationMode.SEMANTIC_IMPUTATION,
            policy_ids=["policy-1"],
            healing_callback=heal,
            client=client,
        )


def test_requests_keep_the_ramen_data_filter_integration_tag() -> None:
    client = Mock(spec=RamenClient)
    client.evaluate_compliance.return_value = {
        "allowed": True,
        "steering": None,
        "receipt_verified": True,
        "policy_ids": ["1006492f-db62-4f46-8775-48b966c5c956"],
    }
    filter_dataframe(
        _dataset(),
        mode=FiltrationMode.STRICT_EXCLUSION,
        bundle_ids=["ramen__shield_core_it"],
        client=client,
    )
    contexts = [call.kwargs["context"] for call in client.evaluate_compliance.call_args_list]
    assert len(contexts) == 3
    assert all(context["integration"] == "ramen-data-filter" for context in contexts)
