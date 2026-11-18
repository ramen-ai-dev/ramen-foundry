"""Dual-mode DataFrame and CSV filtration through the ramen ai API.

Moved from the ramen-data-filter package. The evaluation context keeps the
integration tag ramen-data-filter so existing audit trails stay continuous.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
from ramen_ai import RamenClient


class FiltrationMode(str, Enum):
    """Supported dataset filtration modes."""

    STRICT_EXCLUSION = "strict_exclusion"
    SEMANTIC_IMPUTATION = "semantic_imputation"


class FiltrationError(RuntimeError):
    """Raised when a dataset cannot be evaluated or safely transformed."""


@dataclass(frozen=True)
class FiltrationResult:
    """Filtered data plus row-level evaluation and transformation records."""

    dataframe: pd.DataFrame
    audit_log: pd.DataFrame
    imputation_log: pd.DataFrame


_AUDIT_COLUMNS = (
    "row_position",
    "row_index",
    "allowed",
    "verdict",
    "steering",
    "receipt_verified",
    "policy_ids",
    "response",
)
_IMPUTATION_COLUMNS = (
    "row_position",
    "row_index",
    "columns_changed",
    "steering",
)


def filter_dataframe(
    dataframe: pd.DataFrame,
    *,
    mode: FiltrationMode | str,
    bundle_ids: Sequence[str] | None = None,
    policy_ids: Sequence[str] | None = None,
    remediable_columns: Sequence[str] | None = None,
    healing_callback: Callable[[dict[str, Any], str], dict[str, Any]] | None = None,
    client: RamenClient | None = None,
    api_key: str | None = None,
    provider_key: str | None = None,
    provider_name: str | None = None,
    context: dict[str, str] | None = None,
    base_url: str = "https://api.ramenai.dev",
    timeout: float = 30.0,
) -> FiltrationResult:
    """Evaluate and filter a DataFrame row by row.

    Strict exclusion retains only rows with an allowed verdict. Semantic
    imputation calls ``healing_callback`` for each blocked JSON row and keeps
    the callback's cured row. Without a callback, semantic mode safely falls
    back to strict exclusion. ``remediable_columns`` optionally constrains
    which values the callback may change.

    API, transport, and callback failures stop processing. Partial output is
    never returned or written.
    """
    if not isinstance(dataframe, pd.DataFrame):
        raise TypeError("dataframe must be a pandas DataFrame")
    if not dataframe.columns.is_unique:
        raise ValueError("dataframe columns must be unique")
    try:
        selected_mode = FiltrationMode(mode)
    except ValueError as exc:
        supported = ", ".join(item.value for item in FiltrationMode)
        raise ValueError(f"mode must be one of: {supported}") from exc

    selected_bundles = _validated_identifiers("bundle_ids", bundle_ids)
    selected_policies = _validated_identifiers("policy_ids", policy_ids)
    if not selected_bundles and not selected_policies:
        raise ValueError("Provide at least one of 'bundle_ids' or 'policy_ids'.")

    allowed_columns = _validated_columns(dataframe, remediable_columns)
    if healing_callback is not None and not callable(healing_callback):
        raise TypeError("healing_callback must be callable or None")
    if (
        selected_mode is FiltrationMode.SEMANTIC_IMPUTATION
        and healing_callback is not None
        and not all(isinstance(column, str) for column in dataframe.columns)
    ):
        raise ValueError("semantic healing requires string dataframe columns")

    evaluation_context = dict(context or {})
    if not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in evaluation_context.items()
    ):
        raise TypeError("context keys and values must be strings")
    evaluation_context["integration"] = "ramen-data-filter"
    evaluation_context.setdefault("source", "dataframe")

    owned_client: RamenClient | None = None
    active_client = client
    if active_client is None:
        resolved_api_key = api_key or os.environ.get("RAMEN_API_KEY", "")
        if not resolved_api_key:
            raise FiltrationError(
                "RAMEN_API_KEY is not set; dataset filtration cannot be evaluated"
            )
        owned_client = RamenClient(
            api_key=resolved_api_key,
            base_url=base_url,
            timeout=timeout,
        )
        active_client = owned_client

    records: list[dict[str, Any]] = []
    try:
        for position, (row_index, row) in enumerate(dataframe.iterrows()):
            serialized_row = json.dumps(
                row.to_dict(), sort_keys=True, default=str
            )
            input_row = json.loads(serialized_row)
            try:
                response = active_client.evaluate_compliance(
                    input_text=serialized_row,
                    bundle_ids=selected_bundles or None,
                    policy_ids=selected_policies or None,
                    context=evaluation_context,
                    provider_key=(
                        provider_key
                        if provider_key is not None
                        else os.environ.get("OPENAI_API_KEY")
                    ),
                    provider_name=provider_name,
                )
            except Exception as exc:
                raise FiltrationError(
                    f"ramen-ai evaluation failed for row position {position}"
                ) from exc

            if not isinstance(response, Mapping):
                raise FiltrationError(
                    f"ramen-ai response was not a mapping for row position {position}"
                )
            allowed = response.get("allowed")
            if not isinstance(allowed, bool):
                raise FiltrationError(
                    "ramen-ai response omitted a boolean allowed verdict "
                    f"for row position {position}"
                )
            steering = response.get("steering")
            if steering is not None and not isinstance(steering, str):
                raise FiltrationError(
                    f"ramen-ai steering was not text for row position {position}"
                )
            receipt_verified = response.get("receipt_verified", False)
            if not isinstance(receipt_verified, bool):
                raise FiltrationError(
                    "ramen-ai receipt verification state was not boolean "
                    f"for row position {position}"
                )
            resolved_policy_ids = response.get("policy_ids", [])
            if not isinstance(resolved_policy_ids, list) or not all(
                isinstance(policy_id, str) and policy_id
                for policy_id in resolved_policy_ids
            ):
                raise FiltrationError(
                    "ramen-ai policy_ids was not a list of non-empty strings "
                    f"for row position {position}"
                )
            records.append(
                {
                    "row_position": position,
                    "row_index": row_index,
                    "allowed": allowed,
                    "verdict": "[ALLOWED]" if allowed else "[BLOCKED]",
                    "steering": steering,
                    "receipt_verified": receipt_verified,
                    "policy_ids": resolved_policy_ids.copy(),
                    "response": dict(response),
                    "input_row": input_row,
                }
            )
    finally:
        if owned_client is not None:
            owned_client.close()

    audit_log = pd.DataFrame(records, columns=_AUDIT_COLUMNS)
    allowed_positions = [
        record["row_position"] for record in records if record["allowed"]
    ]

    if (
        selected_mode is FiltrationMode.STRICT_EXCLUSION
        or healing_callback is None
    ):
        filtered = dataframe.iloc[allowed_positions].copy().reset_index(drop=True)
        return FiltrationResult(
            dataframe=filtered,
            audit_log=audit_log,
            imputation_log=pd.DataFrame(columns=_IMPUTATION_COLUMNS),
        )

    healed, imputation_log = _heal_blocked_rows(
        dataframe=dataframe,
        records=records,
        remediable_columns=allowed_columns,
        healing_callback=healing_callback,
    )
    return FiltrationResult(
        dataframe=healed,
        audit_log=audit_log,
        imputation_log=imputation_log,
    )


def filter_csv(
    source_path: str | Path,
    destination_path: str | Path,
    **kwargs: Any,
) -> FiltrationResult:
    """Read a CSV, filter it through :func:`filter_dataframe`, and write output."""
    source = Path(source_path)
    destination = Path(destination_path)
    dataframe = pd.read_csv(source)
    context = dict(kwargs.pop("context", {}) or {})
    context["source"] = "csv"
    result = filter_dataframe(dataframe, context=context, **kwargs)

    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
        result.dataframe.to_csv(temporary_path, index=False)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return result


def _validated_identifiers(
    name: str, values: Sequence[str] | None
) -> list[str]:
    if values is None:
        return []
    if isinstance(values, (str, bytes)):
        raise TypeError(f"{name} must be a sequence of non-empty strings")
    normalized = list(values)
    if not all(isinstance(value, str) and value.strip() for value in normalized):
        raise ValueError(f"{name} must contain only non-empty strings")
    return normalized


def _validated_columns(
    dataframe: pd.DataFrame, columns: Sequence[str] | None
) -> list[str]:
    if columns is None:
        return []
    if isinstance(columns, (str, bytes)):
        raise TypeError("remediable_columns must be a sequence of column names")
    normalized = list(dict.fromkeys(columns))
    if not all(isinstance(column, str) and column for column in normalized):
        raise ValueError("remediable_columns must contain non-empty strings")
    missing = [column for column in normalized if column not in dataframe.columns]
    if missing:
        raise ValueError(f"remediable columns are missing from dataframe: {missing}")
    casefolded = [column.casefold() for column in normalized]
    if len(casefolded) != len(set(casefolded)):
        raise ValueError("remediable column names must be unique ignoring case")
    return normalized


def _heal_blocked_rows(
    *,
    dataframe: pd.DataFrame,
    records: list[dict[str, Any]],
    remediable_columns: list[str],
    healing_callback: Callable[[dict[str, Any], str], dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    healed = dataframe.copy()
    log_records: list[dict[str, Any]] = []

    for record in records:
        if record["allowed"]:
            continue

        position = record["row_position"]
        original_row = deepcopy(record["input_row"])
        steering = record["steering"] or ""
        try:
            cured_row = healing_callback(deepcopy(original_row), steering)
        except Exception as exc:
            raise FiltrationError(
                f"healing callback failed for row position {position}"
            ) from exc

        if not isinstance(cured_row, dict):
            raise FiltrationError(
                "healing callback must return a dict "
                f"for row position {position}"
            )
        try:
            cured_row = json.loads(
                json.dumps(cured_row, sort_keys=True, allow_nan=False)
            )
        except (TypeError, ValueError) as exc:
            raise FiltrationError(
                "healing callback must return a JSON-compatible row "
                f"for row position {position}"
            ) from exc
        if set(cured_row) != set(original_row):
            raise FiltrationError(
                "healing callback must preserve the row schema "
                f"for row position {position}"
            )

        changed_columns = tuple(
            column
            for column in dataframe.columns
            if not _json_values_equal(original_row[column], cured_row[column])
        )
        if not changed_columns:
            raise FiltrationError(
                f"healing callback did not change blocked row position {position}"
            )

        disallowed = [
            column
            for column in changed_columns
            if remediable_columns and column not in remediable_columns
        ]
        if disallowed:
            raise FiltrationError(
                "healing callback changed columns outside remediable_columns "
                f"at row position {position}: {disallowed}"
            )

        for column in changed_columns:
            column_position = dataframe.columns.get_loc(column)
            healed.iat[position, column_position] = cured_row[column]
            stored_value = healed.iat[position, column_position]
            if not _json_values_equal(stored_value, cured_row[column]):
                raise FiltrationError(
                    "dataframe could not preserve healed value for column "
                    f"'{column}' at row position {position}"
                )

        log_records.append(
            {
                "row_position": position,
                "row_index": record["row_index"],
                "columns_changed": changed_columns,
                "steering": steering,
            }
        )

    return healed, pd.DataFrame(log_records, columns=_IMPUTATION_COLUMNS)


def _json_values_equal(left: Any, right: Any) -> bool:
    return _json_comparison_value(left) == _json_comparison_value(right)


def _json_comparison_value(value: Any) -> str:
    item = getattr(value, "item", None)
    if callable(item):
        try:
            value = item()
        except (TypeError, ValueError):
            pass
    return json.dumps(value, sort_keys=True, default=str)
