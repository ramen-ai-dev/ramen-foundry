# Dataset filtration (`ramen_foundry.data`)

Dual-mode pandas and CSV filtration for RAG ingestion scrubbing and MLOps
dataset sanitization. Every row is evaluated through the ramen ai semantic
firewall before it can enter a retrieval corpus, training set, analytics store,
or downstream data pipeline.

Use **Strict Exclusion** to remove blocked rows, or provide an optional
**Semantic Imputation (Healing)** callback to cure blocked rows under your own
runtime and model choices. Without a callback, blocked rows are always dropped.
Typical policies can detect PII, data-exfiltration vectors, poisoned retrieval
context, prompt injection, and other unsafe dataset content.

This module moved here from the `ramen-data-filter` package. The import path is
now `ramen_foundry.data`, and the API is unchanged. Requests keep the evaluation
context tag `integration: "ramen-data-filter"` so existing audit trails stay
continuous.

Requires Python 3.10 or newer.

## Installation

pandas is an optional extra, so installs that only need the governance nodes do
not pull it in:

```bash
pip install "ramen-foundry[data]"
```

`import ramen_foundry.data` raises an `ImportError` that names this command when
pandas is missing. It does not import LangGraph.

## API key

Get a ramen ai API key at [ramenai.dev/pricing](https://ramenai.dev/pricing). The
Free Starter tier (1,000 evaluations per month, BYOK) includes the Core IT
Security bundle.

## Configuration / BYOK

```bash
export RAMEN_API_KEY=ramen_ak_...   # required, except with trial=True
export OPENAI_API_KEY=sk-...        # Starter/Professional BYOK
```

`RAMEN_API_KEY` authenticates the evaluation request. The Starter and
Professional tiers also require an LLM provider key. The module reads
`OPENAI_API_KEY` at evaluation time and `ramen-ai-core` forwards it as the
`X-Provider-Key` header. Without a provider key, those tiers return
`402 Payment Required`.

Pass `provider_name="anthropic"`, `"google"`, `"synthetic"`, or
`"hyperbolic"` with the corresponding `provider_key` when OpenAI is not the
provider. Enterprise tiers use platform-managed provider keys and do not need
`OPENAI_API_KEY`.

Credentials are read from environment variables or passed at runtime. Never
store them in source code, notebooks, CSV files, or committed configuration.

---

## Usage

### Strict Exclusion

Strict exclusion evaluates each row and returns a new DataFrame containing only
`[ALLOWED]` rows. The returned index is reset; the source DataFrame is not
modified.

```python
import pandas as pd
from ramen_foundry.data import FiltrationMode, filter_dataframe

records = pd.DataFrame(
    [
        {"document_id": "doc-1", "content": "Approved public documentation"},
        {"document_id": "doc-2", "content": "Retrieved untrusted context"},
    ]
)

result = filter_dataframe(
    records,
    mode=FiltrationMode.STRICT_EXCLUSION,
    bundle_ids=["ramen__shield_core_it"],
)

clean_records = result.dataframe
print(result.audit_log[["row_index", "verdict", "receipt_verified"]])
```

### Optional Semantic Imputation (Healing)

Mode B is opt-in. Supply a callback with this contract:

```python
Callable[[dict, str], dict]
```

The callback receives a copy of the blocked JSON row and the ramen ai steering
string, then returns the cured JSON row. `ramen_foundry.data` does not call an LLM
framework itself: the callback can use deterministic rules, an internal
service, or a model chosen by the operator.

The returned dictionary must preserve the original keys and change at least one
value. When `remediable_columns` is supplied, changing any other column fails
closed. Allowed rows remain byte-for-value unchanged.

```python
from ramen_foundry.data import FiltrationMode, filter_dataframe


def heal_record(row: dict, steering: str) -> dict:
    cured = dict(row)
    cured["content"] = "[REDACTED BY DATA POLICY]"
    return cured


result = filter_dataframe(
    records,
    mode=FiltrationMode.SEMANTIC_IMPUTATION,
    policy_ids=["<POLICY_UUID>"],
    remediable_columns=["content"],
    healing_callback=heal_record,
)

sanitized_records = result.dataframe
print(result.imputation_log)
```

If `healing_callback` is omitted, semantic mode defaults to Strict Exclusion and
drops every blocked row. If the callback raises, returns a non-dictionary,
changes the schema, makes no change, or modifies a disallowed column, the
operation raises `FiltrationError` and returns no partial result.

For continuous training workloads, see the
**[Just-In-Time Training Pipeline](./data-jit-training-pipeline.md)** guide for
PyTorch `DataLoader` and TensorFlow `tf.data.Dataset` prefetch patterns.

### CSV pipeline

```python
from ramen_foundry.data import FiltrationMode, filter_csv

result = filter_csv(
    "rag-ingestion.csv",
    "rag-ingestion-sanitized.csv",
    mode=FiltrationMode.STRICT_EXCLUSION,
    bundle_ids=["ramen__shield_core_it"],
)
```

The destination is written only after every row has been evaluated and the
transformation succeeds. Existing destination files are replaced by Pandas.

---

## How it works

1. Serializes each row as stable, key-sorted JSON.
2. Calls `RamenClient.evaluate_compliance` with configured bundles or policies.
3. Records the verdict, steering, resolved policy IDs, receipt verification
   state, and complete SDK response in `audit_log`.
4. In Strict Exclusion mode, drops blocked rows. In Semantic Imputation mode,
   drops blocked rows when no callback is configured; otherwise passes each
   blocked JSON row and steering string to the operator's healing callback.
5. Validates callback output and returns a new DataFrame plus transformation
   metadata; CSV usage writes that DataFrame atomically with `index=False`.

The pipeline is fail-closed. Missing credentials, transport failures, malformed
responses, and callback failures raise `FiltrationError`; partial filtered
output is never returned or written.

---

## API reference

### `filter_dataframe(dataframe, *, mode, ...) -> FiltrationResult`

| Parameter | Type | Description |
|---|---|---|
| `dataframe` | `pandas.DataFrame` | Source records. The input is not modified. |
| `mode` | `FiltrationMode \| str` | `strict_exclusion` or `semantic_imputation`. |
| `bundle_ids` | `Sequence[str]` | ramen ai bundle slugs. At least one bundle or policy is required. |
| `policy_ids` | `Sequence[str]` | Explicit policy UUIDs. May be combined with bundles. |
| `remediable_columns` | `Sequence[str]` | Optional guard limiting which columns the healing callback may change. |
| `healing_callback` | `Callable[[dict, str], dict] \| None` | Optional blocked-row healer. Without it, blocked rows are strictly excluded. |
| `client` | `RamenClient \| TrialComplianceClient` | Optional injected client, primarily for controlled runtimes and tests. |
| `api_key` | `str` | Runtime override; defaults to `RAMEN_API_KEY`. |
| `trial` | `bool` | `True` evaluates on the keyless public trial instead of an API key (see below). Default `False`. |
| `provider_key` | `str` | BYOK override; defaults to `OPENAI_API_KEY`. |
| `provider_name` | `str` | Optional BYOK provider routing hint. |
| `context` | `dict[str, str]` | Additional audit metadata. |
| `base_url` | `str` | ramen ai API base URL. |
| `timeout` | `float` | HTTP timeout in seconds. |

### Keyless trial

With no API key, `trial=True` (or `python examples/filter_dataset.py --trial`) sends each row to the public trial (`RamenClient.trial()`, `POST /api/v1/paas/try`) through `ramen_foundry.TrialComplianceClient`. It reads no credentials and sends none. The trial evaluates one bundle per request, allows about 10 requests per day per network, and rejects a request body over 2,048 bytes, so a row must serialize to well under 2,000 bytes once JSON-escaped. `policy_ids` raise `ValueError`; `context`, `provider_key` and `provider_name` are not sent. After the first row, filtration stops early if the remaining allowance cannot cover the rest of the file.

### Errors

`FiltrationError` messages name the failing row position and the underlying cause, for example `ramen-ai evaluation failed for row position 3: TrialLimitError: The keyless trial allowance is used up (scope: network). It resets at ...`. The cause text is clipped to 300 characters and has credentials removed; the original exception stays available as `__cause__`. A healing-callback failure names only the exception type, because callback messages often quote the row.

### `filter_csv(source_path, destination_path, **kwargs) -> FiltrationResult`

Reads the source with `pandas.read_csv`, applies `filter_dataframe`, and writes
the resulting DataFrame with `index=False`.

### `FiltrationResult`

| Attribute | Description |
|---|---|
| `dataframe` | Filtered or imputed DataFrame. |
| `audit_log` | One row per evaluation with verdict and SDK response metadata. |
| `imputation_log` | One row per transformed record; empty in strict mode. |

Public exports: `FiltrationError`, `FiltrationMode`, `FiltrationResult`,
`filter_dataframe`, and `filter_csv`.

---

## Available bundles

| Bundle slug | Useful dataset coverage |
|---|---|
| `ramen__shield_core_it` | Secret exfiltration, prompt leakage, jailbreaks, destructive instructions, and indirect prompt injection. |
| `ramen__eu_ai_act_baseline` | EU AI Act data-governance, prohibited-practice, and transparency controls. |
| `ramen__fintech_banking_invariance` | Credit adverse action notices and wire or payment instructions in financial datasets. |
| `ramen__industrial_iot_actuation_invariance` | Actuation and control instructions in industrial or robotics telemetry datasets. |
| `ramen__financial_authorization_invariance` | Spend, clearance and approval records that carry authorization claims. |
| `ramen__contractual_eligibility_invariance` | Claim and eligibility records that carry coverage, exclusion or evidence claims. |
| `ramen__compliance_provenance_invariance` | Regulated records that must carry verified evidence tokens. |

The fintech banking, industrial actuation, financial authorization, contractual eligibility and compliance provenance bundles are enterprise tier.

Pass explicit `policy_ids` for custom PII, ingestion, data-quality, or internal
MLOps controls. Bundle and policy details are available at
[ramenai.dev/pricing](https://ramenai.dev/pricing).

---

## Limitations

- Evaluation adds one API request per row. Large datasets should be processed in
  controlled batches with rate limits and resumability at the orchestration
  layer.
- Semantic healing creates operator-supplied values; it does not recover original
  truth or establish legal compliance, statistical fairness, or model quality.
- Healed rows are not automatically re-evaluated. Validate and, where required,
  re-evaluate transformed output before production ingestion or training.
- Callback quality, determinism, latency, and provider costs belong to the
  operator. Preserve source data and review `audit_log` and `imputation_log` in
  an access-controlled audit store.

## Running the tests

```bash
pip install -e ".[data]" pytest
python -m pytest tests/test_data_filtration.py -v
```

The tests use a mocked `RamenClient`; they make no network calls and require no
credentials.
