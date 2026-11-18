# Just-In-Time Training Pipelines

Install with `pip install "ramen-foundry[data]"`. See [dataset filtration](./data-filtration.md) for the API.

`ramen_foundry.data` can govern records immediately before a training step
without making PyTorch or TensorFlow package dependencies. The filter itself is
synchronous because each row must receive a policy verdict before it can cross
the boundary. Training pipelines create concurrency around that boundary:
input workers evaluate future records while the accelerator trains on an
already-approved batch.

This pattern is **Just-In-Time (JIT) filtration**, not online model inference.
The CPU input pipeline owns policy evaluation, optional healing, decoding, and
batch assembly. The GPU consumes only completed batches.

## Pipeline shape

```text
source records
    -> bounded CPU input workers
    -> ramen ai verdict per row
    -> exclude, or optional callback healing
    -> tensor conversion and batching
    -> prefetch queue
    -> GPU training step
```

Keep the queue bounded. An unbounded queue can exceed API rate limits, consume
unbounded memory, or continue evaluating data after a training job has failed.
Size worker counts and prefetch depth against observed policy latency, API rate
limits, CPU capacity, and the time of one accelerator step.

A useful starting estimate is:

```text
required concurrent evaluations
    ~= evaluation latency / accelerator step time * records per step
```

Measure the real pipeline before increasing concurrency. More workers do not
improve throughput after the API, network, or CPU becomes saturated.

## Shared healing callback

Both examples use an ordinary Python function. It can call your own model or
rules engine, but `ramen_foundry.data` does not install or require an LLM SDK.
The callback must return the same keys as the input row.

```python
def heal_record(row: dict, steering: str) -> dict:
    cured = dict(row)
    if "content" in steering.casefold():
        cured["content"] = "[REDACTED BY DATA POLICY]"
    return cured
```

Use `remediable_columns` to prevent a callback from changing fields outside an
approved set. If the callback raises, changes the row schema, makes no change,
or changes a disallowed column, filtration raises `FiltrationError` and the
batch is not emitted. If no callback is supplied, blocked rows are excluded.

## PyTorch `DataLoader`

`DataLoader` workers run ahead of the training loop. `num_workers` controls the
number of worker processes and `prefetch_factor` controls how many batches each
worker prepares in advance. Construct the `RamenClient` inside each worker;
HTTP clients must not be created in the parent and then shared across forked or
spawned processes.

```python
import os
from collections.abc import Iterator

import pandas as pd
import torch
from ramen_ai import RamenClient
from ramen_foundry.data import FiltrationMode, filter_dataframe
from torch.utils.data import DataLoader, IterableDataset, get_worker_info


class GovernedRows(IterableDataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __iter__(self) -> Iterator[dict]:
        worker = get_worker_info()
        worker_id = worker.id if worker else 0
        worker_count = worker.num_workers if worker else 1

        # Striding prevents IterableDataset workers from evaluating duplicates.
        assigned_rows = self.rows[worker_id::worker_count]

        with RamenClient(api_key=os.environ["RAMEN_API_KEY"]) as client:
            for row in assigned_rows:
                result = filter_dataframe(
                    pd.DataFrame([row]),
                    mode=FiltrationMode.SEMANTIC_IMPUTATION,
                    bundle_ids=["ramen__shield_core_it"],
                    remediable_columns=["content"],
                    healing_callback=heal_record,
                    client=client,
                    provider_key=os.environ.get("OPENAI_API_KEY"),
                )
                if result.dataframe.empty:
                    continue

                # Replace with application-specific tokenization/tensorization.
                yield result.dataframe.iloc[0].to_dict()


def collate_records(rows: list[dict]) -> dict[str, torch.Tensor]:
    return {
        "features": torch.tensor(
            [row["features"] for row in rows], dtype=torch.float32
        ),
        "labels": torch.tensor(
            [row["label"] for row in rows], dtype=torch.long
        ),
    }


loader = DataLoader(
    GovernedRows(training_rows),
    batch_size=64,
    collate_fn=collate_records,
    num_workers=4,
    prefetch_factor=2,
    persistent_workers=True,
    pin_memory=True,
)

for batch in loader:
    features = batch["features"].to("cuda", non_blocking=True)
    labels = batch["labels"].to("cuda", non_blocking=True)
    loss = training_step(features, labels)
```

Operational notes:

- `num_workers * prefetch_factor` is the approximate number of batches that may
  be prepared ahead. Multiply by `batch_size` to estimate prefetched samples,
  then start small to avoid request bursts.
- A top-level callback is picklable under multiprocessing `spawn`; a lambda or
  nested closure may not be.
- `persistent_workers=True` preserves worker processes across epochs, but the
  context manager shown closes and recreates each worker's client per iterator.
  It does not preserve the HTTP connection pool across epochs.
- Worker order is not an audit identity. Persist source IDs, input hashes,
  verdicts, and receipts rather than relying on arrival order.
- A `FiltrationError` propagating from a worker stops iteration. Do not catch it
  and emit the original blocked row.

## TensorFlow `tf.data.Dataset`

TensorFlow can execute Python filtration through `tf.py_function`. Batch source
records before the Python boundary so one `filter_dataframe` call reuses one
client across several row evaluations. Parallel `map` calls prepare multiple
future batches, and `prefetch` overlaps those calls with accelerator work.

```python
import json
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from ramen_foundry.data import FiltrationMode, filter_dataframe

FILTER_BATCH_SIZE = 32
TRAIN_BATCH_SIZE = 128


def filter_json_batch(values: tf.Tensor) -> tuple[np.ndarray, np.ndarray]:
    rows = [json.loads(value.decode("utf-8")) for value in values.numpy()]
    result = filter_dataframe(
        pd.DataFrame(rows),
        mode=FiltrationMode.SEMANTIC_IMPUTATION,
        bundle_ids=["ramen__shield_core_it"],
        remediable_columns=["content"],
        healing_callback=heal_record,
        api_key=os.environ["RAMEN_API_KEY"],
        provider_key=os.environ.get("OPENAI_API_KEY"),
    )
    payloads = [
        json.dumps(row, sort_keys=True).encode("utf-8")
        for row in result.dataframe.to_dict(orient="records")
    ]
    return np.asarray(payloads, dtype=np.bytes_), np.int64(len(payloads))


def govern_batch(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
    payloads, count = tf.py_function(
        filter_json_batch,
        [values],
        Tout=(tf.string, tf.int64),
    )
    payloads.set_shape([None])
    count.set_shape([])
    return payloads, count


source = tf.data.Dataset.from_tensor_slices(
    [json.dumps(row, sort_keys=True) for row in training_rows]
)

dataset = (
    source
    .batch(FILTER_BATCH_SIZE)
    .map(
        govern_batch,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False,
    )
    .flat_map(lambda payloads, count: tf.data.Dataset.from_tensor_slices(payloads))
    .map(decode_and_tensorize, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(TRAIN_BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

model.fit(dataset, epochs=3)
```

`decode_and_tensorize` is application-specific: parse each healed JSON string,
tokenize text or extract numeric features, and return tensors matching the
model signature. The `count` output makes the variable-size filtration boundary
explicit for monitoring, even though `flat_map` consumes only `payloads`.

`tf.py_function` executes Python and is not serialized as a portable TensorFlow
graph. Use this pattern in a Python training service, not in a SavedModel that
must run without Python. Set a fixed `num_parallel_calls` instead of
`AUTOTUNE` when API concurrency must have a hard ceiling.

## Determinism, audit, and failure handling

- Remote policy evaluation can change when policies or provider models change.
  Pin the policy IDs or bundle configuration used for a training run and record
  them with the model artifact.
- Persist `audit_log` and `imputation_log` in an access-controlled store before
  releasing a batch if your governance process requires replay or provenance.
- Do not retry blocked verdicts. Retry only transient transport failures, with
  bounded exponential backoff outside the filter and with idempotent batch IDs.
- A healed row is not automatically re-evaluated. Re-run filtration on healed
  output when policy requires proof that the transformed row is now allowed.
- Track exclusion and healing rates. A sudden increase can indicate poisoned
  input, policy drift, or a failing upstream data source.
- Keep a CPU-ready queue large enough to hide normal policy latency but small
  enough that cancellation and policy changes take effect promptly.

For offline or reproducible training, run filtration as a separate preparation
job, persist the approved dataset and receipts, and train from that immutable
snapshot. Use JIT filtration when source data arrives continuously or policy
checks must occur immediately before consumption.
