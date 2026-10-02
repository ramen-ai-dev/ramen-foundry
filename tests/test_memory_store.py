"""Persistence and retrieval tests for the episodic memory stores."""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import unittest
import uuid
from dataclasses import replace
from pathlib import Path

from ramen_foundry import (
    BaseEpisodicMemoryStore,
    CorrectionExemplar,
    JSONFileMemoryStore,
    SQLiteMemoryStore,
)
from ramen_foundry.core.memory import fingerprint_task

TASK = "Wire $150,000 to Harbor Equipment LLC"
RECEIPT = {
    "id": "rcpt-0001",
    "schema_version": "5.0",
    "kid": "ramen_pk_v1",
    "signature": "c2lnbmF0dXJl",
    "canonical_payload": '{"id":"rcpt-0001","verdict":1}',
    "statutory_anchors": ["UCC § 4A-202"],
    "attestation": None,
}


def _exemplar(
    *,
    task: str = TASK,
    tool_name: str = "dispatch_wire",
    created_at: str | None = None,
    receipt_id: str | None = "rcpt-0001",
    receipt: dict | None = RECEIPT,
) -> CorrectionExemplar:
    exemplar = CorrectionExemplar.create(
        task=task,
        tool_name=tool_name,
        failed_arguments={"amount_usd": 150000.0, "nested": {"co_signer": None}},
        violation_reason="High-value wire lacks dual-control authorization.",
        primary_statutory_anchor="UCC § 4A-202",
        steering_directive="Attach an Ed25519 co-signer signature.",
        repaired_arguments={"amount_usd": 150000.0, "co_signer_signature": "ed25519:ab"},
        receipt_id=receipt_id,
        receipt=receipt if receipt_id is not None else None,
    )
    return replace(exemplar, created_at=created_at) if created_at else exemplar


class CorrectionExemplarTests(unittest.TestCase):
    def test_create_populates_identity_fingerprint_and_timestamp(self) -> None:
        exemplar = _exemplar()
        uuid.UUID(exemplar.exemplar_id)
        self.assertEqual(exemplar.task_fingerprint, fingerprint_task(TASK))
        self.assertEqual(len(exemplar.task_fingerprint), 64)
        self.assertTrue(exemplar.created_at.endswith("+00:00"))

    def test_round_trips_through_dict(self) -> None:
        exemplar = _exemplar()
        self.assertEqual(CorrectionExemplar.from_dict(exemplar.to_dict()), exemplar)

    def test_rejects_invalid_fields(self) -> None:
        exemplar = _exemplar()
        invalid = (
            {"exemplar_id": "not-a-uuid"},
            {"task_fingerprint": "abc"},
            {"tool_name": " "},
            {"created_at": "2026-10-01T00:00:00"},
            {"receipt_id": ""},
        )
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                replace(exemplar, **changes)
        with self.assertRaises(ValueError):
            fingerprint_task("")


class _StoreContract:
    """Behaviour every BaseEpisodicMemoryStore implementation must satisfy."""

    def make_store(self) -> BaseEpisodicMemoryStore:
        raise NotImplementedError

    def reopen_store(self) -> BaseEpisodicMemoryStore:
        raise NotImplementedError

    def test_record_then_retrieve_persists_all_fields(self) -> None:
        store = self.make_store()
        exemplar = _exemplar()
        store.record_correction(exemplar)

        reopened = self.reopen_store()
        retrieved = reopened.retrieve_relevant_exemplars(
            fingerprint_task(TASK), "dispatch_wire"
        )
        self.assertEqual(retrieved, [exemplar])
        self.assertEqual(retrieved[0].receipt, RECEIPT)

    def test_retrieval_filters_by_task_and_tool_newest_first_with_limit(self) -> None:
        store = self.make_store()
        oldest = _exemplar(created_at="2026-10-01T00:00:01+00:00")
        middle = _exemplar(created_at="2026-10-01T00:00:02+00:00")
        newest = _exemplar(created_at="2026-10-01T00:00:03+00:00")
        other_task = _exemplar(task="Different task")
        other_tool = _exemplar(tool_name="issue_credit_adverse_action")
        for exemplar in (middle, other_task, newest, other_tool, oldest):
            store.record_correction(exemplar)

        retrieved = store.retrieve_relevant_exemplars(
            fingerprint_task(TASK), "dispatch_wire", limit=2
        )
        self.assertEqual(retrieved, [newest, middle])
        self.assertEqual(
            len(store.retrieve_relevant_exemplars(fingerprint_task(TASK), "dispatch_wire")),
            3,
        )
        self.assertEqual(
            store.retrieve_relevant_exemplars(fingerprint_task("unknown"), "dispatch_wire"),
            [],
        )

    def test_null_receipt_id_is_preserved(self) -> None:
        store = self.make_store()
        exemplar = _exemplar(receipt_id=None)
        store.record_correction(exemplar)
        retrieved = store.retrieve_relevant_exemplars(fingerprint_task(TASK), "dispatch_wire")
        self.assertIsNone(retrieved[0].receipt_id)
        self.assertIsNone(retrieved[0].receipt)

    def test_duplicate_exemplar_id_is_rejected(self) -> None:
        store = self.make_store()
        exemplar = _exemplar()
        store.record_correction(exemplar)
        with self.assertRaises(ValueError):
            store.record_correction(exemplar)

    def test_invalid_limit_is_rejected(self) -> None:
        store = self.make_store()
        for limit in (0, -1, True):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                store.retrieve_relevant_exemplars(fingerprint_task(TASK), "dispatch_wire", limit)

    def test_concurrent_writes_are_all_persisted(self) -> None:
        store = self.make_store()
        exemplars = [_exemplar() for _ in range(24)]
        errors: list[BaseException] = []

        def write(exemplar: CorrectionExemplar) -> None:
            try:
                store.record_correction(exemplar)
            except BaseException as error:  # noqa: BLE001
                errors.append(error)

        threads = [threading.Thread(target=write, args=(e,)) for e in exemplars]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        self.assertEqual(errors, [])
        retrieved = self.reopen_store().retrieve_relevant_exemplars(
            fingerprint_task(TASK), "dispatch_wire", limit=100
        )
        self.assertEqual(
            {e.exemplar_id for e in retrieved},
            {e.exemplar_id for e in exemplars},
        )


class JSONFileMemoryStoreTests(_StoreContract, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "agent_memory.json"

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def make_store(self) -> JSONFileMemoryStore:
        return JSONFileMemoryStore(self.path)

    def reopen_store(self) -> JSONFileMemoryStore:
        return JSONFileMemoryStore(self.path)

    def test_default_path_and_missing_file(self) -> None:
        self.assertEqual(JSONFileMemoryStore().path, Path("agent_memory.json"))
        self.assertEqual(
            self.make_store().retrieve_relevant_exemplars(fingerprint_task(TASK), "dispatch_wire"),
            [],
        )
        self.assertFalse(self.path.exists())

    def test_writes_versioned_document_without_leftover_temp_files(self) -> None:
        store = self.make_store()
        store.record_correction(_exemplar())
        store.record_correction(_exemplar())
        document = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(document["version"], 1)
        self.assertEqual(len(document["exemplars"]), 2)
        self.assertEqual(sorted(p.name for p in self.path.parent.iterdir()), ["agent_memory.json"])

    def test_corrupt_file_raises_instead_of_being_overwritten(self) -> None:
        self.path.write_text("{not json", encoding="utf-8")
        store = self.make_store()
        with self.assertRaises(ValueError):
            store.record_correction(_exemplar())
        self.assertEqual(self.path.read_text(encoding="utf-8"), "{not json")

    def test_unserialisable_arguments_leave_store_untouched(self) -> None:
        store = self.make_store()
        store.record_correction(_exemplar())
        before = self.path.read_text(encoding="utf-8")
        bad = replace(_exemplar(), repaired_arguments={"value": object()})
        with self.assertRaises(TypeError):
            store.record_correction(bad)
        self.assertEqual(self.path.read_text(encoding="utf-8"), before)


class SQLiteMemoryStoreTests(_StoreContract, unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "ramen_memory.db"
        self._stores: list[SQLiteMemoryStore] = []

    def tearDown(self) -> None:
        for store in self._stores:
            store.close()
        self._tmp.cleanup()

    def make_store(self) -> SQLiteMemoryStore:
        store = SQLiteMemoryStore(self.path)
        self._stores.append(store)
        return store

    def reopen_store(self) -> SQLiteMemoryStore:
        return self.make_store()

    def test_default_path(self) -> None:
        self.assertEqual(SQLiteMemoryStore.__init__.__defaults__, ("ramen_memory.db",))

    def test_schema_has_lookup_index(self) -> None:
        self.make_store()
        with sqlite3.connect(self.path) as connection:
            indexes = connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'index' "
                "AND tbl_name = 'correction_exemplars'"
            ).fetchall()
            columns = [
                row[2]
                for row in connection.execute(
                    "PRAGMA index_info('idx_correction_exemplars_lookup')"
                )
            ]
        self.assertIn(("idx_correction_exemplars_lookup",), indexes)
        self.assertEqual(columns, ["task_fingerprint", "tool_name", "created_at"])

    def test_pre_receipt_database_is_migrated(self) -> None:
        legacy = sqlite3.connect(self.path)
        legacy.execute(
            "CREATE TABLE correction_exemplars (exemplar_id TEXT PRIMARY KEY, "
            "task_fingerprint TEXT NOT NULL, tool_name TEXT NOT NULL, failed_arguments TEXT NOT NULL, "
            "violation_reason TEXT NOT NULL, primary_statutory_anchor TEXT NOT NULL, "
            "steering_directive TEXT NOT NULL, repaired_arguments TEXT NOT NULL, receipt_id TEXT, "
            "created_at TEXT NOT NULL)"
        )
        old = _exemplar(receipt_id=None)
        legacy.execute(
            "INSERT INTO correction_exemplars VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                old.exemplar_id, old.task_fingerprint, old.tool_name, json.dumps(old.failed_arguments),
                old.violation_reason, old.primary_statutory_anchor, old.steering_directive,
                json.dumps(old.repaired_arguments), None, old.created_at,
            ),
        )
        legacy.commit()
        legacy.close()

        store = self.make_store()
        new = _exemplar(created_at="2999-01-01T00:00:00+00:00")
        store.record_correction(new)
        retrieved = store.retrieve_relevant_exemplars(fingerprint_task(TASK), "dispatch_wire", limit=5)
        self.assertEqual([e.exemplar_id for e in retrieved], [new.exemplar_id, old.exemplar_id])
        self.assertEqual(retrieved[0].receipt, RECEIPT)
        self.assertIsNone(retrieved[1].receipt)

    def test_closed_store_raises(self) -> None:
        store = self.make_store()
        store.close()
        with self.assertRaises(RuntimeError):
            store.record_correction(_exemplar())


if __name__ == "__main__":
    unittest.main()
