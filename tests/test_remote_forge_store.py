"""HTTP contract tests for RemoteForgeMemoryStore (no network access)."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from typing import Any
from unittest import mock

import httpx

from ramen_foundry import CorrectionExemplar, RemoteForgeMemoryStore
from ramen_foundry.core.memory import fingerprint_task

BASE_URL = "https://forge.example.test"
TASK = "Formulate an adverse action notice for credit application APP-99214."
TOOL = "issue_credit_adverse_action"
TOKEN = "forge-test-token"
RECEIPT_ID = "8d1f2c4e-5a6b-4c7d-8e9f-0a1b2c3d4e5f"
RECEIPT: dict[str, Any] = {
    "id": RECEIPT_ID,
    "schema_version": "5.0",
    "kid": "ramen_pk_v1",
    "signature": "c2lnbmF0dXJl",
    "canonical_payload": json.dumps({"id": RECEIPT_ID, "verdict": 1}),
    "statutory_anchors": [],
    "attestation": None,
}


def _exemplar(
    *,
    task: str = TASK,
    tool_name: str = TOOL,
    receipt: dict[str, Any] | None = RECEIPT,
) -> CorrectionExemplar:
    return CorrectionExemplar.create(
        task=task,
        tool_name=tool_name,
        failed_arguments={"reg_b_reason_codes": ["REGIONAL_ECONOMIC_VOLATILITY_ZIP_CODE"]},
        violation_reason="Denial cites ZIP code as adverse factor.",
        primary_statutory_anchor="ECOA Regulation B",
        steering_directive="Use documented neutral creditworthiness factors.",
        repaired_arguments={"reg_b_reason_codes": ["INSUFFICIENT_LIQUIDITY"]},
        receipt_id=RECEIPT_ID if receipt is not None else None,
        receipt=receipt,
    )


def _served(exemplar: CorrectionExemplar) -> dict[str, Any]:
    """Return the record shape served by GET /api/v1/exemplars."""
    return exemplar.to_dict() | {
        "domain": "fintech",
        "task_description": exemplar.task_description,
        "tier": "community",
    }


def _response(status: int, body: Any, method: str = "GET") -> httpx.Response:
    request = httpx.Request(method, f"{BASE_URL}/api/v1/exemplars")
    content = body if isinstance(body, bytes) else json.dumps(body).encode()
    return httpx.Response(status, content=content, request=request)


class RetrieveTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = RemoteForgeMemoryStore(base_url=f"{BASE_URL}/", domain="fintech", timeout_sec=2.5)
        self.fingerprint = fingerprint_task(TASK)

    def retrieve(self, response: httpx.Response | Exception, limit: int = 3) -> tuple[list[CorrectionExemplar], mock.MagicMock]:
        with mock.patch("ramen_foundry.core.memory.httpx.get") as get:
            if isinstance(response, Exception):
                get.side_effect = response
            else:
                get.return_value = response
            return self.store.retrieve_relevant_exemplars(self.fingerprint, TOOL, limit), get

    def test_calls_query_url_and_deserializes_exemplars(self) -> None:
        exemplar = _exemplar()
        body = {"success": True, "count": 1, "exemplars": [_served(exemplar)]}
        retrieved, get = self.retrieve(_response(200, body), limit=2)

        get.assert_called_once()
        args, kwargs = get.call_args
        self.assertEqual(args[0], f"{BASE_URL}/api/v1/exemplars")
        self.assertEqual(
            kwargs["params"],
            {"domain": "fintech", "tool_name": TOOL, "task_fingerprint": self.fingerprint, "limit": "2"},
        )
        self.assertEqual(kwargs["timeout"], 2.5)
        self.assertNotIn("Authorization", kwargs["headers"])
        self.assertEqual(retrieved, [exemplar])
        self.assertEqual(retrieved[0].task_description, TASK)
        self.assertEqual(retrieved[0].repaired_arguments, exemplar.repaired_arguments)

    def test_network_failures_fail_open(self) -> None:
        failures: list[httpx.Response | Exception] = [
            httpx.ConnectTimeout("timed out"),
            httpx.ReadTimeout("timed out"),
            httpx.ConnectError("dns failure"),
            _response(503, {"success": False}),
            _response(200, b"<html>not json</html>"),
            _response(200, [{"unexpected": "array"}]),
        ]
        for failure in failures:
            with self.subTest(failure=repr(failure)):
                with self.assertLogs("ramen_foundry.core.memory", level="WARNING"):
                    retrieved, _ = self.retrieve(failure)
                self.assertEqual(retrieved, [])

    def test_invalid_or_mismatched_records_are_skipped(self) -> None:
        valid = _exemplar()
        bad_uuid = _served(_exemplar()) | {"exemplar_id": "not-a-uuid"}
        wrong_tool = _served(_exemplar(tool_name="dispatch_wire"))
        wrong_domain = _served(_exemplar()) | {"domain": "industrial_iot"}
        tampered = _served(_exemplar()) | {"task_description": "Tampered description"}
        body = {"exemplars": [bad_uuid, wrong_tool, wrong_domain, tampered, _served(valid)]}

        with self.assertLogs("ramen_foundry.core.memory", level="WARNING"):
            retrieved, _ = self.retrieve(_response(200, body))
        self.assertEqual(retrieved, [valid])

    def test_different_task_fingerprint_with_matching_tool_is_accepted(self) -> None:
        rephrased = _exemplar(task="Draft the Reg B denial letter for APP-99214.")
        self.assertNotEqual(rephrased.task_fingerprint, self.fingerprint)
        body = {"exemplars": [_served(rephrased)]}
        retrieved, _ = self.retrieve(_response(200, body))
        self.assertEqual(retrieved, [rephrased])

    def test_retrieval_without_task_fingerprint(self) -> None:
        exemplar = _exemplar(task="Any phrasing of the adverse-action task")
        body = {"exemplars": [_served(exemplar)]}
        with mock.patch("ramen_foundry.core.memory.httpx.get") as get:
            get.return_value = _response(200, body)
            retrieved = self.store.retrieve_relevant_exemplars(tool_name=TOOL)
        self.assertEqual(
            get.call_args.kwargs["params"],
            {"domain": "fintech", "limit": "3", "tool_name": TOOL},
        )
        self.assertEqual(retrieved, [exemplar])

    def test_query_is_forwarded_as_q(self) -> None:
        exemplar = _exemplar(task="Stage solvent canisters near the burner line.", tool_name="place_material")
        body = {"exemplars": [_served(exemplar)]}
        with mock.patch("ramen_foundry.core.memory.httpx.get") as get:
            get.return_value = _response(200, body)
            retrieved = self.store.retrieve_relevant_exemplars(query="burner")
        self.assertEqual(get.call_args.kwargs["params"], {"domain": "fintech", "limit": "3", "q": "burner"})
        self.assertEqual(retrieved, [exemplar])
        url = httpx.Request("GET", get.call_args.args[0], params=get.call_args.kwargs["params"]).url
        self.assertEqual(url.params["q"], "burner")

    def test_invalid_optional_filters_are_rejected_without_network(self) -> None:
        cases = [
            {"query": "   "},
            {"query": "x" * 101},
            {"tool_name": ""},
            {"task_fingerprint": ""},
        ]
        with mock.patch("ramen_foundry.core.memory.httpx.get") as get:
            for kwargs in cases:
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    self.store.retrieve_relevant_exemplars(**kwargs)
        get.assert_not_called()

    def test_limit_is_validated_and_enforced(self) -> None:
        for limit in (0, -1, True, 51):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                self.store.retrieve_relevant_exemplars(self.fingerprint, TOOL, limit)
        body = {"exemplars": [_served(_exemplar()) for _ in range(3)]}
        retrieved, _ = self.retrieve(_response(200, body), limit=1)
        self.assertEqual(len(retrieved), 1)


class RecordTests(unittest.TestCase):
    def setUp(self) -> None:
        self.store = RemoteForgeMemoryStore(base_url=BASE_URL, write_token=TOKEN, domain="fintech")

    def record(self, exemplar: CorrectionExemplar, response: httpx.Response | Exception) -> mock.MagicMock:
        with mock.patch("ramen_foundry.core.memory.httpx.post") as post:
            if isinstance(response, Exception):
                post.side_effect = response
            else:
                post.return_value = response
            self.store.record_correction(exemplar)
            return post

    def test_sends_bearer_token_and_normalized_payload(self) -> None:
        exemplar = _exemplar()
        post = self.record(exemplar, _response(201, {"success": True}, "POST"))

        post.assert_called_once()
        args, kwargs = post.call_args
        self.assertEqual(args[0], f"{BASE_URL}/api/v1/exemplars")
        self.assertEqual(kwargs["headers"]["Authorization"], f"Bearer {TOKEN}")
        expected = exemplar.to_dict() | {"domain": "fintech", "task_description": TASK}
        self.assertEqual(kwargs["json"], expected)
        self.assertEqual(
            set(kwargs["json"]),
            {
                "exemplar_id", "domain", "task_description", "task_fingerprint", "tool_name",
                "failed_arguments", "violation_reason", "primary_statutory_anchor",
                "steering_directive", "repaired_arguments", "receipt_id", "created_at",
                "receipt",
            },
        )
        self.assertEqual(kwargs["json"]["task_fingerprint"], fingerprint_task(TASK))
        self.assertEqual(kwargs["json"]["receipt"], RECEIPT)
        self.assertEqual(kwargs["json"]["receipt_id"], RECEIPT["id"])
        self.assertEqual(json.loads(json.dumps(kwargs["json"]))["receipt"], RECEIPT)

    def test_exemplar_without_receipt_is_skipped_without_network(self) -> None:
        with mock.patch("ramen_foundry.core.memory.httpx.post") as post:
            with self.assertLogs("ramen_foundry.core.memory", level="WARNING") as logs:
                self.store.record_correction(_exemplar(receipt=None))
        post.assert_not_called()
        self.assertIn("no Schema V5 receipt", logs.output[0])

    def test_receipt_rejection_detail_is_reported(self) -> None:
        rejected = _response(
            422,
            {
                "success": False,
                "error": {"code": "INVALID_CRYPTOGRAPHIC_RECEIPT", "message": "Exemplar rejected"},
                "details": ["signed verdict is not 1 (the evaluated call was blocked)"],
            },
            "POST",
        )
        with self.assertRaisesRegex(RuntimeError, "INVALID_CRYPTOGRAPHIC_RECEIPT.*signed verdict is not 1"):
            self.record(_exemplar(), rejected)

    def test_refreshed_200_is_accepted_and_logged_as_a_refresh(self) -> None:
        refreshed = _response(200, {"success": True, "exemplar_id": "stored-id", "refreshed": True}, "POST")
        with self.assertLogs("ramen_foundry.core.memory", level="INFO") as logs:
            self.record(_exemplar(), refreshed)
        self.assertTrue(any("refreshed the receipt" in line for line in logs.output))

    def test_duplicate_409_is_ignored(self) -> None:
        self.record(_exemplar(), _response(409, {"success": False}, "POST"))

    def test_rejection_and_transport_failures_raise(self) -> None:
        rejected = _response(422, {"error": "exemplar rejected", "details": ["bad field"]}, "POST")
        with self.assertRaisesRegex(RuntimeError, "HTTP 422.*bad field"):
            self.record(_exemplar(), rejected)
        with self.assertRaisesRegex(RuntimeError, "write failed"):
            self.record(_exemplar(), httpx.ConnectTimeout("timed out"))

    def test_read_only_store_refuses_writes_without_network(self) -> None:
        store = RemoteForgeMemoryStore(base_url=BASE_URL)
        with mock.patch("ramen_foundry.core.memory.httpx.post") as post:
            with self.assertRaises(PermissionError):
                store.record_correction(_exemplar())
        post.assert_not_called()

    def test_exemplar_without_task_description_is_refused(self) -> None:
        rehydrated = CorrectionExemplar.from_dict(_exemplar().to_dict())
        with mock.patch("ramen_foundry.core.memory.httpx.post") as post:
            with self.assertRaisesRegex(ValueError, "task_description"):
                self.store.record_correction(rehydrated)
        post.assert_not_called()


class ConstructorTests(unittest.TestCase):
    def test_defaults(self) -> None:
        store = RemoteForgeMemoryStore()
        self.assertEqual(store.base_url, "https://ramen-forge.ramenai.workers.dev")
        self.assertEqual(store.domain, "fintech")
        self.assertEqual(store.timeout_sec, 5.0)
        self.assertIsNone(store.write_token)

    def test_repr_never_exposes_token(self) -> None:
        store = RemoteForgeMemoryStore(write_token=TOKEN)
        self.assertNotIn(TOKEN, repr(store))

    def test_rejects_insecure_url_bad_domain_and_timeout(self) -> None:
        with self.assertRaises(ValueError):
            RemoteForgeMemoryStore(base_url="http://forge.example.test")
        RemoteForgeMemoryStore(base_url="http://localhost:8787")
        for domain in ("FinTech", "x", "fin tech"):
            with self.subTest(domain=domain), self.assertRaises(ValueError):
                RemoteForgeMemoryStore(domain=domain)
        for timeout in (0, -1, True):
            with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                RemoteForgeMemoryStore(timeout_sec=timeout)


class ReceiptFieldTests(unittest.TestCase):
    def test_receipt_round_trips_through_dict_and_forge_record(self) -> None:
        exemplar = _exemplar()
        self.assertEqual(exemplar.to_dict()["receipt"], RECEIPT)
        self.assertEqual(CorrectionExemplar.from_dict(exemplar.to_dict()).receipt, RECEIPT)
        self.assertEqual(CorrectionExemplar.from_dict(_served(exemplar)).receipt, RECEIPT)

    def test_missing_receipt_is_omitted_from_to_dict(self) -> None:
        self.assertNotIn("receipt", _exemplar(receipt=None).to_dict())

    def test_invalid_receipts_are_rejected(self) -> None:
        exemplar = _exemplar()
        for receipt in (["not", "a", "dict"], {"id": "other-id"}, {"blob": object()}):
            with self.subTest(receipt=receipt), self.assertRaises(ValueError):
                replace(exemplar, receipt=receipt)


class TaskDescriptionTests(unittest.TestCase):
    def test_task_description_is_not_persisted_by_to_dict(self) -> None:
        exemplar = _exemplar()
        self.assertEqual(exemplar.task_description, TASK)
        self.assertNotIn("task_description", exemplar.to_dict())

    def test_mismatched_task_description_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            CorrectionExemplar.from_dict(_served(_exemplar()) | {"task_description": "other"})


if __name__ == "__main__":
    unittest.main()
