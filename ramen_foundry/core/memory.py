"""Episodic memory for governed tool-call repairs.

A :class:`CorrectionExemplar` captures one closed-loop repair: the tool call
that ramen-ai blocked, the primary statutory anchor and steering directive that
explained the block, and the repaired arguments that were subsequently allowed.
Stores index exemplars by task fingerprint and tool name so a planner can
retrieve prior repairs before its first attempt.

Exemplars persist tool arguments verbatim. Hosts must place store files on
storage with access controls appropriate to the data their tools handle.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import sqlite3
import tempfile
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import httpx

logger = logging.getLogger(__name__)

_JSON_STORE_FORMAT_VERSION = 1
_EXEMPLAR_FIELDS = (
    "exemplar_id",
    "task_fingerprint",
    "tool_name",
    "failed_arguments",
    "violation_reason",
    "primary_statutory_anchor",
    "steering_directive",
    "repaired_arguments",
    "receipt_id",
    "created_at",
)
_SQLITE_COLUMNS = (*_EXEMPLAR_FIELDS, "receipt")


def fingerprint_task(task: str) -> str:
    """Return the SHA-256 hex digest of a UTF-8 encoded task string."""
    if not isinstance(task, str) or not task.strip():
        raise ValueError("task must be a non-blank string")
    return hashlib.sha256(task.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class CorrectionExemplar:
    """One successful repair of a previously blocked tool call."""

    exemplar_id: str
    task_fingerprint: str
    tool_name: str
    failed_arguments: dict[str, Any] = field(repr=False)
    violation_reason: str
    primary_statutory_anchor: str
    steering_directive: str
    repaired_arguments: dict[str, Any] = field(repr=False)
    receipt_id: str | None
    created_at: str
    # Original task text, kept in memory only. It is excluded from to_dict(),
    # equality, and the local stores; RemoteForgeMemoryStore sends it because
    # ramen forge requires task_fingerprint == SHA-256(task_description).
    task_description: str | None = field(default=None, compare=False, repr=False)
    # Complete Schema V5 receipt from the ramen-ai evaluation of the repaired
    # call. ramen forge verifies it and rejects records without one.
    receipt: dict[str, Any] | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        try:
            uuid.UUID(self.exemplar_id)
        except (TypeError, ValueError, AttributeError) as error:
            raise ValueError("exemplar_id must be a UUID string") from error
        if (
            not isinstance(self.task_fingerprint, str)
            or len(self.task_fingerprint) != 64
            or any(char not in "0123456789abcdef" for char in self.task_fingerprint)
        ):
            raise ValueError("task_fingerprint must be a lowercase SHA-256 hex digest")
        for name in (
            "tool_name",
            "violation_reason",
            "primary_statutory_anchor",
            "steering_directive",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a non-blank string")
        for name in ("failed_arguments", "repaired_arguments"):
            if not isinstance(getattr(self, name), dict):
                raise ValueError(f"{name} must be a dict")
        if self.receipt_id is not None and (
            not isinstance(self.receipt_id, str) or not self.receipt_id.strip()
        ):
            raise ValueError("receipt_id must be None or a non-blank string")
        try:
            parsed = datetime.fromisoformat(self.created_at)
        except (TypeError, ValueError) as error:
            raise ValueError("created_at must be an ISO 8601 timestamp") from error
        if parsed.tzinfo is None:
            raise ValueError("created_at must include a UTC offset")
        if self.task_description is not None:
            if not isinstance(self.task_description, str) or not self.task_description.strip():
                raise ValueError("task_description must be None or a non-blank string")
            if fingerprint_task(self.task_description) != self.task_fingerprint:
                raise ValueError("task_fingerprint does not match SHA-256(task_description)")
        if self.receipt is not None:
            if not isinstance(self.receipt, dict):
                raise ValueError("receipt must be None or a dict")
            try:
                json.dumps(self.receipt)
            except (TypeError, ValueError) as error:
                raise ValueError("receipt must be JSON-serialisable") from error
            receipt_key = self.receipt.get("id")
            if receipt_key is not None and self.receipt_id is not None and receipt_key != self.receipt_id:
                raise ValueError("receipt_id does not match receipt['id']")

    @classmethod
    def create(
        cls,
        *,
        task: str,
        tool_name: str,
        failed_arguments: dict[str, Any],
        violation_reason: str,
        primary_statutory_anchor: str,
        steering_directive: str,
        repaired_arguments: dict[str, Any],
        receipt_id: str | None,
        receipt: dict[str, Any] | None = None,
    ) -> CorrectionExemplar:
        """Build an exemplar with a fresh UUID, task fingerprint, and UTC timestamp."""
        return cls(
            exemplar_id=str(uuid.uuid4()),
            task_fingerprint=fingerprint_task(task),
            tool_name=tool_name,
            failed_arguments=dict(failed_arguments),
            violation_reason=violation_reason,
            primary_statutory_anchor=primary_statutory_anchor,
            steering_directive=steering_directive,
            repaired_arguments=dict(repaired_arguments),
            receipt_id=receipt_id,
            created_at=datetime.now(timezone.utc).isoformat(),
            task_description=task,
            receipt=dict(receipt) if receipt is not None else None,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the exemplar.

        ``task_description`` is deliberately excluded so local stores never
        persist raw task text. ``receipt`` is included only when present.
        """
        data = asdict(self)
        data.pop("task_description")
        if data["receipt"] is None:
            data.pop("receipt")
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorrectionExemplar:
        """Rebuild an exemplar from :meth:`to_dict` output, validating every field.

        Unknown keys are ignored. An optional ``task_description`` (as served by
        ramen forge) is kept when present.
        """
        if not isinstance(data, dict):
            raise ValueError("exemplar record must be a JSON object")
        missing = [name for name in _EXEMPLAR_FIELDS if name not in data]
        if missing:
            raise ValueError(f"exemplar record is missing fields: {', '.join(missing)}")
        return cls(
            **{name: data[name] for name in _EXEMPLAR_FIELDS},
            task_description=data.get("task_description"),
            receipt=data.get("receipt"),
        )


class BaseEpisodicMemoryStore(ABC):
    """Persistence contract for repair exemplars."""

    @abstractmethod
    def record_correction(self, exemplar: CorrectionExemplar) -> None:
        """Persist one exemplar. Duplicate ``exemplar_id`` values must be rejected."""

    @abstractmethod
    def retrieve_relevant_exemplars(
        self,
        task_fingerprint: str,
        tool_name: str,
        limit: int = 3,
    ) -> list[CorrectionExemplar]:
        """Return up to ``limit`` matching exemplars, newest first."""


def _validate_limit(limit: int) -> None:
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("limit must be a positive integer")


def _encode_arguments(arguments: dict[str, Any]) -> str:
    return json.dumps(arguments, sort_keys=True, separators=(",", ":"))


class JSONFileMemoryStore(BaseEpisodicMemoryStore):
    """Store exemplars in a single JSON document with atomic replacement writes.

    Writes are serialised with an in-process lock and committed by writing a
    temporary file in the target directory, fsyncing it, and ``os.replace``-ing
    it over the store, so readers never observe a partially written document.
    The lock does not coordinate separate processes sharing one file.
    """

    def __init__(self, path: str | os.PathLike[str] = "agent_memory.json") -> None:
        self._path = Path(path)
        self._lock = threading.RLock()

    @property
    def path(self) -> Path:
        """Return the backing JSON file path."""
        return self._path

    def record_correction(self, exemplar: CorrectionExemplar) -> None:
        if not isinstance(exemplar, CorrectionExemplar):
            raise TypeError("exemplar must be a CorrectionExemplar")
        record = exemplar.to_dict()
        _encode_arguments(record["failed_arguments"])
        _encode_arguments(record["repaired_arguments"])
        with self._lock:
            records = self._read_records()
            if any(existing["exemplar_id"] == exemplar.exemplar_id for existing in records):
                raise ValueError(f"exemplar_id already recorded: {exemplar.exemplar_id}")
            records.append(record)
            self._write_records(records)

    def retrieve_relevant_exemplars(
        self,
        task_fingerprint: str,
        tool_name: str,
        limit: int = 3,
    ) -> list[CorrectionExemplar]:
        _validate_limit(limit)
        with self._lock:
            records = self._read_records()
        matches = [
            (index, CorrectionExemplar.from_dict(record))
            for index, record in enumerate(records)
            if record.get("task_fingerprint") == task_fingerprint
            and record.get("tool_name") == tool_name
        ]
        matches.sort(key=lambda item: (item[1].created_at, item[0]), reverse=True)
        return [exemplar for _, exemplar in matches[:limit]]

    def _read_records(self) -> list[dict[str, Any]]:
        try:
            raw = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        try:
            document = json.loads(raw)
        except json.JSONDecodeError as error:
            raise ValueError(f"episodic memory file is not valid JSON: {self._path}") from error
        if (
            not isinstance(document, dict)
            or document.get("version") != _JSON_STORE_FORMAT_VERSION
            or not isinstance(document.get("exemplars"), list)
        ):
            raise ValueError(f"episodic memory file has an unsupported format: {self._path}")
        return list(document["exemplars"])

    def _write_records(self, records: list[dict[str, Any]]) -> None:
        directory = self._path.parent
        directory.mkdir(parents=True, exist_ok=True)
        document = {"version": _JSON_STORE_FORMAT_VERSION, "exemplars": records}
        file_descriptor, temp_name = tempfile.mkstemp(
            prefix=f".{self._path.name}.",
            suffix=".tmp",
            dir=directory,
        )
        try:
            with os.fdopen(file_descriptor, "w", encoding="utf-8") as handle:
                json.dump(document, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_name, self._path)
        except BaseException:
            try:
                os.unlink(temp_name)
            except FileNotFoundError:
                pass
            raise


class SQLiteMemoryStore(BaseEpisodicMemoryStore):
    """Store exemplars in an embedded SQLite database.

    Lookups are served by a composite index on
    ``(task_fingerprint, tool_name, created_at)``. One connection is shared
    behind a lock so the store can be used from multiple threads; call
    :meth:`close` (or use the store as a context manager) when finished.
    """

    _SCHEMA = (
        """
        CREATE TABLE IF NOT EXISTS correction_exemplars (
            exemplar_id TEXT PRIMARY KEY,
            task_fingerprint TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            failed_arguments TEXT NOT NULL,
            violation_reason TEXT NOT NULL,
            primary_statutory_anchor TEXT NOT NULL,
            steering_directive TEXT NOT NULL,
            repaired_arguments TEXT NOT NULL,
            receipt_id TEXT,
            created_at TEXT NOT NULL,
            receipt TEXT
        )
        """,
        """
        CREATE INDEX IF NOT EXISTS idx_correction_exemplars_lookup
        ON correction_exemplars (task_fingerprint, tool_name, created_at DESC)
        """,
    )

    def __init__(self, path: str | os.PathLike[str] = "ramen_memory.db") -> None:
        self._path = str(path)
        self._lock = threading.RLock()
        self._connection: sqlite3.Connection | None = sqlite3.connect(
            self._path,
            check_same_thread=False,
        )
        try:
            with self._connection:
                for statement in self._SCHEMA:
                    self._connection.execute(statement)
                columns = {
                    row[1]
                    for row in self._connection.execute("PRAGMA table_info(correction_exemplars)")
                }
                if "receipt" not in columns:
                    # Databases created by 0.2.0-0.2.2 predate the receipt column.
                    self._connection.execute("ALTER TABLE correction_exemplars ADD COLUMN receipt TEXT")
        except sqlite3.Error:
            self._connection.close()
            self._connection = None
            raise

    @property
    def path(self) -> str:
        """Return the backing database path."""
        return self._path

    def record_correction(self, exemplar: CorrectionExemplar) -> None:
        if not isinstance(exemplar, CorrectionExemplar):
            raise TypeError("exemplar must be a CorrectionExemplar")
        row = (
            exemplar.exemplar_id,
            exemplar.task_fingerprint,
            exemplar.tool_name,
            _encode_arguments(exemplar.failed_arguments),
            exemplar.violation_reason,
            exemplar.primary_statutory_anchor,
            exemplar.steering_directive,
            _encode_arguments(exemplar.repaired_arguments),
            exemplar.receipt_id,
            exemplar.created_at,
            json.dumps(exemplar.receipt, sort_keys=True) if exemplar.receipt is not None else None,
        )
        with self._lock:
            connection = self._require_connection()
            try:
                with connection:
                    connection.execute(
                        f"INSERT INTO correction_exemplars ({', '.join(_SQLITE_COLUMNS)}) "
                        f"VALUES ({', '.join('?' for _ in _SQLITE_COLUMNS)})",
                        row,
                    )
            except sqlite3.IntegrityError as error:
                raise ValueError(
                    f"exemplar_id already recorded: {exemplar.exemplar_id}"
                ) from error

    def retrieve_relevant_exemplars(
        self,
        task_fingerprint: str,
        tool_name: str,
        limit: int = 3,
    ) -> list[CorrectionExemplar]:
        _validate_limit(limit)
        with self._lock:
            rows = self._require_connection().execute(
                f"SELECT {', '.join(_SQLITE_COLUMNS)} FROM correction_exemplars "
                "WHERE task_fingerprint = ? AND tool_name = ? "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (task_fingerprint, tool_name, limit),
            ).fetchall()
        exemplars: list[CorrectionExemplar] = []
        for row in rows:
            record = dict(zip(_SQLITE_COLUMNS, row))
            record["failed_arguments"] = json.loads(record["failed_arguments"])
            record["repaired_arguments"] = json.loads(record["repaired_arguments"])
            record["receipt"] = json.loads(record["receipt"]) if record["receipt"] is not None else None
            exemplars.append(CorrectionExemplar.from_dict(record))
        return exemplars

    def close(self) -> None:
        """Close the underlying database connection."""
        with self._lock:
            if self._connection is not None:
                self._connection.close()
                self._connection = None

    def __enter__(self) -> SQLiteMemoryStore:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def _require_connection(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("SQLiteMemoryStore is closed")
        return self._connection


_FORGE_EXEMPLARS_PATH = "/api/v1/exemplars"
_FORGE_MAX_LIMIT = 50
_FORGE_MAX_QUERY_LENGTH = 100
_FORGE_DOMAIN_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{1,63}$")
_LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


class RemoteForgeMemoryStore(BaseEpisodicMemoryStore):
    """Share exemplars through a ramen forge service over HTTP.

    Reads go to ``GET /api/v1/exemplars`` and are public. They fail open: a
    network error, timeout, HTTP error, or malformed response is logged and
    returns ``[]``, because memory is advisory and every call is still
    evaluated by the ramen-ai policy boundary. Every filter except the store's
    ``domain`` is optional: ``task_fingerprint`` asks for an exact-task match,
    and ``query`` is a keyword search the forge runs over task descriptions,
    violation rules, and steering directives, so lessons are recalled across
    phrasings. Records that fail :class:`CorrectionExemplar` validation, or
    that belong to a different domain or requested ``tool_name``, are skipped.

    Writes go to ``POST /api/v1/exemplars`` with ``Authorization: Bearer``. They
    require ``write_token`` and an exemplar that carries its
    ``task_description`` (exemplars built with :meth:`CorrectionExemplar.create`
    do). A duplicate ``exemplar_id`` (HTTP 409) is ignored. Any other failure
    raises, so :class:`RamenSteerNode` reports it as ``memory_error``.

    Contributed records, including the task description and tool arguments,
    become publicly readable. Treat retrieved exemplars as untrusted guidance.
    """

    def __init__(
        self,
        base_url: str = "https://ramen-forge.ramenai.workers.dev",
        write_token: str | None = None,
        domain: str = "fintech",
        timeout_sec: float = 5.0,
    ) -> None:
        parts = urlsplit(base_url)
        if parts.scheme != "https" and not (
            parts.scheme == "http" and parts.hostname in _LOCAL_HOSTS
        ):
            raise ValueError("base_url must use https (http is allowed only for localhost)")
        if not isinstance(domain, str) or not _FORGE_DOMAIN_RE.match(domain):
            raise ValueError("domain must be a lowercase slug (a-z, 0-9, '_' or '-', 2-64 characters)")
        if isinstance(timeout_sec, bool) or not isinstance(timeout_sec, (int, float)) or timeout_sec <= 0:
            raise ValueError("timeout_sec must be a positive number")
        self.base_url = base_url.rstrip("/")
        self.write_token = write_token if write_token and write_token.strip() else None
        self.domain = domain
        self.timeout_sec = float(timeout_sec)

    def __repr__(self) -> str:
        # Never include the write token.
        return (
            f"{type(self).__name__}(base_url={self.base_url!r}, domain={self.domain!r}, "
            f"writable={self.write_token is not None})"
        )

    def retrieve_relevant_exemplars(
        self,
        task_fingerprint: str | None = None,
        tool_name: str | None = None,
        limit: int = 3,
        query: str | None = None,
    ) -> list[CorrectionExemplar]:
        _validate_limit(limit)
        if limit > _FORGE_MAX_LIMIT:
            raise ValueError(f"limit must be at most {_FORGE_MAX_LIMIT}")
        url = f"{self.base_url}{_FORGE_EXEMPLARS_PATH}"
        params: dict[str, str] = {"domain": self.domain, "limit": str(limit)}
        if tool_name is not None:
            if not isinstance(tool_name, str) or not tool_name.strip():
                raise ValueError("tool_name must be None or a non-blank string")
            params["tool_name"] = tool_name
        if task_fingerprint is not None:
            if not isinstance(task_fingerprint, str) or not task_fingerprint.strip():
                raise ValueError("task_fingerprint must be None or a non-blank string")
            params["task_fingerprint"] = task_fingerprint
        if query is not None:
            if not isinstance(query, str) or not query.strip():
                raise ValueError("query must be None or a non-blank string")
            if len(query.strip()) > _FORGE_MAX_QUERY_LENGTH:
                raise ValueError(f"query must be at most {_FORGE_MAX_QUERY_LENGTH} characters")
            params["q"] = query.strip()
        try:
            response = httpx.get(
                url,
                params=params,
                headers={"Accept": "application/json"},
                timeout=self.timeout_sec,
            )
            response.raise_for_status()
            body = response.json()
        except (httpx.HTTPError, ValueError) as error:
            logger.warning("ramen-foundry: ramen forge retrieval failed; continuing without exemplars: %s", error)
            return []

        records = body.get("exemplars") if isinstance(body, dict) else None
        if not isinstance(records, list):
            logger.warning("ramen-foundry: ramen forge returned an unexpected response shape; ignoring it")
            return []

        exemplars: list[CorrectionExemplar] = []
        for record in records:
            try:
                exemplar = CorrectionExemplar.from_dict(record)
            except (TypeError, ValueError) as error:
                logger.warning("ramen-foundry: skipping invalid ramen forge exemplar: %s", error)
                continue
            # Fingerprints are deliberately not compared: a lesson recorded for a
            # differently phrased task is still useful for the same tool and domain.
            record_domain = record.get("domain")
            if (tool_name is not None and exemplar.tool_name != tool_name) or (
                record_domain is not None and record_domain != self.domain
            ):
                logger.warning(
                    "ramen-foundry: skipping ramen forge exemplar %s that does not match the query",
                    exemplar.exemplar_id,
                )
                continue
            exemplars.append(exemplar)
        return exemplars[:limit]

    def record_correction(self, exemplar: CorrectionExemplar) -> None:
        if not isinstance(exemplar, CorrectionExemplar):
            raise TypeError("exemplar must be a CorrectionExemplar")
        if exemplar.receipt is None:
            # ramen forge rejects receipt-less records with HTTP 422, so local or
            # mock exemplars are skipped rather than sent.
            logger.warning(
                "ramen-foundry: exemplar %s has no Schema V5 receipt; skipping ramen forge upload",
                exemplar.exemplar_id,
            )
            return
        if self.write_token is None:
            raise PermissionError("RemoteForgeMemoryStore is read-only: no write_token configured")
        if exemplar.task_description is None:
            raise ValueError(
                "exemplar has no task_description; ramen forge requires "
                "task_fingerprint == SHA-256(task_description). Build exemplars "
                "with CorrectionExemplar.create(task=...)."
            )
        payload = exemplar.to_dict() | {
            "domain": self.domain,
            "task_description": exemplar.task_description,
        }
        try:
            response = httpx.post(
                f"{self.base_url}{_FORGE_EXEMPLARS_PATH}",
                json=payload,
                headers={
                    "Authorization": f"Bearer {self.write_token}",
                    "Accept": "application/json",
                },
                timeout=self.timeout_sec,
            )
        except httpx.HTTPError as error:
            raise RuntimeError(f"ramen forge write failed: {error}") from error

        if response.status_code == 409:
            logger.info("ramen-foundry: ramen forge already holds exemplar %s", exemplar.exemplar_id)
            return
        if response.status_code not in (200, 201):
            raise RuntimeError(
                f"ramen forge rejected exemplar {exemplar.exemplar_id} "
                f"(HTTP {response.status_code}): {_forge_error_detail(response)}"
            )
        refreshed = False
        try:
            body = response.json()
            refreshed = isinstance(body, dict) and body.get("refreshed") is True
        except ValueError:
            pass
        if refreshed:
            # Same lesson already stored: ramen forge kept its text and id and
            # refreshed only the receipt, signature, and canonical payload.
            logger.info("ramen-foundry: ramen forge refreshed the receipt of an existing lesson (HTTP 200)")
        else:
            logger.info(
                "ramen-foundry: contributed exemplar %s to ramen forge (HTTP %s)",
                exemplar.exemplar_id,
                response.status_code,
            )


def _forge_error_detail(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return response.text[:200] or "no response body"
    if not isinstance(body, dict):
        return "unexpected response body"
    error = body.get("error")
    if isinstance(error, dict):
        # Receipt rejections use {"code": ..., "message": ...}.
        detail = f"{error.get('code') or 'error'}: {error.get('message') or 'unknown error'}"
    else:
        detail = str(error or "unknown error")
    details = body.get("details")
    if isinstance(details, list) and details:
        detail += ": " + "; ".join(str(item) for item in details[:5])
    return detail
