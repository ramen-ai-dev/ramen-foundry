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
import os
import sqlite3
import tempfile
import threading
import uuid
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation of the exemplar."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CorrectionExemplar:
        """Rebuild an exemplar from :meth:`to_dict` output, validating every field."""
        if not isinstance(data, dict):
            raise ValueError("exemplar record must be a JSON object")
        missing = [name for name in _EXEMPLAR_FIELDS if name not in data]
        if missing:
            raise ValueError(f"exemplar record is missing fields: {', '.join(missing)}")
        return cls(**{name: data[name] for name in _EXEMPLAR_FIELDS})


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
            created_at TEXT NOT NULL
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
        )
        with self._lock:
            connection = self._require_connection()
            try:
                with connection:
                    connection.execute(
                        f"INSERT INTO correction_exemplars ({', '.join(_EXEMPLAR_FIELDS)}) "
                        f"VALUES ({', '.join('?' for _ in _EXEMPLAR_FIELDS)})",
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
                f"SELECT {', '.join(_EXEMPLAR_FIELDS)} FROM correction_exemplars "
                "WHERE task_fingerprint = ? AND tool_name = ? "
                "ORDER BY created_at DESC, rowid DESC LIMIT ?",
                (task_fingerprint, tool_name, limit),
            ).fetchall()
        exemplars: list[CorrectionExemplar] = []
        for row in rows:
            record = dict(zip(_EXEMPLAR_FIELDS, row))
            record["failed_arguments"] = json.loads(record["failed_arguments"])
            record["repaired_arguments"] = json.loads(record["repaired_arguments"])
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
