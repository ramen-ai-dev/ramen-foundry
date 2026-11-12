"""Fixtures and hooks for the serial live tests. Helpers live in ``tests.live._support``.

Opt in with ``RAMEN_LIVE_TESTS=1`` and select the marker::

    RAMEN_LIVE_TESTS=1 python -m pytest -m live tests/live -s
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.live._support import MATRIX, SpacedClient, api_key


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    live_dir = Path(__file__).resolve().parent
    skip = None
    if os.environ.get("RAMEN_LIVE_TESTS") != "1":
        skip = pytest.mark.skip(reason="live tests are opt-in: set RAMEN_LIVE_TESTS=1 and run with -m live")
    elif api_key() is None:
        skip = pytest.mark.skip(reason="RAMEN_API_KEY is not available")
    if skip is None:
        return
    for item in items:
        if live_dir in Path(str(item.fspath)).resolve().parents:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _verify_receipts_with_test_key() -> None:
    """Override the offline fixture: live receipts must verify under the production keys."""


@pytest.fixture(scope="session")
def live_client() -> Iterator[SpacedClient]:
    key = api_key()
    assert key, "RAMEN_API_KEY is not available"
    client = SpacedClient(key)
    try:
        yield client
    finally:
        client.close()


def pytest_terminal_summary(terminalreporter: Any) -> None:
    if not MATRIX:
        return
    terminalreporter.section("live verification matrix")
    for row in MATRIX:
        terminalreporter.write_line(
            f"{'PASS' if row['passed'] else 'FAIL'} | {row['template']:<24} | {row['scenario']:<58} | "
            f"expected {row['expected']:<5} | {row['votes']:<22} | {row['receipt']:<28} | {row['median_ms']} ms"
        )
