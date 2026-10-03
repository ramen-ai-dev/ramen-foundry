"""Shared pytest configuration."""

from __future__ import annotations

import pytest

import ramen_foundry.core.receipts as receipts
from tests.receipt_fixtures import TEST_PUBLIC_KEYS


@pytest.fixture(autouse=True)
def _verify_receipts_with_test_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the default receipt key map at the test signing key.

    Fake clients sign receipts with a generated key, so the tool nodes'
    signed-ALLOW check runs for real in every test instead of being bypassed.
    """
    monkeypatch.setattr(receipts, "AUDIT_PUBLIC_KEYS", TEST_PUBLIC_KEYS)
