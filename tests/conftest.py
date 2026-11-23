"""Shared pytest configuration."""

from __future__ import annotations

import socket

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
@pytest.fixture(autouse=True)
def _isolate_shadow_mode_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep a developer's RAMEN_SHADOW_MODE from turning enforcement off in tests."""
    monkeypatch.delenv("RAMEN_SHADOW_MODE", raising=False)


@pytest.fixture(autouse=True)
def _no_network_outside_live_tests(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse every socket connection in offline tests.

    Default tests never contact api.ramenai.dev, including the keyless trial route
    (``/api/v1/paas/try``), whose daily allowance is small and shared per network.
    Tests marked ``live`` are opt-in and keep real network access.
    """
    if request.node.get_closest_marker("live"):
        return

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("an offline test attempted a network connection")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
