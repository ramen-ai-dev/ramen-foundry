"""Shared fakes for the governed-gate template tests."""

from __future__ import annotations

import base64
import json
from typing import Any

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from langchain_core.tools import StructuredTool
from pydantic import ConfigDict, create_model

from tests.receipt_fixtures import signed_data


class FakeClient:
    """Stands in for ``RamenClient.evaluate_compliance`` with a fixed verdict."""

    def __init__(self, *, allowed: bool = True) -> None:
        self.allowed = allowed
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def evaluate_compliance(self, input_text: str, **kwargs: Any) -> dict[str, Any]:
        self.calls.append((input_text, kwargs))
        return {
            "allowed": self.allowed,
            "receipt_verified": True,
            "receipt_reason": None,
            "steering": None if self.allowed else "Blocked by the policy bundle.",
            "data": signed_data(input_text, allowed=self.allowed),
        }


def _b64url(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode().rstrip("=")


def public_key_hex(private_key: Ed25519PrivateKey) -> str:
    return private_key.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw
    ).hex()


def mint_token(private_key: Ed25519PrivateKey, claims: dict[str, Any], *, kid: str = "k1") -> str:
    """Return a ``v1`` evidence token signed over the claims bytes."""

    claims_bytes = json.dumps(claims, sort_keys=True, separators=(",", ":")).encode()
    signature = private_key.sign(b"ramen-evidence-v1\n" + claims_bytes)
    return f"v1.{kid}.{_b64url(claims_bytes)}.{_b64url(signature)}"


def stub_tool(name: str, executions: list[dict[str, Any]], *fields: str) -> StructuredTool:
    """A tool that records the arguments it receives for the declared ``fields``.

    LangChain passes a tool only the parameters its schema declares.
    """

    def run(**kwargs: Any) -> str:
        executions.append({"tool": name, **kwargs})
        return "ok"

    return StructuredTool.from_function(
        func=run,
        name=name,
        description=f"stub for {name}",
        args_schema=create_model(
            f"{name}_args", __config__=ConfigDict(extra="allow"), **{f: (Any, None) for f in fields}
        ),
    )
