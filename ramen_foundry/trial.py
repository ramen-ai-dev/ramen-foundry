"""Keyless public trial adapter for ramen foundry.

``RamenClient.trial()`` returns a :class:`ramen_ai.RamenTrialClient`, which exposes
``try_evaluate`` rather than ``evaluate_compliance``. :class:`TrialComplianceClient`
adapts it to the ``evaluate_compliance`` call that :class:`~ramen_foundry.RamenToolNode`,
the governed templates and :func:`ramen_foundry.data.filter_dataframe` make, so a
developer without an API key can run live semantic evaluations::

    from ramen_foundry import DevboxShieldAgent, TrialComplianceClient

    with TrialComplianceClient() as client:
        agent = DevboxShieldAgent(client=client, tools={"inspect_directory": inspect_directory})

The trial evaluates one bundle per request, from the bundles the service allows,
with a hard 2,048-byte request body. It sends no credentials and no context,
and it ignores ``provider_key`` and ``provider_name``. This module imports
neither LangGraph nor pandas.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from typing import Any

from ramen_ai import RamenClient, RamenTrialClient, TrialLimitError

TRIAL_SIGNUP_URL = "https://ramenai.dev"


def describe_trial_limit(error: TrialLimitError) -> str:
    """Return a one-line description of an exhausted trial allowance."""
    scope = getattr(error, "scope", "unknown")
    reset = getattr(error, "reset_epoch_seconds", None)
    if isinstance(reset, int) and not isinstance(reset, bool):
        when = datetime.fromtimestamp(reset, tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    else:
        when = "00:00 UTC"
    signup = getattr(error, "signup_url", None)
    url = signup if isinstance(signup, str) and signup.startswith(TRIAL_SIGNUP_URL) else TRIAL_SIGNUP_URL
    return (
        f"The keyless trial allowance is used up (scope: {scope}). It resets at {when}. "
        f"Get an API key at {url} to continue."
    )


class TrialComplianceClient:
    """Adapt the keyless trial client to the ``evaluate_compliance`` interface.

    ``evaluate_compliance`` raises ``ValueError`` before any request unless exactly
    one bundle id is given and no ``policy_ids`` are. ``context``, ``attestation``,
    ``provider_key`` and ``provider_name`` are accepted for call compatibility and
    never sent. Every verdict carries a locally verified ``trial_<uuid>`` receipt
    (the SDK fails closed otherwise) and a ``trial`` mapping with the remaining
    daily allowance.

    The adapter keeps the most recent exhausted-allowance error in
    :attr:`last_limit`, because governed nodes flatten exceptions into text.
    """

    def __init__(self, client: RamenTrialClient | None = None, **client_options: Any) -> None:
        if client is not None and client_options:
            raise TypeError("pass either an existing trial client or client options, not both")
        self._owns_client = client is None
        self._client: RamenTrialClient = client if client is not None else RamenClient.trial(**client_options)
        self.last_limit: TrialLimitError | None = None
        self.last_quota: Mapping[str, int] = {}

    def evaluate_compliance(
        self,
        input_text: str,
        *,
        bundle_ids: Sequence[str] | None = None,
        policy_ids: Sequence[str] | None = None,
        context: Mapping[str, str] | None = None,
        attestation: Any = None,
        provider_key: str | None = None,
        provider_name: str | None = None,
    ) -> Any:
        """Evaluate ``input_text`` against exactly one trial bundle."""
        del context, attestation, provider_key, provider_name  # never sent
        if policy_ids:
            raise ValueError("the keyless trial evaluates bundles only; policy_ids need an API key")
        bundles = list(bundle_ids or [])
        if len(bundles) != 1:
            raise ValueError("the keyless trial evaluates exactly one bundle per request")
        try:
            verdict = self._client.try_evaluate(input_text, bundle_id=bundles[0])
        except TrialLimitError as error:
            self.last_limit = error
            raise
        self.last_quota = dict(verdict.get("trial") or {})
        return verdict

    def close(self) -> None:
        """Close the underlying client when this adapter created it."""
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> TrialComplianceClient:
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()
