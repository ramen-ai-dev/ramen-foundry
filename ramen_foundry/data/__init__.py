"""Dataset filtration: evaluate every row of a DataFrame or CSV through ramen ai.

Requires the optional ``data`` extra (pandas)::

    pip install "ramen-foundry[data]"

``import ramen_foundry.data`` does not import LangGraph.
"""
from __future__ import annotations

try:
    import pandas  # noqa: F401
except ImportError as error:  # pragma: no cover - exercised through a subprocess test
    raise ImportError(
        "ramen_foundry.data needs pandas. Install it with: pip install \"ramen-foundry[data]\""
    ) from error

from .filtration import (
    FiltrationError,
    FiltrationMode,
    FiltrationResult,
    filter_csv,
    filter_dataframe,
)

__all__ = [
    "FiltrationError",
    "FiltrationMode",
    "FiltrationResult",
    "filter_csv",
    "filter_dataframe",
]
