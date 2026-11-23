"""Filter a CSV through the ramen ai dataset filtration boundary.

Every row is evaluated against one bundle and only allowed rows are written
(strict exclusion). The sample file ships with the repository, so this runs as is.

Run from the repository root::

    pip install -e ".[data]"
    python examples/filter_dataset.py --trial          # no API key: the keyless public trial
    RAMEN_API_KEY=... python examples/filter_dataset.py

The keyless trial allows about 10 requests per day per network and one request
per row, so use it with small files such as the five-row sample. Pass your own
files and bundle with ``SOURCE``, ``DESTINATION`` and ``--bundle``, for example
``--bundle ramen__shield_core_it``.
"""
from __future__ import annotations

import argparse
import os
import sys
from collections.abc import Sequence
from pathlib import Path

from ramen_ai import TrialLimitError

from ramen_foundry.data import FiltrationError, FiltrationMode, filter_csv
from ramen_foundry.trial import describe_trial_limit

SAMPLE_CSV = Path(__file__).resolve().parent / "data" / "sample_documents.csv"
DEFAULT_BUNDLE = "ramen__shield_core_it"


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("source", nargs="?", default=str(SAMPLE_CSV), help="input CSV (default: the bundled sample)")
    parser.add_argument(
        "destination",
        nargs="?",
        default="filtered_documents.csv",
        help="output CSV with the allowed rows (default: filtered_documents.csv)",
    )
    parser.add_argument(
        "--bundle",
        action="append",
        default=None,
        help=f"bundle slug to evaluate against (default: {DEFAULT_BUNDLE}); repeat for several with an API key",
    )
    parser.add_argument(
        "--trial",
        action="store_true",
        help="use the keyless public trial instead of RAMEN_API_KEY (one bundle, small files)",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    try:
        result = filter_csv(
            args.source,
            args.destination,
            mode=FiltrationMode.STRICT_EXCLUSION,
            bundle_ids=args.bundle or [DEFAULT_BUNDLE],
            trial=args.trial,
        )
    except FiltrationError as error:
        if args.trial and isinstance(error.__cause__, TrialLimitError):
            print(f"Partial run, nothing written: {describe_trial_limit(error.__cause__)}")
            return 0
        print(f"filtration failed: {error}", file=sys.stderr)
        if not args.trial and not os.environ.get("RAMEN_API_KEY"):
            print("Tip: add --trial to run on the keyless public trial.", file=sys.stderr)
        return 1
    print(result.audit_log["verdict"].value_counts().to_dict())
    print(f"Wrote {len(result.dataframe)} allowed rows to {args.destination}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
