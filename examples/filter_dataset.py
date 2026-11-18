"""Filter a CSV through the ramen ai dataset filtration boundary.

Requires: pip install "ramen-foundry[data]" and RAMEN_API_KEY.
"""

from __future__ import annotations

import argparse

from ramen_foundry.data import FiltrationMode, filter_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source")
    parser.add_argument("destination")
    parser.add_argument(
        "--mode",
        choices=[mode.value for mode in FiltrationMode],
        default=FiltrationMode.STRICT_EXCLUSION.value,
    )
    parser.add_argument("--bundle", action="append", required=True)
    parser.add_argument("--remediable-column", action="append", default=[])
    args = parser.parse_args()

    result = filter_csv(
        args.source,
        args.destination,
        mode=args.mode,
        bundle_ids=args.bundle,
        remediable_columns=args.remediable_column,
    )
    print(result.audit_log["verdict"].value_counts().to_dict())


if __name__ == "__main__":
    main()
