#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dce_analyzer.apollo import APOLLO_DATASETS, load_apollo_long  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Apollo example data and convert to long format."
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(APOLLO_DATASETS.keys()),
        default="swiss_route_choice",
    )
    parser.add_argument(
        "--raw-out",
        type=Path,
        default=None,
        help="Optional raw CSV output path.",
    )
    parser.add_argument(
        "--long-out",
        type=Path,
        default=None,
        help="Optional long CSV output path.",
    )
    return parser.parse_args()


def default_outputs(dataset: str) -> tuple[Path, Path]:
    raw = Path(f"data/raw/apollo_{dataset}.csv")
    long = Path(f"data/processed/apollo_{dataset}_long.csv")
    return raw, long


def main() -> None:
    args = parse_args()
    raw_default, long_default = default_outputs(args.dataset)
    raw_out = args.raw_out or raw_default
    long_out = args.long_out or long_default

    long_df = load_apollo_long(
        dataset=args.dataset,
        raw_output_path=raw_out,
        long_output_path=long_out,
    )
    print(f"Dataset: {args.dataset}")
    print(f"Saved raw Apollo file:  {raw_out}")
    print(f"Saved long-format file: {long_out}")
    print(f"Long-format shape: {long_df.shape}")
    print("Columns:")
    print(", ".join(long_df.columns))


if __name__ == "__main__":
    main()
