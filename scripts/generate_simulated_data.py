#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dce_analyzer.simulate import generate_simulated_dce  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a simulated long-format choice dataset.")
    parser.add_argument("--n-individuals", type=int, default=300)
    parser.add_argument("--n-tasks", type=int, default=8)
    parser.add_argument("--n-alts", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/generated/simulated_choice_data.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--truth-out",
        type=Path,
        default=Path("data/generated/simulated_truth.json"),
        help="Output JSON path for true simulation parameters.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = generate_simulated_dce(
        n_individuals=args.n_individuals,
        n_tasks=args.n_tasks,
        n_alts=args.n_alts,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.truth_out.parent.mkdir(parents=True, exist_ok=True)
    output.data.to_csv(args.out, index=False)
    with open(args.truth_out, "w", encoding="utf-8") as handle:
        json.dump(output.true_parameters, handle, indent=2, default=str)

    print(f"Saved simulated data to: {args.out}")
    print(f"Saved true parameters to: {args.truth_out}")
    print(f"Rows: {len(output.data)}, Columns: {len(output.data.columns)}")


if __name__ == "__main__":
    main()
