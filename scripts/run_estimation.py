#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dce_analyzer.config import ModelSpec, VariableSpec  # noqa: E402
from dce_analyzer.pipeline import estimate_dataframe, save_estimation_outputs  # noqa: E402


def parse_variable_specs(raw: str) -> list[VariableSpec]:
    """
    Parse:
      "price:fixed,time:normal,comfort:fixed"
    """
    variables: list[VariableSpec] = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        parts = [p.strip() for p in token.split(":")]
        if len(parts) == 1:
            col, dist = parts[0], "fixed"
        elif len(parts) == 2:
            col, dist = parts
        else:
            raise ValueError(
                f"Could not parse variable token '{token}'. Use name[:distribution]."
            )
        variables.append(VariableSpec(name=col, column=col, distribution=dist))
    if not variables:
        raise ValueError("No variables were provided.")
    return variables


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run discrete-choice estimation from a long-format CSV file."
    )
    parser.add_argument("--data", type=Path, required=True, help="Input long-format CSV")
    parser.add_argument("--id-col", required=True)
    parser.add_argument("--task-col", required=True)
    parser.add_argument("--alt-col", required=True)
    parser.add_argument("--choice-col", required=True)
    parser.add_argument(
        "--variables",
        required=True,
        help="Comma-separated variable list: col[:dist], e.g., price:normal,time:normal",
    )
    parser.add_argument("--model", choices=["mixed", "conditional", "latent_class"], default="mixed")
    parser.add_argument("--draws", type=int, default=200)
    parser.add_argument("--n-classes", type=int, default=2, help="Number of latent classes (only for latent_class model)")
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--out-prefix", type=Path, default=Path("outputs/model"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = pd.read_csv(args.data)
    variables = parse_variable_specs(args.variables)
    spec = ModelSpec(
        id_col=args.id_col,
        task_col=args.task_col,
        alt_col=args.alt_col,
        choice_col=args.choice_col,
        variables=variables,
        n_draws=args.draws,
        n_classes=args.n_classes,
    )

    result = estimate_dataframe(
        df=df,
        spec=spec,
        model_type=args.model,
        maxiter=args.maxiter,
        seed=args.seed,
    )
    save_estimation_outputs(result.estimation, args.out_prefix)

    summary = result.estimation.summary_dict()
    print("Estimation finished.")
    print(f"Converged: {summary['success']}")
    print(f"Message: {summary['message']}")
    print(f"Log-likelihood: {summary['log_likelihood']:.4f}")
    print(f"AIC: {summary['aic']:.2f}")
    print(f"BIC: {summary['bic']:.2f}")
    print(f"Iterations: {summary['optimizer_iterations']}")
    print(f"Runtime (s): {summary['runtime_seconds']:.2f}")
    print(f"Saved outputs with prefix: {args.out_prefix}")


if __name__ == "__main__":
    main()
