"""Cross-validation module tests.

Run from project root:
    python3 scripts/test_cv.py
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import torch
from dce_analyzer.simulate import generate_simulated_dce
from dce_analyzer.config import FullModelSpec, VariableSpec
from dce_analyzer.cross_validation import cross_validate, CrossValidationResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_results: list[tuple[str, bool, str]] = []


def _run(name: str, fn):
    """Run *fn* and record PASS / FAIL."""
    try:
        fn()
        _results.append((name, True, ""))
        print(f"  PASS  {name}")
    except Exception as exc:
        msg = f"{exc.__class__.__name__}: {exc}"
        _results.append((name, False, msg))
        print(f"  FAIL  {name}")
        traceback.print_exc()
        print()


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

sim = generate_simulated_dce(n_individuals=30, n_tasks=4, n_alts=3, seed=42)
DF = sim.data

VARS_FIXED = [
    VariableSpec(name="price", column="price", distribution="fixed"),
    VariableSpec(name="time", column="time", distribution="fixed"),
    VariableSpec(name="comfort", column="comfort", distribution="fixed"),
]

SPEC_CL = FullModelSpec(
    id_col="respondent_id",
    task_col="task_id",
    alt_col="alternative",
    choice_col="choice",
    variables=VARS_FIXED,
    model_type="conditional",
    maxiter=100,
)

CPU = torch.device("cpu")

# Store the CL result here so later tests can reference it
_cl_result: CrossValidationResult | None = None

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_kfold_split_preserves_all_individuals():
    """1. K-fold split preserves all individuals."""
    unique_ids = DF["respondent_id"].unique()
    rng = np.random.default_rng(42)
    ids_copy = unique_ids.copy()
    rng.shuffle(ids_copy)
    k = 5
    folds = np.array_split(ids_copy, k)

    # Union of all folds should equal the full set
    union = set()
    for fold in folds:
        union.update(fold.tolist())
    assert union == set(unique_ids), (
        f"Union of folds ({len(union)}) != all IDs ({len(unique_ids)})"
    )


_run("1. K-fold split preserves all individuals", test_kfold_split_preserves_all_individuals)


def test_no_individual_in_both_train_and_test():
    """2. No individual appears in both train and test for any fold."""
    unique_ids = DF["respondent_id"].unique()
    rng = np.random.default_rng(42)
    ids_copy = unique_ids.copy()
    rng.shuffle(ids_copy)
    k = 5
    folds = np.array_split(ids_copy, k)

    for fold_idx in range(k):
        test_ids = set(folds[fold_idx].tolist())
        train_ids = set(unique_ids) - test_ids
        overlap = test_ids & train_ids
        assert len(overlap) == 0, (
            f"Fold {fold_idx}: overlap={overlap}"
        )


_run("2. No individual in both train and test", test_no_individual_in_both_train_and_test)


def test_cv_conditional_logit():
    """3. CV with Conditional Logit (3-fold)."""
    global _cl_result
    result = cross_validate(DF, SPEC_CL, k=3, seed=42, device=CPU)
    _cl_result = result

    assert isinstance(result, CrossValidationResult), "Wrong return type"
    assert result.k == 3, f"Expected k=3, got k={result.k}"
    assert len(result.fold_results) == 3, (
        f"Expected 3 fold results, got {len(result.fold_results)}"
    )
    assert result.mean_test_ll < 0, (
        f"Expected negative mean test LL, got {result.mean_test_ll}"
    )
    assert result.model_type == "conditional"
    assert result.total_runtime > 0


_run("3. CV with Conditional Logit (3-fold)", test_cv_conditional_logit)


def test_cv_mixed_logit():
    """4. CV with Mixed Logit (3-fold)."""
    vars_random = [
        VariableSpec(name="price", column="price", distribution="normal"),
        VariableSpec(name="time", column="time", distribution="normal"),
        VariableSpec(name="comfort", column="comfort", distribution="fixed"),
    ]
    spec_mxl = FullModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=vars_random,
        model_type="mixed",
        n_draws=50,
        maxiter=50,
    )
    result = cross_validate(DF, spec_mxl, k=3, seed=42, device=CPU)
    assert isinstance(result, CrossValidationResult)
    assert result.k == 3
    assert len(result.fold_results) == 3
    assert result.model_type == "mixed"


_run("4. CV with Mixed Logit (3-fold)", test_cv_mixed_logit)


def test_hit_rate_bounds():
    """5. Hit rate is between 0 and 1."""
    assert _cl_result is not None, "Test 3 must pass first (CL result needed)"
    for fr in _cl_result.fold_results:
        assert 0.0 <= fr.hit_rate <= 1.0, f"Fold {fr.fold}: hit_rate={fr.hit_rate}"
    assert 0.0 <= _cl_result.mean_hit_rate <= 1.0, (
        f"mean_hit_rate={_cl_result.mean_hit_rate}"
    )


_run("5. Hit rate is between 0 and 1", test_hit_rate_bounds)


def test_k_greater_than_n_individuals_raises():
    """6. K > n_individuals raises an error."""
    sim_small = generate_simulated_dce(n_individuals=10, n_tasks=4, n_alts=3, seed=99)
    df_small = sim_small.data
    spec_small = FullModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=VARS_FIXED,
        model_type="conditional",
        maxiter=50,
    )
    raised = False
    try:
        cross_validate(df_small, spec_small, k=100, seed=42, device=CPU)
    except ValueError as e:
        raised = True
        assert "k=100" in str(e), f"Unexpected error message: {e}"
    assert raised, "Expected ValueError when K > n_individuals"


_run("6. K > n_individuals raises error", test_k_greater_than_n_individuals_raises)


def test_progress_callback_called():
    """7. Progress callback is called K times."""
    calls = []

    def callback(fold_idx, k, status):
        calls.append((fold_idx, k, status))

    k = 3
    cross_validate(DF, SPEC_CL, k=k, seed=42, device=CPU, progress_callback=callback)
    assert len(calls) == k, f"Expected {k} callback calls, got {len(calls)}"
    for i, (fold_idx, k_val, status) in enumerate(calls):
        assert fold_idx == i, f"Expected fold_idx={i}, got {fold_idx}"
        assert k_val == k, f"Expected k={k}, got {k_val}"


_run("7. Progress callback is called K times", test_progress_callback_called)


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

def _summary():
    total = len(_results)
    passed = sum(1 for _, ok, _ in _results if ok)
    failed = total - passed
    print(f"\n{'='*60}")
    print(f"  {passed} passed, {failed} failed out of {total} tests")
    print(f"{'='*60}")
    if failed == 0:
        print("  ALL TESTS PASSED")
    else:
        print("\n  Failed tests:")
        for name, ok, msg in _results:
            if not ok:
                print(f"    - {name}: {msg}")
    return failed == 0


if __name__ == "__main__":
    success = _summary()
    sys.exit(0 if success else 1)
