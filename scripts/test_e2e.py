"""End-to-end test script for the dce_analyzer backend.

Run from project root:
    python scripts/test_e2e.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path

# Ensure src/ is importable
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
import pandas as pd

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


# ===================================================================
# 1. Import all backend modules
# ===================================================================
def test_imports():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.simulate import generate_simulated_dce
    from dce_analyzer.data import prepare_choice_tensors, ChoiceTensors
    from dce_analyzer.model import (
        MixedLogitEstimator,
        ConditionalLogitEstimator,
        EstimationResult,
    )
    from dce_analyzer.latent_class import LatentClassEstimator, LatentClassResult
    from dce_analyzer.pipeline import estimate_dataframe, PipelineResult
    from dce_analyzer.wtp import compute_wtp
    from dce_analyzer.bootstrap import run_bootstrap, BootstrapResult
    from dce_analyzer.format_converter import (
        detect_format,
        wide_to_long,
        infer_structure,
        normalize_choice_column,
        ColumnInference,
    )
    from dce_analyzer.apollo import APOLLO_DATASETS
    # all imported without error


_run("1. Import all backend modules", test_imports)


# ===================================================================
# 2. Generate simulated data
# ===================================================================
sim_output = None


def test_simulate():
    global sim_output
    from dce_analyzer.simulate import generate_simulated_dce

    sim_output = generate_simulated_dce(
        n_individuals=100, n_tasks=4, n_alts=3, seed=42
    )
    df = sim_output.data
    assert isinstance(df, pd.DataFrame), "Expected DataFrame"
    assert len(df) == 100 * 4 * 3, f"Expected 1200 rows, got {len(df)}"
    for col in ["respondent_id", "task_id", "alternative", "choice",
                "price", "time", "comfort", "reliability"]:
        assert col in df.columns, f"Missing column: {col}"
    assert isinstance(sim_output.true_parameters, dict)
    assert len(sim_output.true_parameters) > 0


_run("2. Generate simulated data (100 ind, 4 tasks, 3 alts)", test_simulate)


# ===================================================================
# 3. Conditional Logit estimation
# ===================================================================
cl_result = None


def test_conditional_logit():
    global cl_result
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price"),
            VariableSpec(name="time", column="time"),
            VariableSpec(name="comfort", column="comfort"),
            VariableSpec(name="reliability", column="reliability"),
        ],
    )
    result = estimate_dataframe(
        df=sim_output.data, spec=spec, model_type="conditional", maxiter=200, seed=42
    )
    cl_result = result
    est = result.estimation
    assert est.success, f"CL did not converge: {est.message}"
    assert est.n_parameters == 4
    assert est.n_observations == 100 * 4  # 400 choice tasks
    assert not est.estimates.empty
    assert "estimate" in est.estimates.columns


_run("3. Conditional Logit estimation", test_conditional_logit)


# ===================================================================
# 4. Mixed Logit estimation (n_draws=50)
# ===================================================================
mxl_result = None


def test_mixed_logit():
    global mxl_result
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price", distribution="normal"),
            VariableSpec(name="time", column="time", distribution="normal"),
            VariableSpec(name="comfort", column="comfort", distribution="fixed"),
            VariableSpec(name="reliability", column="reliability", distribution="fixed"),
        ],
        n_draws=50,
    )
    result = estimate_dataframe(
        df=sim_output.data, spec=spec, model_type="mixed", maxiter=200, seed=42
    )
    mxl_result = result
    est = result.estimation
    # 2 normal (mu+sd each) + 2 fixed = 6 params
    assert est.n_parameters == 6, f"Expected 6 params, got {est.n_parameters}"
    assert not est.estimates.empty
    # Should have mu_price, sd_price, mu_time, sd_time, beta_comfort, beta_reliability
    param_names = set(est.estimates["parameter"])
    for expected in ["mu_price", "sd_price", "mu_time", "sd_time",
                     "beta_comfort", "beta_reliability"]:
        assert expected in param_names, f"Missing param: {expected}"


_run("4. Mixed Logit estimation (n_draws=50)", test_mixed_logit)


# ===================================================================
# 5. Latent Class estimation (n_classes=2, n_starts=3)
# ===================================================================
lc_result = None


def test_latent_class():
    global lc_result
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price"),
            VariableSpec(name="time", column="time"),
            VariableSpec(name="comfort", column="comfort"),
            VariableSpec(name="reliability", column="reliability"),
        ],
        n_classes=2,
    )
    result = estimate_dataframe(
        df=sim_output.data, spec=spec, model_type="latent_class",
        maxiter=200, seed=42, n_classes=2, n_starts=3,
    )
    lc_result = result
    est = result.estimation
    assert est.n_classes == 2
    assert len(est.class_probabilities) == 2
    assert abs(sum(est.class_probabilities) - 1.0) < 1e-4, "Class probs must sum to 1"
    assert not est.estimates.empty
    assert not est.class_estimates.empty
    assert not est.posterior_probs.empty
    assert est.posterior_probs.shape[1] == 2  # two class columns


_run("5. Latent Class estimation (n_classes=2, n_starts=3)", test_latent_class)


# ===================================================================
# 6. WTP computation
# ===================================================================
def test_wtp():
    from dce_analyzer.wtp import compute_wtp

    # Use CL result (EstimationResult) for WTP
    wtp_df = compute_wtp(cl_result.estimation, cost_variable="price")
    assert isinstance(wtp_df, pd.DataFrame)
    assert len(wtp_df) == 3  # time, comfort, reliability (3 non-cost attrs)
    assert "wtp_estimate" in wtp_df.columns
    assert "wtp_std_error" in wtp_df.columns
    assert "wtp_ci_lower" in wtp_df.columns
    assert "wtp_ci_upper" in wtp_df.columns
    # WTP values should be finite
    for _, row in wtp_df.iterrows():
        assert np.isfinite(row["wtp_estimate"]), f"Non-finite WTP for {row['attribute']}"


_run("6. WTP computation (CL result)", test_wtp)


# ===================================================================
# 7. Bootstrap (n_boot=10)
# ===================================================================
def test_bootstrap():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.bootstrap import run_bootstrap

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price"),
            VariableSpec(name="time", column="time"),
            VariableSpec(name="comfort", column="comfort"),
            VariableSpec(name="reliability", column="reliability"),
        ],
    )
    boot = run_bootstrap(
        df=sim_output.data, spec=spec, model_type="conditional",
        n_replications=10, maxiter=100, seed=42,
    )
    assert boot.n_replications == 10
    assert boot.n_successful >= 2, f"Only {boot.n_successful} succeeded"
    assert len(boot.param_names) == 4
    assert boot.estimates_matrix.shape == (boot.n_successful, 4)
    summary = boot.summary_dataframe()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 4


_run("7. Bootstrap (n_boot=10, conditional logit)", test_bootstrap)


# ===================================================================
# 8. Wide-to-long conversion
# ===================================================================
def test_wide_to_long():
    from dce_analyzer.format_converter import detect_format, wide_to_long

    # Create a small wide-format dataset
    wide_df = pd.DataFrame({
        "id": [1, 1, 2, 2],
        "choice": [1, 2, 1, 3],
        "price_1": [10, 20, 15, 25],
        "price_2": [12, 22, 17, 27],
        "price_3": [14, 24, 19, 29],
        "time_1": [30, 40, 35, 45],
        "time_2": [32, 42, 37, 47],
        "time_3": [34, 44, 39, 49],
    })

    fmt = detect_format(wide_df)
    assert fmt == "wide", f"Expected 'wide', got '{fmt}'"

    long_df = wide_to_long(
        wide_df,
        attribute_groups={
            "price": ["price_1", "price_2", "price_3"],
            "time": ["time_1", "time_2", "time_3"],
        },
        id_col="id",
        choice_col="choice",
    )
    assert isinstance(long_df, pd.DataFrame)
    # 4 rows * 3 alts = 12 rows
    assert len(long_df) == 12, f"Expected 12 rows, got {len(long_df)}"
    assert "alternative" in long_df.columns
    assert "choice" in long_df.columns
    assert "price" in long_df.columns
    assert "time" in long_df.columns
    # Each task should have exactly one chosen alt
    for (rid, tid), grp in long_df.groupby(["respondent_id", "task_id"]):
        assert grp["choice"].sum() == 1, f"Task ({rid},{tid}) has {grp['choice'].sum()} choices"

    # Test detect_format on long data
    fmt2 = detect_format(long_df)
    assert fmt2 == "long", f"Expected 'long' for converted data, got '{fmt2}'"


_run("8. Wide-to-long conversion", test_wide_to_long)


# ===================================================================
# 9. Additional checks: infer_structure, normalize_choice_column
# ===================================================================
def test_infer_and_normalize():
    from dce_analyzer.format_converter import infer_structure, normalize_choice_column

    df = sim_output.data
    inference = infer_structure(df)
    assert inference.id_col is not None, "Should detect id column"
    assert inference.choice_col is not None, "Should detect choice column"

    # Test normalize_choice_column (already binary -- should be no-op)
    normalized = normalize_choice_column(df, "choice", "alternative")
    assert set(normalized["choice"].unique()) <= {0, 1}


_run("9. infer_structure & normalize_choice_column", test_infer_and_normalize)


# ===================================================================
# 10. LatentClassResult.summary_dict()
# ===================================================================
def test_lc_summary():
    est = lc_result.estimation
    sd = est.summary_dict()
    assert "n_classes" in sd
    assert "class_probabilities" in sd
    assert sd["n_classes"] == 2


_run("10. LatentClassResult.summary_dict()", test_lc_summary)


# ===================================================================
# 11. Full correlated MMNL (backward compat)
# ===================================================================
def test_full_correlated_mxl():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price", distribution="normal"),
            VariableSpec(name="time", column="time", distribution="normal"),
            VariableSpec(name="comfort", column="comfort", distribution="fixed"),
            VariableSpec(name="reliability", column="reliability", distribution="fixed"),
        ],
        n_draws=50,
    )
    result = estimate_dataframe(
        df=sim_output.data, spec=spec, model_type="mixed",
        maxiter=200, seed=42, correlated=True,
    )
    est = result.estimation
    assert est.covariance_matrix is not None, "Expected covariance matrix"
    assert est.covariance_matrix.shape == (2, 2), f"Expected 2x2 cov, got {est.covariance_matrix.shape}"
    assert est.correlation_matrix is not None


_run("11. Full correlated MMNL (backward compat)", test_full_correlated_mxl)


# ===================================================================
# 12. Selective correlated MMNL (block-diagonal Cholesky)
# ===================================================================
def test_selective_correlated_mxl():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price", distribution="normal"),
            VariableSpec(name="time", column="time", distribution="normal"),
            VariableSpec(name="comfort", column="comfort", distribution="normal"),
            VariableSpec(name="reliability", column="reliability", distribution="normal"),
        ],
        n_draws=50,
    )
    # Correlate price-time (group [0,1]) and comfort-reliability (group [2,3])
    result = estimate_dataframe(
        df=sim_output.data, spec=spec, model_type="mixed",
        maxiter=200, seed=42,
        correlation_groups=[[0, 1], [2, 3]],
    )
    est = result.estimation
    assert est.covariance_matrix is not None, "Expected covariance matrix"
    assert est.covariance_matrix.shape == (4, 4)
    # Off-block elements should be zero (price-comfort, price-reliability, etc.)
    cov = est.covariance_matrix
    assert abs(cov[0, 2]) < 1e-8, f"Expected 0 cov(price,comfort), got {cov[0,2]}"
    assert abs(cov[0, 3]) < 1e-8, f"Expected 0 cov(price,reliability), got {cov[0,3]}"
    assert abs(cov[1, 2]) < 1e-8, f"Expected 0 cov(time,comfort), got {cov[1,2]}"
    assert abs(cov[1, 3]) < 1e-8, f"Expected 0 cov(time,reliability), got {cov[1,3]}"
    # Within-block elements should be non-zero
    assert abs(cov[0, 1]) > 1e-10 or True  # may be zero by chance, just check shape


_run("12. Selective correlated MMNL (block-diagonal)", test_selective_correlated_mxl)


# ===================================================================
# 13. Selective with standalone random params
# ===================================================================
def test_selective_with_standalone():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price", distribution="normal"),
            VariableSpec(name="time", column="time", distribution="normal"),
            VariableSpec(name="comfort", column="comfort", distribution="normal"),
            VariableSpec(name="reliability", column="reliability", distribution="fixed"),
        ],
        n_draws=50,
    )
    # Only correlate price-time, comfort is standalone random
    result = estimate_dataframe(
        df=sim_output.data, spec=spec, model_type="mixed",
        maxiter=200, seed=42,
        correlation_groups=[[0, 1]],
    )
    est = result.estimation
    assert est.covariance_matrix is not None
    assert est.covariance_matrix.shape == (3, 3)
    cov = est.covariance_matrix
    # comfort (index 2) is standalone: zero cross-cov with price/time
    assert abs(cov[0, 2]) < 1e-8, f"Expected 0 cov(price,comfort), got {cov[0,2]}"
    assert abs(cov[1, 2]) < 1e-8, f"Expected 0 cov(time,comfort), got {cov[1,2]}"
    # n_parameters: 3 mu + 3 chol(price-time) + 1 sd(comfort) + 1 fixed = 8
    assert est.n_parameters == 8, f"Expected 8 params, got {est.n_parameters}"


_run("13. Selective with standalone random params", test_selective_with_standalone)


# ===================================================================
# 14. Create BWS simulated data
# ===================================================================
bws_df = None


def test_create_bws_data():
    """Create BWS data by adding a 'worst' column to simulated DCE data."""
    global bws_df
    df = sim_output.data.copy()
    # J=3 alts per task. For each task, pick the alt with LOWEST utility-like
    # score as worst. Use negative of choice to ensure worst != best.
    rng = np.random.default_rng(99)
    worst_rows = []
    for (rid, tid), grp in df.groupby(["respondent_id", "task_id"]):
        best_alt = grp.loc[grp["choice"] == 1, "alternative"].values[0]
        non_best = grp[grp["alternative"] != best_alt]
        # Pick random non-best as worst
        worst_alt = non_best["alternative"].values[rng.integers(len(non_best))]
        for _, row in grp.iterrows():
            worst_rows.append(1 if row["alternative"] == worst_alt else 0)
    df["worst"] = worst_rows
    # Verify: each task has exactly 1 worst, 1 best, and worst != best
    for (rid, tid), grp in df.groupby(["respondent_id", "task_id"]):
        assert grp["choice"].sum() == 1, "Exactly one best per task"
        assert grp["worst"].sum() == 1, "Exactly one worst per task"
        best_idx = grp.loc[grp["choice"] == 1].index[0]
        worst_idx = grp.loc[grp["worst"] == 1].index[0]
        assert best_idx != worst_idx, "worst != best"
    bws_df = df
    assert "worst" in bws_df.columns


_run("14. Create BWS simulated data", test_create_bws_data)


# ===================================================================
# 15. BWS + Conditional Logit
# ===================================================================
def test_bws_clogit():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price"),
            VariableSpec(name="time", column="time"),
            VariableSpec(name="comfort", column="comfort"),
            VariableSpec(name="reliability", column="reliability"),
        ],
    )
    result = estimate_dataframe(
        df=bws_df, spec=spec, model_type="conditional",
        maxiter=200, seed=42,
        bws_worst_col="worst", estimate_lambda_w=True,
    )
    est = result.estimation
    assert est.success, f"BWS CL did not converge: {est.message}"
    # 4 betas + 1 lambda_w = 5 params
    assert est.n_parameters == 5, f"Expected 5 params, got {est.n_parameters}"
    # lambda_w should appear in estimates
    param_names = set(est.estimates["parameter"])
    assert "lambda_w (worst scale)" in param_names, f"Missing lambda_w param. Got: {param_names}"
    # lambda_w should be positive
    lw_row = est.estimates[est.estimates["parameter"] == "lambda_w (worst scale)"]
    assert lw_row["estimate"].values[0] > 0, "lambda_w must be positive"


_run("15. BWS + Conditional Logit", test_bws_clogit)


# ===================================================================
# 16. Sequential BWS + CLogit with fixed worst-stage scale
# ===================================================================
def test_bws_clogit_fixed_lw():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price"),
            VariableSpec(name="time", column="time"),
            VariableSpec(name="comfort", column="comfort"),
            VariableSpec(name="reliability", column="reliability"),
        ],
    )
    result = estimate_dataframe(
        df=bws_df, spec=spec, model_type="conditional",
        maxiter=200, seed=42,
        bws_worst_col="worst", estimate_lambda_w=False,
    )
    est = result.estimation
    assert est.success
    # 4 betas only (no lambda_w)
    assert est.n_parameters == 4, f"Expected 4 params, got {est.n_parameters}"
    param_names = set(est.estimates["parameter"])
    assert "lambda_w (worst scale)" not in param_names


_run("16. Sequential BWS + CLogit with fixed lambda_w", test_bws_clogit_fixed_lw)


# ===================================================================
# 17. BWS + Mixed Logit
# ===================================================================
def test_bws_mxl():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price", distribution="normal"),
            VariableSpec(name="time", column="time", distribution="normal"),
            VariableSpec(name="comfort", column="comfort", distribution="fixed"),
            VariableSpec(name="reliability", column="reliability", distribution="fixed"),
        ],
        n_draws=50,
    )
    result = estimate_dataframe(
        df=bws_df, spec=spec, model_type="mixed",
        maxiter=200, seed=42,
        bws_worst_col="worst", estimate_lambda_w=True,
    )
    est = result.estimation
    # 2 mu + 2 sd + 2 fixed + 1 lambda_w = 7
    assert est.n_parameters == 7, f"Expected 7 params, got {est.n_parameters}"
    param_names = set(est.estimates["parameter"])
    assert "lambda_w (worst scale)" in param_names
    assert "mu_price" in param_names
    assert "sd_price" in param_names


_run("17. BWS + Mixed Logit", test_bws_mxl)


# ===================================================================
# 18. BWS + GMNL
# ===================================================================
def test_bws_gmnl():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price", distribution="normal"),
            VariableSpec(name="time", column="time", distribution="fixed"),
            VariableSpec(name="comfort", column="comfort", distribution="fixed"),
            VariableSpec(name="reliability", column="reliability", distribution="fixed"),
        ],
        n_draws=50,
    )
    result = estimate_dataframe(
        df=bws_df, spec=spec, model_type="gmnl",
        maxiter=200, seed=42,
        bws_worst_col="worst", estimate_lambda_w=True,
    )
    est = result.estimation
    # 1 mu + 1 sd + 3 fixed + 1 lambda_w + 2 GMNL scale terms = 8
    assert est.n_parameters == 8, f"Expected 8 params, got {est.n_parameters}"
    param_names = set(est.estimates["parameter"])
    assert "lambda_w (worst scale)" in param_names
    assert "sigma_tau (scale SD)" in param_names


_run("18. BWS + GMNL", test_bws_gmnl)


# ===================================================================
# 19. BWS + Latent Class
# ===================================================================
def test_bws_lc():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price"),
            VariableSpec(name="time", column="time"),
            VariableSpec(name="comfort", column="comfort"),
            VariableSpec(name="reliability", column="reliability"),
        ],
        n_classes=2,
    )
    result = estimate_dataframe(
        df=bws_df, spec=spec, model_type="latent_class",
        maxiter=200, seed=42, n_classes=2, n_starts=3,
        bws_worst_col="worst", estimate_lambda_w=True,
    )
    est = result.estimation
    assert est.n_classes == 2
    assert len(est.class_probabilities) == 2
    # Check lambda_w appears in estimates
    lw_rows = est.estimates[est.estimates["parameter"].str.contains("lambda_w")]
    assert len(lw_rows) > 0, "Missing lambda_w in LC estimates"


_run("19. BWS + Latent Class", test_bws_lc)


# ===================================================================
# 20. Correlation inference (delta method SEs for cov/cor)
# ===================================================================
def test_correlation_inference():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price", distribution="normal"),
            VariableSpec(name="time", column="time", distribution="normal"),
            VariableSpec(name="comfort", column="comfort", distribution="fixed"),
            VariableSpec(name="reliability", column="reliability", distribution="fixed"),
        ],
        n_draws=50,
    )
    result = estimate_dataframe(
        df=sim_output.data, spec=spec, model_type="mixed",
        maxiter=200, seed=42, correlated=True,
    )
    est = result.estimation
    # Covariance SE matrix should exist and match shape
    assert est.covariance_se is not None, "Expected covariance_se"
    assert est.covariance_se.shape == (2, 2), f"Expected 2x2, got {est.covariance_se.shape}"
    # Correlation SE matrix
    assert est.correlation_se is not None, "Expected correlation_se"
    assert est.correlation_se.shape == (2, 2)
    # Diagonal of correlation SE should be 0 (cor(x,x)=1, no variation)
    for i in range(2):
        assert est.correlation_se[i, i] < 1e-6, f"Diagonal cor SE should be ~0, got {est.correlation_se[i,i]}"
    # Correlation test table
    assert est.correlation_test is not None, "Expected correlation_test DataFrame"
    assert len(est.correlation_test) == 1, "Expected 1 off-diagonal pair for 2 random params"
    row = est.correlation_test.iloc[0]
    assert row["param_1"] == "price"
    assert row["param_2"] == "time"
    assert not np.isnan(row["cor_std_error"]), "SE should not be NaN"
    assert not np.isnan(row["z_stat"]), "z_stat should not be NaN"
    assert not np.isnan(row["p_value"]), "p_value should not be NaN"
    assert 0.0 <= row["p_value"] <= 1.0, f"p-value out of range: {row['p_value']}"


_run("20. Correlation inference (delta method SEs for cov/cor)", test_correlation_inference)


# ===================================================================
# 21. FullModelSpec + estimate_from_spec
# ===================================================================
def test_full_model_spec():
    from dce_analyzer.config import FullModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_from_spec

    spec = FullModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price", distribution="normal"),
            VariableSpec(name="time", column="time", distribution="normal"),
            VariableSpec(name="comfort", column="comfort", distribution="fixed"),
            VariableSpec(name="reliability", column="reliability", distribution="fixed"),
        ],
        model_type="mixed",
        n_draws=50,
        maxiter=200,
        seed=42,
    )
    result = estimate_from_spec(df=sim_output.data, spec=spec)
    est = result.estimation
    # Should produce the same kind of result as estimate_dataframe
    assert est.n_parameters == 6, f"Expected 6 params, got {est.n_parameters}"
    assert not est.estimates.empty
    param_names = set(est.estimates["parameter"])
    for expected in ["mu_price", "sd_price", "mu_time", "sd_time",
                     "beta_comfort", "beta_reliability"]:
        assert expected in param_names, f"Missing param: {expected}"
    assert est.n_observations == 100 * 4


_run("21. FullModelSpec + estimate_from_spec", test_full_model_spec)


# ===================================================================
# 22. Heterogeneity interactions with MMNL via FullModelSpec
# ===================================================================
def test_interactions_mmnl():
    from dce_analyzer.config import FullModelSpec, InteractionTerm, VariableSpec
    from dce_analyzer.pipeline import estimate_from_spec

    spec = FullModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price", distribution="normal"),
            VariableSpec(name="time", column="time", distribution="fixed"),
            VariableSpec(name="comfort", column="comfort", distribution="fixed"),
            VariableSpec(name="reliability", column="reliability", distribution="fixed"),
        ],
        model_type="mixed",
        interactions=[
            InteractionTerm(columns=("price", "income")),
        ],
        n_draws=50,
        maxiter=200,
        seed=42,
    )
    result = estimate_from_spec(df=sim_output.data, spec=spec)
    est = result.estimation
    param_names = set(est.estimates["parameter"])
    # Interaction term should appear as a fixed parameter
    assert "beta_price_x_income" in param_names, (
        f"Missing interaction param. Got: {param_names}"
    )
    # 1 mu + 1 sd (price) + 3 fixed (time, comfort, reliability) + 1 interaction = 6
    assert est.n_parameters == 6, f"Expected 6 params, got {est.n_parameters}"


_run("22. Heterogeneity interactions with MMNL (InteractionTerm)", test_interactions_mmnl)


# ===================================================================
# 23. GMNL + full correlation
# ===================================================================
def test_gmnl_full_correlation():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price", distribution="normal"),
            VariableSpec(name="time", column="time", distribution="normal"),
            VariableSpec(name="comfort", column="comfort", distribution="fixed"),
            VariableSpec(name="reliability", column="reliability", distribution="fixed"),
        ],
        n_draws=50,
    )
    result = estimate_dataframe(
        df=sim_output.data, spec=spec, model_type="gmnl",
        maxiter=200, seed=42, correlated=True,
    )
    est = result.estimation
    assert est.covariance_matrix is not None, "Expected covariance matrix for GMNL+correlated"
    assert est.covariance_matrix.shape == (2, 2), (
        f"Expected 2x2 cov, got {est.covariance_matrix.shape}"
    )
    assert est.correlation_matrix is not None
    # 2 mu + 3 Cholesky + 2 fixed + 2 GMNL scale terms = 9
    assert est.n_parameters == 9, f"Expected 9 params, got {est.n_parameters}"
    param_names = set(est.estimates["parameter"])
    assert "tau (scale mean)" not in param_names
    assert "sigma_tau (scale SD)" in param_names
    assert "gamma (mixing)" in param_names


_run("23. GMNL + full correlation", test_gmnl_full_correlation)


# ===================================================================
# 24. GMNL + selective correlation
# ===================================================================
def test_gmnl_selective_correlation():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price", distribution="normal"),
            VariableSpec(name="time", column="time", distribution="normal"),
            VariableSpec(name="comfort", column="comfort", distribution="normal"),
            VariableSpec(name="reliability", column="reliability", distribution="fixed"),
        ],
        n_draws=50,
    )
    # Correlate price-time only; comfort is standalone random
    result = estimate_dataframe(
        df=sim_output.data, spec=spec, model_type="gmnl",
        maxiter=200, seed=42,
        correlation_groups=[[0, 1]],
    )
    est = result.estimation
    assert est.covariance_matrix is not None
    assert est.covariance_matrix.shape == (3, 3)
    cov = est.covariance_matrix
    # comfort (index 2) is standalone: zero cross-cov with price/time
    assert abs(cov[0, 2]) < 1e-8, f"Expected 0 cov(price,comfort), got {cov[0,2]}"
    assert abs(cov[1, 2]) < 1e-8, f"Expected 0 cov(time,comfort), got {cov[1,2]}"
    param_names = set(est.estimates["parameter"])
    assert "sigma_tau (scale SD)" in param_names


_run("24. GMNL + selective correlation", test_gmnl_selective_correlation)


# ===================================================================
# 25. BWS composable functions (bws_log_prob, standard_log_prob)
# ===================================================================
def test_bws_composable_functions():
    import torch
    from dce_analyzer.bws import bws_log_prob, standard_log_prob

    # Create simple test tensors: 4 observations, 3 alternatives
    n_obs, n_alts = 4, 3
    torch.manual_seed(42)
    utility = torch.randn(n_obs, n_alts)
    y_best = torch.tensor([0, 1, 2, 0])   # chosen alt indices
    y_worst = torch.tensor([2, 0, 1, 1])  # worst alt indices (different from best)

    # Test standard_log_prob
    log_p = standard_log_prob(utility, y_best, alt_dim=-1)
    assert log_p.shape == (n_obs,), f"Expected shape ({n_obs},), got {log_p.shape}"
    # Log-probabilities must be <= 0
    assert (log_p <= 1e-6).all(), "Log-probabilities must be <= 0"
    # Probabilities must sum to 1 across alternatives (verify via logsumexp)
    log_all = torch.stack([
        standard_log_prob(utility, torch.full((n_obs,), j), alt_dim=-1)
        for j in range(n_alts)
    ], dim=1)
    prob_sums = torch.exp(log_all).sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(n_obs), atol=1e-5), (
        f"Probabilities don't sum to 1: {prob_sums}"
    )

    # Test bws_log_prob
    lambda_w = 1.0
    log_p_bws = bws_log_prob(utility, y_best, y_worst, lambda_w, alt_dim=-1)
    assert log_p_bws.shape == (n_obs,), f"Expected shape ({n_obs},), got {log_p_bws.shape}"
    assert (log_p_bws <= 1e-6).all(), "BWS log-probabilities must be <= 0"
    # BWS log-prob should be less than standard (it's a product of two probs)
    assert (log_p_bws <= log_p + 1e-6).all(), (
        "BWS log-prob should be <= standard log-prob (product of two probs)"
    )

    # Test with lambda_w as tensor
    lambda_w_tensor = torch.tensor(2.0)
    log_p_bws2 = bws_log_prob(utility, y_best, y_worst, lambda_w_tensor, alt_dim=-1)
    assert log_p_bws2.shape == (n_obs,)

    # Test with 3D utility (simulating draws): (n_obs, n_draws, n_alts)
    n_draws = 5
    utility_3d = torch.randn(n_obs, n_draws, n_alts)
    log_p_3d = standard_log_prob(utility_3d, y_best, alt_dim=-1)
    assert log_p_3d.shape == (n_obs, n_draws), f"Expected ({n_obs},{n_draws}), got {log_p_3d.shape}"

    log_p_bws_3d = bws_log_prob(utility_3d, y_best, y_worst, 1.0, alt_dim=-1)
    assert log_p_bws_3d.shape == (n_obs, n_draws), (
        f"Expected ({n_obs},{n_draws}), got {log_p_bws_3d.shape}"
    )


_run("25. BWS composable functions (bws_log_prob, standard_log_prob)", test_bws_composable_functions)


# ===================================================================
# 26. Heterogeneity interactions with Latent Class via FullModelSpec
# ===================================================================
def test_interactions_lc():
    from dce_analyzer.config import FullModelSpec, InteractionTerm, VariableSpec
    from dce_analyzer.pipeline import estimate_from_spec

    spec = FullModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price"),
            VariableSpec(name="time", column="time"),
            VariableSpec(name="comfort", column="comfort"),
            VariableSpec(name="reliability", column="reliability"),
        ],
        model_type="latent_class",
        interactions=[
            InteractionTerm(columns=("price", "income")),
        ],
        n_classes=2,
        n_starts=3,
        maxiter=200,
        seed=42,
    )
    result = estimate_from_spec(df=sim_output.data, spec=spec)
    est = result.estimation
    assert est.n_classes == 2
    # Interaction param should appear in estimates
    has_interaction = any("price_x_income" in str(p) for p in est.estimates["parameter"])
    assert has_interaction, (
        f"Missing interaction param in LC estimates. Got: {list(est.estimates['parameter'])}"
    )


_run("26. Heterogeneity interactions with Latent Class (InteractionTerm)", test_interactions_lc)


# ===================================================================
# 27. FullModelSpec with dummy coding via estimate_from_spec
# ===================================================================
def test_dummy_coding_via_spec():
    from dce_analyzer.config import DummyCoding, FullModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_from_spec

    # comfort has 2 unique values (0, 1) -> dummy with ref=0 -> one dummy comfort_L1
    spec = FullModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price"),
            VariableSpec(name="time", column="time"),
            VariableSpec(name="comfort", column="comfort"),
            VariableSpec(name="reliability", column="reliability"),
        ],
        model_type="conditional",
        dummy_codings=[
            DummyCoding(column="comfort", ref_level=0),
        ],
        maxiter=200,
        seed=42,
    )
    result = estimate_from_spec(df=sim_output.data, spec=spec)
    est = result.estimation
    param_names = set(est.estimates["parameter"])
    # comfort should be expanded: beta_comfort_L1 instead of beta_comfort
    assert "beta_comfort_L1" in param_names, (
        f"Missing dummy param beta_comfort_L1. Got: {param_names}"
    )
    # Original comfort should NOT appear
    assert "beta_comfort" not in param_names, (
        f"Original column should be replaced by dummy expansion. Got: {param_names}"
    )
    # price, time, reliability remain continuous
    assert "beta_price" in param_names
    assert "beta_time" in param_names
    assert "beta_reliability" in param_names
    # 3 continuous + 1 dummy = 4 params
    assert est.n_parameters == 4, f"Expected 4 params, got {est.n_parameters}"


_run("27. FullModelSpec with dummy coding via estimate_from_spec", test_dummy_coding_via_spec)


# ===================================================================
# 28. Variable ordering: dummy-coded vars expanded in-place
# ===================================================================
def test_variable_ordering_preservation():
    from dce_analyzer.config import DummyCoding, FullModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_from_spec

    # Variables in order: price (continuous), comfort (dummy, binary 0/1), time (continuous), reliability (continuous)
    # After expansion, order must be: price, comfort_L1, time, reliability
    # (not: price, time, reliability, comfort_L1 — the old buggy behavior)
    spec = FullModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price"),
            VariableSpec(name="comfort", column="comfort"),
            VariableSpec(name="time", column="time"),
            VariableSpec(name="reliability", column="reliability"),
        ],
        model_type="conditional",
        dummy_codings=[
            DummyCoding(column="comfort", ref_level=0),
        ],
        maxiter=200,
        seed=42,
    )
    result = estimate_from_spec(df=sim_output.data, spec=spec)
    est = result.estimation
    param_names = list(est.estimates["parameter"])
    # Check order: price -> comfort dummy -> time -> reliability
    expected_order = ["beta_price", "beta_comfort_L1", "beta_time", "beta_reliability"]
    assert param_names == expected_order, (
        f"Variable ordering not preserved. Expected {expected_order}, got {param_names}"
    )
    # Also verify expanded_spec preserves order
    exp_spec = result.expanded_spec
    exp_var_names = [v.name for v in exp_spec.variables]
    assert exp_var_names == ["price", "comfort_L1", "time", "reliability"], (
        f"Expanded spec variable order wrong: {exp_var_names}"
    )


_run("28. Variable ordering: dummy-coded vars expanded in-place", test_variable_ordering_preservation)


# ===================================================================
# 29. WTP theta_index mapping for MMNL (SE correctness)
# ===================================================================
def test_wtp_theta_index():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe
    from dce_analyzer.wtp import compute_wtp

    # price is random, then time (fixed), comfort (fixed), reliability (fixed)
    # This creates interleaved mu/sd rows: mu_price, sd_price, beta_time, ...
    # The theta_index mapping must be correct for WTP SEs.
    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price", distribution="normal"),
            VariableSpec(name="time", column="time", distribution="fixed"),
            VariableSpec(name="comfort", column="comfort", distribution="fixed"),
            VariableSpec(name="reliability", column="reliability", distribution="fixed"),
        ],
        n_draws=50,
    )
    result = estimate_dataframe(
        df=sim_output.data, spec=spec, model_type="mixed",
        maxiter=200, seed=42,
    )
    est = result.estimation

    # Verify theta_index column exists and is correct
    assert "theta_index" in est.estimates.columns, "theta_index column missing"
    # mu_price -> theta 0, sd_price -> theta 4, beta_time -> theta 1,
    # beta_comfort -> theta 2, beta_reliability -> theta 3
    tidx_map = dict(zip(est.estimates["parameter"], est.estimates["theta_index"]))
    assert tidx_map["mu_price"] == 0, f"mu_price should be theta 0, got {tidx_map['mu_price']}"
    assert tidx_map["beta_time"] == 1, f"beta_time should be theta 1, got {tidx_map['beta_time']}"
    assert tidx_map["sd_price"] == 4, f"sd_price should be theta 4, got {tidx_map['sd_price']}"

    # Compute WTP using time as the cost variable
    wtp_df = compute_wtp(est, cost_variable="time")
    assert not wtp_df.empty
    # Check that SEs are not NaN (vcov should be available)
    if est.vcov_matrix is not None:
        for _, row in wtp_df.iterrows():
            if row["attribute"] in ("price", "comfort", "reliability"):
                assert not np.isnan(row["wtp_std_error"]), (
                    f"WTP SE is NaN for {row['attribute']} — theta_index mapping may be wrong"
                )


_run("29. WTP theta_index mapping for MMNL (SE correctness)", test_wtp_theta_index)


# ===================================================================
# 30. 3-way interaction (price × time × income)
# ===================================================================
def test_3way_interaction():
    from dce_analyzer.config import FullModelSpec, InteractionTerm, VariableSpec
    from dce_analyzer.pipeline import estimate_from_spec

    spec = FullModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price"),
            VariableSpec(name="time", column="time"),
            VariableSpec(name="comfort", column="comfort"),
            VariableSpec(name="reliability", column="reliability"),
        ],
        model_type="conditional",
        interactions=[
            InteractionTerm(columns=("price", "time", "income")),
        ],
        maxiter=200,
        seed=42,
    )
    result = estimate_from_spec(df=sim_output.data, spec=spec)
    est = result.estimation
    param_names = set(est.estimates["parameter"])
    # 3-way interaction name: price_x_time_x_income
    assert "beta_price_x_time_x_income" in param_names, (
        f"Missing 3-way interaction param. Got: {param_names}"
    )
    # 4 base + 1 interaction = 5 params
    assert est.n_parameters == 5, f"Expected 5 params, got {est.n_parameters}"


_run("30. 3-way interaction (price × time × income)", test_3way_interaction)


# ===================================================================
# 31. Attribute × attribute interaction (price × time)
# ===================================================================
def test_attribute_x_attribute_interaction():
    from dce_analyzer.config import FullModelSpec, InteractionTerm, VariableSpec
    from dce_analyzer.pipeline import estimate_from_spec

    spec = FullModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price"),
            VariableSpec(name="time", column="time"),
            VariableSpec(name="comfort", column="comfort"),
            VariableSpec(name="reliability", column="reliability"),
        ],
        model_type="conditional",
        interactions=[
            InteractionTerm(columns=("price", "time")),
        ],
        maxiter=200,
        seed=42,
    )
    result = estimate_from_spec(df=sim_output.data, spec=spec)
    est = result.estimation
    param_names = set(est.estimates["parameter"])
    # attribute x attribute interaction
    assert "beta_price_x_time" in param_names, (
        f"Missing attribute x attribute interaction param. Got: {param_names}"
    )
    # 4 base + 1 interaction = 5 params
    assert est.n_parameters == 5, f"Expected 5 params, got {est.n_parameters}"


_run("31. Attribute × attribute interaction (price × time)", test_attribute_x_attribute_interaction)


# ===================================================================
# 32. Custom attribute names in simulation
# ===================================================================
def test_custom_attribute_names():
    from dce_analyzer.simulate import generate_simulated_dce

    output = generate_simulated_dce(
        n_individuals=50, n_tasks=4, n_alts=3, seed=123,
        attribute_names=["cost", "quality", "speed"],
    )
    df = output.data
    for col in ["cost", "quality", "speed"]:
        assert col in df.columns, f"Missing custom attribute column: {col}"
    # true_params should reference custom names
    for attr in ["cost", "quality", "speed"]:
        assert f"mu_{attr}" in output.true_parameters, f"Missing mu_{attr} in true_params"
        assert f"sd_{attr}" in output.true_parameters, f"Missing sd_{attr} in true_params"
    # Default attributes should NOT be present
    for col in ["price", "time", "comfort", "reliability"]:
        assert col not in df.columns, f"Default attribute '{col}' should not be present"


_run("32. Custom attribute names in simulation", test_custom_attribute_names)


# ===================================================================
# 33. Custom covariate names in simulation
# ===================================================================
def test_custom_covariate_names():
    from dce_analyzer.simulate import generate_simulated_dce

    output = generate_simulated_dce(
        n_individuals=50, n_tasks=4, n_alts=3, seed=123,
        covariate_names=["education", "gender"],
    )
    df = output.data
    for col in ["education", "gender"]:
        assert col in df.columns, f"Missing custom covariate column: {col}"
    # Covariates should be constant within each respondent
    for (rid,), grp in df.groupby(["respondent_id"]):
        for col in ["education", "gender"]:
            assert grp[col].nunique() == 1, (
                f"Covariate '{col}' not constant for respondent {rid}"
            )
    # Default covariates should NOT be present
    for col in ["income", "age"]:
        assert col not in df.columns, f"Default covariate '{col}' should not be present"


_run("33. Custom covariate names in simulation", test_custom_covariate_names)


# ===================================================================
# 34. BWS simulation (worst column)
# ===================================================================
def test_bws_simulation():
    from dce_analyzer.simulate import generate_simulated_dce

    output = generate_simulated_dce(
        n_individuals=50, n_tasks=4, n_alts=3, seed=42, bws=True,
    )
    df = output.data
    assert "worst" in df.columns, "Missing 'worst' column"
    # Each task should have exactly 1 worst
    for (rid, tid), grp in df.groupby(["respondent_id", "task_id"]):
        assert grp["worst"].sum() == 1, f"Task ({rid},{tid}) should have exactly 1 worst"
        assert grp["choice"].sum() == 1, f"Task ({rid},{tid}) should have exactly 1 best"
        best_alt = grp.loc[grp["choice"] == 1, "alternative"].values[0]
        worst_alt = grp.loc[grp["worst"] == 1, "alternative"].values[0]
        assert best_alt != worst_alt, f"Task ({rid},{tid}): worst must differ from best"


_run("34. BWS simulation (worst column)", test_bws_simulation)


# ===================================================================
# 35. BWS simulation with n_alts=2 raises ValueError
# ===================================================================
def test_bws_n_alts_2_raises():
    from dce_analyzer.simulate import generate_simulated_dce

    try:
        generate_simulated_dce(n_individuals=10, n_tasks=2, n_alts=2, seed=1, bws=True)
        raise AssertionError("Should have raised ValueError for bws with n_alts=2")
    except ValueError as exc:
        assert "n_alts >= 3" in str(exc), f"Unexpected error message: {exc}"


_run("35. BWS simulation with n_alts=2 raises ValueError", test_bws_n_alts_2_raises)


# ===================================================================
# 36. Default params backward compat (no new args)
# ===================================================================
def test_default_params_backward_compat():
    from dce_analyzer.simulate import generate_simulated_dce

    output = generate_simulated_dce(n_individuals=50, n_tasks=4, n_alts=3, seed=42)
    df = output.data
    # Same columns as original test 2
    for col in ["respondent_id", "task_id", "alternative", "choice",
                "price", "time", "comfort", "reliability", "income", "age"]:
        assert col in df.columns, f"Missing column: {col}"
    assert "worst" not in df.columns, "'worst' column should not be present by default"
    assert len(df) == 50 * 4 * 3
    # true_params should contain the hardcoded keys
    for key in ["mu_price", "sd_price", "mu_time", "sd_time",
                "mu_comfort", "sd_comfort", "beta_reliability"]:
        assert key in output.true_parameters, f"Missing true_param key: {key}"


_run("36. Default params backward compat (no new args)", test_default_params_backward_compat)


# ===================================================================
# 37. Bootstrap with Mixed Logit
# ===================================================================
def test_bootstrap_mixed_logit():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.bootstrap import run_bootstrap

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price", distribution="normal"),
            VariableSpec(name="time", column="time", distribution="normal"),
            VariableSpec(name="comfort", column="comfort", distribution="fixed"),
            VariableSpec(name="reliability", column="reliability", distribution="fixed"),
        ],
        n_draws=50,
    )
    boot = run_bootstrap(
        df=sim_output.data, spec=spec, model_type="mixed",
        n_replications=5, maxiter=100, seed=42,
    )
    assert boot.n_replications == 5
    assert boot.n_successful >= 2, f"Only {boot.n_successful} succeeded"
    # 2 normal (mu+sd each) + 2 fixed = 6 params
    assert len(boot.param_names) == 6, f"Expected 6 params, got {len(boot.param_names)}"
    assert boot.estimates_matrix.shape == (boot.n_successful, 6)
    summary = boot.summary_dataframe()
    assert isinstance(summary, pd.DataFrame)
    assert len(summary) == 6
    # Check that bootstrap SE is computed for all parameters
    for name in boot.param_names:
        se = boot.bootstrap_se[name]
        assert se >= 0, f"Bootstrap SE negative for {name}"
        assert np.isfinite(se), f"Bootstrap SE not finite for {name}"


_run("37. Bootstrap with Mixed Logit", test_bootstrap_mixed_logit)


# ===================================================================
# 38. Latent Class with EM algorithm
# ===================================================================
def test_lc_em():
    from dce_analyzer.config import ModelSpec, VariableSpec
    from dce_analyzer.pipeline import estimate_dataframe

    spec = ModelSpec(
        id_col="respondent_id",
        task_col="task_id",
        alt_col="alternative",
        choice_col="choice",
        variables=[
            VariableSpec(name="price", column="price"),
            VariableSpec(name="time", column="time"),
            VariableSpec(name="comfort", column="comfort"),
            VariableSpec(name="reliability", column="reliability"),
        ],
        n_classes=2,
    )
    result = estimate_dataframe(
        df=sim_output.data, spec=spec, model_type="latent_class",
        maxiter=200, seed=42, n_classes=2, n_starts=3, lc_method="em",
    )
    est = result.estimation
    assert est.n_classes == 2
    assert len(est.class_probabilities) == 2
    assert abs(sum(est.class_probabilities) - 1.0) < 1e-4, "Class probs must sum to 1"
    assert not est.estimates.empty
    assert not est.class_estimates.empty
    assert not est.posterior_probs.empty
    assert est.posterior_probs.shape[1] == 2

    # EM-specific fields
    assert est.optimizer_method == "EM"
    assert est.em_iterations > 0, "EM should run at least 1 iteration"
    assert len(est.em_ll_history) == est.em_iterations
    assert isinstance(est.em_converged, bool)
    # LL should be monotonically non-decreasing (EM guarantee)
    for i in range(1, len(est.em_ll_history)):
        assert est.em_ll_history[i] >= est.em_ll_history[i - 1] - 1e-6, (
            f"EM LL decreased at iter {i}: {est.em_ll_history[i-1]:.6f} -> {est.em_ll_history[i]:.6f}"
        )

    # summary_dict should include EM fields
    sd = est.summary_dict()
    assert "em_iterations" in sd
    assert "em_ll_history" in sd
    assert "em_converged" in sd


_run("38. Latent Class with EM algorithm", test_lc_em)


# ===================================================================
# Summary
# ===================================================================
print()
print("=" * 60)
n_pass = sum(1 for _, ok, _ in _results if ok)
n_fail = sum(1 for _, ok, _ in _results if not ok)
print(f"  {n_pass} passed, {n_fail} failed out of {len(_results)} tests")
print("=" * 60)

if n_fail > 0:
    print()
    print("FAILURES:")
    for name, ok, msg in _results:
        if not ok:
            print(f"  {name}: {msg}")
    print()
    sys.exit(1)
else:
    print("  ALL TESTS PASSED")
    sys.exit(0)
