"""分析侠 — Page 4: Model Comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils import init_session_state, require_data, sidebar_branding, slowbro_next_step  # noqa: E402

init_session_state()
sidebar_branding()

st.header("Compare")
st.caption("Compare multiple estimated models side-by-side.")

require_data()

history: list[dict] = st.session_state.model_history
if len(history) < 2:
    st.warning(
        "At least **two** models are needed for comparison. "
        "Go to the **Model** page and run additional estimations."
    )
    if history:
        st.info(f"Currently stored: {len(history)} model(s).")
    st.stop()

# ── model selection ────────────────────────────────────────────────
model_names = [e.get("label", e.get("name", "unknown")) for e in history]
selected = st.multiselect(
    "Select models to compare",
    options=model_names,
    default=model_names,
    help="Choose two or more models from your estimation history to compare side-by-side.",
)
if len(selected) < 2:
    st.info("Select at least two models.")
    st.stop()

entries = [e for e in history if e.get("label", e.get("name", "unknown")) in selected]

# ── 1  Fit metric comparison table ─────────────────────────────────
st.divider()
st.subheader("1. Model fit comparison")

fit_rows = []
for e in entries:
    est = e["estimation"]
    fit_rows.append(
        {
            "Model": e.get("label", e.get("name", "unknown")),
            "Type": e["model_type"],
            "Log-likelihood": round(est.log_likelihood, 3),
            "AIC": round(est.aic, 2),
            "BIC": round(est.bic, 2),
            "Parameters": est.n_parameters,
            "Observations": est.n_observations,
            "Converged": est.success,
            "Runtime (s)": round(est.runtime_seconds, 2),
        }
    )

fit_df = pd.DataFrame(fit_rows)

# Highlight the row with the lowest BIC
best_bic_model = fit_df.loc[fit_df["BIC"].idxmin(), "Model"]


def _highlight_best(row: pd.Series) -> list[str]:
    if row["Model"] == best_bic_model:
        return ["background-color: rgba(76, 175, 80, 0.15)"] * len(row)
    return [""] * len(row)


st.dataframe(
    fit_df.style.apply(_highlight_best, axis=1),
    use_container_width=True,
    hide_index=True,
)

with st.expander("Understanding AIC and BIC"):
    st.markdown(
        "**AIC** and **BIC** help you choose between models with different numbers of parameters.\n\n"
        "- Both penalize models for having more parameters (to avoid overfitting).\n"
        "- **Lower values are better** -- they indicate a better balance of fit and simplicity.\n"
        "- **AIC** tends to favor slightly more complex models.\n"
        "- **BIC** applies a stronger penalty for extra parameters, favoring simpler models.\n"
        "- If AIC and BIC disagree, BIC is generally more conservative and may be preferred "
        "when you want a parsimonious model."
    )

# best model recommendation
best_aic_idx = fit_df["AIC"].idxmin()
best_bic_idx = fit_df["BIC"].idxmin()
b1, b2 = st.columns(2)
b1.success(f"Best AIC: **{fit_df.loc[best_aic_idx, 'Model']}** ({fit_df.loc[best_aic_idx, 'AIC']:.2f})")
b2.success(f"Best BIC: **{fit_df.loc[best_bic_idx, 'Model']}** ({fit_df.loc[best_bic_idx, 'BIC']:.2f})")

# ── 2  AIC / BIC bar chart ────────────────────────────────────────
st.divider()
st.subheader("2. AIC / BIC comparison")

chart_df = fit_df.set_index("Model")[["AIC", "BIC"]]
st.bar_chart(chart_df)

# ── 3  Parameter estimates across models ───────────────────────────
st.divider()
st.subheader("3. Parameter estimates across models")

# build a wide table: rows = parameters, columns = model names
all_params: set[str] = set()
for e in entries:
    all_params.update(e["estimation"].estimates["parameter"].tolist())

param_rows = []
for param in sorted(all_params):
    row: dict[str, object] = {"Parameter": param}
    for e in entries:
        est_df = e["estimation"].estimates
        match = est_df[est_df["parameter"] == param]
        if len(match) == 1:
            val = match.iloc[0]["estimate"]
            se = match.iloc[0].get("std_error", float("nan"))
            if pd.notna(se) and se > 0:
                row[e.get("label", e.get("name", "unknown"))] = f"{val:.4f} ({se:.4f})"
            else:
                row[e.get("label", e.get("name", "unknown"))] = f"{val:.4f}"
        else:
            row[e.get("label", e.get("name", "unknown"))] = "—"
    param_rows.append(row)

param_compare_df = pd.DataFrame(param_rows)
st.dataframe(param_compare_df, use_container_width=True, hide_index=True)

# ── 4  Coefficient comparison plot ─────────────────────────────────
st.divider()
st.subheader("4. Coefficient comparison")

common_params = sorted(all_params)
if common_params:
    sel_param = st.selectbox(
        "Select parameter", common_params,
        help="Pick a parameter to see how its estimated value differs across models.",
    )

    coef_data: dict[str, list[float]] = {"Model": [], "Estimate": []}
    for e in entries:
        est_df = e["estimation"].estimates
        match = est_df[est_df["parameter"] == sel_param]
        if len(match) == 1:
            coef_data["Model"].append(e.get("label", e.get("name", "unknown")))
            coef_data["Estimate"].append(float(match.iloc[0]["estimate"]))

    if coef_data["Model"]:
        coef_df = pd.DataFrame(coef_data).set_index("Model")
        st.bar_chart(coef_df)

# ── 5  Likelihood ratio test ──────────────────────────────────────
st.divider()
st.subheader("5. Likelihood ratio test")

with st.expander("What is the likelihood ratio test?"):
    st.markdown(
        "The **Likelihood Ratio (LR) test** checks whether a more complex model provides a "
        "statistically significant improvement over a simpler one.\n\n"
        "- The **restricted model** is the simpler model (fewer parameters).\n"
        "- The **unrestricted model** is the more complex model (more parameters).\n"
        "- The test asks: *does adding those extra parameters significantly improve the fit?*\n"
        "- A **p-value below 0.05** means the complex model is significantly better.\n"
        "- A **p-value above 0.05** means the simpler model is sufficient.\n\n"
        "**Important**: This test is only valid for **nested** models -- where the restricted "
        "model is a special case of the unrestricted model (e.g., Conditional Logit vs. "
        "Mixed Logit with the same variables)."
    )

lr1, lr2 = st.columns(2)
restricted = lr1.selectbox(
    "Restricted model", [e.get("label", e.get("name", "unknown")) for e in entries],
    key="lr_restricted",
    help="The simpler model with fewer parameters (e.g., Conditional Logit).",
)
unrestricted = lr2.selectbox(
    "Unrestricted model", [e.get("label", e.get("name", "unknown")) for e in entries],
    index=min(1, len(entries) - 1), key="lr_unrestricted",
    help="The more complex model with more parameters (e.g., Mixed Logit).",
)

if restricted != unrestricted:
    r_entry = next(e for e in entries if e.get("label", e.get("name", "unknown")) == restricted)
    u_entry = next(e for e in entries if e.get("label", e.get("name", "unknown")) == unrestricted)
    r_ll = r_entry["estimation"].log_likelihood
    u_ll = u_entry["estimation"].log_likelihood
    r_k = r_entry["estimation"].n_parameters
    u_k = u_entry["estimation"].n_parameters

    if u_k <= r_k:
        st.warning("The unrestricted model should have more parameters than the restricted model.")
    else:
        lr_stat = 2 * (u_ll - r_ll)
        df_diff = u_k - r_k
        from scipy.stats import chi2

        p_val = float(1 - chi2.cdf(lr_stat, df_diff))

        l1, l2, l3 = st.columns(3)
        l1.metric("LR statistic", f"{lr_stat:.4f}")
        l2.metric("Degrees of freedom", f"{df_diff}")
        l3.metric("p-value", f"{p_val:.6f}")

        if p_val < 0.05:
            st.success("The unrestricted model is significantly better at the 5% level.")
        else:
            st.info("No significant improvement — the simpler model may be preferred.")
else:
    st.info("Select two different models for the LR test.")

# ── 6  Swait-Louviere pooling test ─────────────────────────────────
st.divider()
st.subheader("6. Swait-Louviere pooling test")

with st.expander("What is the Swait-Louviere test?"):
    st.markdown(
        "The **Swait-Louviere test** checks whether two groups of respondents share "
        "the same preference parameters (utility coefficients).\n\n"
        "- You split the data by a **grouping variable** (e.g., gender, age group).\n"
        "- The same model is estimated on each group separately and on the pooled data.\n"
        "- Test statistic: **-2 * [LL_pooled - (LL_group1 + LL_group2)]**, "
        "distributed as chi-squared(K) where K = number of parameters.\n"
        "- A **p-value below 0.05** means the groups have significantly different preferences.\n"
    )

df_full: pd.DataFrame = st.session_state.df

# Only show columns that could be meaningful grouping variables (not structural/utility)
if entries:
    # Use the first model's spec to identify structural columns
    first_spec = entries[0].get("spec")
    if first_spec:
        structural = {first_spec.id_col, first_spec.task_col, first_spec.alt_col, first_spec.choice_col}
        utility_cols = {v.column for v in first_spec.variables}
    else:
        structural = set()
        utility_cols = set()
    candidate_cols = [
        c for c in df_full.columns
        if c not in structural and c not in utility_cols
        and df_full[c].dropna().nunique() == 2
    ]
else:
    candidate_cols = [c for c in df_full.columns if df_full[c].dropna().nunique() == 2]

if candidate_cols:
    sl_col1, sl_col2 = st.columns(2)
    grouping_col = sl_col1.selectbox(
        "Grouping variable (must have exactly 2 levels)",
        candidate_cols,
        key="sl_grouping",
        help="Select a binary variable to split respondents into two groups.",
    )
    sl_model_label = sl_col2.selectbox(
        "Model to test",
        [e.get("label", "unknown") for e in entries],
        key="sl_model",
        help="Select which estimated model's specification to use for the test.",
    )

    if st.button("Run Swait-Louviere test", use_container_width=True):
        from dce_analyzer.tests import swait_louviere_test  # noqa: E402

        sl_entry = next(e for e in entries if e.get("label", "unknown") == sl_model_label)
        sl_spec = sl_entry["spec"]
        sl_model_type = sl_entry["model_type"]

        # Need to reconstruct _work_df if dummy columns are used
        # Use the spec's variable columns; they should already exist in the estimation df
        # For the test, we need to build the working dataframe with dummy columns
        sl_work_df = df_full.copy()
        for var in sl_spec.variables:
            if var.column not in sl_work_df.columns:
                # Dummy column pattern: originalcol_Lvalue
                # Try to reconstruct from the column name
                parts = var.column.rsplit("_L", 1)
                if len(parts) == 2:
                    orig_col, level_str = parts
                    if orig_col in sl_work_df.columns:
                        try:
                            level_val = type(sl_work_df[orig_col].iloc[0])(level_str)
                        except (ValueError, TypeError):
                            level_val = level_str
                        sl_work_df[var.column] = (sl_work_df[orig_col] == level_val).astype(int)

        with st.spinner("Estimating models on each group and pooled data..."):
            try:
                est_kwargs: dict = dict(maxiter=200, seed=123)
                if sl_model_type == "latent_class":
                    est_kwargs["n_classes"] = sl_spec.n_classes
                    est_kwargs["n_starts"] = 10
                sl_result = swait_louviere_test(
                    df=sl_work_df,
                    spec=sl_spec,
                    grouping_col=grouping_col,
                    model_type=sl_model_type,
                    **est_kwargs,
                )
            except Exception as exc:
                st.error(f"Swait-Louviere test failed: {exc}")
                st.exception(exc)
                st.stop()

        st.markdown("#### Results")
        r1, r2, r3 = st.columns(3)
        r1.metric("Chi-squared statistic", f"{sl_result.chi2_stat:.4f}")
        r2.metric("Degrees of freedom", f"{sl_result.df}")
        r3.metric("p-value", f"{sl_result.p_value:.6f}")

        ll1, ll2, ll3 = st.columns(3)
        ll1.metric("LL (pooled)", f"{sl_result.ll_pooled:.3f}")
        ll2.metric(f"LL ({sl_result.group1_label})", f"{sl_result.ll_group1:.3f}")
        ll3.metric(f"LL ({sl_result.group2_label})", f"{sl_result.ll_group2:.3f}")

        if sl_result.reject_null:
            st.error(sl_result.conclusion)
        else:
            st.success(sl_result.conclusion)
else:
    st.info(
        "No suitable grouping variable found. The Swait-Louviere test requires a "
        "binary column (exactly 2 unique values) that is not a structural or utility variable."
    )

# ── 7  Export ──────────────────────────────────────────────────────
st.divider()
st.subheader("7. Export comparison")

export_bytes = fit_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download comparison table CSV",
    data=export_bytes,
    file_name="model_comparison.csv",
    mime="text/csv",
)

slowbro_next_step("pages/4_🔄_Compare.py")
