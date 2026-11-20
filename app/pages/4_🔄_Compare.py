# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Prefero — Page 4: Model Comparison."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils import init_session_state, require_data, sidebar_branding, slowbro_next_step  # noqa: E402
from dce_analyzer.tests import comparison_outcome, conditional_lr  # noqa: E402

init_session_state()
sidebar_branding()

st.header("Compare")
st.caption("Compare multiple estimated models side-by-side.")

require_data()

history = []
labels_in_use = set()
for run in st.session_state.model_history:
    base = run.get("label", run.get("name", "unknown"))
    label, count = base, 1
    while label in labels_in_use:
        count += 1
        label = f"{base} ({count})"
    labels_in_use.add(label)
    history.append({**run, "label": label})
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
            "AIC": est.aic,
            "BIC": est.bic,
            "Parameters": est.n_parameters,
            "Observations": est.n_observations,
            "Converged": est.success,
            "Runtime (s)": round(est.runtime_seconds, 2),
        }
    )

fit_df = pd.DataFrame(fit_rows)

outcomes = {comparison_outcome(entry) for entry in entries}
same_outcome = len(outcomes) == 1 and None not in outcomes
eligible = fit_df["Converged"] & np.isfinite(fit_df[["AIC", "BIC"]]).all(axis=1)
ranked = fit_df.loc[eligible] if same_outcome else fit_df.iloc[:0]
best_bic_model = None if len(ranked) < 2 else ranked.loc[ranked["BIC"].idxmin(), "Model"]


def _highlight_best(row: pd.Series) -> list[str]:
    if row["Model"] == best_bic_model:
        return ["background-color: rgba(76, 175, 80, 0.15)"] * len(row)
    return [""] * len(row)


st.dataframe(
    fit_df.style.format({"AIC": "{:.2f}", "BIC": "{:.2f}"}).apply(_highlight_best, axis=1),
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

if not same_outcome:
    st.info("Ranking requires the same choice outcomes and estimation sample.")
elif len(ranked) >= 2:
    for column, metric in zip(st.columns(2), ("AIC", "BIC")):
        best = ranked.loc[ranked[metric].idxmin()]
        column.success(f"Best {metric}: **{best['Model']}** ({best[metric]:.2f})")
else:
    st.info("Ranking requires two converged fits with finite information criteria.")

# ── 2  AIC / BIC bar chart ────────────────────────────────────────
st.divider()
st.subheader("2. AIC / BIC comparison")

if len(ranked) >= 2:
    st.bar_chart(ranked.set_index("Model")[["AIC", "BIC"]])

# ── 3  Parameter estimates across models ───────────────────────────
st.divider()
st.subheader("3. Parameter estimates across models")

# build a wide table: rows = parameters, columns = model names
all_params: set[str] = set()
for e in entries:
    all_params.update(e["estimation"].estimates.get("parameter", ()))

param_rows = []
for param in sorted(all_params):
    row: dict[str, object] = {"Parameter": param}
    for e in entries:
        est_df = e["estimation"].estimates
        match = est_df.loc[est_df["parameter"] == param] if "parameter" in est_df else est_df
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

    coef_data = {"Model": [], "Estimate": []}
    for e in entries:
        est_df = e["estimation"].estimates
        if "parameter" not in est_df:
            continue
        match = est_df[est_df["parameter"] == sel_param]
        if len(match) == 1:
            coef_data["Model"].append(e.get("label", e.get("name", "unknown")))
            coef_data["Estimate"].append(float(match.iloc[0]["estimate"]))

    if coef_data["Model"]:
        coef_df = pd.DataFrame(coef_data).set_index("Model")
        st.bar_chart(coef_df)

st.divider()
st.subheader("5. Likelihood ratio test")
st.caption("Use converged, nested conditional logit fits with the same single-choice outcome.")
labels = [entry["label"] for entry in entries]
lr1, lr2 = st.columns(2)
restricted = lr1.selectbox("Restricted model", labels, key="lr_restricted")
unrestricted = lr2.selectbox("Unrestricted model", labels, index=1, key="lr_unrestricted")
if restricted != unrestricted:
    by_label = {entry["label"]: entry for entry in entries}
    try:
        statistic, degrees, p_value = conditional_lr(
            by_label[restricted], by_label[unrestricted], data=st.session_state.df,
        )
    except ValueError as exc:
        st.info(str(exc))
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("LR statistic", f"{statistic:.4f}")
        c2.metric("Degrees of freedom", degrees)
        c3.metric("p-value", f"{p_value:.6f}")

st.divider()
st.subheader("6. Swait–Louviere pooling test")
st.caption("Compare preferences across two groups while allowing the groups to have different error scales.")
pooling_entries = [
    entry for entry in entries
    if entry["model_type"] == "conditional" and entry.get("spec") is not None
    and len(entry["spec"].variables) >= 2
    and getattr(entry.get("full_spec"), "bws_worst_col", None) is None
]
if not pooling_entries:
    st.info("Choose a conditional logit model with at least two coefficients and one choice per task.")
else:
    pooling_label = st.selectbox("Model for pooling test", [entry["label"] for entry in pooling_entries])
    pooling_entry = next(entry for entry in pooling_entries if entry["label"] == pooling_label)
    pooling_spec = pooling_entry["spec"]
    pooling_data = pooling_entry.get("expanded_df")
    if pooling_data is None:
        pooling_data = st.session_state.df
        full_spec = pooling_entry.get("full_spec")
        if full_spec is not None:
            from dce_analyzer.pipeline import prepare_from_spec

            pooling_data, pooling_spec = prepare_from_spec(pooling_data, full_spec)
    excluded = {pooling_spec.id_col, pooling_spec.task_col, pooling_spec.alt_col, pooling_spec.choice_col}
    excluded.update(v.column for v in pooling_spec.variables)
    group_cols = [col for col in pooling_data if col not in excluded and pooling_data[col].nunique() == 2]
    if not group_cols:
        st.info("This dataset needs a binary respondent-level grouping column.")
    else:
        group_column = st.selectbox("Grouping variable", group_cols)
        if st.button("Run Swait–Louviere test"):
            from dce_analyzer.tests import swait_louviere_test

            try:
                result = swait_louviere_test(pooling_data, pooling_spec, group_column)
            except ValueError as exc:
                st.error(str(exc))
            else:
                col1, col2, col3 = st.columns(3)
                col1.metric("Preference test p-value", f"{result.p_value:.4f}")
                col2.metric("Relative scale (group 2 / group 1)", f"{result.relative_scale:.3f}")
                col3.metric("Equal-scale p-value", f"{result.scale_p_value:.4f}" if not result.reject_null else "—")
                st.write(result.conclusion)
                if result.reject_null:
                    st.caption("The scale comparison assumes common preferences.")
                st.caption(f"Group 1 is {result.group1_label}. Group 2 is {result.group2_label}.")

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

# ── 8  Cross-validation ──────────────────────────────────────────
st.divider()
st.subheader("8. Cross-validation")

with st.expander("What is cross-validation?"):
    st.markdown(
        "**K-fold cross-validation** splits individuals into K groups (folds). "
        "For each fold, the model is estimated on K-1 folds and tested on the held-out fold.\n\n"
        "- **Out-of-sample LL**: How well the model predicts choices it wasn't trained on. "
        "Higher (less negative) is better.\n"
        "- **Hit rate**: The proportion of correctly predicted choices on held-out data.\n"
        "- Cross-validation helps detect **overfitting** — a model that fits training data well "
        "but performs poorly on new data."
    )

# Model selector — pick any model from history (not just the selected subset)
all_model_labels = [e.get("label", e.get("name", "unknown")) for e in history]
cv_model_label = st.selectbox(
    "Select a model for cross-validation",
    options=all_model_labels,
    key="cv_model_selector",
    help="Choose a model from your estimation history to cross-validate.",
)

cv_col1, cv_col2, cv_col3 = st.columns(3)
cv_k = cv_col1.number_input("Number of folds (K)", min_value=3, max_value=10, value=5, key="cv_k_input")
cv_seed = cv_col2.number_input("Random seed", min_value=1, value=42, key="cv_seed_input")
cv_run = cv_col3.button("Run Cross-Validation", use_container_width=True, key="cv_run_btn")

if "cv_results" not in st.session_state:
    st.session_state["cv_results"] = {}

if cv_run:
    cv_entry = next(e for e in history if e.get("label", e.get("name", "unknown")) == cv_model_label)

    # Build FullModelSpec from entry
    full_spec = cv_entry.get("full_spec")
    if full_spec is None:
        from dce_analyzer.config import FullModelSpec as _FullModelSpec
        _sp = cv_entry["spec"]
        full_spec = _FullModelSpec(
            id_col=_sp.id_col, task_col=_sp.task_col,
            alt_col=_sp.alt_col, choice_col=_sp.choice_col,
            variables=_sp.variables, model_type=cv_entry["model_type"],
            n_draws=_sp.n_draws, n_classes=_sp.n_classes,
            membership_cols=_sp.membership_cols,
        )

    from dce_analyzer.cross_validation import cross_validate  # noqa: E402

    progress_bar = st.progress(0)
    status_text = st.empty()

    def _cv_progress(fold: int, total: int, msg: str) -> None:
        progress_bar.progress((fold + 1) / total)
        status_text.text(msg)

    try:
        cv_result = cross_validate(
            df=st.session_state.df,
            spec=full_spec,
            k=int(cv_k),
            seed=int(cv_seed),
            progress_callback=_cv_progress,
        )
        st.session_state["cv_results"][cv_model_label] = cv_result
        progress_bar.progress(1.0)
        scored = sum(f.converged for f in cv_result.fold_results)
        status_text.text(f"Scored {scored} of {cv_result.k} folds.")
    except Exception as exc:
        progress_bar.empty()
        status_text.empty()
        st.error(f"Cross-validation failed: {exc}")

# Display results from session state
if st.session_state.get("cv_results"):
    for label, cv_res in st.session_state["cv_results"].items():
        st.markdown(f"#### CV results: {label}")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Mean out-of-sample LL", f"{cv_res.mean_test_ll:.3f}")
        m2.metric("Mean LL per obs", f"{cv_res.mean_test_ll_per_obs:.4f}")
        m3.metric("Mean hit rate", f"{cv_res.mean_hit_rate:.3%}")
        m4.metric("Total runtime (s)", f"{cv_res.total_runtime:.1f}")

        # Per-fold table
        fold_rows = []
        for fr in cv_res.fold_results:
            fold_rows.append({
                "Fold": fr.fold + 1,
                "Train LL": round(fr.train_ll, 3),
                "Test LL": round(fr.test_ll, 3),
                "Hit rate": round(fr.hit_rate, 4),
                "Train N (ind)": fr.train_n_ind,
                "Test N (ind)": fr.test_n_ind,
                "Train N (obs)": fr.train_n_obs,
                "Test N (obs)": fr.test_n_obs,
                "Converged": fr.converged,
                "Fit message": fr.message,
                "Runtime (s)": round(fr.runtime, 2),
            })
        fold_df = pd.DataFrame(fold_rows)
        st.dataframe(fold_df, use_container_width=True, hide_index=True)

        # Per-fold bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[f"Fold {r['Fold']}" for r in fold_rows],
            y=[r["Test LL"] for r in fold_rows],
            marker_color="steelblue",
        ))
        fig.update_layout(
            title="Out-of-sample log-likelihood by fold",
            xaxis_title="Fold",
            yaxis_title="Test LL",
            template="plotly_dark",
        )
        st.plotly_chart(fig, use_container_width=True, key=f"cv_fold_chart_{label}")

    # Comparison table if multiple models have CV results
    if len(st.session_state["cv_results"]) > 1:
        st.markdown("#### Cross-validation comparison")
        compare_rows = []
        for lbl, res in st.session_state["cv_results"].items():
            compare_rows.append({
                "Model": lbl,
                "K": res.k,
                "Mean test LL": round(res.mean_test_ll, 3),
                "Mean LL/obs": round(res.mean_test_ll_per_obs, 4),
                "Mean hit rate": round(res.mean_hit_rate, 4),
                "Runtime (s)": round(res.total_runtime, 1),
            })
        cv_compare_df = pd.DataFrame(compare_rows)
        st.dataframe(cv_compare_df, use_container_width=True, hide_index=True)

slowbro_next_step("pages/4_🔄_Compare.py")
