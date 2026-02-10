"""分析侠 — Page 3: Results Visualization."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── path setup ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from scipy.stats import norm as sp_norm, lognorm  # noqa: E402
from dce_analyzer.model import EstimationResult  # noqa: E402
from dce_analyzer.latent_class import LatentClassResult  # noqa: E402
from dce_analyzer.wtp import compute_wtp  # noqa: E402
from utils import init_session_state, require_data, sidebar_branding, slowbro_next_step  # noqa: E402

init_session_state()
sidebar_branding()

st.header("Results")
st.caption("Inspect model fit metrics, parameter estimates, WTP, and predicted choice shares.")

require_data()

# ── guard: need model results ─────────────────────────────────────
if st.session_state.model_results is None:
    st.warning("No model results yet. Go to the **Model** page to run an estimation first.")
    st.stop()

res = st.session_state.model_results
estimation = res["estimation"]
model_type = res.get("model_type", "unknown")
run_label = res.get("label", "model")
spec = res.get("spec")

is_lc = isinstance(estimation, LatentClassResult)


# ── Helpers ────────────────────────────────────────────────────────

def _significance(p: float) -> str:
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    if p < 0.1:
        return "."
    return ""


est_df = estimation.estimates.drop(columns=["theta_index"], errors="ignore").copy()
has_se = "std_error" in est_df.columns and est_df["std_error"].notna().any()
if has_se:
    est_df["sig"] = est_df["p_value"].apply(_significance)

# ── Model fit summary (always visible above tabs) ─────────────────
st.divider()

with st.expander("Understanding fit metrics"):
    st.markdown(
        "- **Log-Likelihood (LL)**: Measures how well the model explains the observed choices. "
        "Higher (less negative) is better. Values closer to zero indicate a better fit.\n"
        "- **AIC** (Akaike Information Criterion): Balances fit against model complexity. "
        "Lower AIC = better model. Penalizes models with more parameters to avoid overfitting.\n"
        "- **BIC** (Bayesian Information Criterion): Similar to AIC but with a stronger penalty "
        "for extra parameters, especially with large samples. Lower BIC = better model.\n"
        "- **Converged**: Whether the optimizer found a stable solution. "
        "If the model did not converge, results may be unreliable."
    )

if estimation.success:
    st.success(f"Converged: {estimation.message}")
else:
    st.warning(f"Did not converge: {estimation.message}")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Log-Likelihood", f"{estimation.log_likelihood:,.3f}")
m2.metric("AIC", f"{estimation.aic:,.2f}")
m3.metric("BIC", f"{estimation.bic:,.2f}")
m4.metric("Iterations", f"{estimation.optimizer_iterations}")
m5.metric("Runtime (s)", f"{estimation.runtime_seconds:.2f}")

info_cols = st.columns(3)
info_cols[0].metric("Parameters", estimation.n_parameters)
info_cols[1].metric("Observations", estimation.n_observations)
info_cols[2].metric("Individuals", estimation.n_individuals)

# LC convergence diagnostics on Results page
if is_lc:
    _n_attempted = getattr(estimation, "n_starts_attempted", 0)
    _n_succeeded = getattr(estimation, "n_starts_succeeded", 0)
    _all_lls = getattr(estimation, "all_start_lls", [])
    _best_idx = getattr(estimation, "best_start_index", -1)
    _opt_method = getattr(estimation, "optimizer_method", "L-BFGS-B")

    if _n_attempted > 0:
        with st.expander("Convergence diagnostics"):
            st.markdown(f"**Optimizer:** {_opt_method}")
            _cd1, _cd2, _cd3 = st.columns(3)
            _cd1.metric("Random starts attempted", _n_attempted)
            _cd2.metric("Starts succeeded", _n_succeeded)
            _cd3.metric("Best start index", _best_idx + 1 if _best_idx >= 0 else "N/A")

            _valid_lls = [ll for ll in _all_lls if not (isinstance(ll, float) and ll != ll)]
            if len(_valid_lls) > 1:
                _ll_range = max(_valid_lls) - min(_valid_lls)
                st.markdown(
                    f"**LL range across starts:** {min(_valid_lls):.3f} to "
                    f"{max(_valid_lls):.3f} (spread: {_ll_range:.3f})"
                )
                if _ll_range > 1.0:
                    st.caption(
                        "A large spread in LL values across starts suggests "
                        "multiple local optima. Consider increasing random starts."
                    )
                else:
                    st.caption(
                        "LL values are close across starts, suggesting a "
                        "well-identified global optimum."
                    )

                _ll_df = pd.DataFrame({
                    "Start": list(range(1, len(_all_lls) + 1)),
                    "Log-Likelihood": _all_lls,
                })
                _fig_ll = px.bar(
                    _ll_df.dropna(subset=["Log-Likelihood"]),
                    x="Start", y="Log-Likelihood",
                    title="Log-Likelihood by Random Start",
                )
                if _best_idx >= 0:
                    _fig_ll.add_hline(
                        y=estimation.log_likelihood,
                        line_dash="dash", line_color="green",
                        annotation_text="Best",
                    )
                st.plotly_chart(_fig_ll, use_container_width=True, key="results_lc_ll_starts")

# ═══════════════════════════════════════════════════════════════════
#  Tabbed results
# ═══════════════════════════════════════════════════════════════════
st.divider()

wtp_df: pd.DataFrame | None = None

if is_lc:
    # ── Latent Class tabs ─────────────────────────────────────────
    tab_estimates, tab_classes, tab_posterior = st.tabs(
        ["Estimates", "Class Analysis", "Posterior Membership"]
    )

    with tab_estimates:
        st.subheader("Parameter estimates")

        with st.expander("How to read the estimates table"):
            st.markdown(
                "- **estimate**: The coefficient value. A **negative** coefficient means that attribute "
                "reduces the probability of choosing an alternative; a **positive** value increases it.\n"
                "- **std_error**: How precisely the coefficient is estimated. Smaller = more precise.\n"
                "- **z_stat**: The estimate divided by its standard error.\n"
                "- **p_value**: The probability of seeing this result by chance. "
                "Values below 0.05 are typically considered statistically significant.\n"
                "- **sig** (stars): A quick visual indicator of significance level.\n"
                "- **ci_lower / ci_upper**: The 95% confidence interval."
            )

        st.dataframe(est_df, use_container_width=True, hide_index=True)
        if has_se:
            st.caption("Significance codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
        else:
            st.caption("Latent Class model -- standard errors are not computed for individual class parameters.")

    with tab_classes:
        st.subheader("Class membership probabilities")
        st.caption(
            "Each class represents a distinct segment of respondents with similar preferences. "
            "The probability shows the estimated share of the population belonging to each segment."
        )

        pi_df = pd.DataFrame({
            "Class": [f"Class {i+1}" for i in range(estimation.n_classes)],
            "Probability": estimation.class_probabilities,
        })
        c1, c2 = st.columns([1, 1])
        with c1:
            st.dataframe(pi_df, use_container_width=True, hide_index=True)
        with c2:
            fig_pie = px.pie(pi_df, names="Class", values="Probability", title="Class Shares")
            st.plotly_chart(fig_pie, use_container_width=True, key="lc_pie")

        st.divider()
        st.subheader("Class-specific parameter estimates")
        st.caption(
            "Each class has its own set of coefficients, reflecting that segment's preferences. "
            "Compare values across classes to see how segments differ in their trade-offs."
        )

        class_est = estimation.class_estimates
        pivot = class_est.pivot(index="parameter", columns="class_id", values="estimate")
        pivot.columns = [f"Class {c}" for c in pivot.columns]
        st.dataframe(pivot, use_container_width=True)

        st.markdown("#### Per-class coefficient comparison")
        fig_bar = px.bar(
            class_est, x="parameter", y="estimate",
            color=class_est["class_id"].astype(str), barmode="group",
            labels={"estimate": "Coefficient", "parameter": "Variable", "color": "Class"},
            title="Coefficient Estimates by Class",
        )
        fig_bar.update_layout(legend_title_text="Class", height=450)
        st.plotly_chart(fig_bar, use_container_width=True, key="lc_coef_bar")

    with tab_posterior:
        st.subheader("Posterior class membership")
        st.caption(
            "After estimation, each individual is assigned a probability of belonging to each class "
            "based on their observed choices. Higher max probabilities indicate clearer class separation."
        )

        posterior = estimation.posterior_probs
        assigned_class = posterior.idxmax(axis=1)
        class_counts = assigned_class.value_counts().sort_index()

        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(
                posterior.max(axis=1), nbins=30,
                labels={"value": "Max posterior probability", "count": "Count"},
                title="Distribution of max posterior probability",
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True, key="lc_post_hist")
        with c2:
            count_df = pd.DataFrame({"Class": class_counts.index, "Count": class_counts.values})
            fig_count = px.bar(count_df, x="Class", y="Count", title="Assigned class counts")
            fig_count.update_layout(height=400)
            st.plotly_chart(fig_count, use_container_width=True, key="lc_post_bar")

    # Export
    st.divider()
    st.subheader("Export")

    e1, e2, e3, e4 = st.columns(4)
    with e1:
        csv_bytes = estimation.estimates.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download estimates CSV",
            data=csv_bytes,
            file_name=f"{run_label}_estimates.csv",
            mime="text/csv",
        )
    with e2:
        summary_bytes = json.dumps(estimation.summary_dict(), indent=2, default=str).encode("utf-8")
        st.download_button(
            label="Download summary JSON",
            data=summary_bytes,
            file_name=f"{run_label}_summary.json",
            mime="application/json",
        )
    with e3:
        post_csv = posterior.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download class assignments CSV",
            data=post_csv,
            file_name=f"{run_label}_posterior.csv",
            mime="text/csv",
        )
    with e4:
        class_csv = class_est.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download class estimates CSV",
            data=class_csv,
            file_name=f"{run_label}_class_estimates.csv",
            mime="text/csv",
        )

    # HTML report for LC
    st.divider()
    st.subheader("Summary Report")

    report_parts = [
        "<html><head><style>",
        "body { font-family: sans-serif; margin: 2em; }",
        "table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
        "th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }",
        "th { background-color: #f5f5f5; }",
        "</style></head><body>",
        f"<h1>Estimation Report: {run_label}</h1>",
        f"<p>Model type: <b>{model_type}</b> ({estimation.n_classes} classes)</p>",
        "<h2>Model Fit</h2>",
        "<table><tr>",
        f"<th>Log-Likelihood</th><td>{estimation.log_likelihood:.4f}</td>",
        f"<th>AIC</th><td>{estimation.aic:.2f}</td>",
        f"<th>BIC</th><td>{estimation.bic:.2f}</td>",
        "</tr><tr>",
        f"<th>Parameters</th><td>{estimation.n_parameters}</td>",
        f"<th>Observations</th><td>{estimation.n_observations}</td>",
        f"<th>Individuals</th><td>{estimation.n_individuals}</td>",
        "</tr></table>",
        "<h2>Class Membership Probabilities</h2>",
        pi_df.to_html(index=False),
        "<h2>Class-Specific Estimates</h2>",
        pivot.to_html(),
        "<h2>All Parameter Estimates</h2>",
        estimation.estimates.to_html(index=False),
    ]
    report_parts.append("</body></html>")
    report_html = "\n".join(report_parts)

    st.download_button(
        label="Download HTML Report",
        data=report_html.encode("utf-8"),
        file_name=f"{run_label}_report.html",
        mime="text/html",
    )

else:
    # ── CL / MXL path with tabs ──────────────────────────────────
    tab_estimates, tab_wtp, tab_predictions = st.tabs(
        ["Estimates", "WTP", "Predictions"]
    )

    with tab_estimates:
        st.subheader("Parameter estimates")

        with st.expander("How to read the estimates table"):
            st.markdown(
                "- **estimate**: The coefficient value. A **negative** coefficient means that attribute "
                "reduces the probability of choosing an alternative; a **positive** value increases it.\n"
                "- **std_error**: How precisely the coefficient is estimated. Smaller = more precise.\n"
                "- **z_stat**: The estimate divided by its standard error.\n"
                "- **p_value**: The probability of seeing this result by chance. "
                "Values below 0.05 are typically considered statistically significant.\n"
                "- **sig** (stars): A quick visual indicator of significance level.\n"
                "- **ci_lower / ci_upper**: The 95% confidence interval."
            )

        if has_se:
            display_cols = [
                "parameter", "distribution", "estimate", "std_error",
                "z_stat", "p_value", "ci_lower", "ci_upper", "sig",
            ]
            display_df = est_df[[c for c in display_cols if c in est_df.columns]]
            st.dataframe(display_df, use_container_width=True, hide_index=True)
            st.caption("Significance codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")
        else:
            st.dataframe(est_df, use_container_width=True, hide_index=True)
            st.caption("Standard errors not available for this model run.")

        st.markdown("#### Coefficient plot")
        plot_df = est_df.copy()
        if has_se:
            fig_coef = go.Figure()
            fig_coef.add_trace(
                go.Bar(
                    y=plot_df["parameter"],
                    x=plot_df["estimate"],
                    orientation="h",
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=(plot_df["ci_upper"] - plot_df["estimate"]).tolist(),
                        arrayminus=(plot_df["estimate"] - plot_df["ci_lower"]).tolist(),
                    ),
                    marker_color=[
                        "#2196F3" if p < 0.05 else "#90CAF9"
                        for p in plot_df["p_value"]
                    ],
                    hovertemplate=(
                        "<b>%{y}</b><br>"
                        "Estimate: %{x:.4f}<br>"
                        "<extra></extra>"
                    ),
                )
            )
            fig_coef.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_coef.update_layout(
                title="Parameter Estimates with 95% Confidence Intervals",
                xaxis_title="Estimate",
                yaxis_title="",
                height=max(350, len(plot_df) * 55),
                showlegend=False,
            )
            st.plotly_chart(fig_coef, use_container_width=True, key="coef_plot")
        else:
            fig_coef = px.bar(
                plot_df,
                y="parameter",
                x="estimate",
                orientation="h",
                title="Parameter Estimates",
            )
            fig_coef.add_vline(x=0, line_dash="dash", line_color="gray")
            fig_coef.update_layout(
                height=max(350, len(plot_df) * 55),
                xaxis_title="Estimate",
                yaxis_title="",
            )
            st.plotly_chart(fig_coef, use_container_width=True, key="coef_plot_noci")

        # Covariance and correlation matrices for correlated MMNL
        if getattr(estimation, "covariance_matrix", None) is not None:
            names = estimation.random_param_names or []

            st.markdown("#### Covariance matrix (random parameters)")
            cov_df = pd.DataFrame(estimation.covariance_matrix, index=names, columns=names)
            st.dataframe(cov_df, use_container_width=True)

            if getattr(estimation, "covariance_se", None) is not None:
                with st.expander("Covariance standard errors"):
                    cov_se_df = pd.DataFrame(estimation.covariance_se, index=names, columns=names)
                    st.dataframe(cov_se_df, use_container_width=True)

            st.markdown("#### Correlation matrix (random parameters)")
            cor_df = pd.DataFrame(estimation.correlation_matrix, index=names, columns=names)
            st.dataframe(cor_df, use_container_width=True)

            if getattr(estimation, "correlation_se", None) is not None:
                with st.expander("Correlation standard errors"):
                    cor_se_df = pd.DataFrame(estimation.correlation_se, index=names, columns=names)
                    st.dataframe(cor_se_df, use_container_width=True)

            if getattr(estimation, "correlation_test", None) is not None:
                st.markdown("#### Correlation significance tests (H\u2080: \u03c1 = 0)")
                _ct = estimation.correlation_test.copy()
                _ct["sig"] = _ct["p_value"].apply(_significance)
                _ct_display = _ct.rename(columns={
                    "param_1": "Parameter 1",
                    "param_2": "Parameter 2",
                    "covariance": "Cov",
                    "cov_std_error": "Cov SE",
                    "correlation": "\u03c1",
                    "cor_std_error": "SE(\u03c1)",
                    "z_stat": "z",
                    "p_value": "p-value",
                    "sig": "Sig.",
                })
                st.dataframe(_ct_display, use_container_width=True, hide_index=True)
                st.caption("Significance codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")

    with tab_wtp:
        st.subheader("Willingness to Pay (WTP)")
        st.caption(
            "WTP measures how much a respondent would pay for a one-unit "
            "improvement in an attribute: WTP = -beta_attribute / beta_cost."
        )

        param_names = est_df["parameter"].tolist()
        candidate_vars = []
        for p in param_names:
            if p.startswith("beta_") or p.startswith("mu_"):
                var_name = p.split("_", 1)[1]
                candidate_vars.append(var_name)

        seen: set[str] = set()
        candidate_vars_unique: list[str] = []
        for v in candidate_vars:
            if v not in seen:
                seen.add(v)
                candidate_vars_unique.append(v)

        if len(candidate_vars_unique) >= 2:
            cost_guesses = ["price", "cost", "fee", "travel_cost"]
            default_cost = candidate_vars_unique[0]
            for guess in cost_guesses:
                if guess in candidate_vars_unique:
                    default_cost = guess
                    break

            cost_variable = st.selectbox(
                "Select cost / price variable for WTP",
                options=candidate_vars_unique,
                index=candidate_vars_unique.index(default_cost),
                help="Choose the monetary variable (e.g. price, cost, fee) that serves as the denominator for WTP calculations.",
            )

            try:
                wtp_df = compute_wtp(estimation, cost_variable=cost_variable)

                if not wtp_df.empty:
                    st.dataframe(wtp_df, use_container_width=True, hide_index=True)

                    wtp_has_se = wtp_df["wtp_std_error"].notna().any()
                    if wtp_has_se:
                        fig_wtp = go.Figure()
                        fig_wtp.add_trace(
                            go.Bar(
                                y=wtp_df["attribute"],
                                x=wtp_df["wtp_estimate"],
                                orientation="h",
                                error_x=dict(
                                    type="data",
                                    symmetric=False,
                                    array=(wtp_df["wtp_ci_upper"] - wtp_df["wtp_estimate"]).tolist(),
                                    arrayminus=(wtp_df["wtp_estimate"] - wtp_df["wtp_ci_lower"]).tolist(),
                                ),
                                marker_color="#4CAF50",
                                hovertemplate=(
                                    "<b>%{y}</b><br>"
                                    "WTP: %{x:.4f}<br>"
                                    "<extra></extra>"
                                ),
                            )
                        )
                        fig_wtp.add_vline(x=0, line_dash="dash", line_color="gray")
                        fig_wtp.update_layout(
                            title=f"WTP (relative to {cost_variable})",
                            xaxis_title="WTP Estimate",
                            yaxis_title="",
                            height=max(300, len(wtp_df) * 65),
                            showlegend=False,
                        )
                    else:
                        fig_wtp = px.bar(
                            wtp_df,
                            y="attribute",
                            x="wtp_estimate",
                            orientation="h",
                            title=f"WTP (relative to {cost_variable})",
                        )
                        fig_wtp.add_vline(x=0, line_dash="dash", line_color="gray")
                        fig_wtp.update_layout(
                            height=max(300, len(wtp_df) * 65),
                            xaxis_title="WTP Estimate",
                            yaxis_title="",
                        )
                    st.plotly_chart(fig_wtp, use_container_width=True, key="wtp_plot")
                else:
                    st.info("No WTP results to display (no non-cost attributes found).")
            except Exception as exc:
                st.error(f"WTP computation failed: {exc}")
        else:
            st.info("WTP requires at least two estimated parameters (one cost, one attribute).")

    with tab_predictions:
        st.subheader("Predicted choice shares")

        # Use expanded df if available (handles dummy coding / interactions)
        df: pd.DataFrame = res.get("expanded_df", st.session_state.df)
        if df is None:
            df = st.session_state.df
        if spec is not None:
            try:
                from dce_analyzer.data import prepare_choice_tensors  # noqa: E402
                import torch

                tensors = prepare_choice_tensors(df, spec, device=torch.device("cpu"))

                X_np = tensors.X.cpu().numpy()
                betas = []
                for v in spec.variables:
                    param_name = f"beta_{v.name}" if v.distribution == "fixed" else f"mu_{v.name}"
                    match = est_df[est_df["parameter"] == param_name]
                    if len(match) == 1:
                        betas.append(float(match.iloc[0]["estimate"]))
                    else:
                        betas.append(0.0)
                betas_np = np.array(betas)

                V = X_np @ betas_np
                exp_V = np.exp(V - V.max(axis=1, keepdims=True))
                probs = exp_V / exp_V.sum(axis=1, keepdims=True)
                pred_shares = probs.mean(axis=0)

                y_np = tensors.y.cpu().numpy()
                actual_counts = np.bincount(y_np, minlength=tensors.n_alts)
                actual_shares = actual_counts / actual_counts.sum()

                alt_labels = [f"Alt {i + 1}" for i in range(tensors.n_alts)]
                share_df = pd.DataFrame({
                    "Alternative": alt_labels,
                    "Actual Share": actual_shares,
                    "Predicted Share": pred_shares,
                })

                c1, c2 = st.columns(2)
                with c1:
                    st.dataframe(share_df, use_container_width=True, hide_index=True)

                with c2:
                    share_melt = share_df.melt(
                        id_vars="Alternative",
                        value_vars=["Actual Share", "Predicted Share"],
                        var_name="Type",
                        value_name="Share",
                    )
                    fig_share = px.bar(
                        share_melt,
                        x="Alternative",
                        y="Share",
                        color="Type",
                        barmode="group",
                        title="Predicted vs Actual Choice Shares",
                        color_discrete_map={
                            "Actual Share": "#FF7043",
                            "Predicted Share": "#42A5F5",
                        },
                    )
                    fig_share.update_layout(yaxis_tickformat=".1%", height=400)
                    st.plotly_chart(fig_share, use_container_width=True, key="share_plot")

            except Exception as exc:
                st.warning(f"Could not compute predicted shares: {exc}")
        else:
            st.info("Model specification not available for prediction.")

    # Export
    st.divider()
    st.subheader("Export")

    e1, e2, e3 = st.columns(3)

    with e1:
        csv_bytes = estimation.estimates.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download estimates CSV",
            data=csv_bytes,
            file_name=f"{run_label}_estimates.csv",
            mime="text/csv",
        )

    with e2:
        summary_bytes = json.dumps(estimation.summary_dict(), indent=2, default=str).encode("utf-8")
        st.download_button(
            label="Download summary JSON",
            data=summary_bytes,
            file_name=f"{run_label}_summary.json",
            mime="application/json",
        )

    with e3:
        if wtp_df is not None and not wtp_df.empty:
            wtp_csv = wtp_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download WTP CSV",
                data=wtp_csv,
                file_name=f"{run_label}_wtp.csv",
                mime="text/csv",
            )
        else:
            st.download_button(
                label="Download WTP CSV",
                data=b"",
                file_name=f"{run_label}_wtp.csv",
                mime="text/csv",
                disabled=True,
            )

    # HTML summary report
    st.divider()
    st.subheader("Summary Report")

    report_parts = [
        "<html><head><style>",
        "body { font-family: sans-serif; margin: 2em; }",
        "table { border-collapse: collapse; width: 100%; margin: 1em 0; }",
        "th, td { border: 1px solid #ccc; padding: 8px; text-align: left; }",
        "th { background-color: #f5f5f5; }",
        ".metric { display: inline-block; margin: 0.5em 1em; }",
        "</style></head><body>",
        f"<h1>Estimation Report: {run_label}</h1>",
        f"<p>Model type: <b>{model_type}</b></p>",
        "<h2>Model Fit</h2>",
        "<table><tr>",
        f"<th>Log-Likelihood</th><td>{estimation.log_likelihood:.4f}</td>",
        f"<th>AIC</th><td>{estimation.aic:.2f}</td>",
        f"<th>BIC</th><td>{estimation.bic:.2f}</td>",
        "</tr><tr>",
        f"<th>Parameters</th><td>{estimation.n_parameters}</td>",
        f"<th>Observations</th><td>{estimation.n_observations}</td>",
        f"<th>Individuals</th><td>{estimation.n_individuals}</td>",
        "</tr></table>",
        "<h2>Parameter Estimates</h2>",
        estimation.estimates.to_html(index=False),
    ]

    if wtp_df is not None and not wtp_df.empty:
        report_parts.append("<h2>Willingness to Pay</h2>")
        report_parts.append(wtp_df.to_html(index=False))

    report_parts.append("</body></html>")
    report_html = "\n".join(report_parts)

    st.download_button(
        label="Download HTML Report",
        data=report_html.encode("utf-8"),
        file_name=f"{run_label}_report.html",
        mime="text/html",
    )

# ═══════════════════════════════════════════════════════════════════
#  Cross-model distribution comparison
# ═══════════════════════════════════════════════════════════════════
history: list[dict] = st.session_state.get("model_history", [])

# Find MMNL runs that have random parameters (mu_ and sd_ rows)
_mxl_runs = []
for _entry in history:
    if _entry.get("model_type") != "mixed":
        continue
    _est = _entry["estimation"]
    _est_tbl = _est.estimates
    if _est_tbl["parameter"].str.startswith("mu_").any():
        _mxl_runs.append(_entry)

if _mxl_runs:
    st.divider()
    st.subheader("Distribution Comparison Across Models")
    st.caption(
        "For each random parameter in a Mixed Logit run, the estimated distribution "
        "(normal or lognormal) is shown as a density curve. Point estimates from "
        "Conditional Logit runs appear as dashed vertical lines; Latent Class "
        "class-specific estimates appear as dotted vertical lines."
    )

    # Gather CLogit and LC estimates for overlay
    _cl_estimates: list[dict] = []
    _lc_estimates: list[dict] = []

    for _entry in history:
        if _entry.get("model_type") == "conditional":
            _lbl = _entry.get("label", "CL")
            for _, _row in _entry["estimation"].estimates.iterrows():
                _pname = _row["parameter"]
                if _pname.startswith("beta_"):
                    _var_name = _pname.split("_", 1)[1]
                    _cl_estimates.append({"label": _lbl, "param": _var_name, "value": _row["estimate"]})

        elif _entry.get("model_type") == "latent_class":
            _lbl = _entry.get("label", "LC")
            _lc_est = _entry["estimation"]
            if hasattr(_lc_est, "class_estimates") and _lc_est.class_estimates is not None:
                for _, _row in _lc_est.class_estimates.iterrows():
                    _var_name = _row["parameter"]
                    _lc_estimates.append({
                        "label": _lbl,
                        "class_id": int(_row["class_id"]),
                        "param": _var_name,
                        "value": _row["estimate"],
                    })

    # Let user pick which MXL run to visualize if multiple
    _mxl_labels = [e.get("label", f"MXL {i+1}") for i, e in enumerate(_mxl_runs)]
    if len(_mxl_labels) > 1:
        _selected_mxl_label = st.selectbox(
            "Select Mixed Logit run",
            _mxl_labels,
            key="dist_compare_mxl_select",
        )
        _selected_idx = _mxl_labels.index(_selected_mxl_label)
    else:
        _selected_idx = 0

    _mxl_entry = _mxl_runs[_selected_idx]
    _mxl_est = _mxl_entry["estimation"]
    _mxl_tbl = _mxl_est.estimates

    # Extract random parameter info: name, mu, sd, distribution
    _random_params = []
    _mu_rows = _mxl_tbl[_mxl_tbl["parameter"].str.startswith("mu_")]
    for _, _mu_row in _mu_rows.iterrows():
        _vname = _mu_row["parameter"].split("_", 1)[1]
        _sd_row = _mxl_tbl[_mxl_tbl["parameter"] == f"sd_{_vname}"]
        if _sd_row.empty:
            continue
        _mu_val = float(_mu_row["estimate"])
        _sd_val = float(_sd_row.iloc[0]["estimate"])
        _dist_type = str(_mu_row.get("distribution", "normal"))
        _random_params.append({
            "name": _vname,
            "mu": _mu_val,
            "sd": _sd_val,
            "distribution": _dist_type,
        })

    if not _random_params:
        st.info("No random parameters found in the selected Mixed Logit run.")
    else:
        _n_cols = min(2, len(_random_params))
        _plot_cols = st.columns(_n_cols)

        for _pidx, _rp in enumerate(_random_params):
            with _plot_cols[_pidx % _n_cols]:
                _fig = go.Figure()
                _mu, _sd, _dist = _rp["mu"], _rp["sd"], _rp["distribution"]
                _vname = _rp["name"]

                if _dist == "lognormal":
                    if _sd > 0:
                        _x_lo = max(1e-6, lognorm.ppf(0.005, s=_sd, scale=np.exp(_mu)))
                        _x_hi = lognorm.ppf(0.995, s=_sd, scale=np.exp(_mu))
                    else:
                        _x_lo = np.exp(_mu) * 0.5
                        _x_hi = np.exp(_mu) * 1.5
                    _x_vals = np.linspace(_x_lo, _x_hi, 300)
                    _y_vals = lognorm.pdf(_x_vals, s=_sd, scale=np.exp(_mu)) if _sd > 0 else np.zeros_like(_x_vals)
                    _dist_label = "Lognormal"
                else:
                    if _sd > 0:
                        _x_lo = _mu - 4 * _sd
                        _x_hi = _mu + 4 * _sd
                    else:
                        _x_lo = _mu - 1.0
                        _x_hi = _mu + 1.0
                    _x_vals = np.linspace(_x_lo, _x_hi, 300)
                    _y_vals = sp_norm.pdf(_x_vals, loc=_mu, scale=_sd) if _sd > 0 else np.zeros_like(_x_vals)
                    _dist_label = "Normal"

                _fig.add_trace(go.Scatter(
                    x=_x_vals, y=_y_vals,
                    mode="lines",
                    name=f"MXL {_dist_label}(mu={_mu:.3f}, sd={_sd:.3f})",
                    line=dict(color="#2196F3", width=2),
                    fill="tozeroy",
                    fillcolor="rgba(33,150,243,0.15)",
                ))

                # Overlay CLogit point estimates
                _cl_colors = ["#E91E63", "#9C27B0", "#FF5722"]
                _ci = 0
                for _cl in _cl_estimates:
                    if _cl["param"] == _vname:
                        _color = _cl_colors[_ci % len(_cl_colors)]
                        _ci += 1
                        _fig.add_vline(
                            x=_cl["value"], line_dash="dash", line_color=_color,
                            annotation_text=f"{_cl['label']}: {_cl['value']:.3f}",
                            annotation_position="top right",
                        )

                # Overlay LC class estimates
                _lc_colors = ["#4CAF50", "#FF9800", "#00BCD4", "#795548", "#607D8B"]
                _li = 0
                for _lc in _lc_estimates:
                    if _lc["param"] == _vname:
                        _color = _lc_colors[_li % len(_lc_colors)]
                        _li += 1
                        _fig.add_vline(
                            x=_lc["value"], line_dash="dot", line_color=_color,
                            annotation_text=f"{_lc['label']} C{_lc['class_id']}: {_lc['value']:.3f}",
                            annotation_position="top left",
                        )

                _fig.update_layout(
                    title=f"{_vname}",
                    xaxis_title="Coefficient value",
                    yaxis_title="Density",
                    height=350,
                    showlegend=True,
                    legend=dict(font=dict(size=9), yanchor="top", y=0.99),
                    margin=dict(t=40, b=40),
                )
                st.plotly_chart(_fig, use_container_width=True, key=f"dist_cmp_{_vname}_{_pidx}")

slowbro_next_step("pages/3_📈_Results.py")
