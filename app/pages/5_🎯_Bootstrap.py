# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Prefero — Page 5: Bootstrap Inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ── path setup ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dce_analyzer.bootstrap import run_bootstrap  # noqa: E402
from utils import init_session_state, require_data, sidebar_branding, slowbro_next_step  # noqa: E402

init_session_state()
sidebar_branding()

st.header("Bootstrap")
st.caption("Non-parametric bootstrap inference for robust standard errors.")

require_data()

if st.session_state.model_results is None:
    st.warning("No model results yet. Go to the **Model** page to run an estimation first.")
    st.stop()

_model_res = st.session_state.model_results
estimation = _model_res["estimation"]
df: pd.DataFrame = st.session_state.df
run_id = _model_res.get("run_id")
if not estimation.success or not np.isfinite(estimation.log_likelihood):
    st.warning("Fit a model that converges before bootstrapping.")
    st.stop()


# ── Significance helpers ──────────────────────────────────────────

def _boot_significance(p: float) -> str:
    """Bootstrap significance markers using daggers."""
    if pd.isna(p):
        return ""
    if p < 0.01:
        return "\u2020\u2020\u2020"
    if p < 0.05:
        return "\u2020\u2020"
    if p < 0.1:
        return "\u2020"
    return ""


# ── 1  Configuration ──────────────────────────────────────────────
st.subheader("1. Bootstrap configuration")

with st.expander("What is bootstrap and why use it?"):
    st.markdown(
        "**Bootstrap** estimates uncertainty by refitting the model on samples of respondents.\n\n"
        "**How it works:**\n"
        "1. Randomly draw a new sample of individuals **with replacement** from your data "
        "(some individuals appear multiple times, others not at all).\n"
        "2. Re-estimate the model on this new sample.\n"
        "3. Repeat many times (e.g., 100-500 times).\n"
        "4. The spread of estimates across replications tells you how stable your results are.\n\n"
        "**Why use it?**\n"
        "- Provides **standard errors** from the spread of fitted estimates.\n"
        "- Gives **confidence intervals** that account for the actual shape of the data.\n"
        "- Useful when you suspect the analytical standard errors may be unreliable."
    )

c1, c2 = st.columns(2)
n_reps = c1.slider(
    "Number of replications", 20, 500, 100, step=10,
    help="More successful replications give more stable standard errors and intervals, and take longer.",
)
boot_seed = c2.number_input(
    "Bootstrap seed", min_value=1, value=42, step=1,
    help="A fixed seed ensures reproducible results. Using the same seed will produce identical bootstrap samples.",
)

spec = _model_res.get("full_spec")
if spec is None:
    st.info("Run the model again to save the complete fitting options for bootstrap.")
    st.stop()
label = _model_res.get("label", "model")
st.info(f"Bootstrap will refit **{label}** ({spec.model_type}).")

# ── 2  Run bootstrap ──────────────────────────────────────────────
st.divider()
if st.button("Run Bootstrap", type="primary", use_container_width=True):
    status_container = st.empty()
    progress_bar = st.progress(0, text="Preparing bootstrap...")

    def _update(current: int, total: int) -> None:
        pct = current / total
        progress_bar.progress(
            pct,
            text=f"Replication {current} of {total} ({pct:.0%})",
        )

    st.session_state["_estimation_running"] = True
    try:
        boot = run_bootstrap(
            df=df,
            spec=spec,
            n_replications=n_reps,
            seed=int(boot_seed),
            progress_callback=_update,
            fitted_result=estimation,
        )
    except Exception as exc:
        progress_bar.empty()
        st.session_state["_estimation_running"] = False
        st.error(f"Bootstrap failed: {exc}")
        try:
            from community_db import log_activity
            log_activity("bootstrap_error")
        except Exception:
            pass
        st.stop()

    st.session_state["_estimation_running"] = False
    progress_bar.progress(1.0, text="Complete")
    st.session_state.bootstrap_results = boot
    st.session_state.bootstrap_run_id = run_id
    try:
        from community_db import log_activity
        log_activity("bootstrap_complete")
    except Exception:
        pass

    if boot.n_successful == boot.n_replications:
        status_container.success(
            f"All {boot.n_replications} replications completed successfully."
        )
    else:
        status_container.warning(
            f"{boot.n_successful} of {boot.n_replications} replications succeeded. "
            f"{boot.n_replications - boot.n_successful} did not produce finite, converged fits."
        )

# ── 3  Display results ────────────────────────────────────────────
boot = st.session_state.bootstrap_results
if boot is None or st.session_state.get("bootstrap_run_id") != run_id:
    st.info("Configure and run the bootstrap above to see results.")
    st.stop()

st.divider()
st.subheader("2. Bootstrap summary")
st.caption(f"{boot.n_successful} of {boot.n_replications} replications produced finite, converged fits.")

with st.expander("Understanding bootstrap results"):
    st.markdown(
        "- **bootstrap_mean**: The average coefficient value across all bootstrap samples. "
        "Should be close to the original estimate if the model is stable.\n"
        "- **bootstrap_se**: The standard deviation of the coefficient across bootstrap samples. "
        "This is the bootstrap standard error -- a robust measure of estimation uncertainty.\n"
        "- **ci_lower / ci_upper**: The 2.5th and 97.5th percentiles of the bootstrap distribution, "
        "forming a percentile interval.\n"
        "- **asymptotic_se**: The analytical standard error from the original model, shown for "
        "comparison. Large discrepancies may indicate the analytical approximation is unreliable.\n"
        "- **boot_sig**: An approximate significance marker based on z = estimate / bootstrap_se."
    )

summary = boot.summary_dataframe()

# add z-stat, p-value and bootstrap significance
from scipy.stats import norm as sp_norm  # noqa: E402

_boot_z = summary["original"] / summary["bootstrap_se"].where(summary["bootstrap_se"] > 0)
_boot_p = 2.0 * sp_norm.sf(np.abs(_boot_z))
summary["boot_z_stat"] = _boot_z
summary["boot_p_value"] = _boot_p
summary["boot_sig"] = summary["boot_p_value"].apply(_boot_significance)

# add asymptotic SE for comparison
if "std_error" in estimation.estimates.columns:
    asym_se = dict(zip(estimation.estimates["parameter"], estimation.estimates["std_error"]))
else:
    asym_se = {}
summary["asymptotic_se"] = summary["parameter"].map(asym_se)
st.dataframe(summary, use_container_width=True, hide_index=True)
st.caption(
    "Bootstrap significance codes: "
    "\u2020\u2020\u2020 p<0.01, \u2020\u2020 p<0.05, \u2020 p<0.1"
)

# ── 4  Distribution histograms ────────────────────────────────────
st.divider()
st.subheader("3. Bootstrap distributions")
st.caption(
    "The histogram shows how the coefficient estimate varies across bootstrap samples. "
    "A narrow, symmetric distribution indicates a stable, well-estimated parameter. "
    "A wide or skewed distribution suggests more uncertainty."
)

selected_param = st.selectbox(
    "Select parameter", boot.param_names,
    help="Choose a parameter to view its bootstrap sampling distribution.",
)
if selected_param:
    idx = boot.param_names.index(selected_param)
    col_data = boot.estimates_matrix[:, idx]

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Distribution of `{selected_param}`** ({boot.n_successful} replications)")
        import plotly.express as px

        st.plotly_chart(px.histogram(x=col_data, nbins=20, labels={"x": selected_param}))

    with c2:
        st.markdown(f"**Statistics for `{selected_param}`**")
        stats = {
            "Original estimate": boot.original_estimates.get(selected_param, float("nan")),
            "Bootstrap mean": float(np.mean(col_data)),
            "Bootstrap SE": boot.bootstrap_se[selected_param],
            "95% CI lower": boot.percentile_ci[selected_param][0],
            "95% CI upper": boot.percentile_ci[selected_param][1],
        }
        if selected_param in asym_se:
            stats["Asymptotic SE"] = asym_se[selected_param]
        st.dataframe(pd.Series(stats, name="Value"), use_container_width=True)

# ── 5  Export ──────────────────────────────────────────────────────
st.divider()
st.subheader("4. Export")

d1, d2 = st.columns(2)
with d1:
    csv_bytes = summary.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download bootstrap summary CSV",
        data=csv_bytes,
        file_name="bootstrap_summary.csv",
        mime="text/csv",
    )
with d2:
    raw_df = pd.DataFrame(boot.estimates_matrix, columns=boot.param_names)
    raw_bytes = raw_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download raw bootstrap estimates CSV",
        data=raw_bytes,
        file_name="bootstrap_raw_estimates.csv",
        mime="text/csv",
    )

slowbro_next_step("pages/5_🎯_Bootstrap.py")
