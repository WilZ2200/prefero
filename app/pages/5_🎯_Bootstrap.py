"""分析侠 — Page 5: Bootstrap Inference."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Prefero - Bootstrap", page_icon="🎯", layout="wide")

# ── path setup ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from dce_analyzer.bootstrap import run_bootstrap  # noqa: E402
from dce_analyzer.config import ModelSpec, VariableSpec  # noqa: E402
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
        "**Bootstrap** is a resampling technique that gives you robust measures of uncertainty "
        "for your model estimates, without relying on mathematical assumptions about the "
        "distribution of your data.\n\n"
        "**How it works:**\n"
        "1. Randomly draw a new sample of individuals **with replacement** from your data "
        "(some individuals appear multiple times, others not at all).\n"
        "2. Re-estimate the model on this new sample.\n"
        "3. Repeat many times (e.g., 100-500 times).\n"
        "4. The spread of estimates across replications tells you how stable your results are.\n\n"
        "**Why use it?**\n"
        "- Provides **robust standard errors** that don't rely on large-sample approximations.\n"
        "- Gives **confidence intervals** that account for the actual shape of the data.\n"
        "- Useful when you suspect the analytical standard errors may be unreliable."
    )

c1, c2 = st.columns(2)
n_reps = c1.slider(
    "Number of replications", 20, 500, 100, step=10,
    help="More replications give more precise standard errors but take longer. 100-200 is usually sufficient; use 500 for publication-quality results.",
)
boot_seed = c2.number_input(
    "Bootstrap seed", min_value=1, value=42, step=1,
    help="A fixed seed ensures reproducible results. Using the same seed will produce identical bootstrap samples.",
)

# We need the model spec to re-run estimation.
# Retrieve the last model run from history if available.
history = st.session_state.model_history
if not history:
    st.warning("No model in history. Run a model on the **Model** page first.")
    st.stop()

last_entry = history[-1]
st.info(
    f"Bootstrap will re-use the most recent model configuration: "
    f"**{last_entry.get('label', last_entry.get('name', 'model'))}** ({last_entry['model_type']})"
)

# ── reconstruct spec from the estimation ───────────────────────────
# The user must have columns matching what was used. We'll ask for
# the column roles in case the data changed, but default to the
# detection logic.

_ROLE_CANDIDATES = {
    "id": ["respondent_id", "id", "ID"],
    "task": ["task_id", "task"],
    "alt": ["alternative", "alt"],
    "choice": ["choice", "chosen"],
}


def _auto_detect(candidates: list[str]) -> str:
    lowered = {c.lower(): c for c in df.columns}
    for t in candidates:
        if t.lower() in lowered:
            return lowered[t.lower()]
    return df.columns[0]


# infer variable specs from the estimation estimates table
param_df = estimation.estimates
param_names_raw = param_df["parameter"].tolist()
# extract variable names: beta_X -> X, mu_X -> X, sd_X -> X
_seen: dict[str, str] = {}
has_dist_col = "distribution" in param_df.columns
for p in param_names_raw:
    for prefix in ("beta_", "mu_", "sd_"):
        if p.startswith(prefix):
            varname = p[len(prefix):]
            if has_dist_col:
                dist = param_df.loc[param_df["parameter"] == p, "distribution"].iloc[0]
            else:
                dist = "fixed" if p.startswith("beta_") else "normal"
            if varname not in _seen:
                _seen[varname] = dist

variable_specs = [
    VariableSpec(name=v, column=v, distribution=d)
    for v, d in _seen.items()
    if v in df.columns
]

if not variable_specs:
    st.error("Could not reconstruct variable specifications from the last estimation.")
    st.stop()

# Try to reuse the spec from the last model run if available
_last_spec = last_entry.get("spec")
if _last_spec is not None:
    spec = _last_spec
else:
    spec = ModelSpec(
        id_col=_auto_detect(_ROLE_CANDIDATES["id"]),
        task_col=_auto_detect(_ROLE_CANDIDATES["task"]),
        alt_col=_auto_detect(_ROLE_CANDIDATES["alt"]),
        choice_col=_auto_detect(_ROLE_CANDIDATES["choice"]),
        variables=variable_specs,
        n_draws=200,
    )

model_type = last_entry["model_type"]

# ── Extract extra kwargs from the FullModelSpec (if available) ────
_full_spec = last_entry.get("full_spec")
_boot_extra_kwargs: dict = {}
if _full_spec is not None:
    if getattr(_full_spec, "correlated", False):
        _boot_extra_kwargs["correlated"] = True
    _cg = getattr(_full_spec, "correlation_groups", None)
    if _cg is not None:
        _boot_extra_kwargs["correlation_groups"] = _cg
    if model_type == "latent_class":
        _boot_extra_kwargs["n_classes"] = getattr(_full_spec, "n_classes", 2)
        _boot_extra_kwargs["n_starts"] = getattr(_full_spec, "n_starts", 10)
        _mc = getattr(_full_spec, "membership_cols", None)
        if _mc:
            _boot_extra_kwargs["membership_cols"] = _mc
    _bws = getattr(_full_spec, "bws_worst_col", None)
    if _bws:
        _boot_extra_kwargs["bws_worst_col"] = _bws
        _boot_extra_kwargs["estimate_lambda_w"] = getattr(_full_spec, "estimate_lambda_w", True)

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
            model_type=model_type,
            n_replications=n_reps,
            maxiter=200,
            seed=int(boot_seed),
            progress_callback=_update,
            **_boot_extra_kwargs,
        )
    except Exception as exc:
        progress_bar.empty()
        st.session_state["_estimation_running"] = False
        st.error(f"Bootstrap failed: {exc}")
        try:
            from community_db import log_activity
            log_activity(st.session_state.get("username", "anonymous"), "bootstrap_error", f"error={str(exc)[:200]}")
        except Exception:
            pass
        st.exception(exc)
        st.stop()

    st.session_state["_estimation_running"] = False
    progress_bar.progress(1.0, text="Complete")
    st.session_state.bootstrap_results = boot
    try:
        from community_db import log_activity
        log_activity(st.session_state.get("username", "anonymous"), "bootstrap_complete", f"n_reps={boot.n_replications}, n_success={boot.n_successful}, seed={int(boot_seed)}")
    except Exception:
        pass

    if boot.n_successful == boot.n_replications:
        status_container.success(
            f"All {boot.n_replications} replications completed successfully."
        )
    else:
        status_container.warning(
            f"{boot.n_successful} of {boot.n_replications} replications succeeded. "
            f"{boot.n_replications - boot.n_successful} failed to converge."
        )

# ── 3  Display results ────────────────────────────────────────────
boot = st.session_state.bootstrap_results
if boot is None:
    st.info("Configure and run the bootstrap above to see results.")
    st.stop()

st.divider()
st.subheader("2. Bootstrap summary")

with st.expander("Understanding bootstrap results"):
    st.markdown(
        "- **bootstrap_mean**: The average coefficient value across all bootstrap samples. "
        "Should be close to the original estimate if the model is stable.\n"
        "- **bootstrap_se**: The standard deviation of the coefficient across bootstrap samples. "
        "This is the bootstrap standard error -- a robust measure of estimation uncertainty.\n"
        "- **ci_lower / ci_upper**: The 2.5th and 97.5th percentiles of the bootstrap distribution, "
        "forming a 95% confidence interval. If this interval does not include zero, the effect "
        "is statistically significant.\n"
        "- **asymptotic_se**: The analytical standard error from the original model, shown for "
        "comparison. Large discrepancies may indicate the analytical approximation is unreliable.\n"
        "- **boot_sig**: Bootstrap significance using z = estimate / bootstrap_se."
    )

summary = boot.summary_dataframe()

# add z-stat, p-value and bootstrap significance
from scipy.stats import norm as sp_norm  # noqa: E402

_boot_z = summary["original"] / summary["bootstrap_se"]
_boot_p = 2.0 * (1.0 - sp_norm.cdf(np.abs(_boot_z)))
summary["boot_z_stat"] = _boot_z
summary["boot_p_value"] = _boot_p
summary["boot_sig"] = _boot_p.apply(_boot_significance)

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
        hist_series = pd.Series(col_data, name=selected_param)
        st.bar_chart(hist_series.value_counts().sort_index().head(50))

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
