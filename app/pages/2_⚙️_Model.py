"""分析侠 — Page 2: Model Configuration and Estimation."""

from __future__ import annotations

import json
import sys
from pathlib import Path

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

from dce_analyzer.config import (  # noqa: E402
    DummyCoding,
    FullModelSpec,
    InteractionTerm,
    VariableSpec,
)
from dce_analyzer.data import get_device_info  # noqa: E402
from dce_analyzer.pipeline import estimate_from_spec  # noqa: E402
from utils import init_session_state, require_data, sidebar_branding  # noqa: E402


def _connected_components(pairs: list[tuple[int, int]], n: int) -> list[list[int]]:
    """Compute connected components from selected correlation pairs."""
    adj: dict[int, set[int]] = {i: set() for i in range(n)}
    for a, b in pairs:
        adj[a].add(b)
        adj[b].add(a)
    visited: set[int] = set()
    components: list[list[int]] = []
    for i in range(n):
        if i not in visited and adj[i]:
            comp: list[int] = []
            queue = [i]
            while queue:
                node = queue.pop(0)
                if node in visited:
                    continue
                visited.add(node)
                comp.append(node)
                for nb in adj[node]:
                    if nb not in visited:
                        queue.append(nb)
            if len(comp) >= 2:
                components.append(sorted(comp))
    return components

init_session_state()
sidebar_branding()

# ── Page header ────────────────────────────────────────────────────
st.header("Model")
st.caption("Configure utility variables, choose a model type, and run estimation.")

require_data()

df: pd.DataFrame = st.session_state.df


# ── helpers ────────────────────────────────────────────────────────
def _guess_col(columns: list[str], candidates: list[str], role: str | None = None) -> str:
    """Find a column by name candidates, checking inferred_columns first."""
    inferred = st.session_state.get("inferred_columns", {})
    if role and inferred.get(role) in columns:
        return inferred[role]
    lowered = {c.lower(): c for c in columns}
    for target in candidates:
        if target.lower() in lowered:
            return lowered[target.lower()]
    return columns[0]


columns = df.columns.tolist()

# ── 1  Column role assignment ──────────────────────────────────────
st.subheader("1. Column roles")
st.markdown("Assign the structural columns in your dataset.")

if st.session_state.get("inferred_columns"):
    st.caption("Pre-filled from auto-detect on the Data page.")

r1, r2, r3, r4 = st.columns(4)
id_col = r1.selectbox(
    "ID column",
    columns,
    index=columns.index(_guess_col(columns, ["respondent_id", "id", "ID"], "id")),
)
task_col = r2.selectbox(
    "Task column",
    columns,
    index=columns.index(_guess_col(columns, ["task_id", "task"], "task")),
)
alt_col = r3.selectbox(
    "Alternative column",
    columns,
    index=columns.index(_guess_col(columns, ["alternative", "alt"], "alt")),
)
choice_col = r4.selectbox(
    "Choice column",
    columns,
    index=columns.index(_guess_col(columns, ["choice", "chosen"], "choice")),
)

# ── BWS (Best-Worst Scaling) mode ────────────────────────────────
st.divider()
bws_mode = st.checkbox(
    "BWS (Best-Worst Scaling) data",
    value=False,
    key="bws_mode",
    help="Enable if your data contains both best AND worst choices per task. "
         "Requires at least 3 alternatives per task (J >= 3).",
)
bws_worst_col: str | None = None
bws_estimate_lambda_w: bool = True

if bws_mode:
    # Auto-detect worst column candidates
    _worst_candidates = [c for c in columns if c not in {id_col, task_col, alt_col, choice_col}]
    _worst_default = _guess_col(columns, ["worst", "worst_choice", "least_preferred"], None)
    if _worst_default not in _worst_candidates:
        _worst_default = _worst_candidates[0] if _worst_candidates else columns[0]

    bws_c1, bws_c2 = st.columns(2)
    with bws_c1:
        bws_worst_col = st.selectbox(
            "Worst choice column",
            _worst_candidates,
            index=_worst_candidates.index(_worst_default) if _worst_default in _worst_candidates else 0,
            key="bws_worst_col",
            help="Column indicating the worst (least preferred) alternative in each task. "
                 "Same format as the choice column (binary 0/1 or label).",
        )
    with bws_c2:
        bws_estimate_lambda_w = st.checkbox(
            "Estimate lambda_w (worst scale parameter)",
            value=True,
            key="bws_estimate_lw",
            help="If checked, estimates a scale parameter lambda_w for worst choices. "
                 "lambda_w > 1 means worst choices are more deterministic; lambda_w < 1 means noisier. "
                 "If unchecked, lambda_w = 1 (equivalent to MaxDiff specification).",
        )
    st.caption(
        "BWS uses **sequential best-first** likelihood: "
        "P(best) x P(worst | best removed). "
        "The existing choice column is treated as the **best** choice."
    )

# ── 2  Variable selection and coding ──────────────────────────────
st.divider()
st.subheader("2. Utility variables")

# Allow selecting any numeric column (DCE attributes are typically numeric-coded)
structural_cols = {id_col, task_col, alt_col, choice_col}
numeric_columns = [c for c in columns if pd.api.types.is_numeric_dtype(df[c]) and c not in structural_cols]
default_features = [
    c
    for c in [
        "price", "time", "comfort", "reliability",
        "travel_time", "travel_cost", "headway", "changes",
    ]
    if c in numeric_columns
]
if not default_features and numeric_columns:
    default_features = numeric_columns[: min(4, len(numeric_columns))]

feature_cols = st.multiselect(
    "Select variables for the utility function",
    options=numeric_columns,
    default=default_features,
    help="Select the attribute columns to include in the utility specification.",
)

if len(feature_cols) == 0:
    st.warning("Pick at least one utility variable.")
    st.stop()

# ── Per-variable coding type: Continuous vs Dummy ──────────────────
st.markdown("**Variable coding**")
st.caption(
    "For each variable, choose **Continuous** (single coefficient, assumes linear effect) "
    "or **Dummy** (one coefficient per level, flexible non-linear effect). "
    "Dummy coding is standard for categorical DCE attributes."
)

coding_map: dict[str, str] = {}  # col -> "continuous" | "dummy"
ref_levels: dict[str, object] = {}  # col -> reference level value

n_coding_cols = min(4, len(feature_cols))
coding_cols = st.columns(n_coding_cols)

for idx, col in enumerate(feature_cols):
    with coding_cols[idx % n_coding_cols]:
        unique_vals = sorted(df[col].dropna().unique())
        n_unique = len(unique_vals)
        # default to dummy if few unique values (typical categorical attribute)
        default_idx = 1 if 2 <= n_unique <= 10 else 0
        coding = st.selectbox(
            f"{col}",
            ["Continuous", "Dummy"],
            index=default_idx,
            key=f"coding_{col}",
            help=f"{n_unique} unique values: {unique_vals[:8]}{'...' if n_unique > 8 else ''}",
        )
        coding_map[col] = coding.lower()

        if coding == "Dummy":
            ref = st.selectbox(
                f"Reference level",
                unique_vals,
                index=0,
                key=f"ref_{col}",
                help="The omitted baseline category. Other levels are estimated relative to this.",
            )
            ref_levels[col] = ref

# ── Build dummy coding specs (backend will expand columns) ─────────
_dummy_codings: list[DummyCoding] = []
expanded_feature_cols: list[str] = []  # expanded column names for UI display
_dummy_info: dict[str, list[str]] = {}  # original col -> list of dummy col names

for col in feature_cols:
    if coding_map[col] == "dummy":
        dc = DummyCoding(column=col, ref_level=ref_levels[col])
        _dummy_codings.append(dc)
        dummy_names, _ = dc.expand(df)
        expanded_feature_cols.extend(dummy_names)
        _dummy_info[col] = dummy_names
    else:
        expanded_feature_cols.append(col)

# Show summary of expanded variables
with st.expander("Variable specification summary", expanded=False):
    summary_rows = []
    for col in feature_cols:
        if coding_map[col] == "dummy":
            ref = ref_levels[col]
            n_dummies = len(_dummy_info[col])
            summary_rows.append({
                "Variable": col,
                "Coding": "Dummy",
                "Reference": str(ref),
                "Coefficients": n_dummies,
                "Columns": ", ".join(_dummy_info[col]),
            })
        else:
            summary_rows.append({
                "Variable": col,
                "Coding": "Continuous",
                "Reference": "—",
                "Coefficients": 1,
                "Columns": col,
            })
    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)
    st.caption(f"Total parameters to estimate: **{len(expanded_feature_cols)}**")

# ── 3  Model type and settings ────────────────────────────────────
st.divider()
st.subheader("3. Model type and settings")

# Show detected hardware
st.info(f"Compute device: **{get_device_info()}**")

model_type_label = st.radio(
    "Select model type",
    ["Conditional Logit", "Mixed Logit", "GMNL", "Latent Class"],
    horizontal=True,
    key="model_type_radio",
)
model_type_map = {
    "Conditional Logit": "conditional",
    "Mixed Logit": "mixed",
    "GMNL": "gmnl",
    "Latent Class": "latent_class",
}
model_type = model_type_map[model_type_label]

if bws_mode:
    if model_type == "conditional":
        st.info("BWS + Conditional Logit: lambda_w is fully identified. Good baseline.")
    elif model_type == "mixed":
        st.info(
            "BWS + Mixed Logit: lambda_w must be a fixed scalar (not random). "
            "It is identified separately from the random coefficient distributions."
        )
    elif model_type == "gmnl":
        st.warning(
            "BWS + GMNL: lambda_w is identified separately from sigma_tau (individual scale), "
            "but both must be fixed parameters. Monitor convergence carefully."
        )
    elif model_type == "latent_class":
        st.info(
            "BWS + Latent Class: lambda_w is shared across all classes. "
            "Per-class lambda_w is theoretically identified but increases parameter count."
        )

dist_map: dict[str, str] = {}

if model_type == "conditional":
    st.caption(
        "All coefficients are fixed across respondents. "
        "Fast to estimate, good baseline model."
    )
    s1, s2 = st.columns(2)
    maxiter = s1.slider("Max optimizer iterations", 20, 500, 200, step=10, key="cl_maxiter")
    est_seed = s2.number_input("Estimation seed", min_value=1, value=123, step=1, key="cl_seed")
    for col in expanded_feature_cols:
        dist_map[col] = "fixed"
    n_draws = 1
    n_classes = 2
    n_starts = 10

elif model_type == "gmnl":
    st.caption(
        "Generalized Multinomial Logit (Fiebig et al. 2010). "
        "Extends Mixed Logit with individual-level scale heterogeneity. "
        "Nests both S-MNL (pure scale) and MMNL as special cases."
    )

    st.markdown("**Distribution assumptions**")
    st.caption("Set distributions for each variable. At least one random variable is required.")
    with st.expander("What do the distribution options mean?"):
        st.markdown(
            "- **fixed**: The coefficient is the same for all respondents.\n"
            "- **normal**: Varies across respondents following a normal distribution.\n"
            "- **lognormal**: exp(normal), ensuring always positive values."
        )

    gmnl_dist_cols = st.columns(min(4, len(feature_cols)))
    for idx, col in enumerate(feature_cols):
        with gmnl_dist_cols[idx % len(gmnl_dist_cols)]:
            default_dist_idx = 0
            dist_val = st.selectbox(
                f"{col}" + (" (dummy)" if coding_map.get(col) == "dummy" else ""),
                ["fixed", "normal", "lognormal"],
                index=default_dist_idx,
                key=f"gmnl_dist_{col}",
            )
            if coding_map.get(col) == "dummy" and col in _dummy_info:
                for dc in _dummy_info[col]:
                    dist_map[dc] = dist_val
            else:
                dist_map[col] = dist_val

    st.markdown("**Estimation settings**")
    gs1, gs2, gs3 = st.columns(3)
    n_draws = gs1.slider("Halton draws", 20, 2000, 200, step=10, key="gmnl_draws")
    maxiter = gs2.slider("Max optimizer iterations", 20, 500, 200, step=10, key="gmnl_maxiter")
    est_seed = gs3.number_input("Estimation seed", min_value=1, value=123, step=1, key="gmnl_seed")

    with st.expander("About GMNL scale parameters"):
        st.markdown(
            "The GMNL model estimates three additional parameters:\n"
            "- **tau** (scale mean): controls the average scale of utility.\n"
            "- **sigma_tau** (scale SD): individual variation in scale.\n"
            "- **gamma** (mixing, 0-1): gamma=0 is pure scale heterogeneity (S-MNL), "
            "gamma=1 is GMNL-II (closest to standard MMNL)."
        )

    # ── Correlation structure for GMNL ────────────────────────────────
    _gmnl_random_expanded = [c for c in expanded_feature_cols if dist_map.get(c, "fixed") != "fixed"]
    mxl_correlated = False
    mxl_correlation_groups: list[list[int]] | None = None

    if len(_gmnl_random_expanded) >= 2:
        st.markdown("**Correlation structure**")
        _gmnl_corr_mode = st.radio(
            "Random parameter correlations",
            ["Independent", "Full correlation", "Selective (pick pairs)"],
            horizontal=True,
            key="gmnl_corr_mode",
            help="Independent: each random parameter varies independently. "
                 "Full: all random parameters are correlated (Cholesky). "
                 "Selective: choose specific pairs to correlate.",
        )
        if _gmnl_corr_mode == "Full correlation":
            mxl_correlated = True
        elif _gmnl_corr_mode == "Selective (pick pairs)":
            _n_rand = len(_gmnl_random_expanded)
            _all_corr_pairs = [
                (i, j, _gmnl_random_expanded[i], _gmnl_random_expanded[j])
                for i in range(_n_rand)
                for j in range(i + 1, _n_rand)
            ]
            _selected_corr: list[tuple[int, int]] = []
            _n_pair_cols = min(4, max(1, len(_all_corr_pairs)))
            _pair_cols = st.columns(_n_pair_cols)
            for _pidx, (_i, _j, _ni, _nj) in enumerate(_all_corr_pairs):
                with _pair_cols[_pidx % _n_pair_cols]:
                    if st.checkbox(f"{_ni} ↔ {_nj}", key=f"gmnl_corr_{_ni}_{_nj}"):
                        _selected_corr.append((_i, _j))
            if _selected_corr:
                mxl_correlated = True
                _groups = _connected_components(_selected_corr, _n_rand)
                mxl_correlation_groups = _groups
                _group_labels = [
                    [_gmnl_random_expanded[k] for k in g] for g in _groups
                ]
                st.caption(f"Correlation blocks: {_group_labels}")
            else:
                st.caption("No pairs selected — using independent structure.")
    elif len(_gmnl_random_expanded) == 1:
        st.caption("Only one random parameter — correlation not applicable.")

    n_classes = 2
    n_starts = 10

elif model_type == "mixed":
    st.caption(
        "Allows coefficients to vary continuously across respondents "
        "(preference heterogeneity). Uses simulated maximum likelihood with Halton draws."
    )

    st.markdown("**Distribution assumptions**")
    st.caption("Set distributions for each variable (or group of dummies). "
               "Dummy-coded variables are typically kept **fixed**.")
    with st.expander("What do the distribution options mean?"):
        st.markdown(
            "- **fixed**: The coefficient is the same for all respondents.\n"
            "- **normal**: Varies across respondents following a normal distribution.\n"
            "- **lognormal**: exp(normal), ensuring always positive values."
        )

    # Show distribution selector per original variable (applies to all its dummies)
    mxl_dist_cols = st.columns(min(4, len(feature_cols)))
    for idx, col in enumerate(feature_cols):
        with mxl_dist_cols[idx % len(mxl_dist_cols)]:
            default_dist_idx = 0  # fixed by default for dummy-coded
            dist_val = st.selectbox(
                f"{col}" + (" (dummy)" if coding_map.get(col) == "dummy" else ""),
                ["fixed", "normal", "lognormal"],
                index=default_dist_idx,
                key=f"dist_{col}",
            )
            # Apply distribution to all expanded columns from this variable
            if coding_map.get(col) == "dummy" and col in _dummy_info:
                for dc in _dummy_info[col]:
                    dist_map[dc] = dist_val
            else:
                dist_map[col] = dist_val

    st.markdown("**Estimation settings**")
    ms1, ms2, ms3 = st.columns(3)
    n_draws = ms1.slider("Halton draws", 20, 2000, 200, step=10, key="mxl_draws")
    maxiter = ms2.slider("Max optimizer iterations", 20, 500, 200, step=10, key="mxl_maxiter")
    est_seed = ms3.number_input("Estimation seed", min_value=1, value=123, step=1, key="mxl_seed")

    # ── Correlation structure: Independent / Full / Selective ────────
    _random_expanded = [c for c in expanded_feature_cols if dist_map.get(c, "fixed") != "fixed"]
    mxl_correlated = False
    mxl_correlation_groups: list[list[int]] | None = None

    if len(_random_expanded) >= 2:
        st.markdown("**Correlation structure**")
        _corr_mode = st.radio(
            "Random parameter correlations",
            ["Independent", "Full correlation", "Selective (pick pairs)"],
            horizontal=True,
            key="mxl_corr_mode",
            help="Independent: each random parameter varies independently. "
                 "Full: all random parameters are correlated (Cholesky). "
                 "Selective: choose specific pairs to correlate.",
        )
        if _corr_mode == "Full correlation":
            mxl_correlated = True
        elif _corr_mode == "Selective (pick pairs)":
            _n_rand = len(_random_expanded)
            _all_corr_pairs = [
                (i, j, _random_expanded[i], _random_expanded[j])
                for i in range(_n_rand)
                for j in range(i + 1, _n_rand)
            ]
            _selected_corr: list[tuple[int, int]] = []
            _n_pair_cols = min(4, max(1, len(_all_corr_pairs)))
            _pair_cols = st.columns(_n_pair_cols)
            for _pidx, (_i, _j, _ni, _nj) in enumerate(_all_corr_pairs):
                with _pair_cols[_pidx % _n_pair_cols]:
                    if st.checkbox(f"{_ni} ↔ {_nj}", key=f"corr_{_ni}_{_nj}"):
                        _selected_corr.append((_i, _j))
            if _selected_corr:
                mxl_correlated = True
                _groups = _connected_components(_selected_corr, _n_rand)
                mxl_correlation_groups = _groups
                _group_labels = [
                    [_random_expanded[k] for k in g] for g in _groups
                ]
                st.caption(f"Correlation blocks: {_group_labels}")
            else:
                st.caption("No pairs selected — using independent structure.")
    elif len(_random_expanded) == 1:
        st.caption("Only one random parameter — correlation not applicable.")

    n_classes = 2
    n_starts = 10

else:  # latent_class
    st.caption(
        "Assumes Q discrete segments of respondents, each with distinct "
        "fixed preferences. Useful for market segmentation."
    )
    ls1, ls2, ls3, ls4 = st.columns(4)
    n_classes = ls1.slider("Number of classes (Q)", 2, 5, 2, key="lc_classes")
    n_starts = ls2.slider("Random starts", 5, 20, 10, key="lc_starts")
    maxiter = ls3.slider("Max optimizer iterations", 20, 500, 200, step=10, key="lc_maxiter")
    est_seed = ls4.number_input("Estimation seed", min_value=1, value=123, step=1, key="lc_seed")

    # Membership covariates: columns constant within each respondent
    st.markdown("**Membership variables (demographics)**")
    st.caption(
        "Optionally select individual-level covariates that explain class membership. "
        "Only columns that are constant within each respondent are shown."
    )
    _candidate_membership_cols = [
        c for c in columns
        if c not in structural_cols
        and c not in expanded_feature_cols
        and c not in feature_cols
    ]
    # Filter to columns constant within respondent groups
    _constant_cols: list[str] = []
    for c in _candidate_membership_cols:
        try:
            if df.groupby(id_col)[c].nunique().max() == 1:
                _constant_cols.append(c)
        except Exception:
            pass
    lc_membership_cols: list[str] = st.multiselect(
        "Select membership covariates",
        options=_constant_cols,
        default=[],
        key="lc_membership_cols",
        help="These variables enter the class membership function. "
             "They must be constant within each respondent (e.g. age, income, gender).",
    )

    for col in expanded_feature_cols:
        dist_map[col] = "fixed"
    n_draws = 1

# ── Defaults for variables not set by every model type branch ─────
if model_type not in ("mixed", "gmnl"):
    mxl_correlated = False
    mxl_correlation_groups = None
if model_type != "latent_class":
    lc_membership_cols = []

# ── Interaction terms (N-way, any columns) — all models ──────────
st.divider()
st.subheader("Interaction terms (optional)")
st.caption(
    "Add interaction terms by multiplying 2 or more columns together. "
    "Columns can be attributes, demographics, or any numeric column. "
    "Works with all model types."
)

# Available columns: expanded feature cols + all other numeric non-structural columns
_interaction_available_cols = list(expanded_feature_cols)
for c in columns:
    if (
        c not in structural_cols
        and c not in _interaction_available_cols
        and c not in feature_cols
        and pd.api.types.is_numeric_dtype(df[c])
    ):
        _interaction_available_cols.append(c)

# Session state for interaction terms
if "interaction_terms" not in st.session_state:
    st.session_state.interaction_terms = []

# Add new interaction term via form
with st.form("add_interaction_form", clear_on_submit=True):
    _inter_cols = st.multiselect(
        "Select columns to interact",
        options=_interaction_available_cols,
        default=[],
        key="new_interaction_cols",
        help="Pick 2 or more columns. Their product will be added as a new variable.",
    )
    _submitted = st.form_submit_button("Add interaction term")
    if _submitted:
        if len(_inter_cols) < 2:
            st.warning("Select at least 2 columns for an interaction term.")
        else:
            new_term = tuple(_inter_cols)
            if new_term not in st.session_state.interaction_terms:
                st.session_state.interaction_terms.append(new_term)
                st.rerun()
            else:
                st.info("This interaction term already exists.")

# Display existing terms with remove buttons
_het_interactions: list[InteractionTerm] = []
if st.session_state.interaction_terms:
    st.markdown("**Current interaction terms:**")
    _terms_to_keep: list[tuple] = []
    for idx, term in enumerate(st.session_state.interaction_terms):
        label = " × ".join(term)
        c_label, c_remove = st.columns([4, 1])
        c_label.markdown(f"- `{label}`")
        if c_remove.button("Remove", key=f"remove_inter_{idx}"):
            pass  # skip this term
        else:
            _terms_to_keep.append(term)
            _het_interactions.append(InteractionTerm(columns=term))
    if len(_terms_to_keep) != len(st.session_state.interaction_terms):
        st.session_state.interaction_terms = _terms_to_keep
        st.rerun()
    st.caption(f"{len(_het_interactions)} interaction term(s) configured.")

# ── Sidebar: model history count ──────────────────────────────────
history: list[dict] = st.session_state.model_history
if history:
    st.sidebar.divider()
    st.sidebar.metric("Saved models", len(history))
    st.sidebar.markdown("**Model history**")
    for entry in history:
        st.sidebar.caption(f"- {entry.get('label', 'model')}")


# ── helpers for result display ────────────────────────────────────
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


def _show_results(estimation, run_label: str, header_suffix: str = "") -> None:
    """Render fit metrics, parameter table, and download buttons."""
    if estimation.success:
        st.success(f"Converged: {estimation.message}")
    else:
        st.warning(f"Did not converge: {estimation.message}")

    st.markdown(f"#### Model fit{header_suffix}")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Log-Likelihood", f"{estimation.log_likelihood:,.3f}")
    m2.metric("AIC", f"{estimation.aic:,.2f}")
    m3.metric("BIC", f"{estimation.bic:,.2f}")
    m4.metric("Iterations", f"{estimation.optimizer_iterations}")
    m5.metric("Runtime (s)", f"{estimation.runtime_seconds:.2f}")

    st.markdown(f"#### Parameter estimates{header_suffix}")
    display_df = estimation.estimates.copy()
    if "p_value" in display_df.columns:
        display_df["sig"] = display_df["p_value"].apply(_significance)
    display_df = display_df.drop(columns=["theta_index"], errors="ignore")
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    st.caption("Significance codes: *** p<0.001, ** p<0.01, * p<0.05, . p<0.1")

    # Show covariance and correlation matrices for correlated MMNL
    if getattr(estimation, "covariance_matrix", None) is not None:
        names = estimation.random_param_names or []
        st.markdown(f"#### Covariance matrix (random parameters){header_suffix}")
        cov_df = pd.DataFrame(estimation.covariance_matrix, index=names, columns=names)
        st.dataframe(cov_df, use_container_width=True)

        st.markdown(f"#### Correlation matrix (random parameters){header_suffix}")
        cor_df = pd.DataFrame(estimation.correlation_matrix, index=names, columns=names)
        st.dataframe(cor_df, use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        csv_bytes = estimation.estimates.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download estimates CSV",
            data=csv_bytes,
            file_name=f"{run_label}_estimates.csv",
            mime="text/csv",
        )
    with d2:
        summary_bytes = json.dumps(estimation.summary_dict(), indent=2, default=str).encode("utf-8")
        st.download_button(
            label="Download summary JSON",
            data=summary_bytes,
            file_name=f"{run_label}_summary.json",
            mime="application/json",
        )


def _show_lc_results(estimation, run_label: str) -> None:
    """Render Latent Class specific results."""
    import plotly.express as px

    if estimation.success:
        st.success(f"Converged: {estimation.message}")
    else:
        st.warning(f"Did not converge: {estimation.message}")

    # Fit metrics
    st.markdown("#### Model fit")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Log-Likelihood", f"{estimation.log_likelihood:,.3f}")
    m2.metric("AIC", f"{estimation.aic:,.2f}")
    m3.metric("BIC", f"{estimation.bic:,.2f}")
    m4.metric("Iterations", f"{estimation.optimizer_iterations}")
    m5.metric("Runtime (s)", f"{estimation.runtime_seconds:.2f}")

    # Class membership probabilities
    st.markdown("#### Class membership probabilities")
    pi_df = pd.DataFrame({
        "Class": [f"Class {i+1}" for i in range(estimation.n_classes)],
        "Probability": estimation.class_probabilities,
    })
    c1, c2 = st.columns([1, 1])
    with c1:
        st.dataframe(pi_df, use_container_width=True, hide_index=True)
    with c2:
        fig_pie = px.pie(pi_df, names="Class", values="Probability", title="Class Shares")
        st.plotly_chart(fig_pie, use_container_width=True)

    # Membership coefficients (if covariates were used)
    if getattr(estimation, "membership_estimates", None) is not None:
        st.markdown("#### Membership function coefficients")
        st.caption(
            "Coefficients explaining class membership probabilities "
            "(relative to Class 1 as reference)."
        )
        mem_est = estimation.membership_estimates
        mem_pivot = mem_est.pivot(index="variable", columns="class_id", values="estimate")
        mem_pivot.columns = [f"Class {c}" for c in mem_pivot.columns]
        st.dataframe(mem_pivot, use_container_width=True)

    # Class-specific parameter estimates
    st.markdown("#### Class-specific parameter estimates")
    class_est = estimation.class_estimates
    pivot = class_est.pivot(index="parameter", columns="class_id", values="estimate")
    pivot.columns = [f"Class {c}" for c in pivot.columns]
    st.dataframe(pivot, use_container_width=True)

    # Coefficient comparison plot
    st.markdown("#### Per-class coefficient comparison")
    fig_bar = px.bar(
        class_est, x="parameter", y="estimate",
        color=class_est["class_id"].astype(str), barmode="group",
        labels={"estimate": "Coefficient", "parameter": "Variable", "color": "Class"},
        title="Coefficient Estimates by Class",
    )
    fig_bar.update_layout(legend_title_text="Class")
    st.plotly_chart(fig_bar, use_container_width=True)

    # Posterior class membership
    st.markdown("#### Posterior class membership")
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
        st.plotly_chart(fig_hist, use_container_width=True)
    with c2:
        count_df = pd.DataFrame({"Class": class_counts.index, "Count": class_counts.values})
        fig_count = px.bar(count_df, x="Class", y="Count", title="Assigned class counts")
        st.plotly_chart(fig_count, use_container_width=True)

    # Full parameter table
    st.markdown("#### All parameter estimates")
    st.dataframe(estimation.estimates, use_container_width=True, hide_index=True)

    # Downloads
    st.markdown("#### Export")
    d1, d2, d3 = st.columns(3)
    with d1:
        csv_bytes = estimation.estimates.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download estimates CSV",
            data=csv_bytes,
            file_name=f"{run_label}_estimates.csv",
            mime="text/csv",
        )
    with d2:
        post_csv = posterior.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download class assignments CSV",
            data=post_csv,
            file_name=f"{run_label}_posterior.csv",
            mime="text/csv",
        )
    with d3:
        summary_bytes = json.dumps(estimation.summary_dict(), indent=2, default=str).encode("utf-8")
        st.download_button(
            label="Download summary JSON",
            data=summary_bytes,
            file_name=f"{run_label}_summary.json",
            mime="application/json",
        )


# ── 4  Run estimation ────────────────────────────────────────────
st.divider()
st.subheader("4. Run estimation")

if st.button("Run Estimation", type="primary", use_container_width=True):
    # Build VariableSpec list from original feature columns.
    # For dummy-coded variables, use the original column name as a placeholder;
    # the backend will expand them using dummy_codings.
    # For continuous variables, use the column directly.
    variables = []
    for col in feature_cols:
        if coding_map[col] == "dummy":
            # Placeholder: backend replaces with expanded dummies.
            # Distribution from the UI dist selector applies to all dummies.
            dummy_expanded = _dummy_info[col]
            dist = dist_map.get(dummy_expanded[0], "fixed") if dummy_expanded else "fixed"
            variables.append(VariableSpec(name=col, column=col, distribution=dist))
        else:
            dist = dist_map.get(col, "fixed")
            variables.append(VariableSpec(name=col, column=col, distribution=dist))

    # Build FullModelSpec — one object captures everything
    full_spec = FullModelSpec(
        id_col=id_col,
        task_col=task_col,
        alt_col=alt_col,
        choice_col=choice_col,
        variables=variables,
        model_type=model_type,
        dummy_codings=_dummy_codings,
        interactions=_het_interactions,
        correlated=mxl_correlated,
        correlation_groups=mxl_correlation_groups,
        bws_worst_col=bws_worst_col if bws_mode else None,
        estimate_lambda_w=bws_estimate_lambda_w if bws_mode else True,
        n_classes=int(n_classes),
        membership_cols=lc_membership_cols if lc_membership_cols else None,
        n_draws=int(n_draws),
        maxiter=int(maxiter),
        seed=int(est_seed),
        n_starts=int(n_starts),
    )

    # Pass the original DataFrame; backend handles dummy expansion
    est_df = df

    # Build spinner message
    if model_type == "latent_class":
        spinner_msg = f"Estimating {n_classes}-class model with {n_starts} random starts..."
    else:
        spinner_msg = "Estimating model — this may take a minute..."

    with st.spinner(spinner_msg):
        try:
            result = estimate_from_spec(df=est_df, spec=full_spec)
        except Exception as exc:
            st.error(f"Estimation failed: {exc}")
            st.exception(exc)
            st.stop()

    estimation = result.estimation

    # Auto-generate run label
    prefix_map = {"conditional": "CL", "mixed": "MXL", "gmnl": "GMNL", "latent_class": "LC"}
    prefix = prefix_map[model_type]
    existing_count = sum(1 for h in history if h.get("model_type") == model_type)
    if model_type == "latent_class":
        run_label = f"{prefix} Run {existing_count + 1} (Q={n_classes})"
    else:
        run_label = f"{prefix} Run {existing_count + 1}"

    # Store as current result (use expanded spec/df for Results page compatibility)
    st.session_state.model_results = {
        "spec": result.expanded_spec or full_spec.to_model_spec(),
        "full_spec": full_spec,
        "model_type": model_type,
        "estimation": estimation,
        "label": run_label,
        "expanded_df": result.expanded_df,
    }

    # Append to history for comparison page
    st.session_state.model_history.append({
        "label": run_label,
        "model_type": model_type,
        "spec": result.expanded_spec or full_spec.to_model_spec(),
        "full_spec": full_spec,
        "estimation": estimation,
    })

    # Also store LC-specific result
    if model_type == "latent_class":
        st.session_state.lc_result = {
            "estimation": estimation,
            "label": run_label,
        }

    st.success(f"Model **{run_label}** estimated successfully.")

    # Show results with appropriate display
    if model_type == "latent_class":
        _show_lc_results(estimation, run_label)
    else:
        _show_results(estimation, run_label)

# ── Show last run results on rerun ────────────────────────────────
elif st.session_state.model_results is not None:
    res = st.session_state.model_results
    est = res["estimation"]
    label = res.get("label", "model")
    if res.get("model_type") == "latent_class":
        _show_lc_results(est, label)
    else:
        _show_results(est, label, header_suffix=" (last run)")

# ── LC: BIC comparison tool ───────────────────────────────────────
if model_type == "latent_class":
    st.divider()
    st.subheader("Optimal number of classes")
    st.markdown(
        "Automatically estimate models with Q = 2, 3, 4, 5 classes and compare BIC."
    )

    if st.button("Run BIC comparison (Q = 2..5)", use_container_width=True):
        import plotly.express as px

        bic_variables = []
        for col in feature_cols:
            bic_variables.append(VariableSpec(name=col, column=col, distribution="fixed"))

        bic_rows: list[dict] = []
        best_bic = float("inf")
        best_q = 2
        progress = st.progress(0, text="Starting class comparison...")

        for i, q in enumerate([2, 3, 4, 5]):
            progress.progress(i / 4, text=f"Estimating Q = {q}...")
            bic_spec = FullModelSpec(
                id_col=id_col, task_col=task_col, alt_col=alt_col,
                choice_col=choice_col, variables=bic_variables,
                model_type="latent_class",
                dummy_codings=_dummy_codings,
                n_classes=q, n_starts=int(n_starts),
                maxiter=int(maxiter), seed=int(est_seed),
                membership_cols=lc_membership_cols if lc_membership_cols else None,
                bws_worst_col=bws_worst_col if bws_mode else None,
                estimate_lambda_w=bws_estimate_lambda_w if bws_mode else True,
            )
            try:
                result = estimate_from_spec(df=df, spec=bic_spec)
                est = result.estimation
                bic_rows.append({
                    "Q": q, "Log-Likelihood": round(est.log_likelihood, 3),
                    "AIC": round(est.aic, 2), "BIC": round(est.bic, 2),
                    "Parameters": est.n_parameters, "Converged": est.success,
                })
                if est.bic < best_bic:
                    best_bic = est.bic
                    best_q = q
            except Exception as exc:
                bic_rows.append({
                    "Q": q, "Log-Likelihood": None, "AIC": None,
                    "BIC": None, "Parameters": None, "Converged": False,
                })
                st.warning(f"Q = {q} failed: {exc}")

        progress.progress(1.0, text="Done!")

        bic_df = pd.DataFrame(bic_rows)
        st.session_state.lc_bic_comparison = bic_df
        st.session_state.lc_best_q = best_q

        st.dataframe(bic_df, use_container_width=True, hide_index=True)

        valid = bic_df.dropna(subset=["BIC"])
        if not valid.empty:
            fig_bic = px.line(valid, x="Q", y="BIC", markers=True, title="BIC by Number of Classes")
            fig_bic.add_vline(x=best_q, line_dash="dash", line_color="green",
                              annotation_text=f"Best Q = {best_q}")
            st.plotly_chart(fig_bic, use_container_width=True)

        st.info(f"Recommended number of classes: **Q = {best_q}**")

    elif st.session_state.get("lc_bic_comparison") is not None:
        import plotly.express as px

        bic_df = st.session_state.lc_bic_comparison
        best_q = st.session_state.lc_best_q
        st.dataframe(bic_df, use_container_width=True, hide_index=True)
        valid = bic_df.dropna(subset=["BIC"])
        if not valid.empty:
            fig_bic = px.line(valid, x="Q", y="BIC", markers=True, title="BIC by Number of Classes")
            fig_bic.add_vline(x=best_q, line_dash="dash", line_color="green",
                              annotation_text=f"Best Q = {best_q}")
            st.plotly_chart(fig_bic, use_container_width=True)
        st.info(f"Recommended number of classes: **Q = {best_q}**")

# ── Show saved model history ──────────────────────────────────────
if st.session_state.model_history:
    st.divider()
    st.subheader("Saved model runs")
    for i, entry in enumerate(st.session_state.model_history, 1):
        est = entry["estimation"]
        st.markdown(
            f"**{i}. {entry.get('label', 'model')}** ({entry.get('model_type', '?')}) "
            f"— LL: {est.log_likelihood:.3f}, AIC: {est.aic:.2f}, "
            f"BIC: {est.bic:.2f}"
        )
