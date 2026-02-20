"""Prefero — Page 2: Model Configuration and Estimation."""

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
from utils import init_session_state, require_data, sidebar_branding, slowbro_next_step  # noqa: E402


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
st.markdown(
    "Select the **attributes that enter the utility function**. "
    "Each selected attribute will receive its own coefficient (beta) to be estimated. "
    "Structural columns (ID, task, alternative, choice) are excluded automatically."
)

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
    maxiter = 2000
    est_seed = st.number_input(
        "Estimation seed", min_value=1, value=123, step=1, key="cl_seed",
        help="Random seed ensures replicability — it has no real meaning. Pick your lucky number!",
    )
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

    gmnl_variant_label = st.selectbox(
        "GMNL variant",
        ["GMNL (General)", "S-MNL (gamma=0)", "GMNL-II (gamma=1)"],
        key="gmnl_variant_select",
        help=(
            "**GMNL (General):** gamma is freely estimated (Fiebig et al. 2010). "
            "**S-MNL:** gamma fixed at 0 — pure scale heterogeneity only; "
            "all taste variation comes from individual scale differences. "
            "**GMNL-II:** gamma fixed at 1 — scale multiplies the full "
            "individual-specific beta (mean + deviation)."
        ),
    )
    _gmnl_variant_map = {
        "GMNL (General)": "general",
        "S-MNL (gamma=0)": "s-mnl",
        "GMNL-II (gamma=1)": "gmnl-ii",
    }
    gmnl_variant = _gmnl_variant_map[gmnl_variant_label]

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
    maxiter = 2000
    gs1, gs2 = st.columns(2)
    n_draws = gs1.slider("Halton draws", 20, 2000, 200, step=10, key="gmnl_draws")
    est_seed = gs2.number_input(
        "Estimation seed", min_value=1, value=123, step=1, key="gmnl_seed",
        help="Random seed ensures replicability — it has no real meaning. Pick your lucky number!",
    )

    with st.expander("About GMNL scale parameters"):
        st.markdown(
            "The GMNL model estimates additional scale parameters:\n"
            "- **tau** (scale mean): controls the average scale of utility.\n"
            "- **sigma_tau** (scale SD): individual variation in scale.\n"
            "- **gamma** (mixing, 0-1): gamma=0 is pure scale heterogeneity (S-MNL), "
            "gamma=1 is GMNL-II (closest to standard MMNL).\n\n"
            "When using **S-MNL** or **GMNL-II**, gamma is fixed and not estimated — "
            "reducing the parameter count by 1."
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
    maxiter = 2000
    ms1, ms2 = st.columns(2)
    n_draws = ms1.slider("Halton draws", 20, 2000, 200, step=10, key="mxl_draws")
    est_seed = ms2.number_input(
        "Estimation seed", min_value=1, value=123, step=1, key="mxl_seed",
        help="Random seed ensures replicability — it has no real meaning. Pick your lucky number!",
    )

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
    maxiter = 300  # EM converges in 50-200 iters; 300 is a safe cap
    ls1, ls2, ls3 = st.columns(3)
    n_classes = ls1.slider("Number of classes (Q)", 2, 5, 2, key="lc_classes")
    n_starts = ls2.slider("Random starts", 5, 20, 10, key="lc_starts")
    est_seed = ls3.number_input(
        "Estimation seed", min_value=1, value=123, step=1, key="lc_seed",
        help="Random seed ensures replicability — it has no real meaning. Pick your lucky number!",
    )

    lc_method_label = st.radio(
        "Estimation method",
        ["EM (Recommended)", "Direct (L-BFGS-B)"],
        horizontal=True,
        key="lc_method_radio",
        help="EM: Expectation-Maximization algorithm — standard for latent class models, "
             "monotonic LL increase, generally more robust. "
             "Direct: Full L-BFGS-B optimization of the joint log-likelihood.",
    )
    lc_method = "em" if lc_method_label.startswith("EM") else "direct"

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
if model_type != "gmnl":
    gmnl_variant = "general"
if model_type != "latent_class":
    lc_membership_cols = []
    lc_method = "em"

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
    _sidebar_delete_idx: int | None = None
    for _si, entry in enumerate(history):
        _sb_label, _sb_del = st.sidebar.columns([5, 1])
        with _sb_label:
            st.caption(f"{entry.get('label', 'model')}")
        with _sb_del:
            if st.button("✕", key=f"sidebar_del_{_si}", help="Delete this model"):
                _sidebar_delete_idx = _si
    if _sidebar_delete_idx is not None:
        _deleted = st.session_state.model_history.pop(_sidebar_delete_idx)
        _cur = st.session_state.model_results
        if _cur is not None and _cur.get("label") == _deleted.get("label"):
            st.session_state.model_results = None
        st.rerun()


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
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Log-Likelihood", f"{estimation.log_likelihood:,.3f}")
    m2.metric("AIC", f"{estimation.aic:,.2f}")
    m3.metric("BIC", f"{estimation.bic:,.2f}")
    m4.metric(
        "Runtime (s)", f"{estimation.runtime_seconds:.2f}",
        help="This is the time Slowbro spent on core estimation (number crunching). "
             "Slowbro also handles other chores — data preparation, parameter expansion, "
             "post-processing, and standard error computation — so the actual wall-clock "
             "time you experienced is slightly longer.",
    )

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
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Log-Likelihood", f"{estimation.log_likelihood:,.3f}")
    m2.metric("AIC", f"{estimation.aic:,.2f}")
    m3.metric("BIC", f"{estimation.bic:,.2f}")
    m4.metric(
        "Runtime (s)", f"{estimation.runtime_seconds:.2f}",
        help="This is the time Slowbro spent on core estimation (number crunching). "
             "Slowbro also handles other chores — data preparation, parameter expansion, "
             "post-processing, and standard error computation — so the actual wall-clock "
             "time you experienced is slightly longer.",
    )

    # Convergence diagnostics
    n_attempted = getattr(estimation, "n_starts_attempted", 0)
    n_succeeded = getattr(estimation, "n_starts_succeeded", 0)
    all_lls = getattr(estimation, "all_start_lls", [])
    best_idx = getattr(estimation, "best_start_index", -1)
    opt_method = getattr(estimation, "optimizer_method", "L-BFGS-B")

    if n_attempted > 0:
        with st.expander("Convergence diagnostics"):
            st.markdown(f"**Optimizer:** {opt_method}")
            c_d1, c_d2, c_d3 = st.columns(3)
            c_d1.metric("Random starts attempted", n_attempted)
            c_d2.metric("Starts succeeded", n_succeeded)
            c_d3.metric("Best start index", best_idx + 1 if best_idx >= 0 else "N/A")

            # EM-specific diagnostics
            em_iters = getattr(estimation, "em_iterations", 0)
            em_history = getattr(estimation, "em_ll_history", [])
            em_conv = getattr(estimation, "em_converged", False)

            if em_iters > 0:
                e_d1, e_d2 = st.columns(2)
                e_d1.metric("EM iterations (best start)", em_iters)
                e_d2.metric("EM converged", "Yes" if em_conv else "No")

                if len(em_history) > 1:
                    em_df = pd.DataFrame({
                        "Iteration": list(range(1, len(em_history) + 1)),
                        "Log-Likelihood": em_history,
                    })
                    fig_em = px.line(
                        em_df, x="Iteration", y="Log-Likelihood",
                        title="EM Convergence (best start)",
                        markers=True,
                    )
                    st.plotly_chart(fig_em, use_container_width=True)

            valid_lls = [ll for ll in all_lls if not (isinstance(ll, float) and ll != ll)]
            if len(valid_lls) > 1:
                ll_range = max(valid_lls) - min(valid_lls)
                st.markdown(
                    f"**LL range across starts:** {min(valid_lls):.3f} to "
                    f"{max(valid_lls):.3f} (spread: {ll_range:.3f})"
                )
                if ll_range > 1.0:
                    st.caption(
                        "A large spread in LL values across starts suggests "
                        "multiple local optima. Consider increasing random starts."
                    )
                else:
                    st.caption(
                        "LL values are close across starts, suggesting a "
                        "well-identified global optimum."
                    )

                ll_df = pd.DataFrame({
                    "Start": list(range(1, len(all_lls) + 1)),
                    "Log-Likelihood": all_lls,
                })
                fig_ll = px.bar(
                    ll_df.dropna(subset=["Log-Likelihood"]),
                    x="Start", y="Log-Likelihood",
                    title="Log-Likelihood by Random Start",
                )
                if best_idx >= 0:
                    fig_ll.add_hline(
                        y=estimation.log_likelihood,
                        line_dash="dash", line_color="green",
                        annotation_text="Best",
                    )
                st.plotly_chart(fig_ll, use_container_width=True)

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

# ── Auto-suggest a descriptive model name ──────────────────────
def _suggest_model_name() -> str:
    """Build a descriptive default name from the current model configuration."""
    prefix_map = {"conditional": "CL", "mixed": "MXL", "gmnl": "GMNL", "latent_class": "LC"}
    prefix = prefix_map[model_type]
    if model_type == "gmnl" and gmnl_variant != "general":
        variant_labels = {"s-mnl": "S-MNL", "gmnl-ii": "GMNL-II"}
        prefix = variant_labels.get(gmnl_variant, prefix)
    parts: list[str] = [prefix]

    if model_type in ("mixed", "gmnl"):
        # Summarise distribution counts: e.g. "2N+1F" for 2 normal + 1 fixed
        dist_counts: dict[str, int] = {}
        for _c in expanded_feature_cols:
            d = dist_map.get(_c, "fixed")
            abbr = {"fixed": "F", "normal": "N", "lognormal": "LN"}.get(d, d[0].upper())
            dist_counts[abbr] = dist_counts.get(abbr, 0) + 1
        dist_summary = "+".join(f"{v}{k}" for k, v in dist_counts.items())
        if dist_summary:
            parts.append(dist_summary)

        # Correlation structure
        if mxl_correlated:
            if mxl_correlation_groups:
                parts.append("sel-corr")
            else:
                parts.append("full-corr")
        else:
            parts.append("indep")

    elif model_type == "latent_class":
        parts.append(f"Q={n_classes}")
        parts.append(f"{len(expanded_feature_cols)}vars")

    elif model_type == "conditional":
        parts.append(f"{len(expanded_feature_cols)}vars")

    # BWS flag
    if bws_mode:
        parts.append("+BWS")

    # Interaction count
    if _het_interactions:
        parts.append(f"+{len(_het_interactions)}int")

    return " ".join(parts)

_suggested_name = _suggest_model_name()
# Auto-update model name when specification changes
if st.session_state.get("_prev_suggested_name") != _suggested_name:
    st.session_state.model_run_name = _suggested_name
    st.session_state._prev_suggested_name = _suggested_name

run_name = st.text_input(
    "Model name",
    key="model_run_name",
    help="A descriptive name for this model run. Auto-suggested based on your settings; feel free to edit.",
)

# ── Custom starting values (all model types) ─────────────────────

def _build_param_names() -> list[dict[str, str]]:
    """Build the parameter name list for the current model configuration.

    Returns a list of dicts with 'index', 'parameter', 'description'.
    """
    rows: list[dict[str, str]] = []
    idx = 0

    if model_type == "conditional":
        for col in expanded_feature_cols:
            rows.append({"#": str(idx), "Parameter": f"beta_{col}",
                         "Description": f"Coefficient for {col}"})
            idx += 1

    elif model_type in ("mixed", "gmnl"):
        # Mean parameters
        for col in expanded_feature_cols:
            d = dist_map.get(col, "fixed")
            if d == "fixed":
                rows.append({"#": str(idx), "Parameter": f"mu_{col}",
                             "Description": f"Fixed coefficient for {col}"})
                idx += 1
            else:
                rows.append({"#": str(idx), "Parameter": f"mu_{col}",
                             "Description": f"Mean of {d} coefficient for {col}"})
                idx += 1

        # SD / Cholesky parameters
        _random_cols = [c for c in expanded_feature_cols if dist_map.get(c, "fixed") != "fixed"]
        if mxl_correlated and len(_random_cols) >= 2:
            # Cholesky lower-triangular elements
            if mxl_correlation_groups:
                # Selective: block-diagonal Cholesky
                for group in mxl_correlation_groups:
                    g_names = [_random_cols[i] for i in group if i < len(_random_cols)]
                    for i, ni in enumerate(g_names):
                        for j in range(i + 1):
                            nj = g_names[j]
                            if i == j:
                                rows.append({"#": str(idx), "Parameter": f"L[{ni},{nj}]",
                                             "Description": f"Cholesky diagonal: SD of {ni} (raw, softplus-transformed)"})
                            else:
                                rows.append({"#": str(idx), "Parameter": f"L[{ni},{nj}]",
                                             "Description": f"Cholesky off-diag: covariance {ni}-{nj}"})
                            idx += 1
                # Standalone random params (not in any group)
                _grouped_indices = {i for g in mxl_correlation_groups for i in g}
                for ri, col in enumerate(_random_cols):
                    if ri not in _grouped_indices:
                        rows.append({"#": str(idx), "Parameter": f"sd_{col}",
                                     "Description": f"SD of {col} (raw, softplus-transformed)"})
                        idx += 1
            else:
                # Full correlation: one Cholesky block for all random vars
                for i, ni in enumerate(_random_cols):
                    for j in range(i + 1):
                        nj = _random_cols[j]
                        if i == j:
                            rows.append({"#": str(idx), "Parameter": f"L[{ni},{nj}]",
                                         "Description": f"Cholesky diagonal: SD of {ni} (raw, softplus-transformed)"})
                        else:
                            rows.append({"#": str(idx), "Parameter": f"L[{ni},{nj}]",
                                         "Description": f"Cholesky off-diag: covariance {ni}-{nj}"})
                        idx += 1
        else:
            # Independent: one SD per random var
            for col in _random_cols:
                d = dist_map.get(col, "fixed")
                rows.append({"#": str(idx), "Parameter": f"sd_{col}",
                             "Description": f"SD of {d} coefficient for {col} (raw, softplus-transformed)"})
                idx += 1

        if model_type == "gmnl":
            rows.append({"#": str(idx), "Parameter": "tau",
                         "Description": "Scale mean (0 = unit scale)"})
            idx += 1
            rows.append({"#": str(idx), "Parameter": "raw_sigma_tau",
                         "Description": "Scale SD (raw, softplus-transformed)"})
            idx += 1
            if gmnl_variant == "general":
                rows.append({"#": str(idx), "Parameter": "raw_gamma",
                             "Description": "Mixing parameter (raw, sigmoid -> [0,1])"})
                idx += 1

    elif model_type == "latent_class":
        for q in range(1, n_classes + 1):
            for col in expanded_feature_cols:
                rows.append({"#": str(idx), "Parameter": f"beta_{col}_class{q}",
                             "Description": f"Coefficient for {col} in class {q}"})
                idx += 1
        if n_classes > 1:
            if lc_membership_cols:
                for q in range(2, n_classes + 1):
                    rows.append({"#": str(idx), "Parameter": f"gamma_class{q}",
                                 "Description": f"Intercept for class {q} membership (vs class 1)"})
                    idx += 1
                    for mc in lc_membership_cols:
                        rows.append({"#": str(idx), "Parameter": f"delta_{mc}_class{q}",
                                     "Description": f"Effect of {mc} on class {q} membership"})
                        idx += 1
            else:
                for q in range(2, n_classes + 1):
                    rows.append({"#": str(idx), "Parameter": f"gamma_class{q}",
                                 "Description": f"Log-odds of class {q} vs class 1"})
                    idx += 1

    # BWS lambda_w (appended last for all model types)
    if bws_mode and bws_estimate_lambda_w:
        rows.append({"#": str(idx), "Parameter": "raw_lambda_w",
                     "Description": "Worst-choice scale (raw, softplus-transformed)"})
        idx += 1

    return rows


_param_info = _build_param_names()

with st.expander("Custom starting values (advanced)"):
    st.caption(
        "Optionally provide your own initial parameter vector (theta_0). "
        "Fill in the **Starting value** column below — leave blank for default initialization. "
        "For Latent Class, this replaces the first random start; remaining starts are still random."
    )
    st.markdown(f"**Total parameters:** {len(_param_info)}")

    _start_df = pd.DataFrame({
        "#": [r["#"] for r in _param_info],
        "Parameter": [r["Parameter"] for r in _param_info],
        "Description": [r["Description"] for r in _param_info],
        "Starting value": [None] * len(_param_info),
    })

    _edited_start = st.data_editor(
        _start_df,
        use_container_width=True,
        hide_index=True,
        disabled=["#", "Parameter", "Description"],
        column_config={
            "#": st.column_config.TextColumn(width="small"),
            "Parameter": st.column_config.TextColumn(width="medium"),
            "Description": st.column_config.TextColumn(width="large"),
            "Starting value": st.column_config.NumberColumn(
                format="%.6f",
                help="Enter your initial guess for this parameter. Leave blank for default.",
            ),
        },
        key="custom_start_editor",
    )

    st.caption(
        "**Note:** Parameters marked *(raw, softplus-transformed)* are on an internal scale. "
        "For example, softplus(0) ≈ 0.69, softplus(-1) ≈ 0.31. The final reported value "
        "is the transformed (positive) quantity."
    )

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

    # Parse custom starting values from the data_editor
    _parsed_custom_start: list[float] | None = None
    _sv_col = _edited_start["Starting value"]
    if _sv_col.notna().any():
        if _sv_col.notna().all():
            _parsed_custom_start = _sv_col.tolist()
        else:
            st.error(
                "Custom starting values must be provided for **all** parameters or **none**. "
                f"You filled {_sv_col.notna().sum()} of {len(_sv_col)} values."
            )
            st.stop()

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
        gmnl_variant=gmnl_variant,
        bws_worst_col=bws_worst_col if bws_mode else None,
        estimate_lambda_w=bws_estimate_lambda_w if bws_mode else True,
        n_classes=int(n_classes),
        membership_cols=lc_membership_cols if lc_membership_cols else None,
        lc_method=lc_method if model_type == "latent_class" else "em",
        n_draws=int(n_draws),
        maxiter=int(maxiter),
        seed=int(est_seed),
        n_starts=int(n_starts),
        custom_start=_parsed_custom_start,
    )

    # ── Early validation: BWS requires J >= 3 ──────────────────
    if bws_mode:
        _n_alts_per_task = df.groupby([id_col, task_col]).size()
        if _n_alts_per_task.min() < 3:
            st.error(
                "BWS requires at least 3 alternatives per task (J >= 3). "
                f"Your data has tasks with only {_n_alts_per_task.min()} alternatives."
            )
            st.stop()

    # Pass the original DataFrame; backend handles dummy expansion
    est_df = df

    # ── Slowbro waiting UI ──────────────────────────────────────
    import random as _rng
    from trivia import TRIVIA_FACTS

    _facts = TRIVIA_FACTS.copy()
    _rng.shuffle(_facts)
    _display_facts = _facts[:25]  # show 25 facts in rotation

    _trivia_items = "".join(
        f'<div class="trivia-item">{fact}</div>' for fact in _display_facts
    )

    _waiting_placeholder = st.empty()
    _waiting_placeholder.markdown(
        f'''
        <style>
        .slowbro-waiting {{
            margin: 16px 0;
            max-width: 520px;
        }}
        .trivia-box {{
            height: 44px;
            overflow: hidden;
            border-left: 3px solid #4CAF50;
            padding-left: 12px;
            position: relative;
        }}
        .trivia-roll {{
            animation: roll-up {len(_display_facts) * 8}s linear infinite;
        }}
        @keyframes roll-up {{
            0%   {{ transform: translateY(0); }}
            100% {{ transform: translateY(-50%); }}
        }}
        .trivia-item {{
            height: 44px;
            display: flex;
            align-items: center;
            font-size: 0.82rem;
            color: #999;
            line-height: 1.3;
        }}
        /* Switch top-right Slowbro to working state */
        [data-testid="stHeader"]::after {{
            content: "working on caffeine..." !important;
        }}
        </style>
        <div class="slowbro-waiting">
            <div style="font-size:0.85rem; color:#aaa; margin-bottom:8px;">Slowbro is working...</div>
            <div class="trivia-box">
                <div class="trivia-roll">
                    {_trivia_items}
                    {_trivia_items}
                </div>
            </div>
        </div>
        ''',
        unsafe_allow_html=True,
    )

    # ── Run estimation (no st.spinner) ──────────────────────
    st.session_state["_estimation_running"] = True
    try:
        result = estimate_from_spec(df=est_df, spec=full_spec)
    except Exception as exc:
        _waiting_placeholder.empty()
        st.session_state["_estimation_running"] = False
        st.error(f"Estimation failed: {exc}")
        try:
            from community_db import log_activity
            import traceback as _tb
            _tb_str = _tb.format_exc()
            log_activity(st.session_state.get("username", "anonymous"), "estimation_error", f"model={model_type}, error={str(exc)[:200]}")
            logger_msg = f"Estimation error: {_tb_str}"
            import logging
            logging.getLogger("prefero").warning(logger_msg)
        except Exception:
            pass
        st.stop()

    st.session_state["_estimation_running"] = False
    _waiting_placeholder.empty()

    estimation = result.estimation

    # Use the user-provided model name (or the auto-suggested default)
    run_label = run_name.strip() if run_name.strip() else _suggested_name

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
    try:
        from community_db import log_activity
        log_activity(st.session_state.get("username", "anonymous"), "estimation_success", f"model={model_type}, label={run_label}, LL={estimation.log_likelihood:.2f}, AIC={estimation.aic:.2f}, runtime={estimation.runtime_seconds:.1f}s")
    except Exception:
        pass

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

# ── Save to Profile button (always visible when results exist) ────
if st.session_state.model_results is not None:
    _username = st.session_state.get("username", "")
    if _username:
        _n_saved = len(st.session_state.get("saved_models", []))
        if _n_saved >= 10:
            st.warning("You've reached the limit of 10 saved models. Delete one from your profile to save more.")
        else:
            _cur_label = st.session_state.model_results.get("label", "model")
            # Check if already saved
            _already_saved = any(
                sm.get("label") == _cur_label
                for sm in st.session_state.get("saved_models", [])
            )
            if _already_saved:
                st.info(f"Model **{_cur_label}** is already saved to your profile.")
            else:
                if st.button("💾 Save to Profile", key="_save_to_profile", type="primary", use_container_width=True):
                    from model_store import save_model, load_models
                    _entry = st.session_state.model_results
                    if save_model(_username, _entry):
                        st.session_state.saved_models = load_models(_username)
                        st.success(f"Model '{_cur_label}' saved to your profile!")
                        st.rerun()
                    else:
                        st.error("Could not save — you may have reached the 10-model limit.")

# ── LC: BIC comparison tool ───────────────────────────────────────
if model_type == "latent_class":
    st.divider()
    st.subheader("Optimal number of classes")
    st.markdown(
        "Automatically estimate models with Q = 2, 3, 4, 5 classes and compare BIC."
    )

    if st.button("Run BIC comparison (Q = 2..5)", use_container_width=True):
        import plotly.express as px

        try:
            from community_db import log_activity
            log_activity(st.session_state.get("username", "anonymous"), "bic_comparison", "Q_range=2-5")
        except Exception:
            pass

        st.session_state["_estimation_running"] = True
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
                lc_method=lc_method,
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

        st.session_state["_estimation_running"] = False
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
    _delete_idx: int | None = None
    for i, entry in enumerate(st.session_state.model_history):
        est = entry["estimation"]
        _info_col, _del_col = st.columns([6, 1])
        with _info_col:
            st.markdown(
                f"**{i + 1}. {entry.get('label', 'model')}** ({entry.get('model_type', '?')}) "
                f"— LL: {est.log_likelihood:.3f}, AIC: {est.aic:.2f}, "
                f"BIC: {est.bic:.2f}"
            )
        with _del_col:
            if st.button("Delete", key=f"delete_model_{i}"):
                _delete_idx = i

    if _delete_idx is not None:
        _deleted = st.session_state.model_history.pop(_delete_idx)
        # Clear current results if they match the deleted model
        _cur = st.session_state.model_results
        if _cur is not None and _cur.get("label") == _deleted.get("label"):
            st.session_state.model_results = None
        st.rerun()

# ── Saved models from profile ──────────────────────────────────
_saved = st.session_state.get("saved_models", [])
if _saved:
    st.divider()
    st.subheader("Saved Models (Profile)")
    st.caption("These models are saved to your profile and persist across sessions.")
    _profile_delete_idx: int | None = None
    _profile_load_idx: int | None = None
    for i, sm in enumerate(_saved):
        est = sm.get("estimation")
        _info_c, _load_c, _del_c = st.columns([5, 1, 1])
        with _info_c:
            _saved_at = sm.get("saved_at", "")
            if _saved_at:
                _saved_at = _saved_at[:10]  # just date
            st.markdown(
                f"**{i+1}. {sm.get('label', 'model')}** ({sm.get('model_type', '?')}) "
                f"-- LL: {est.log_likelihood:.3f}, AIC: {est.aic:.2f}, BIC: {est.bic:.2f} "
                f"<span style='color:gray;font-size:0.8em;'>({_saved_at})</span>",
                unsafe_allow_html=True,
            )
        with _load_c:
            if st.button("Load", key=f"_load_saved_{i}"):
                _profile_load_idx = i
        with _del_c:
            if st.button("Delete", key=f"_del_saved_{i}"):
                _profile_delete_idx = i

    if _profile_load_idx is not None:
        _loaded = _saved[_profile_load_idx]
        st.session_state.model_results = {
            "spec": _loaded.get("spec"),
            "full_spec": _loaded.get("full_spec"),
            "model_type": _loaded.get("model_type"),
            "estimation": _loaded["estimation"],
            "label": _loaded.get("label"),
            "expanded_df": None,
        }
        if _loaded not in st.session_state.model_history:
            st.session_state.model_history.append(_loaded)
        st.rerun()

    if _profile_delete_idx is not None:
        _username = st.session_state.get("username", "")
        from model_store import delete_saved_model, load_models
        if delete_saved_model(_username, _profile_delete_idx):
            st.session_state.saved_models = load_models(_username)
        st.rerun()

slowbro_next_step("pages/2_⚙️_Model.py")
