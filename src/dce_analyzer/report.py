# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Standalone HTML report generator for DCE estimation results.

Generates a professional, self-contained HTML report with:
  - Interactive plotly charts (via CDN)
  - Collapsible <details>/<summary> sections for lengthy content
  - Utility function formula
  - Coefficient plot, WTP bar chart, distribution curves
  - Latent class charts (pie, bar, posterior histogram)
  - Predicted vs actual choice shares
  - Full model specification details (dummy coding, interactions, correlations)
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any

import numpy as np
import pandas as pd

from .config import FullModelSpec, ModelSpec, VariableSpec
from .latent_class import LatentClassResult
from .model import EstimationResult

# Optional dependencies for charts
try:
    import plotly.graph_objects as go
    import plotly.io as pio

    _HAS_PLOTLY = True
except ImportError:
    _HAS_PLOTLY = False

try:
    from scipy.stats import norm as sp_norm, lognorm

    _HAS_SCIPY = True
except ImportError:
    _HAS_SCIPY = False

# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

_MODEL_TYPE_NAMES = {
    "conditional": "Conditional Logit",
    "mixed": "Mixed Logit",
    "gmnl": "Generalised Mixed Logit (GMNL)",
    "latent_class": "Latent Class Logit",
}


def _format_number(x: Any, decimals: int = 4) -> str:
    """Format a number for display, handling NaN / None gracefully."""
    if x is None:
        return "&mdash;"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if math.isnan(f) or math.isinf(f):
        return "&mdash;"
    return f"{f:.{decimals}f}"


def _significance(p: Any) -> str:
    """Return significance stars for a p-value."""
    try:
        pv = float(p)
    except (TypeError, ValueError):
        return ""
    if math.isnan(pv):
        return ""
    if pv < 0.001:
        return "***"
    if pv < 0.01:
        return "**"
    if pv < 0.05:
        return "*"
    if pv < 0.1:
        return "."
    return ""


def _esc(text: Any) -> str:
    """HTML-escape a value."""
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _html_table(
    df: pd.DataFrame,
    *,
    num_cols: set[str] | None = None,
    fmt: int = 4,
    caption: str = "",
    css_class: str = "",
) -> str:
    """Convert a DataFrame to a styled HTML table string."""
    if num_cols is None:
        num_cols = {c for c in df.columns if df[c].dtype.kind in "fiuc"}

    cls_attr = f' class="{css_class}"' if css_class else ""
    parts: list[str] = [f"<table{cls_attr}>"]
    if caption:
        parts.append(f"<caption>{_esc(caption)}</caption>")

    parts.append("<thead><tr>")
    for col in df.columns:
        align = ' class="num"' if col in num_cols else ""
        parts.append(f"<th{align}>{_esc(col)}</th>")
    parts.append("</tr></thead>")

    parts.append("<tbody>")
    for _, row in df.iterrows():
        parts.append("<tr>")
        for col in df.columns:
            val = row[col]
            if col in num_cols:
                parts.append(f'<td class="num">{_format_number(val, fmt)}</td>')
            else:
                parts.append(f"<td>{_esc(val)}</td>")
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def _matrix_table(matrix: np.ndarray, names: list[str]) -> str:
    """Render a square numpy matrix as an HTML table with row/column headers."""
    parts: list[str] = ["<table>", "<thead><tr><th></th>"]
    for n in names:
        parts.append(f'<th class="num">{_esc(n)}</th>')
    parts.append("</tr></thead><tbody>")
    for i, rname in enumerate(names):
        parts.append(f"<tr><td><strong>{_esc(rname)}</strong></td>")
        for j in range(len(names)):
            parts.append(f'<td class="num">{_format_number(matrix[i, j], 4)}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Plotly chart helper
# ---------------------------------------------------------------------------


def _plotly_div(fig: Any, height: int = 400) -> str:
    """Convert a plotly figure to an embeddable HTML div string.

    Uses ``include_plotlyjs=False`` — the CDN script is included once in the
    ``<head>`` of the report HTML.
    """
    if not _HAS_PLOTLY:
        return ""
    fig.update_layout(
        margin=dict(l=60, r=30, t=50, b=50),
        height=height,
        font=dict(
            family=(
                "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, "
                "'Helvetica Neue', Arial, sans-serif"
            ),
            size=13,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return pio.to_html(fig, full_html=False, include_plotlyjs=False)


# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

_CSS = """\
:root {
  --clr-primary: #1a365d;
  --clr-accent: #2563eb;
  --clr-bg: #f7f8fa;
  --clr-white: #ffffff;
  --clr-border: #e2e8f0;
  --clr-text: #1e293b;
  --clr-muted: #64748b;
}
*, *::before, *::after { box-sizing: border-box; }
html { font-size: 15px; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  color: var(--clr-text);
  background: var(--clr-bg);
  margin: 0; padding: 0;
  line-height: 1.55;
}
.container { max-width: 960px; margin: 0 auto; padding: 2rem 1.5rem; }

/* Title block */
.title-block {
  background: var(--clr-primary);
  color: #fff;
  padding: 2.5rem 2rem 2rem;
  border-radius: 8px;
  margin-bottom: 2rem;
}
.title-block h1 { margin: 0 0 1rem; font-size: 1.8rem; font-weight: 700; line-height: 1.25; }
.title-block .meta p { margin: 0.2rem 0; opacity: 0.85; font-size: 0.95rem; }

/* Section */
.section {
  background: var(--clr-white);
  border: 1px solid var(--clr-border);
  border-radius: 8px;
  padding: 1.5rem 1.75rem;
  margin-bottom: 1.5rem;
}
.section h2 {
  color: var(--clr-accent);
  font-size: 1.2rem;
  margin: 0 0 1rem;
  padding-bottom: 0.5rem;
  border-bottom: 2px solid var(--clr-accent);
}
.section h3 {
  font-size: 1rem;
  color: var(--clr-primary);
  margin: 1rem 0 0.5rem;
}

/* Tables */
table { width: 100%; border-collapse: collapse; margin: 0.75rem 0; font-size: 0.9rem; }
th, td { padding: 0.5rem 0.75rem; border-bottom: 1px solid var(--clr-border); text-align: left; }
th { background: var(--clr-primary); color: #fff; font-weight: 600; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.03em; }
tbody tr:nth-child(even) { background: var(--clr-bg); }
tbody tr:hover { background: #eef2ff; }
.num { text-align: right; font-variant-numeric: tabular-nums; }
caption { caption-side: bottom; text-align: left; font-size: 0.8rem; color: var(--clr-muted); padding-top: 0.5rem; }
.sig-bold { font-weight: 700; }

/* Metric cards */
.metric-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 0.75rem; margin-bottom: 1rem; }
.metric-card {
  background: var(--clr-bg);
  border: 1px solid var(--clr-border);
  border-radius: 6px;
  padding: 0.75rem 1rem;
  text-align: center;
}
.metric-card .label { font-size: 0.75rem; text-transform: uppercase; color: var(--clr-muted); letter-spacing: 0.05em; margin-bottom: 0.25rem; }
.metric-card .value { font-size: 1.25rem; font-weight: 700; color: var(--clr-primary); }

/* Collapsible details */
details {
  margin: 0.75rem 0;
  border: 1px solid var(--clr-border);
  border-radius: 6px;
  overflow: hidden;
}
details summary {
  cursor: pointer;
  font-weight: 600;
  font-size: 0.95rem;
  color: var(--clr-accent);
  padding: 0.65rem 1rem;
  background: var(--clr-bg);
  user-select: none;
}
details summary:hover { background: #eef2ff; }
details > .detail-body {
  padding: 0.75rem 1rem;
}

/* Chart wrapper */
.chart-wrapper {
  margin: 1rem 0;
  border: 1px solid var(--clr-border);
  border-radius: 6px;
  padding: 0.5rem;
  overflow: hidden;
}

/* Utility formula */
.formula-box {
  background: var(--clr-bg);
  border: 1px solid var(--clr-border);
  border-radius: 6px;
  padding: 1rem 1.25rem;
  font-size: 0.95rem;
  overflow-x: auto;
  line-height: 1.8;
  font-family: 'Georgia', 'Times New Roman', serif;
}
.formula-box sub { font-size: 0.75em; }

/* Footer */
.footer { text-align: center; font-size: 0.8rem; color: var(--clr-muted); padding: 1.5rem 0 2rem; }
.footer a { color: var(--clr-accent); text-decoration: none; }

/* Print */
@media print {
  body { background: #fff; }
  .container { max-width: 100%; padding: 0; }
  .title-block { border-radius: 0; break-after: avoid; }
  .section { box-shadow: none; border: 1px solid #ccc; break-inside: avoid; }
  .page-break { page-break-before: always; }
  table { font-size: 0.8rem; }
  th { background: #333 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  tbody tr:nth-child(even) { background: #f2f2f2 !important; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  details { break-inside: avoid; }
  details[open] > .detail-body { break-inside: auto; }
  .chart-wrapper { break-inside: avoid; }
}
"""

# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _build_title(run_label: str, data_label: str, model_type: str, now: str) -> str:
    mt_name = _MODEL_TYPE_NAMES.get(model_type, model_type)
    return f"""\
<div class="title-block">
  <h1>Discrete Choice Experiment<br>Analysis Report</h1>
  <div class="meta">
    <p><strong>Model:</strong> {_esc(run_label)}</p>
    <p><strong>Type:</strong> {_esc(mt_name)}</p>
    <p><strong>Dataset:</strong> {_esc(data_label)}</p>
    <p><strong>Generated:</strong> {_esc(now)}</p>
  </div>
</div>
"""


def _build_executive_summary(
    estimation: EstimationResult | LatentClassResult,
    model_type: str,
    full_spec: FullModelSpec | None,
) -> str:
    mt_name = _MODEL_TYPE_NAMES.get(model_type, model_type)
    n_params = getattr(estimation, "n_parameters", "?")
    n_obs = getattr(estimation, "n_observations", "?")
    n_ind = getattr(estimation, "n_individuals", "?")
    ll = getattr(estimation, "log_likelihood", None)
    success = getattr(estimation, "success", None)
    converge_word = "converged" if success else "did not converge"

    est_df = getattr(estimation, "estimates", None)
    n_sig = 0
    n_total = 0
    if est_df is not None and "p_value" in est_df.columns:
        pvals = est_df["p_value"].dropna()
        n_total = len(pvals)
        n_sig = int((pvals < 0.05).sum())

    lines: list[str] = []
    lines.append(
        f"A <strong>{_esc(mt_name)}</strong> model was estimated with "
        f"<strong>{n_params}</strong> parameters on <strong>{n_obs}</strong> "
        f"observations from <strong>{n_ind}</strong> individuals."
    )
    lines.append(
        f"The model <strong>{converge_word}</strong> with a log-likelihood of "
        f"<strong>{_format_number(ll, 3)}</strong>."
    )
    if n_total > 0:
        lines.append(
            f"{n_sig} of {n_total} parameters are statistically significant "
            f"at the 5% level."
        )

    # Model-specific summary additions
    fs = full_spec
    if model_type in ("mixed", "gmnl") and fs:
        n_draws = getattr(fs, "n_draws", None)
        corr = getattr(fs, "correlated", False)
        corr_str = "with correlated random parameters" if corr else "with independent random parameters"
        if n_draws:
            lines.append(
                f"Estimation used <strong>{n_draws}</strong> Halton draws {corr_str}."
            )
        if model_type == "gmnl":
            lines.append("The GMNL model extends Mixed Logit with a scale heterogeneity parameter.")

    if model_type == "latent_class" and isinstance(estimation, LatentClassResult):
        nc = getattr(estimation, "n_classes", "?")
        lines.append(f"The model estimated <strong>{nc}</strong> latent classes.")

    # BWS
    if fs and getattr(fs, "bws_worst_col", None):
        lines.append("Best-Worst Scaling (BWS) was enabled for this estimation.")

    body = "</p><p>".join(lines)
    return f"""\
<div class="section">
  <h2>Executive Summary</h2>
  <p>{body}</p>
</div>
"""


def _build_data_summary(
    data_label: str,
    data_shape: tuple[int, int],
    estimation: EstimationResult | LatentClassResult,
) -> str:
    n_ind = getattr(estimation, "n_individuals", "&mdash;")
    n_obs = getattr(estimation, "n_observations", "&mdash;")

    rows = [
        {"Metric": "Dataset", "Value": _esc(data_label)},
        {"Metric": "Rows", "Value": str(data_shape[0])},
        {"Metric": "Columns", "Value": str(data_shape[1])},
        {"Metric": "Individuals", "Value": str(n_ind)},
        {"Metric": "Observations (choice tasks)", "Value": str(n_obs)},
    ]
    df = pd.DataFrame(rows)
    tbl = _html_table(df, num_cols=set())
    return f"""\
<div class="section">
  <h2>Data Summary</h2>
  {tbl}
</div>
"""


def _build_model_spec(
    model_type: str,
    spec: ModelSpec | FullModelSpec,
    full_spec: FullModelSpec | None,
) -> str:
    mt_name = _MODEL_TYPE_NAMES.get(model_type, model_type)
    fs = full_spec or (spec if isinstance(spec, FullModelSpec) else None)

    # ── Variables table ──────────────────────────────────────────────
    var_rows = []
    variables = getattr(spec, "variables", [])
    for v in variables:
        var_rows.append({
            "Variable": _esc(v.name),
            "Column": _esc(v.column),
            "Distribution": _esc(v.distribution),
        })
    var_df = pd.DataFrame(var_rows) if var_rows else None

    # ── Specification details ────────────────────────────────────────
    detail_rows = [{"Setting": "Model type", "Value": mt_name}]

    if model_type in ("mixed", "gmnl"):
        n_draws = getattr(fs, "n_draws", None) or getattr(spec, "n_draws", None)
        if n_draws is not None:
            detail_rows.append({"Setting": "Number of Halton draws", "Value": str(n_draws)})
        corr = getattr(fs, "correlated", False)
        detail_rows.append({"Setting": "Correlation structure", "Value": "Correlated" if corr else "Independent"})

    if model_type == "latent_class":
        nc = getattr(fs, "n_classes", None) or getattr(spec, "n_classes", None)
        if nc is not None:
            detail_rows.append({"Setting": "Number of classes", "Value": str(nc)})
        method = getattr(fs, "lc_method", "em")
        detail_rows.append({"Setting": "Estimation method", "Value": method.upper()})
        mc = getattr(fs, "membership_cols", None) or getattr(spec, "membership_cols", None)
        if mc:
            detail_rows.append({"Setting": "Membership covariates", "Value": ", ".join(mc)})

    # Estimation settings
    maxiter = getattr(fs, "maxiter", None)
    if maxiter is not None:
        detail_rows.append({"Setting": "Max iterations", "Value": str(maxiter)})
    seed = getattr(fs, "seed", None)
    if seed is not None:
        detail_rows.append({"Setting": "Random seed", "Value": str(seed)})

    # BWS
    bws_col = getattr(fs, "bws_worst_col", None)
    if bws_col:
        detail_rows.append({"Setting": "BWS worst column", "Value": _esc(bws_col)})
        est_lam = getattr(fs, "estimate_lambda_w", True)
        detail_rows.append({"Setting": "Estimate lambda_w", "Value": "Yes" if est_lam else "No"})

    detail_df = pd.DataFrame(detail_rows)
    detail_tbl = _html_table(detail_df, num_cols=set())
    var_tbl = _html_table(var_df, num_cols=set()) if var_df is not None else ""

    # ── Utility function formula ─────────────────────────────────────
    formula_html = ""
    if variables:
        terms = []
        for v in variables:
            vn = _esc(v.name)
            vc = _esc(v.column)
            if v.distribution == "fixed":
                terms.append(f"&beta;<sub>{vn}</sub>&middot;{vc}")
            elif v.distribution == "normal":
                terms.append(f"(&mu;<sub>{vn}</sub> + &sigma;<sub>{vn}</sub>&middot;&eta;)&middot;{vc}")
            elif v.distribution == "lognormal":
                terms.append(f"exp(&mu;<sub>{vn}</sub> + &sigma;<sub>{vn}</sub>&middot;&eta;)&middot;{vc}")

        formula_str = " + ".join(terms)
        formula_html = f"""
  <h3>Utility Function</h3>
  <div class="formula-box">
    V<sub>nj</sub> = {formula_str}
  </div>
"""

    # ── Dummy coding details ─────────────────────────────────────────
    dummy_html = ""
    if fs and hasattr(fs, "dummy_codings") and fs.dummy_codings:
        dummy_rows = []
        for dc in fs.dummy_codings:
            dummy_rows.append({
                "Column": _esc(dc.column),
                "Reference Level": _esc(str(dc.ref_level)),
                "Coding": "Dummy (0/1)",
            })
        dummy_df = pd.DataFrame(dummy_rows)
        dummy_tbl = _html_table(dummy_df, num_cols=set())
        dummy_html = f"""
  <details>
    <summary>Dummy Coding Details</summary>
    <div class="detail-body">
      <p style="font-size:0.85rem;color:var(--clr-muted)">
        Categorical variables are dummy-coded with the reference level omitted as the baseline.
        Coefficients represent the effect relative to the reference level.
      </p>
      {dummy_tbl}
    </div>
  </details>
"""

    # ── Interaction terms ────────────────────────────────────────────
    interaction_html = ""
    if fs and hasattr(fs, "interactions") and fs.interactions:
        int_rows = []
        for it in fs.interactions:
            int_rows.append({
                "Interaction Term": _esc(it.name),
                "Components": " &times; ".join(_esc(c) for c in it.columns),
            })
        int_df = pd.DataFrame(int_rows)
        int_tbl = _html_table(int_df, num_cols=set())
        interaction_html = f"""
  <details>
    <summary>Interaction Terms ({len(fs.interactions)})</summary>
    <div class="detail-body">
      <p style="font-size:0.85rem;color:var(--clr-muted)">
        Interaction terms are the product of two or more variables,
        capturing how the effect of one attribute depends on another.
      </p>
      {int_tbl}
    </div>
  </details>
"""

    # ── Correlation groups ───────────────────────────────────────────
    corr_html = ""
    if fs and getattr(fs, "correlated", False) and model_type in ("mixed", "gmnl"):
        cg = getattr(fs, "correlation_groups", None)
        if cg:
            group_rows = []
            for gi, group in enumerate(cg):
                var_names_in_group = []
                for idx in group:
                    if idx < len(variables):
                        var_names_in_group.append(_esc(variables[idx].name))
                    else:
                        var_names_in_group.append(f"var_{idx}")
                group_rows.append({
                    "Group": f"Group {gi + 1}",
                    "Variable Indices": ", ".join(str(i) for i in group),
                    "Variables": ", ".join(var_names_in_group),
                })
            group_df = pd.DataFrame(group_rows)
            group_tbl = _html_table(group_df, num_cols=set())
            corr_html = f"""
  <details>
    <summary>Selective Correlation Groups ({len(cg)} groups)</summary>
    <div class="detail-body">
      <p style="font-size:0.85rem;color:var(--clr-muted)">
        Parameters within each group are allowed to be correlated.
        Parameters in different groups are assumed independent.
      </p>
      {group_tbl}
    </div>
  </details>
"""
        else:
            corr_html = """
  <p style="font-size:0.9rem;margin-top:0.5rem">
    <strong>Correlation:</strong> Full correlation among all random parameters.
  </p>
"""

    return f"""\
<div class="section">
  <h2>Model Specification</h2>
  {detail_tbl}
  {formula_html}
  {"<h3>Variables</h3>" + var_tbl if var_tbl else ""}
  {dummy_html}
  {interaction_html}
  {corr_html}
</div>
"""


def _build_model_fit(
    estimation: EstimationResult | LatentClassResult,
) -> str:
    ll = getattr(estimation, "log_likelihood", None)
    aic = getattr(estimation, "aic", None)
    bic = getattr(estimation, "bic", None)
    n_params = getattr(estimation, "n_parameters", None)
    n_obs = getattr(estimation, "n_observations", None)
    n_ind = getattr(estimation, "n_individuals", None)
    runtime = getattr(estimation, "runtime_seconds", None)
    success = getattr(estimation, "success", None)
    iters = getattr(estimation, "optimizer_iterations", None)

    # Metric cards
    cards = []

    def _card(label: str, value: str) -> str:
        return (
            f'<div class="metric-card">'
            f'<div class="label">{label}</div>'
            f'<div class="value">{value}</div></div>'
        )

    cards.append(_card("Log-Likelihood", _format_number(ll, 3)))
    cards.append(_card("AIC", _format_number(aic, 3)))
    cards.append(_card("BIC", _format_number(bic, 3)))
    cards.append(_card("Parameters", str(n_params) if n_params is not None else "&mdash;"))
    cards.append(_card("Observations", str(n_obs) if n_obs is not None else "&mdash;"))
    cards.append(_card("Individuals", str(n_ind) if n_ind is not None else "&mdash;"))
    cards.append(_card("Convergence", "Yes" if success else "No"))
    if runtime is not None:
        cards.append(_card("Runtime", f"{runtime:.1f}s"))

    card_html = "\n".join(cards)

    # Detailed table in collapsible section
    detail_rows = [
        {"Metric": "Log-Likelihood", "Value": _format_number(ll, 6)},
        {"Metric": "AIC", "Value": _format_number(aic, 4)},
        {"Metric": "BIC", "Value": _format_number(bic, 4)},
        {"Metric": "Number of parameters", "Value": str(n_params) if n_params is not None else "&mdash;"},
        {"Metric": "Observations", "Value": str(n_obs) if n_obs is not None else "&mdash;"},
        {"Metric": "Individuals", "Value": str(n_ind) if n_ind is not None else "&mdash;"},
        {"Metric": "Optimizer iterations", "Value": str(iters) if iters is not None else "&mdash;"},
        {"Metric": "Converged", "Value": "Yes" if success else "No"},
    ]
    if runtime is not None:
        detail_rows.append({"Metric": "Runtime (seconds)", "Value": f"{runtime:.2f}"})
    msg = getattr(estimation, "message", "")
    if msg:
        detail_rows.append({"Metric": "Optimizer message", "Value": _esc(msg)})

    detail_df = pd.DataFrame(detail_rows)
    detail_tbl = _html_table(detail_df, num_cols=set())

    return f"""\
<div class="section page-break">
  <h2>Model Fit</h2>
  <div class="metric-grid">
    {card_html}
  </div>
  <details>
    <summary>Detailed Fit Statistics</summary>
    <div class="detail-body">
      {detail_tbl}
    </div>
  </details>
</div>
"""


def _build_estimates(
    estimation: EstimationResult | LatentClassResult,
) -> str:
    est_df = getattr(estimation, "estimates", None)
    if est_df is None or est_df.empty:
        return ""

    display_df = est_df.copy()
    if "theta_index" in display_df.columns:
        display_df = display_df.drop(columns=["theta_index"])

    cols = list(display_df.columns)
    has_p = "p_value" in cols

    # ── Estimates table with significance stars ──────────────────────
    parts: list[str] = ['<table class="estimates">']
    parts.append("<thead><tr>")
    for c in cols:
        align = ' class="num"' if c not in ("parameter", "distribution") else ""
        nice = {
            "parameter": "Parameter",
            "distribution": "Distribution",
            "estimate": "Estimate",
            "std_error": "Std. Error",
            "z_stat": "z-stat",
            "p_value": "p-value",
            "ci_lower": "CI Lower",
            "ci_upper": "CI Upper",
        }.get(c, c)
        parts.append(f"<th{align}>{_esc(nice)}</th>")
    if has_p:
        parts.append('<th class="num">Sig.</th>')
    parts.append("</tr></thead>")

    parts.append("<tbody>")
    for _, row in display_df.iterrows():
        p_val = row.get("p_value", None)
        sig = _significance(p_val)
        is_sig = sig in ("***", "**", "*")
        parts.append("<tr>")
        for c in cols:
            val = row[c]
            if c in ("parameter", "distribution"):
                cls = ' class="sig-bold"' if (is_sig and c == "parameter") else ""
                parts.append(f"<td{cls}>{_esc(val)}</td>")
            elif c == "p_value":
                parts.append(f'<td class="num">{_format_number(val, 4)}</td>')
            else:
                bold = ' class="num sig-bold"' if (is_sig and c == "estimate") else ' class="num"'
                parts.append(f"<td{bold}>{_format_number(val, 4)}</td>")
        if has_p:
            parts.append(f'<td class="num">{sig}</td>')
        parts.append("</tr>")
    parts.append("</tbody></table>")

    caption = (
        '<p style="font-size:0.8rem;color:var(--clr-muted);margin-top:0.25rem">'
        "Significance codes: *** p&lt;0.001, ** p&lt;0.01, * p&lt;0.05, . p&lt;0.1</p>"
    )

    table_html = "\n".join(parts)

    # ── Coefficient chart ────────────────────────────────────────────
    chart_html = ""
    if _HAS_PLOTLY:
        fig = go.Figure()
        has_se = "std_error" in display_df.columns and display_df["std_error"].notna().any()
        if has_se and "ci_lower" in display_df.columns and "ci_upper" in display_df.columns:
            p_vals = display_df.get("p_value", pd.Series([1.0] * len(display_df)))
            fig.add_trace(go.Bar(
                y=display_df["parameter"],
                x=display_df["estimate"],
                orientation="h",
                error_x=dict(
                    type="data",
                    symmetric=False,
                    array=(display_df["ci_upper"] - display_df["estimate"]).tolist(),
                    arrayminus=(display_df["estimate"] - display_df["ci_lower"]).tolist(),
                ),
                marker_color=[
                    "#2563eb" if (pd.notna(p) and float(p) < 0.05) else "#93c5fd"
                    for p in p_vals
                ],
                hovertemplate="<b>%{y}</b><br>Estimate: %{x:.4f}<extra></extra>",
            ))
        else:
            fig.add_trace(go.Bar(
                y=display_df["parameter"],
                x=display_df["estimate"],
                orientation="h",
                marker_color="#2563eb",
                hovertemplate="<b>%{y}</b><br>Estimate: %{x:.4f}<extra></extra>",
            ))

        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Parameter Estimates with 95% Confidence Intervals",
            xaxis_title="Estimate",
            yaxis_title="",
            showlegend=False,
        )
        chart_height = max(350, len(display_df) * 50)
        chart_div = _plotly_div(fig, height=chart_height)
        chart_html = f'<div class="chart-wrapper">{chart_div}</div>'

    return f"""\
<div class="section page-break">
  <h2>Parameter Estimates</h2>
  {table_html}
  {caption}
  {chart_html}
</div>
"""


def _build_wtp(wtp_df: pd.DataFrame | None) -> str:
    if wtp_df is None or wtp_df.empty:
        return ""

    display = wtp_df.copy()
    # Keep relevant columns
    keep = [c for c in (
        "attribute", "wtp_estimate", "wtp_std_error", "wtp_ci_lower", "wtp_ci_upper",
        "wtp", "wtp_se", "wtp_ci_lower", "wtp_ci_upper",
        "parameter", "estimate", "std_error", "ci_lower", "ci_upper",
    ) if c in display.columns]
    if keep:
        display = display[keep]

    rename = {
        "attribute": "Attribute",
        "parameter": "Parameter",
        "wtp_estimate": "WTP Estimate",
        "wtp_std_error": "Std. Error",
        "wtp_ci_lower": "CI Lower",
        "wtp_ci_upper": "CI Upper",
        "wtp": "WTP Estimate",
        "wtp_se": "Std. Error",
        "estimate": "WTP Estimate",
        "std_error": "Std. Error",
        "ci_lower": "CI Lower",
        "ci_upper": "CI Upper",
    }
    display = display.rename(columns={k: v for k, v in rename.items() if k in display.columns})

    num_cols = {c for c in display.columns if c not in ("Attribute", "Parameter")}
    tbl = _html_table(display, num_cols=num_cols)

    # ── WTP bar chart ────────────────────────────────────────────────
    chart_html = ""
    if _HAS_PLOTLY:
        attr_col = "attribute" if "attribute" in wtp_df.columns else "parameter"
        wtp_val_col = None
        for c in ("wtp_estimate", "wtp", "estimate"):
            if c in wtp_df.columns:
                wtp_val_col = c
                break

        if wtp_val_col and attr_col in wtp_df.columns:
            fig = go.Figure()
            wtp_has_ci = "wtp_ci_lower" in wtp_df.columns and "wtp_ci_upper" in wtp_df.columns

            if wtp_has_ci:
                fig.add_trace(go.Bar(
                    y=wtp_df[attr_col],
                    x=wtp_df[wtp_val_col],
                    orientation="h",
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=(wtp_df["wtp_ci_upper"] - wtp_df[wtp_val_col]).tolist(),
                        arrayminus=(wtp_df[wtp_val_col] - wtp_df["wtp_ci_lower"]).tolist(),
                    ),
                    marker_color="#059669",
                    hovertemplate="<b>%{y}</b><br>WTP: %{x:.4f}<extra></extra>",
                ))
            else:
                fig.add_trace(go.Bar(
                    y=wtp_df[attr_col],
                    x=wtp_df[wtp_val_col],
                    orientation="h",
                    marker_color="#059669",
                    hovertemplate="<b>%{y}</b><br>WTP: %{x:.4f}<extra></extra>",
                ))

            fig.add_vline(x=0, line_dash="dash", line_color="gray")
            fig.update_layout(
                title="Willingness to Pay",
                xaxis_title="WTP Estimate",
                yaxis_title="",
                showlegend=False,
            )
            chart_height = max(300, len(wtp_df) * 60)
            chart_div = _plotly_div(fig, height=chart_height)
            chart_html = f'<div class="chart-wrapper">{chart_div}</div>'

    return f"""\
<div class="section page-break">
  <h2>Willingness to Pay</h2>
  {tbl}
  <p style="font-size:0.85rem;color:var(--clr-muted);margin-top:0.5rem">
    WTP estimates are computed as the ratio of each attribute coefficient to the
    cost/price coefficient. Confidence intervals are derived via the delta method.
  </p>
  {chart_html}
</div>
"""


def _build_cov_cor(
    estimation: EstimationResult | LatentClassResult,
) -> str:
    cov = getattr(estimation, "covariance_matrix", None)
    cor = getattr(estimation, "correlation_matrix", None)
    names = getattr(estimation, "random_param_names", None)
    if cov is None and cor is None:
        return ""

    parts: list[str] = ['<div class="section page-break">', "<h2>Covariance &amp; Correlation</h2>"]

    if cov is not None and names:
        parts.append("<h3>Covariance Matrix</h3>")
        parts.append(_matrix_table(cov, names))

    if cor is not None and names:
        parts.append("<h3>Correlation Matrix</h3>")
        parts.append(_matrix_table(cor, names))

    # SE tables in collapsible sections
    cov_se = getattr(estimation, "covariance_se", None)
    cor_se = getattr(estimation, "correlation_se", None)

    if cov_se is not None and names:
        parts.append(
            '<details><summary>Covariance Standard Errors</summary>'
            f'<div class="detail-body">{_matrix_table(cov_se, names)}</div></details>'
        )
    if cor_se is not None and names:
        parts.append(
            '<details><summary>Correlation Standard Errors</summary>'
            f'<div class="detail-body">{_matrix_table(cor_se, names)}</div></details>'
        )

    # Correlation significance test
    cor_test = getattr(estimation, "correlation_test", None)
    if cor_test is not None and not cor_test.empty:
        parts.append(
            '<details><summary>Correlation Significance Tests (H&#8320;: &rho; = 0)</summary>'
            f'<div class="detail-body">{_html_table(cor_test, fmt=4)}</div></details>'
        )

    parts.append("</div>")
    return "\n".join(parts)


def _build_distributions(
    estimation: EstimationResult | LatentClassResult,
) -> str:
    """Render estimated distribution curves for random parameters (MXL/GMNL)."""
    if not _HAS_PLOTLY or not _HAS_SCIPY:
        return ""

    est_df = getattr(estimation, "estimates", None)
    if est_df is None:
        return ""

    # Find random parameters (those with mu_ and sd_ pairs)
    mu_rows = est_df[est_df["parameter"].str.startswith("mu_")]
    if mu_rows.empty:
        return ""

    random_params = []
    for _, mu_row in mu_rows.iterrows():
        vname = mu_row["parameter"].split("_", 1)[1]
        sd_row = est_df[est_df["parameter"] == f"sd_{vname}"]
        if sd_row.empty:
            continue
        mu_val = float(mu_row["estimate"])
        sd_val = float(sd_row.iloc[0]["estimate"])
        dist_type = str(mu_row.get("distribution", "normal"))
        random_params.append({
            "name": vname,
            "mu": mu_val,
            "sd": abs(sd_val),
            "distribution": dist_type,
        })

    if not random_params:
        return ""

    # Build individual density plots
    chart_parts: list[str] = []
    for rp in random_params:
        mu, sd, dist = rp["mu"], rp["sd"], rp["distribution"]
        vname = rp["name"]

        fig = go.Figure()

        if dist == "lognormal" and sd > 0:
            x_lo = max(1e-6, lognorm.ppf(0.005, s=sd, scale=np.exp(mu)))
            x_hi = lognorm.ppf(0.995, s=sd, scale=np.exp(mu))
            x_vals = np.linspace(x_lo, x_hi, 300)
            y_vals = lognorm.pdf(x_vals, s=sd, scale=np.exp(mu))
            dist_label = "Lognormal"
        elif sd > 0:
            x_lo = mu - 4 * sd
            x_hi = mu + 4 * sd
            x_vals = np.linspace(x_lo, x_hi, 300)
            y_vals = sp_norm.pdf(x_vals, loc=mu, scale=sd)
            dist_label = "Normal"
        else:
            # Zero sd: just a spike at mu
            x_vals = np.array([mu])
            y_vals = np.array([1.0])
            dist_label = "Fixed" if dist == "normal" else dist.title()

        fig.add_trace(go.Scatter(
            x=x_vals.tolist(),
            y=y_vals.tolist(),
            mode="lines",
            name=f"{dist_label}(&mu;={mu:.3f}, &sigma;={sd:.3f})",
            line=dict(color="#2563eb", width=2),
            fill="tozeroy",
            fillcolor="rgba(37,99,235,0.12)",
        ))

        # Add mean line
        fig.add_vline(x=mu, line_dash="dash", line_color="#dc2626",
                       annotation_text=f"&mu;={mu:.3f}")

        fig.update_layout(
            title=f"{_esc(vname)} &mdash; {dist_label}(&mu;={mu:.3f}, &sigma;={sd:.3f})",
            xaxis_title="Coefficient value",
            yaxis_title="Density",
            showlegend=False,
        )
        chart_parts.append(f'<div class="chart-wrapper">{_plotly_div(fig, height=320)}</div>')

    charts_grid = "\n".join(chart_parts)
    return f"""\
<div class="section page-break">
  <h2>Random Parameter Distributions</h2>
  <p style="font-size:0.9rem;color:var(--clr-muted);margin-bottom:1rem">
    For each random parameter in the Mixed Logit model, the estimated distribution
    is shown as a density curve. The dashed line indicates the mean (&mu;).
  </p>
  {charts_grid}
</div>
"""


def _build_predictions(predictions_df: pd.DataFrame | None) -> str:
    """Render predicted vs actual choice shares section."""
    if predictions_df is None or predictions_df.empty:
        return ""

    tbl = _html_table(predictions_df, num_cols={"Actual Share", "Predicted Share"}, fmt=4)

    chart_html = ""
    if _HAS_PLOTLY and "Alternative" in predictions_df.columns:
        share_melt = predictions_df.melt(
            id_vars="Alternative",
            value_vars=["Actual Share", "Predicted Share"],
            var_name="Type",
            value_name="Share",
        )
        fig = go.Figure()
        for stype, color in [("Actual Share", "#f97316"), ("Predicted Share", "#3b82f6")]:
            subset = share_melt[share_melt["Type"] == stype]
            fig.add_trace(go.Bar(
                x=subset["Alternative"],
                y=subset["Share"],
                name=stype,
                marker_color=color,
            ))
        fig.update_layout(
            title="Predicted vs Actual Choice Shares",
            barmode="group",
            xaxis_title="Alternative",
            yaxis_title="Share",
            yaxis_tickformat=".1%",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        chart_div = _plotly_div(fig, height=400)
        chart_html = f'<div class="chart-wrapper">{chart_div}</div>'

    return f"""\
<div class="section">
  <h2>Predicted Choice Shares</h2>
  {tbl}
  {chart_html}
</div>
"""


def _build_latent_class(
    estimation: LatentClassResult,
) -> str:
    parts: list[str] = ['<div class="section page-break">', "<h2>Latent Class Results</h2>"]

    # ── Class membership probabilities ───────────────────────────────
    class_probs = getattr(estimation, "class_probabilities", None)
    if class_probs is not None:
        parts.append("<h3>Class Membership Probabilities</h3>")
        prob_rows = []
        for i, p in enumerate(class_probs):
            prob_rows.append({"Class": f"Class {i + 1}", "Probability": p, "Share (%)": p * 100})
        prob_df = pd.DataFrame(prob_rows)
        parts.append(_html_table(prob_df, num_cols={"Probability", "Share (%)"}, fmt=4))

        # Pie chart
        if _HAS_PLOTLY:
            fig = go.Figure(data=[go.Pie(
                labels=prob_df["Class"],
                values=prob_df["Probability"],
                hole=0.35,
                marker=dict(colors=[
                    "#3b82f6", "#f97316", "#10b981", "#8b5cf6",
                    "#ef4444", "#06b6d4", "#84cc16", "#f59e0b",
                ]),
                textinfo="label+percent",
                textposition="outside",
            )])
            fig.update_layout(title="Class Membership Shares", showlegend=False)
            parts.append(f'<div class="chart-wrapper">{_plotly_div(fig, height=380)}</div>')

    # ── Class-specific estimates ─────────────────────────────────────
    class_est = getattr(estimation, "class_estimates", None)
    if class_est is not None and not class_est.empty:
        parts.append("<h3>Class-Specific Parameter Estimates</h3>")

        # Pivot table
        try:
            pivot = class_est.pivot(index="parameter", columns="class_id", values="estimate")
            pivot.columns = [f"Class {c}" for c in pivot.columns]
            pivot_html = _html_table(
                pivot.reset_index().rename(columns={"parameter": "Parameter"}),
                num_cols={c for c in pivot.columns},
                fmt=4,
            )
            parts.append(pivot_html)
        except Exception:
            display = class_est.copy()
            if "theta_index" in display.columns:
                display = display.drop(columns=["theta_index"])
            num = {c for c in display.columns if display[c].dtype.kind in "fiuc"}
            parts.append(_html_table(display, num_cols=num, fmt=4))

        # Class comparison bar chart
        if _HAS_PLOTLY:
            fig = go.Figure()
            class_ids = sorted(class_est["class_id"].unique())
            colors = ["#3b82f6", "#f97316", "#10b981", "#8b5cf6", "#ef4444", "#06b6d4"]
            for ci, cid in enumerate(class_ids):
                subset = class_est[class_est["class_id"] == cid]
                fig.add_trace(go.Bar(
                    x=subset["parameter"],
                    y=subset["estimate"],
                    name=f"Class {cid}",
                    marker_color=colors[ci % len(colors)],
                ))
            fig.update_layout(
                title="Coefficient Estimates by Class",
                barmode="group",
                xaxis_title="Parameter",
                yaxis_title="Estimate",
                legend_title="Class",
            )
            parts.append(f'<div class="chart-wrapper">{_plotly_div(fig, height=450)}</div>')

    # ── Membership model estimates ───────────────────────────────────
    mem_est = getattr(estimation, "membership_estimates", None)
    if mem_est is not None and not mem_est.empty:
        display = mem_est.copy()
        if "theta_index" in display.columns:
            display = display.drop(columns=["theta_index"])
        num = {c for c in display.columns if display[c].dtype.kind in "fiuc"}
        mem_tbl = _html_table(display, num_cols=num, fmt=4)
        parts.append(
            '<details><summary>Membership Model Estimates</summary>'
            f'<div class="detail-body">{mem_tbl}</div></details>'
        )

    # ── Posterior membership ─────────────────────────────────────────
    posterior = getattr(estimation, "posterior_probs", None)
    if posterior is not None and _HAS_PLOTLY:
        max_probs = posterior.max(axis=1).tolist()
        assigned = posterior.idxmax(axis=1)
        class_counts = assigned.value_counts().sort_index()

        parts.append("<h3>Posterior Class Membership</h3>")
        parts.append(
            '<p style="font-size:0.9rem;color:var(--clr-muted)">'
            "Each individual is assigned a posterior probability of belonging to each class. "
            "Higher max probabilities indicate clearer class separation.</p>"
        )

        # Posterior histogram
        fig_hist = go.Figure(data=[go.Histogram(
            x=max_probs,
            nbinsx=30,
            marker_color="#3b82f6",
            name="Max posterior prob",
        )])
        fig_hist.update_layout(
            title="Distribution of Max Posterior Probability",
            xaxis_title="Max Posterior Probability",
            yaxis_title="Count",
        )
        parts.append(f'<div class="chart-wrapper">{_plotly_div(fig_hist, height=350)}</div>')

        # Assigned class counts
        count_df = pd.DataFrame({
            "Class": class_counts.index.tolist(),
            "Count": class_counts.values.tolist(),
        })
        fig_count = go.Figure(data=[go.Bar(
            x=count_df["Class"],
            y=count_df["Count"],
            marker_color="#10b981",
        )])
        fig_count.update_layout(
            title="Assigned Class Counts",
            xaxis_title="Class",
            yaxis_title="Count",
            showlegend=False,
        )
        parts.append(f'<div class="chart-wrapper">{_plotly_div(fig_count, height=350)}</div>')

    # ── EM diagnostics ───────────────────────────────────────────────
    em_iters = getattr(estimation, "em_iterations", 0)
    if em_iters > 0:
        em_conv = getattr(estimation, "em_converged", False)
        diag_rows = [
            {"Metric": "EM iterations", "Value": str(em_iters)},
            {"Metric": "EM converged", "Value": "Yes" if em_conv else "No"},
        ]
        n_starts = getattr(estimation, "n_starts_attempted", 0)
        n_ok = getattr(estimation, "n_starts_succeeded", 0)
        if n_starts > 0:
            diag_rows.append({"Metric": "Random starts attempted", "Value": str(n_starts)})
            diag_rows.append({"Metric": "Random starts succeeded", "Value": str(n_ok)})
        all_lls = getattr(estimation, "all_start_lls", [])
        valid_lls = [ll for ll in all_lls if not (isinstance(ll, float) and ll != ll)]
        if len(valid_lls) > 1:
            diag_rows.append({"Metric": "Best LL", "Value": f"{max(valid_lls):.3f}"})
            diag_rows.append({"Metric": "Worst LL", "Value": f"{min(valid_lls):.3f}"})
            diag_rows.append({"Metric": "LL spread", "Value": f"{max(valid_lls) - min(valid_lls):.3f}"})

        diag_df = pd.DataFrame(diag_rows)
        diag_tbl = _html_table(diag_df, num_cols=set())

        # LL by start chart
        chart_html = ""
        if _HAS_PLOTLY and len(all_lls) > 1:
            ll_data = pd.DataFrame({
                "Start": list(range(1, len(all_lls) + 1)),
                "Log-Likelihood": all_lls,
            }).dropna(subset=["Log-Likelihood"])

            fig = go.Figure(data=[go.Bar(
                x=ll_data["Start"],
                y=ll_data["Log-Likelihood"],
                marker_color="#6366f1",
            )])
            best_ll = getattr(estimation, "log_likelihood", None)
            if best_ll is not None:
                fig.add_hline(y=best_ll, line_dash="dash", line_color="green",
                              annotation_text="Best")
            fig.update_layout(
                title="Log-Likelihood by Random Start",
                xaxis_title="Start",
                yaxis_title="Log-Likelihood",
            )
            chart_html = f'<div class="chart-wrapper">{_plotly_div(fig, height=350)}</div>'

        parts.append(
            '<details><summary>EM Algorithm &amp; Convergence Diagnostics</summary>'
            f'<div class="detail-body">{diag_tbl}{chart_html}</div></details>'
        )

    parts.append("</div>")
    return "\n".join(parts)


def _build_bootstrap(bootstrap: Any) -> str:
    """Render bootstrap results section."""
    if bootstrap is None:
        return ""

    param_names = getattr(bootstrap, "param_names", [])
    original = getattr(bootstrap, "original_estimates", {})
    boot_se = getattr(bootstrap, "bootstrap_se", {})
    pci = getattr(bootstrap, "percentile_ci", {})
    n_rep = getattr(bootstrap, "n_replications", 0)
    n_ok = getattr(bootstrap, "n_successful", 0)
    est_matrix = getattr(bootstrap, "estimates_matrix", None)

    rows = []
    for name in param_names:
        orig = original.get(name, float("nan"))
        bse = boot_se.get(name, float("nan"))
        ci = pci.get(name, (float("nan"), float("nan")))
        boot_mean = float("nan")
        if est_matrix is not None and len(param_names) > 0:
            idx = param_names.index(name)
            if idx < est_matrix.shape[1]:
                boot_mean = float(np.nanmean(est_matrix[:, idx]))
        rows.append({
            "Parameter": name,
            "Original": orig,
            "Boot Mean": boot_mean,
            "Boot SE": bse,
            "CI Lower (2.5%)": ci[0],
            "CI Upper (97.5%)": ci[1],
        })

    df = pd.DataFrame(rows)
    num_cols = {c for c in df.columns if c != "Parameter"}
    tbl = _html_table(df, num_cols=num_cols, fmt=4)

    # Comparison chart: original vs bootstrap mean
    chart_html = ""
    if _HAS_PLOTLY and len(rows) > 0:
        fig = go.Figure()
        boot_df = pd.DataFrame(rows)
        fig.add_trace(go.Bar(
            y=boot_df["Parameter"],
            x=boot_df["Original"],
            name="Original",
            orientation="h",
            marker_color="#3b82f6",
        ))
        fig.add_trace(go.Bar(
            y=boot_df["Parameter"],
            x=boot_df["Boot Mean"],
            name="Bootstrap Mean",
            orientation="h",
            marker_color="#f97316",
            error_x=dict(
                type="data",
                array=boot_df["Boot SE"].tolist(),
                visible=True,
            ),
        ))
        fig.add_vline(x=0, line_dash="dash", line_color="gray")
        fig.update_layout(
            title="Original vs Bootstrap Estimates",
            barmode="group",
            xaxis_title="Estimate",
            yaxis_title="",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
        )
        chart_height = max(350, len(rows) * 45)
        chart_div = _plotly_div(fig, height=chart_height)
        chart_html = f"""
  <details>
    <summary>Bootstrap Comparison Chart</summary>
    <div class="detail-body"><div class="chart-wrapper">{chart_div}</div></div>
  </details>"""

    # Bootstrap distribution histograms
    hist_html = ""
    if _HAS_PLOTLY and est_matrix is not None and est_matrix.shape[1] > 0:
        hist_parts = []
        for idx, name in enumerate(param_names):
            if idx >= est_matrix.shape[1]:
                break
            col_data = est_matrix[:, idx]
            valid = col_data[~np.isnan(col_data)]
            if len(valid) < 5:
                continue
            fig = go.Figure(data=[go.Histogram(
                x=valid.tolist(),
                nbinsx=40,
                marker_color="#6366f1",
                opacity=0.8,
            )])
            orig_val = original.get(name, None)
            if orig_val is not None and not math.isnan(orig_val):
                fig.add_vline(x=orig_val, line_dash="dash", line_color="#dc2626",
                              annotation_text=f"Original: {orig_val:.4f}")
            fig.update_layout(
                title=f"{name}",
                xaxis_title="Estimate",
                yaxis_title="Count",
                showlegend=False,
            )
            hist_parts.append(f'<div class="chart-wrapper">{_plotly_div(fig, height=300)}</div>')

        if hist_parts:
            all_hists = "\n".join(hist_parts)
            hist_html = f"""
  <details>
    <summary>Bootstrap Distribution Histograms ({len(hist_parts)} parameters)</summary>
    <div class="detail-body">{all_hists}</div>
  </details>"""

    return f"""\
<div class="section page-break">
  <h2>Bootstrap Results</h2>
  <p style="margin-bottom:0.75rem">
    {n_ok} of {n_rep} bootstrap replications completed successfully.
  </p>
  {tbl}
  <p style="font-size:0.85rem;color:var(--clr-muted);margin-top:0.5rem">
    Bootstrap standard errors and percentile confidence intervals are based on
    resampling individuals with replacement. They provide a non-parametric
    alternative to the asymptotic standard errors from the information matrix.
  </p>
  {chart_html}
  {hist_html}
</div>
"""


def _build_footer() -> str:
    return """\
<div class="footer">
  <p>Generated by <strong>&#20998;&#26512;&#20384; Prefero</strong> &mdash; GPU-accelerated DCE Analysis</p>
  <p>Open this file in any web browser. Press Ctrl+P / Cmd+P to save as PDF.</p>
  <p style="font-size:0.75rem; margin-top:0.25rem">
    Interactive charts require an internet connection (Plotly CDN).
  </p>
</div>
"""


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_report_html(
    estimation: EstimationResult | LatentClassResult,
    model_type: str,
    run_label: str,
    data_label: str,
    data_shape: tuple[int, int],
    spec: ModelSpec | FullModelSpec,
    wtp_df: pd.DataFrame | None = None,
    bootstrap: Any = None,
    full_spec: FullModelSpec | None = None,
    predictions_df: pd.DataFrame | None = None,
) -> str:
    """Generate a complete, standalone HTML report string.

    Parameters
    ----------
    estimation : EstimationResult | LatentClassResult
        The fitted model results.
    model_type : str
        One of ``"conditional"``, ``"mixed"``, ``"gmnl"``, ``"latent_class"``.
    run_label : str
        Human-readable model name / label.
    data_label : str
        Human-readable dataset name.
    data_shape : tuple[int, int]
        ``(n_rows, n_cols)`` of the source DataFrame.
    spec : ModelSpec | FullModelSpec
        The model specification used for estimation.
    wtp_df : pd.DataFrame | None
        Optional willingness-to-pay table.
    bootstrap : BootstrapResult | None
        Optional bootstrap inference results.
    full_spec : FullModelSpec | None
        Optional full model specification (provides extra config details).
    predictions_df : pd.DataFrame | None
        Optional predicted vs actual choice shares DataFrame.
        Expected columns: ``Alternative``, ``Actual Share``, ``Predicted Share``.

    Returns
    -------
    str
        A self-contained HTML document with interactive charts.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    sections: list[str] = []
    sections.append(_build_title(run_label, data_label, model_type, now))
    sections.append(_build_executive_summary(estimation, model_type, full_spec))
    sections.append(_build_data_summary(data_label, data_shape, estimation))
    sections.append(_build_model_spec(model_type, spec, full_spec))
    sections.append(_build_model_fit(estimation))
    sections.append(_build_estimates(estimation))

    # Conditional sections
    sections.append(_build_wtp(wtp_df))
    sections.append(_build_cov_cor(estimation))
    sections.append(_build_distributions(estimation))
    sections.append(_build_predictions(predictions_df))

    if isinstance(estimation, LatentClassResult):
        sections.append(_build_latent_class(estimation))

    sections.append(_build_bootstrap(bootstrap))
    sections.append(_build_footer())

    body = "\n".join(s for s in sections if s)

    # Include plotly CDN if plotly is available
    plotly_script = ""
    if _HAS_PLOTLY:
        plotly_script = '<script src="https://cdn.plot.ly/plotly-2.35.0.min.js" charset="utf-8"></script>'

    return f"""\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DCE Analysis Report &mdash; {_esc(run_label)}</title>
{plotly_script}
<style>
{_CSS}
</style>
</head>
<body>
<div class="container">
{body}
</div>
</body>
</html>
"""
