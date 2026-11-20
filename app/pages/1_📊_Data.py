# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Prefero — Page 1: Data Loading and Exploration."""

from __future__ import annotations

import sys
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile

import pandas as pd
import streamlit as st
from openpyxl.utils.exceptions import InvalidFileException

# ── path setup so we can import the backend ────────────────────────
ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dce_analyzer.apollo import load_apollo_long  # noqa: E402
from dce_analyzer.data import chosen_row_mask  # noqa: E402
from dce_analyzer.format_converter import (  # noqa: E402
    _auto_attribute_groups,
    _guess_choice_col,
    _guess_id_col,
    apply_column_mapping,
    detect_format,
    infer_structure,
    load_excel_with_dictionary,
    wide_to_long,
)
from dce_analyzer.simulate import generate_simulated_dce  # noqa: E402

# app-level shared helpers (utils.py sits next to Home.py)
APP_DIR = Path(__file__).resolve().parents[1]
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from utils import init_session_state, sidebar_branding, slowbro_next_step  # noqa: E402

init_session_state()
sidebar_branding()

# ── Page header ────────────────────────────────────────────────────
st.header("Data")
st.caption("Upload, generate, or load a discrete-choice dataset, then explore it.")


def _clear_import() -> None:
    for key in ("_excel_result", "_excel_name", "_wide_raw", "_wide_name",
                "_auto_converted", "_auto_meta", "_auto_error", "_upload_error"):
        st.session_state.pop(key, None)


def _clear_dataset() -> None:
    st.session_state.update(df=None, data_label="", true_params={}, model_results=None,
                            model_history=[], bootstrap_results=None, lc_result=None,
                            lc_bic_comparison=None, lc_best_q=None)
    for key in ("inferred_columns", "interaction_terms", "bootstrap_run_id", "cv_results",
                "_generated_report", "_report_run_id", "_report_notes", "_data_warnings"):
        st.session_state.pop(key, None)


if any(st.session_state.get(key) is not None for key in (
    "df", "_excel_result", "_wide_raw", "_upload_error",
)):
    if st.sidebar.button("Reset Data", type="secondary", use_container_width=True):
        _clear_import()
        _clear_dataset()
        st.session_state.pop("_upload_identity", None)
        st.session_state["_uploader_generation"] = st.session_state.get("_uploader_generation", 0) + 1
        st.rerun()


# ── helpers ────────────────────────────────────────────────────────
def _store(df: pd.DataFrame, label: str, true_params: dict | None = None) -> None:
    _clear_dataset()
    _clear_import()
    st.session_state.df = df
    st.session_state.data_label = label
    st.session_state.true_params = true_params or {}


def _stage_dataframe(df: pd.DataFrame, file_name: str, label_suffix: str = "") -> None:
    if df.empty:
        raise ValueError("The uploaded data sheet has no rows.")
    if detect_format(df) == "long":
        _store(df, f"Uploaded: {file_name}{label_suffix}")
        return

    _clear_import()
    _clear_dataset()
    for key in ("wide_attr_select", "wide_id", "wide_choice"):
        st.session_state.pop(key, None)
    st.session_state["_wide_raw"] = df
    st.session_state["_wide_name"] = file_name
    columns = df.columns.tolist()
    groups = _auto_attribute_groups(columns)
    id_col = _guess_id_col(columns)
    choice_col = _guess_choice_col(columns)
    if not groups or not choice_col:
        return

    messages = []
    try:
        converted = wide_to_long(df, groups, id_col, choice_col, on_exclusion=messages.append)
    except ValueError as exc:
        st.session_state["_auto_error"] = " ".join([*messages, str(exc)])
        return
    st.session_state["_auto_converted"] = converted
    st.session_state["_auto_meta"] = {
        "id_col": id_col, "choice_col": choice_col, "groups": groups, "warnings": messages,
        "automatic": True,
    }


@st.cache_data
def _generate_template_excel() -> bytes:
    """Generate a downloadable Excel template with data and dictionary sheets."""
    import openpyxl
    from openpyxl.styles import Alignment, Font, PatternFill

    wb = openpyxl.Workbook()

    # ── Data sheet ──
    ws_data = wb.active
    ws_data.title = "data"

    headers = ["respondent_id", "task_id", "alternative", "choice",
               "price", "time", "comfort", "income"]
    example_rows = [
        [1, 1, 1, 1, 10, 30, 3, 50000],
        [1, 1, 2, 0, 15, 20, 5, 50000],
        [1, 1, 3, 0, 12, 25, 4, 50000],
        [1, 2, 1, 0, 8, 35, 2, 50000],
        [1, 2, 2, 1, 14, 15, 4, 50000],
        [1, 2, 3, 0, 11, 28, 3, 50000],
        [2, 1, 1, 0, 9, 32, 3, 65000],
        [2, 1, 2, 1, 16, 18, 5, 65000],
    ]

    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")

    for col_idx, header in enumerate(headers, 1):
        cell = ws_data.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for row_idx, row_data in enumerate(example_rows, 2):
        for col_idx, value in enumerate(row_data, 1):
            cell = ws_data.cell(row=row_idx, column=col_idx, value=value)
            cell.alignment = Alignment(horizontal="center")

    for col_idx in range(1, len(headers) + 1):
        ws_data.column_dimensions[openpyxl.utils.get_column_letter(col_idx)].width = 16

    # ── Dictionary sheet ──
    ws_dict = wb.create_sheet("dictionary")

    dict_headers = ["Column", "Description", "Type", "Required"]
    dict_rows = [
        ["respondent_id", "Unique identifier for each respondent", "integer", "Yes"],
        ["task_id", "Choice task number within each respondent", "integer", "Yes"],
        ["alternative", "Alternative number within each task", "integer", "Yes"],
        ["choice", "Binary indicator: 1 if chosen, 0 otherwise", "binary (0/1)", "Yes"],
        ["price", "Price attribute level", "numeric", "Yes (at least 1 attribute)"],
        ["time", "Travel time attribute level", "numeric", "Yes (at least 1 attribute)"],
        ["comfort", "Comfort rating attribute level", "numeric", "Yes (at least 1 attribute)"],
        ["income", "Respondent income (demographic)", "numeric", "Optional"],
        ["worst", "BWS worst choice indicator (0/1)", "binary (0/1)", "Optional"],
    ]

    for col_idx, header in enumerate(dict_headers, 1):
        cell = ws_dict.cell(row=1, column=col_idx, value=header)
        cell.font = header_font
        cell.fill = header_fill
        cell.alignment = Alignment(horizontal="center")

    for row_idx, row_data in enumerate(dict_rows, 2):
        for col_idx, value in enumerate(row_data, 1):
            ws_dict.cell(row=row_idx, column=col_idx, value=value)

    ws_dict.column_dimensions["A"].width = 18
    ws_dict.column_dimensions["B"].width = 45
    ws_dict.column_dimensions["C"].width = 18
    ws_dict.column_dimensions["D"].width = 25

    buf = BytesIO()
    wb.save(buf)
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════
# 1. DATA FORMAT GUIDE
# ══════════════════════════════════════════════════════════════════

with st.expander("Data Format Guide", expanded=True):

    # ── Flowchart ──
    st.markdown(
        """
        <div style="display:flex; align-items:center; justify-content:center; gap:0; margin:18px 0 22px 0; flex-wrap:wrap;">
            <div style="background:#4472C4; color:#fff; padding:8px 18px; border-radius:8px; font-size:0.95em; font-weight:600; white-space:nowrap;">Survey Design</div>
            <div style="font-size:1.4em; color:#888; padding:0 6px;">&rarr;</div>
            <div style="background:#5B9BD5; color:#fff; padding:8px 18px; border-radius:8px; font-size:0.95em; font-weight:600; white-space:nowrap;">Collect Responses</div>
            <div style="font-size:1.4em; color:#888; padding:0 6px;">&rarr;</div>
            <div style="background:#70AD47; color:#fff; padding:8px 18px; border-radius:8px; font-size:0.95em; font-weight:600; white-space:nowrap;">Clean in Excel / CSV</div>
            <div style="font-size:1.4em; color:#888; padding:0 6px;">&rarr;</div>
            <div style="background:#ED7D31; color:#fff; padding:8px 18px; border-radius:8px; font-size:0.95em; font-weight:600; white-space:nowrap;">Upload to Prefero</div>
            <div style="font-size:1.4em; color:#888; padding:0 6px;">&rarr;</div>
            <div style="background:#A855F7; color:#fff; padding:8px 18px; border-radius:8px; font-size:0.95em; font-weight:600; white-space:nowrap;">Analyze!</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── Required columns table ──
    st.markdown("#### Required Columns")
    st.markdown(
        """
| Column | Description | Example | Required? |
|--------|------------|---------|-----------|
| `respondent_id` | Unique identifier for each respondent | 1, 2, 3... | Yes |
| `task_id` | Choice task number within each respondent | 1, 2, 3... | Yes |
| `alternative` | Alternative number within each task | 1, 2, 3 | Yes |
| `choice` | Binary indicator: 1 if chosen, 0 otherwise | 0 or 1 | Yes |
| *[attribute columns]* | Attribute levels (numeric) | price, time, comfort... | Yes (at least 1) |
| *[demographic columns]* | Respondent-level covariates | income, age... | Optional |
| `worst` | BWS worst choice indicator | 0 or 1 | Optional |
"""
    )

    # ── Example long-format data ──
    st.markdown("#### Example: Long Format (ready to use)")
    example_long = pd.DataFrame({
        "respondent_id": [1, 1, 1, 1, 1, 1],
        "task_id": [1, 1, 1, 2, 2, 2],
        "alternative": [1, 2, 3, 1, 2, 3],
        "choice": [1, 0, 0, 0, 1, 0],
        "price": [10, 15, 12, 8, 14, 11],
        "time": [30, 20, 25, 35, 15, 28],
        "comfort": [3, 5, 4, 2, 4, 3],
        "income": [50000, 50000, 50000, 50000, 50000, 50000],
    })
    st.dataframe(example_long, use_container_width=True, hide_index=True)

    # ── Example wide-format data ──
    st.markdown("#### Example: Wide Format (we can convert it)")
    example_wide = pd.DataFrame({
        "respondent_id": [1, 1],
        "choice": [1, 2],
        "price_1": [10, 8],
        "price_2": [15, 14],
        "price_3": [12, 11],
        "time_1": [30, 35],
        "time_2": [20, 15],
        "time_3": [25, 28],
        "income": [50000, 50000],
    })
    st.dataframe(example_wide, use_container_width=True, hide_index=True)
    st.caption("If your data looks like this (wide format), upload it and we will help convert it.")

    # ── Data preparation checklist ──
    st.markdown("#### Data Preparation Checklist")
    st.markdown(
        """
- Each row = one alternative in one task for one respondent
- One and only one alternative chosen per task (`choice` = 1, rest = 0)
- No missing values in structural columns (`respondent_id`, `task_id`, `alternative`, `choice`)
- Attribute values should be numeric
- Demographics are constant within each respondent
- Column names should be clean (no spaces or special characters -- use underscores)
"""
    )

    # ── Excel tips ──
    st.markdown("#### Excel Tips")
    st.markdown(
        """
- Remove any header rows or merged cells before uploading
- Save as `.xlsx` or `.csv`
- Use one sheet for data, optionally a second sheet for a data dictionary
- Make sure column names are in Row 1
"""
    )

    # ── Download template ──
    st.markdown("---")
    template_bytes = _generate_template_excel()
    st.download_button(
        "Download Template (.xlsx)",
        data=template_bytes,
        file_name="prefero_template.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ══════════════════════════════════════════════════════════════════
# 2. PRIMARY UPLOAD
# ══════════════════════════════════════════════════════════════════

st.subheader("Upload Your Choice Data")

uploaded = st.file_uploader(
    "Upload a CSV or .xlsx file",
    type=["csv", "xlsx"],
    key=f"choice_upload_{st.session_state.get('_uploader_generation', 0)}",
)
if uploaded is not None:
    file_name = uploaded.name
    identity = (file_name, getattr(uploaded, "file_id", None) or sha256(uploaded.getbuffer()).hexdigest())
    if st.session_state.get("_upload_identity") != identity:
        _clear_import()
        st.session_state["_upload_identity"] = identity
        try:
            if file_name.lower().endswith(".xlsx"):
                with st.spinner("Reading Excel file..."):
                    excel = load_excel_with_dictionary(uploaded)
                if excel.data.empty:
                    raise ValueError("The uploaded data sheet has no rows.")
                _clear_dataset()
                st.session_state.pop("dict_mapping_editor", None)
                st.session_state["_excel_result"] = excel
                st.session_state["_excel_name"] = file_name
            else:
                _stage_dataframe(pd.read_csv(uploaded), file_name)
        except (ValueError, OSError, BadZipFile, InvalidFileException) as exc:
            st.session_state["_upload_error"] = str(exc)
        else:
            st.rerun()

    if error := st.session_state.get("_upload_error"):
        st.error(f"Could not load the new file. {error}")
        if st.session_state.df is not None:
            st.info("The previous dataset is still active.")
    if excel := st.session_state.get("_excel_result"):
        st.success(
            f"Uploaded **{st.session_state['_excel_name']}**. "
            f"Data sheet **{excel.data_sheet_name}** has {len(excel.data)} rows."
        )
        if excel.dict_sheet_name:
            st.info(f"Dictionary sheet detected: **{excel.dict_sheet_name}**")


# ══════════════════════════════════════════════════════════════════
# 3. ALTERNATIVE DATA SOURCES (collapsed)
# ══════════════════════════════════════════════════════════════════

with st.expander("Alternative: Generate or Load Sample Data", expanded=False):
    tab_simulate, tab_apollo = st.tabs(["Simulate", "Apollo"])

    with tab_simulate:
        c1, c2, c3, c4 = st.columns(4)
        n_individuals = c1.slider("Individuals", 50, 2000, 300, step=50)
        n_tasks = c2.slider("Tasks per individual", 2, 20, 8)
        n_alts = c3.slider("Alternatives", 2, 6, 3)
        sim_seed = c4.number_input(
            "Simulation seed", min_value=1, value=42, step=1,
            help="Random seed ensures replicability — it has no real meaning. Pick your lucky number!",
        )

        sim_attr_input = st.text_input(
            "Attribute names (comma-separated)",
            value="price, time, comfort, reliability",
            help="Leave as default for the built-in simulation, or enter custom names.",
        )
        sim_cov_input = st.text_input(
            "Covariate names (comma-separated)",
            value="income, age",
            help="Covariates are respondent-level and constant within each individual.",
        )
        sim_bws = st.checkbox(
            "Generate BWS (worst-choice) data",
            value=False,
            disabled=n_alts < 3,
            help="Requires at least 3 alternatives." if n_alts < 3 else "Adds a 'worst' column.",
        )

        if st.button("Generate Simulated Dataset", type="primary"):
            # parse attribute / covariate inputs
            _default_attrs = ["price", "time", "comfort", "reliability"]
            _default_covs = ["income", "age"]
            _parsed_attrs = [s.strip() for s in sim_attr_input.split(",") if s.strip()]
            _parsed_covs = [s.strip() for s in sim_cov_input.split(",") if s.strip()]
            _attr_arg = None if _parsed_attrs == _default_attrs else _parsed_attrs
            _cov_arg = None if _parsed_covs == _default_covs else _parsed_covs
            with st.spinner("Generating data..."):
                output = generate_simulated_dce(
                    n_individuals=n_individuals,
                    n_tasks=n_tasks,
                    n_alts=n_alts,
                    seed=int(sim_seed),
                    attribute_names=_attr_arg,
                    covariate_names=_cov_arg,
                    bws=sim_bws,
                )
            _store(output.data, "Simulated Choice Data", output.true_parameters)
            try:
                from community_db import log_activity
                log_activity("data_simulate")
            except Exception:
                pass
            st.success("Simulated dataset generated.")

    with tab_apollo:
        st.markdown(
            "**Swiss Route Choice** — Route choice data from Axhausen et al. (2008). "
            "388 respondents chose between 2 route alternatives over 9 choice tasks. "
            "Attributes include travel time, travel cost, headway, and number of changes."
        )
        if st.button("Load Apollo Dataset", type="primary"):
            with st.spinner("Downloading and transforming Apollo dataset..."):
                df = load_apollo_long("swiss_route_choice")
            _store(df, "Apollo swiss_route_choice")
            try:
                from community_db import log_activity
                log_activity("data_apollo")
            except Exception:
                pass
            st.success("Apollo dataset loaded.")


# ══════════════════════════════════════════════════════════════════
# UPLOAD STEP PROGRESS INDICATOR
# ══════════════════════════════════════════════════════════════════

_has_excel = st.session_state.get("_excel_result") is not None
_has_wide = st.session_state.get("_wide_raw") is not None
_has_data = st.session_state.df is not None

if _has_excel or _has_wide:
    _step_labels = ["Upload", "Mapping", "Conversion", "Done"]
    if _has_excel:
        _current = 1  # Mapping step
    elif _has_wide:
        _current = 2  # Conversion step
    else:
        _current = 3
    _indicator_parts = []
    for i, lbl in enumerate(_step_labels):
        if i < _current:
            _indicator_parts.append(f"~~{lbl}~~")
        elif i == _current:
            _indicator_parts.append(f"**> {lbl} <**")
        else:
            _indicator_parts.append(lbl)
    st.caption(" --- ".join(_indicator_parts))

# ── Excel dictionary mapping UI ───────────────────────────
if st.session_state.get("_excel_result") is not None:
    excel_result = st.session_state["_excel_result"]
    raw_df = excel_result.data
    mapping = excel_result.column_mapping
    excel_name = st.session_state["_excel_name"]

    with st.expander("Column Mapping (from Dictionary Sheet)", expanded=True):
        if mapping:
            st.markdown("**Auto-detected mapping** (coded name -> readable name):")
            # editable mapping table
            map_rows = []
            for coded, readable in mapping.items():
                map_rows.append({"Coded Column": coded, "Readable Name": readable})
            map_df = pd.DataFrame(map_rows)
            edited_map = st.data_editor(
                map_df,
                use_container_width=True,
                hide_index=True,
                num_rows="dynamic",
                key="dict_mapping_editor",
            )

            # also show unmapped columns
            mapped_codes = set(mapping.keys())
            unmapped = [c for c in raw_df.columns if c not in mapped_codes]
            if unmapped:
                st.caption(f"Unmapped columns (kept as-is): {', '.join(unmapped)}")

            col_a, col_b = st.columns(2)
            with col_a:
                if st.button("Apply Mapping & Load", type="primary"):
                    pairs = edited_map.dropna().astype(str).apply(lambda col: col.str.strip())
                    pairs = pairs[(pairs != "").all(axis=1)]
                    source_names = pairs["Coded Column"]
                    if source_names.duplicated().any() or not source_names.isin(raw_df.columns).all():
                        st.error("Each coded column in the mapping must exist and appear only once.")
                    else:
                        final_mapping = dict(zip(source_names, pairs["Readable Name"]))
                        _stage_dataframe(apply_column_mapping(raw_df, final_mapping), excel_name, " (mapped)")
                        st.rerun()

            with col_b:
                if st.button("Skip Mapping & Load Raw"):
                    _stage_dataframe(raw_df, excel_name, " (raw)")
                    st.rerun()
        else:
            if excel_result.dictionary is not None:
                st.warning("No column mapping could be extracted from the dictionary sheet.")
                st.markdown("**Dictionary sheet preview:**")
                st.dataframe(excel_result.dictionary.head(30), use_container_width=True)
            if st.button("Load Data Without Mapping", type="primary"):
                _stage_dataframe(raw_df, excel_name)
                st.rerun()

# ── Auto-conversion confirmation card ─────────────────────
if st.session_state.get("_auto_converted") is not None:
    meta = st.session_state.get("_auto_meta", {})
    auto_long = st.session_state["_auto_converted"]

    st.success("Review the converted data before loading it.")

    with st.container():
        info_parts = []
        if meta.get("id_col"):
            info_parts.append(f"ID column: **{meta['id_col']}**")
        else:
            info_parts.append("ID column: *(auto-generated)*")
        info_parts.append(f"Choice column: **{meta['choice_col']}**")

        groups = meta.get("groups", {})
        if groups:
            group_summary = ", ".join(f"{k}({len(v)})" for k, v in groups.items())
            info_parts.append(f"Attribute groups: {group_summary}")

        shape = meta.get("result_shape", auto_long.shape)
        info_parts.append(f"Result: **{shape[0]}** rows x **{shape[1]}** columns")

        for part in info_parts:
            st.markdown(f"- {part}")

    for message in meta.get("warnings", []):
        st.warning(message)

    acc_col, edit_col = st.columns(2)
    with acc_col:
        if st.button("Accept & Continue", type="primary", key="_auto_accept"):
            conversion = "auto-converted" if meta.get("automatic") else "converted"
            _store(auto_long, f"Uploaded: {st.session_state['_wide_name']} ({conversion})")
            st.session_state["_data_warnings"] = meta.get("warnings", [])
            st.rerun()
    with edit_col:
        if st.button("Edit Settings", key="_auto_edit"):
            # Clear auto-converted, keep _wide_raw for manual UI
            st.session_state.pop("_auto_converted", None)
            st.session_state.pop("_auto_meta", None)
            st.rerun()

    # Show preview
    with st.expander("Preview converted data"):
        st.dataframe(auto_long.head(20), use_container_width=True, hide_index=True)

    st.stop()  # Don't show manual UI while confirmation is displayed

# ── Wide-to-long conversion UI ────────────────────────────
if st.session_state.get("_wide_raw") is not None:
    raw_df = st.session_state["_wide_raw"]
    with st.expander("Wide-to-Long Conversion", expanded=True):
        if error := st.session_state.get("_auto_error"):
            st.warning(f"Automatic conversion could not proceed. {error}")
        auto_groups = _auto_attribute_groups(raw_df.columns.tolist())

        if auto_groups:
            st.markdown("**Detected attribute groups:**")
            for attr, cols in auto_groups.items():
                st.caption(f"- **{attr}**: {', '.join(cols)}")
        else:
            st.warning("Could not auto-detect attribute groups from column names.")

        # let user select which groups to include
        available_attrs = list(auto_groups.keys())
        selected_attrs = st.multiselect(
            "Select attribute groups to include",
            options=available_attrs,
            default=available_attrs,
            key="wide_attr_select",
        )

        w1, w2 = st.columns(2)
        raw_cols = raw_df.columns.tolist()
        wide_id_col = w1.selectbox(
            "ID column (wide)",
            [None] + raw_cols,
            index=([None] + raw_cols).index(_guess_id_col(raw_cols)),
            key="wide_id",
        )
        wide_choice_col = w2.selectbox(
            "Choice column (wide)",
            [None] + raw_cols,
            index=([None] + raw_cols).index(_guess_choice_col(raw_cols)),
            key="wide_choice",
        )

        if st.button("Convert to Long Format", type="primary", disabled=not selected_attrs or not wide_choice_col):
            groups = {k: auto_groups[k] for k in selected_attrs}
            messages = []
            try:
                long_df = wide_to_long(raw_df, groups, wide_id_col, wide_choice_col,
                                       on_exclusion=messages.append)
            except ValueError as exc:
                for message in messages:
                    st.warning(message)
                st.error(f"Conversion failed: {exc}")
            else:
                st.session_state["_auto_converted"] = long_df
                st.session_state["_auto_meta"] = {
                    "id_col": wide_id_col, "choice_col": wide_choice_col, "groups": groups,
                    "warnings": messages, "automatic": False,
                }
                st.rerun()

# ── guard: stop here if nothing loaded yet ─────────────────────────
df: pd.DataFrame | None = st.session_state.df
if df is None:
    st.info("Upload a file above to get started, or generate/load sample data.")
    st.stop()

# ══════════════════════════════════════════════════════════════════
# DATA EXPLORATION (shown after data is loaded)
# ══════════════════════════════════════════════════════════════════

# ── 2  Data Preview ────────────────────────────────────────────────
st.divider()
st.subheader("2. Data preview")
st.markdown(f"**Dataset:** {st.session_state.data_label}  &mdash;  `{df.shape[0]}` rows x `{df.shape[1]}` columns")
for message in st.session_state.get("_data_warnings", []):
    st.warning(message)
st.dataframe(df.head(50), use_container_width=True)

# ground-truth parameters (simulated data only)
if st.session_state.true_params:
    with st.expander("Simulation ground-truth parameters"):
        st.json(st.session_state.true_params)

# ── 3  Panel Structure Summary ─────────────────────────────────────
st.divider()
st.subheader("3. Panel structure")

_guess_cols = {
    "id": ["respondent_id", "id", "resp_id", "person_id", "participant_id"],
    "task": ["task_id", "task", "scenario", "question", "situation"],
    "alt": ["alternative", "alt", "option", "alt_id"],
    "choice": ["choice", "chosen", "selected", "response"],
}


def _find(candidates: list[str]) -> str | None:
    lowered = {str(c).lower(): c for c in df.columns}
    for t in candidates:
        if t.lower() in lowered:
            return lowered[t.lower()]
    return None


detected_id = _find(_guess_cols["id"])
detected_task = _find(_guess_cols["task"])
detected_alt = _find(_guess_cols["alt"])
detected_choice = _find(_guess_cols["choice"])

# Handle missing task_id or alternative columns by generating them
if detected_task is None and detected_id is not None:
    st.warning("No task column detected. Generating `task_id` from row ordering.")
    if detected_alt is not None:
        n_alts = max(1, df[detected_alt].nunique())
        df["task_id"] = df.groupby(detected_id, observed=True).cumcount() // n_alts + 1
    else:
        df["task_id"] = df.groupby(detected_id).cumcount() + 1
    st.session_state.df = df
    detected_task = "task_id"

if detected_alt is None and detected_task is not None and detected_id is not None:
    st.warning("No alternative column detected. Generating `alternative` from row ordering within tasks.")
    df["alternative"] = df.groupby([detected_id, detected_task]).cumcount() + 1
    st.session_state.df = df
    detected_alt = "alternative"

p1, p2, p3, p4 = st.columns(4)
if detected_id:
    p1.metric("Individuals", f"{df[detected_id].nunique():,}")
if detected_id and detected_task:
    n_tasks = len(df[[detected_id, detected_task]].dropna().drop_duplicates())
    p2.metric("Choice tasks", f"{n_tasks:,}")
if detected_alt:
    p3.metric("Alternatives", f"{df[detected_alt].nunique():,}")
p4.metric("Total rows", f"{len(df):,}")

# ── Auto-detect Structure button ─────────────────────────────────
if st.button("Auto-detect Structure"):
    inference = infer_structure(df)
    det_parts = []
    if inference.id_col:
        det_parts.append(f"ID: **{inference.id_col}** ({inference.id_confidence:.0%})")
    if inference.task_col:
        det_parts.append(f"Task: **{inference.task_col}** ({inference.task_confidence:.0%})")
    if inference.alt_col:
        det_parts.append(f"Alt: **{inference.alt_col}** ({inference.alt_confidence:.0%})")
    if inference.choice_col:
        det_parts.append(f"Choice: **{inference.choice_col}** ({inference.choice_confidence:.0%})")
    if det_parts:
        st.info("Detected columns: " + " | ".join(det_parts))
        # store inferred roles for Model page to pick up
        st.session_state["inferred_columns"] = {
            "id": inference.id_col,
            "task": inference.task_col,
            "alt": inference.alt_col,
            "choice": inference.choice_col,
        }
    else:
        st.warning("Could not auto-detect any structural columns.")

# ── 4  Column Types ────────────────────────────────────────────────
st.divider()
st.subheader("4. Column types")

col_info = pd.DataFrame(
    {
        "Column": df.columns,
        "Dtype": [str(df[c].dtype) for c in df.columns],
        "Non-null": [df[c].notna().sum() for c in df.columns],
        "Null": [df[c].isna().sum() for c in df.columns],
        "Unique": [df[c].nunique() for c in df.columns],
    }
)
st.dataframe(col_info, use_container_width=True, hide_index=True)

# ── 5  Descriptive Statistics ──────────────────────────────────────
st.divider()
st.subheader("5. Descriptive statistics")

numeric_cols = df.select_dtypes("number").columns.tolist()
categorical_cols = [c for c in df.columns if c not in numeric_cols]

if numeric_cols:
    st.markdown("**Numeric columns**")
    st.dataframe(df[numeric_cols].describe().T, use_container_width=True)

if categorical_cols:
    st.markdown("**Categorical columns**")
    for col in categorical_cols:
        with st.expander(f"Frequency table: {col}"):
            freq = df[col].value_counts().reset_index()
            freq.columns = [col, "count"]
            st.dataframe(freq, use_container_width=True, hide_index=True)

# ── 6  Missing Values ─────────────────────────────────────────────
missing = df.isnull().sum()
if missing.sum() > 0:
    st.divider()
    st.subheader("6. Missing values")
    missing_df = missing[missing > 0].reset_index()
    missing_df.columns = ["Column", "Missing count"]
    missing_df["% missing"] = (missing_df["Missing count"] / len(df) * 100).round(2)
    st.dataframe(missing_df, use_container_width=True, hide_index=True)

# ── 7  Choice Frequency ───────────────────────────────────────────
if all((detected_id, detected_task, detected_alt, detected_choice)):
    st.divider()
    st.subheader("7. Choice frequency")
    try:
        chosen = chosen_row_mask(df, detected_id, detected_task, detected_alt, detected_choice)
    except ValueError as exc:
        st.warning(f"Choice counts are unavailable. {exc}")
    else:
        alternatives = df[detected_alt].drop_duplicates()
        freq = df.loc[chosen, detected_alt].value_counts().reindex(alternatives, fill_value=0)
        table = freq.rename_axis("Alternative").reset_index(name="Times chosen")
        c_left, c_right = st.columns(2)
        c_left.dataframe(table, use_container_width=True, hide_index=True)
        c_right.bar_chart(table, x="Alternative", y="Times chosen")

# ── 8  Distribution Plots ─────────────────────────────────────────
if numeric_cols:
    st.divider()
    st.subheader("8. Distribution plots")
    selected_col = st.selectbox("Select numeric column", numeric_cols)
    if selected_col:
        c_left, c_right = st.columns(2)
        with c_left:
            st.markdown(f"**Histogram: {selected_col}**")
            hist_data = df[selected_col].dropna()
            st.bar_chart(hist_data.value_counts().sort_index().head(50))
        with c_right:
            st.markdown(f"**Summary: {selected_col}**")
            stats = df[selected_col].describe()
            st.dataframe(stats, use_container_width=True)

slowbro_next_step("pages/1_📊_Data.py")
