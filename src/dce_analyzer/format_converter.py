"""Utilities for detecting and converting DCE data formats."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass, field

import pandas as pd


@dataclass
class ColumnInference:
    """Result of auto-detecting structural columns in a long-format DataFrame."""

    id_col: str | None = None
    id_confidence: float = 0.0
    task_col: str | None = None
    task_confidence: float = 0.0
    alt_col: str | None = None
    alt_confidence: float = 0.0
    choice_col: str | None = None
    choice_confidence: float = 0.0


# ── Format detection ──────────────────────────────────────────────


def detect_format(df: pd.DataFrame) -> str:
    """Return ``'long'`` or ``'wide'`` based on column naming patterns.

    Wide-format heuristic: look for columns that end with ``_1``, ``_2`` (or
    just a trailing digit like ``tt1``, ``tt2``) and share a common prefix.
    If sufficiently many such groups are found the data is likely wide.
    """
    groups = _find_suffix_groups(df.columns.tolist())
    if len(groups) >= 2:
        return "wide"
    return "long"


def _find_suffix_groups(columns: list[str]) -> dict[str, list[int]]:
    """Identify column-name groups sharing a prefix with numeric suffixes.

    Matches patterns like ``price_1, price_2`` and ``tt1, tt2``.
    Returns ``{prefix: [suffixes]}`` for groups with at least two members.
    """
    # pattern: word characters followed by optional _ then one or more digits at end
    pat = re.compile(r"^(.+?)_?(\d+)$")
    buckets: dict[str, set[int]] = {}
    for col in columns:
        m = pat.match(col)
        if m:
            prefix, num = m.group(1), int(m.group(2))
            buckets.setdefault(prefix, set()).add(num)

    return {
        prefix: sorted(nums)
        for prefix, nums in buckets.items()
        if len(nums) >= 2
    }


# ── Wide to long conversion ──────────────────────────────────────


def wide_to_long(
    df: pd.DataFrame,
    attribute_groups: dict[str, list[str]] | None = None,
    id_col: str | None = None,
    choice_col: str | None = None,
    covariate_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Convert a wide-format DCE DataFrame to long format.

    Parameters
    ----------
    df : DataFrame
        Wide-format data where each row is one choice task and alternative-
        specific attributes appear as separate columns (e.g. ``price_1``,
        ``price_2``).
    attribute_groups : dict mapping attribute name to list of wide columns
        For example ``{"price": ["price_1", "price_2"], "time": ["time_1", "time_2"]}``.
        If ``None``, groups are auto-detected from column suffix patterns.
    id_col : str or None
        Respondent identifier column.  If ``None``, auto-detected or generated.
    choice_col : str or None
        Column holding the chosen alternative (1-based label).
        If ``None``, auto-detected.
    covariate_cols : list[str] or None
        Columns that are constant across alternatives (demographics etc.).
        If ``None``, inferred as all columns not in attribute_groups, id_col,
        or choice_col.
    """
    work = df.copy()

    # auto-detect attribute groups if not provided
    if attribute_groups is None:
        attribute_groups = _auto_attribute_groups(work.columns.tolist())
    if not attribute_groups:
        raise ValueError(
            "Could not detect alternative-specific column groups. "
            "Please specify attribute_groups explicitly."
        )

    # determine number of alternatives from group sizes
    n_alts_set = {len(v) for v in attribute_groups.values()}
    if len(n_alts_set) != 1:
        raise ValueError(
            f"Attribute groups have inconsistent sizes: {n_alts_set}. "
            "All groups must have the same number of alternatives."
        )
    n_alts = n_alts_set.pop()

    # auto-detect id column
    if id_col is None:
        id_col = _guess_id_col(work.columns.tolist())
    if id_col is None:
        work["respondent_id"] = range(1, len(work) + 1)
        id_col = "respondent_id"

    # auto-detect choice column
    if choice_col is None:
        choice_col = _guess_choice_col(work.columns.tolist())

    # ── Exclude respondents with ANY missing / invalid choice data ──
    # A missing choice means the respondent didn't answer that task.
    # We drop the entire respondent (not just the row) to keep clean panels.
    import warnings

    if choice_col and choice_col in work.columns:
        n_total_rows = len(work)
        n_total_resp = work[id_col].nunique()

        # find rows where choice is NaN or not in the valid alt range [1..n_alts]
        valid_alts = set(range(1, n_alts + 1))
        bad_mask = work[choice_col].isna()
        if not bad_mask.any():
            # also check for values outside the valid alt range
            bad_mask = bad_mask | ~work[choice_col].isin(valid_alts)

        if bad_mask.any():
            bad_respondents = work.loc[bad_mask, id_col].unique()
            na_rate = len(bad_respondents) / max(n_total_resp, 1)
            work = work[~work[id_col].isin(bad_respondents)].reset_index(drop=True)
            warnings.warn(
                f"Excluded {len(bad_respondents)} respondent(s) "
                f"({na_rate:.1%} of {n_total_resp}) with missing or invalid "
                f"choice data in column '{choice_col}'. "
                f"{work[id_col].nunique()} respondents remaining."
            )

    # generate task_id per respondent
    work["_task_id"] = work.groupby(id_col).cumcount() + 1

    # figure out covariate columns
    all_attr_cols = {c for cols in attribute_groups.values() for c in cols}
    exclude = {id_col, "_task_id"} | all_attr_cols
    if choice_col:
        exclude.add(choice_col)
    if covariate_cols is None:
        covariate_cols = [c for c in work.columns if c not in exclude]

    # build long parts
    parts = []
    for alt_idx in range(n_alts):
        alt_num = alt_idx + 1
        part: dict[str, object] = {
            "respondent_id": work[id_col],
            "task_id": work["_task_id"],
            "alternative": alt_num,
        }

        # choice indicator
        if choice_col:
            part["choice"] = (work[choice_col] == alt_num).astype(int)
        else:
            part["choice"] = 0

        for attr_name, wide_cols in attribute_groups.items():
            part[attr_name] = work[wide_cols[alt_idx]].values

        for cov in covariate_cols:
            part[cov] = work[cov].values

        parts.append(pd.DataFrame(part))

    long_df = (
        pd.concat(parts, ignore_index=True)
        .sort_values(["respondent_id", "task_id", "alternative"])
        .reset_index(drop=True)
    )
    return long_df


def _auto_attribute_groups(columns: list[str]) -> dict[str, list[str]]:
    """Build attribute groups from column-name patterns.

    Supports two common DCE naming conventions:

    1. Suffix pattern: ``price_1, price_2, price_3`` or ``tt1, tt2, tt3``
       → ``{"price": ["price_1", "price_2", "price_3"]}``

    2. DCE matrix pattern: ``a1_x1, a2_x1, a3_x1`` (alt prefix + attribute)
       → ``{"x1": ["a1_x1", "a2_x1", "a3_x1"]}``
       This also handles renamed variants like ``income_a1, income_a2, income_a3``
       → ``{"income": ["income_a1", "income_a2", "income_a3"]}``
    """
    # Try DCE matrix pattern first (a{N}_x{M} style)
    dce_groups = _detect_dce_matrix_groups(columns)
    if dce_groups:
        return dce_groups

    # Fall back to simple suffix pattern
    pat = re.compile(r"^(.+?)_?(\d+)$")
    buckets: dict[str, dict[int, str]] = {}
    for col in columns:
        m = pat.match(col)
        if m:
            prefix, num = m.group(1), int(m.group(2))
            buckets.setdefault(prefix, {})[num] = col

    groups: dict[str, list[str]] = {}
    for prefix, num_map in buckets.items():
        if len(num_map) < 2:
            continue
        sorted_nums = sorted(num_map.keys())
        if sorted_nums != list(range(1, len(sorted_nums) + 1)):
            continue
        groups[prefix] = [num_map[n] for n in sorted_nums]

    return groups


def _detect_dce_matrix_groups(columns: list[str]) -> dict[str, list[str]]:
    """Detect DCE-style alternative×attribute column patterns.

    Recognizes columns like ``a1_x1, a2_x1, a3_x1`` where:
    - ``a1, a2, a3`` are alternative identifiers (prefix)
    - ``x1, x2, ...`` are attribute identifiers (suffix)

    Also handles renamed variants like ``income_a1, income_a2, income_a3``
    where the alternative identifier is at the end.

    Returns groups keyed by attribute, with one column per alternative.
    """
    # Pattern 1: a{N}_x{M} — alternative prefix, attribute suffix
    prefix_pat = re.compile(r"^(a\d+)[_.](.+)$", re.IGNORECASE)
    alt_attr_pairs: list[tuple[str, str, str]] = []  # (alt_id, attr_id, full_col)
    for col in columns:
        m = prefix_pat.match(col)
        if m:
            alt_attr_pairs.append((m.group(1).lower(), m.group(2).lower(), col))

    if alt_attr_pairs:
        alts = sorted(set(p[0] for p in alt_attr_pairs))
        attrs = sorted(set(p[1] for p in alt_attr_pairs))

        if len(alts) >= 2 and len(attrs) >= 1:
            # build lookup: (alt, attr) -> col
            lookup = {(p[0], p[1]): p[2] for p in alt_attr_pairs}
            groups: dict[str, list[str]] = {}
            for attr in attrs:
                cols_for_attr = []
                for alt in alts:
                    if (alt, attr) in lookup:
                        cols_for_attr.append(lookup[(alt, attr)])
                if len(cols_for_attr) == len(alts):
                    groups[attr] = cols_for_attr
            if groups:
                return groups

    # Pattern 2: {name}_a{N} — attribute prefix, alternative suffix
    suffix_pat = re.compile(r"^(.+?)_a(\d+)$", re.IGNORECASE)
    name_alt_pairs: list[tuple[str, str, str]] = []  # (attr_name, alt_id, full_col)
    for col in columns:
        m = suffix_pat.match(col)
        if m:
            name_alt_pairs.append((m.group(1).lower(), m.group(2), col))

    if name_alt_pairs:
        attr_names = sorted(set(p[0] for p in name_alt_pairs))
        alt_ids = sorted(set(p[1] for p in name_alt_pairs))

        if len(alt_ids) >= 2 and len(attr_names) >= 1:
            lookup = {(p[0], p[1]): p[2] for p in name_alt_pairs}
            groups = {}
            for attr in attr_names:
                cols_for_attr = []
                for alt in alt_ids:
                    if (attr, alt) in lookup:
                        cols_for_attr.append(lookup[(attr, alt)])
                if len(cols_for_attr) == len(alt_ids):
                    groups[attr] = cols_for_attr
            if groups:
                return groups

    return {}


def _guess_id_col(columns: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in ["respondent_id", "id", "ID", "resp_id", "person_id"]:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _guess_choice_col(columns: list[str]) -> str | None:
    lowered = {c.lower(): c for c in columns}
    for candidate in ["choice", "chosen", "selected"]:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


# ── Choice column normalization ───────────────────────────────────


def normalize_choice_column(
    df: pd.DataFrame,
    choice_col: str,
    alt_col: str,
) -> pd.DataFrame:
    """Normalize a choice column to binary 0/1 indicator format.

    Handles three input formats:
    - Already binary 0/1: returned as-is.
    - Numeric label (e.g. ``choice=2``): compared against ``alt_col`` values.
    - String label (e.g. ``choice="B"``): compared against ``alt_col`` values.
    """
    out = df.copy()
    choice_vals = out[choice_col]
    unique = choice_vals.unique()

    # already binary
    if set(unique) <= {0, 1, 0.0, 1.0}:
        out[choice_col] = choice_vals.astype(int)
        return out

    # label format: choice column holds the chosen alt code
    out[choice_col] = (out[choice_col] == out[alt_col]).astype(int)
    return out


# ── Structure inference ───────────────────────────────────────────


def infer_structure(df: pd.DataFrame) -> ColumnInference:
    """Auto-detect ID, task, alternative, and choice columns with confidence.

    Confidence is a float in [0, 1].  A score of 1.0 means the column name
    matches a well-known pattern exactly; lower scores indicate weaker signals.
    """
    result = ColumnInference()
    cols = df.columns.tolist()

    # --- ID column ---
    result.id_col, result.id_confidence = _best_match(
        cols, df,
        name_candidates=["respondent_id", "id", "ID", "resp_id", "person_id", "participant_id"],
        heuristic=_id_heuristic,
    )

    # --- Task column ---
    result.task_col, result.task_confidence = _best_match(
        cols, df,
        name_candidates=["task_id", "task", "scenario", "question", "situation"],
        heuristic=_task_heuristic,
    )

    # --- Alternative column ---
    result.alt_col, result.alt_confidence = _best_match(
        cols, df,
        name_candidates=["alternative", "alt", "option", "alt_id"],
        heuristic=_alt_heuristic,
    )

    # --- Choice column ---
    result.choice_col, result.choice_confidence = _best_match(
        cols, df,
        name_candidates=["choice", "chosen", "selected", "response"],
        heuristic=_choice_heuristic,
    )

    return result


def _best_match(
    columns: list[str],
    df: pd.DataFrame,
    name_candidates: list[str],
    heuristic,
) -> tuple[str | None, float]:
    """Find the best column match by name, falling back to a heuristic."""
    lowered = {c.lower(): c for c in columns}

    # exact name match (high confidence)
    for candidate in name_candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()], 1.0

    # partial name match
    for candidate in name_candidates:
        for col_lower, col_orig in lowered.items():
            if candidate.lower() in col_lower:
                return col_orig, 0.7

    # heuristic-based detection
    best_col, best_score = heuristic(df)
    if best_col is not None:
        return best_col, best_score

    return None, 0.0


def _id_heuristic(df: pd.DataFrame) -> tuple[str | None, float]:
    """Detect ID column: many repeated values, integer-like."""
    best, score = None, 0.0
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        n_unique = df[col].nunique()
        repeat_ratio = len(df) / max(n_unique, 1)
        if repeat_ratio >= 2 and n_unique > 1:
            s = min(0.5, 0.1 * repeat_ratio)
            if s > score:
                best, score = col, s
    return best, score


def _task_heuristic(df: pd.DataFrame) -> tuple[str | None, float]:
    """Detect task column: sequential integers within groups."""
    best, score = None, 0.0
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        vals = df[col].dropna().unique()
        if len(vals) < 2:
            continue
        if all(float(v).is_integer() for v in vals):
            mn, mx = int(min(vals)), int(max(vals))
            if mn >= 0 and mx <= 100 and mx - mn + 1 == len(vals):
                s = 0.4
                if s > score:
                    best, score = col, s
    return best, score


def _alt_heuristic(df: pd.DataFrame) -> tuple[str | None, float]:
    """Detect alt column: small set of repeating integers or labels."""
    best, score = None, 0.0
    for col in df.columns:
        n_unique = df[col].nunique()
        if 2 <= n_unique <= 10:
            repeat_ratio = len(df) / n_unique
            if repeat_ratio > 10:
                s = 0.3
                if s > score:
                    best, score = col, s
    return best, score


def _choice_heuristic(df: pd.DataFrame) -> tuple[str | None, float]:
    """Detect choice column: binary 0/1 column."""
    best, score = None, 0.0
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        unique_vals = set(df[col].dropna().unique())
        if unique_vals == {0, 1} or unique_vals == {0.0, 1.0}:
            mean_val = df[col].mean()
            # a good choice column has roughly 1/n_alts fraction of 1s
            if 0.05 < mean_val < 0.95:
                s = 0.5
                if s > score:
                    best, score = col, s
    return best, score


# ── Excel / .xlsx support ────────────────────────────────────────


@dataclass
class ExcelParseResult:
    """Result of parsing a multi-sheet Excel file."""

    data: pd.DataFrame
    dictionary: pd.DataFrame | None = None
    column_mapping: dict[str, str] = field(default_factory=dict)
    data_sheet_name: str = ""
    dict_sheet_name: str = ""
    sheet_names: list[str] = field(default_factory=list)


def load_excel_with_dictionary(
    file: io.BytesIO | str,
    data_sheet: str | int | None = None,
    dict_sheet: str | int | None = None,
) -> ExcelParseResult:
    """Load an Excel file, auto-detecting data and dictionary sheets.

    Heuristic for identifying sheets:
    - The sheet with the most rows is likely the data sheet.
    - A sheet whose name contains "dict", "codebook", "legend", "map",
      "label", or "variable" (case-insensitive) is treated as the dictionary.
    - If only two sheets exist and one is clearly smaller, the smaller one
      is the dictionary.

    Parameters
    ----------
    file : file-like or path
        Excel file to read.
    data_sheet : str, int, or None
        Explicit data sheet name/index. Auto-detected if ``None``.
    dict_sheet : str, int, or None
        Explicit dictionary sheet name/index. Auto-detected if ``None``.
    """
    xls = pd.ExcelFile(file, engine="openpyxl")
    sheet_names = xls.sheet_names

    if len(sheet_names) == 0:
        raise ValueError("Excel file has no sheets.")

    # ── pick data sheet ──
    if data_sheet is not None:
        data_df = pd.read_excel(xls, sheet_name=data_sheet)
        data_sheet_name = data_sheet if isinstance(data_sheet, str) else sheet_names[data_sheet]
    else:
        # largest sheet = data
        row_counts = {}
        for s in sheet_names:
            tmp = pd.read_excel(xls, sheet_name=s)
            row_counts[s] = len(tmp)
        data_sheet_name = max(row_counts, key=row_counts.get)
        data_df = pd.read_excel(xls, sheet_name=data_sheet_name)

    # ── pick dictionary sheet ──
    dict_df: pd.DataFrame | None = None
    dict_sheet_name = ""
    dict_keywords = {"dict", "codebook", "legend", "map", "label", "variable", "key", "lookup"}

    if dict_sheet is not None:
        dict_df = pd.read_excel(xls, sheet_name=dict_sheet)
        dict_sheet_name = dict_sheet if isinstance(dict_sheet, str) else sheet_names[dict_sheet]
    else:
        remaining = [s for s in sheet_names if s != data_sheet_name]
        # try keyword match first
        for s in remaining:
            if any(kw in s.lower() for kw in dict_keywords):
                dict_df = pd.read_excel(xls, sheet_name=s)
                dict_sheet_name = s
                break
        # fallback: if exactly one other sheet, assume it's the dictionary
        if dict_df is None and len(remaining) == 1:
            dict_df = pd.read_excel(xls, sheet_name=remaining[0])
            dict_sheet_name = remaining[0]

    # ── build column mapping from dictionary ──
    column_mapping: dict[str, str] = {}
    if dict_df is not None:
        column_mapping = _parse_dictionary_sheet(dict_df, data_df.columns.tolist())

    return ExcelParseResult(
        data=data_df,
        dictionary=dict_df,
        column_mapping=column_mapping,
        data_sheet_name=data_sheet_name,
        dict_sheet_name=dict_sheet_name,
        sheet_names=sheet_names,
    )


def _strip_html(text: str) -> str:
    """Remove HTML tags and extract plain text."""
    if not isinstance(text, str):
        return str(text)
    # extract text from common patterns like <div ...>text</div>
    text = re.sub(r"<[^>]+>", " ", text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _parse_dictionary_sheet(
    dict_df: pd.DataFrame,
    data_columns: list[str],
) -> dict[str, str]:
    """Extract a code→label mapping from a dictionary sheet.

    Supports several common dictionary layouts:
    1. Hierarchical / Sawtooth-style: variable names in one column, descriptions
       in the next, with level codes below (separated by blank rows). Columns
       may have auto-generated names like ``Unnamed: 1``.
    2. Two-column: [code, label]  or  [variable, description]
    3. Multi-column with identifiable code+label columns

    Returns a dict mapping coded column names to human-readable labels.
    """
    if dict_df.empty or len(dict_df.columns) < 2:
        return {}

    data_col_set = set(data_columns)
    data_col_lower = {c.lower(): c for c in data_columns}

    # ── Strategy 0: Hierarchical / Sawtooth-style dictionary ──
    # These have variable names scattered across rows in one column,
    # with descriptions in an adjacent column. Other rows hold level
    # codes/labels or are blank. Detect by scanning all columns for
    # cells that match data column names.
    hierarchical = _parse_hierarchical_dictionary(dict_df, data_col_set)
    if hierarchical:
        return hierarchical

    # ── Strategy 1: Find two columns where one contains data column names ──
    for code_idx, code_col in enumerate(dict_df.columns):
        code_vals = dict_df[code_col].dropna().astype(str)
        matches = sum(1 for v in code_vals if v in data_col_set or v.lower() in data_col_lower)
        match_ratio = matches / max(len(code_vals), 1)

        if match_ratio >= 0.3 and matches >= 2:
            # this column likely holds the coded names — find the label column
            label_col = _find_label_column(dict_df, code_idx)
            if label_col is not None:
                mapping = {}
                for _, row in dict_df.iterrows():
                    code = str(row[code_col]).strip()
                    label = str(row[label_col]).strip()
                    if code and label and code != "nan" and label != "nan":
                        actual_code = data_col_set & {code} or {data_col_lower.get(code.lower())}
                        actual_code = {c for c in actual_code if c is not None}
                        if actual_code:
                            orig = actual_code.pop()
                            clean_label = _clean_column_name(label)
                            if clean_label and clean_label != orig:
                                mapping[orig] = clean_label
                if mapping:
                    return mapping

    # ── Strategy 2: First col = code, second col = label (simple two-col) ──
    col_a, col_b = dict_df.columns[0], dict_df.columns[1]
    mapping = {}
    for _, row in dict_df.iterrows():
        code = str(row[col_a]).strip()
        label = str(row[col_b]).strip()
        if code in data_col_set and label and label != "nan":
            clean_label = _clean_column_name(label)
            if clean_label and clean_label != code:
                mapping[code] = clean_label
    if len(mapping) >= 2:
        return mapping

    return {}


def _parse_hierarchical_dictionary(
    dict_df: pd.DataFrame,
    data_col_set: set[str],
) -> dict[str, str]:
    """Parse a hierarchical dictionary where variable names and descriptions
    are scattered across rows, typical of Sawtooth / Lighthouse Studio exports.

    Layout pattern (across any pair of adjacent columns):
        Row i:   [variable_name]  [description / label]
        Row i+1: [level_code]     [level_label]          (optional, repeats)
        ...      [blank row]                              (separator)

    The variable_name cell matches a data column name; the adjacent cell in
    the same row holds the human-readable description (possibly with HTML).
    """
    mapping: dict[str, str] = {}
    cols = dict_df.columns.tolist()

    # scan every pair of adjacent columns
    for ci in range(len(cols) - 1):
        var_col = cols[ci]
        desc_col = cols[ci + 1]

        for _, row in dict_df.iterrows():
            var_val = str(row[var_col]).strip() if pd.notna(row[var_col]) else ""
            desc_val = str(row[desc_col]).strip() if pd.notna(row[desc_col]) else ""

            if var_val in data_col_set and desc_val and desc_val != "nan":
                # strip HTML from description
                clean_desc = _strip_html(desc_val)
                clean_label = _clean_column_name(clean_desc)
                if clean_label and clean_label != var_val.lower():
                    # for a{N}_x{M} style columns, prefix the attribute name
                    # with the base attribute to avoid duplicates across alts
                    mapping[var_val] = clean_label

    if len(mapping) < 2:
        return {}

    # deduplicate: if multiple coded columns map to the same label (e.g.
    # a1_x1, a2_x1, a3_x1 all -> "average_annual_income_from_the_product"),
    # append the alt prefix to distinguish them
    label_counts: dict[str, list[str]] = {}
    for code, label in mapping.items():
        label_counts.setdefault(label, []).append(code)

    final: dict[str, str] = {}
    for label, codes in label_counts.items():
        if len(codes) == 1:
            final[codes[0]] = label
        else:
            # extract distinguishing prefix from coded name
            for code in codes:
                # try to extract alt identifier like "a1", "a2" from "a1_x1"
                alt_match = re.match(r"^(a\d+)[_.]", code)
                if alt_match:
                    final[code] = f"{label}_{alt_match.group(1)}"
                else:
                    final[code] = f"{label}_{code}"

    return final


def _find_label_column(dict_df: pd.DataFrame, code_col_idx: int) -> str | None:
    """Find the most likely label/description column in a dictionary sheet."""
    label_keywords = {"label", "name", "description", "desc", "meaning", "variable",
                      "attribute", "full", "readable", "display"}
    cols = dict_df.columns.tolist()

    # check column names for label-like keywords
    for i, col in enumerate(cols):
        if i == code_col_idx:
            continue
        if any(kw in str(col).lower() for kw in label_keywords):
            return col

    # fallback: pick the next column that has string-like content
    for i, col in enumerate(cols):
        if i == code_col_idx:
            continue
        if dict_df[col].dtype == object:
            # prefer columns with longer average string length (descriptions)
            avg_len = dict_df[col].dropna().astype(str).str.len().mean()
            if avg_len > 2:
                return col

    # last resort: just pick the adjacent column
    adj = code_col_idx + 1 if code_col_idx + 1 < len(cols) else code_col_idx - 1
    if 0 <= adj < len(cols):
        return cols[adj]

    return None


def _clean_column_name(label: str, max_words: int = 6) -> str:
    """Convert a human-readable label to a valid, clean column name.

    Long descriptions are truncated to ``max_words`` meaningful words.

    Examples:
        "Travel Time (min)" -> "travel_time_min"
        "Price / Cost" -> "price_cost"
        "Average Annual Income from the product" -> "avg_annual_income"
        "Alternative-Specific Constant" -> "alternative_specific_constant"
    """
    s = label.strip().lower()
    # replace common separators with underscore
    s = re.sub(r"[\s/\-–—.]+", "_", s)
    # remove parentheses, special characters, currency symbols, etc.
    s = re.sub(r"[()[\]{},;:\"'!@#$%^&*+=<>?|~`]+", "", s)
    # collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    # strip leading/trailing underscores
    s = s.strip("_")
    # if result is empty or just numbers, return empty
    if not s or s.isdigit():
        return ""

    # truncate long names: drop filler words, then keep max_words
    filler = {"the", "a", "an", "from", "for", "of", "to", "in", "on",
              "by", "with", "among", "all", "and", "or", "is", "was",
              "are", "been", "be", "its", "this", "that", "which",
              "where", "may", "will", "not", "no", "except", "normally"}
    parts = s.split("_")
    # remove pure filler words (keep at least 2 words)
    meaningful = [w for w in parts if w not in filler]
    if len(meaningful) < 2:
        meaningful = parts
    if len(meaningful) > max_words:
        meaningful = meaningful[:max_words]
    s = "_".join(meaningful)

    # common abbreviations for shorter names
    abbrevs = {
        "average": "avg",
        "number": "num",
        "preferred": "pref",
        "unexpected": "unexp",
        "beneficiaries": "benefic",
        "withdrawal": "withdraw",
        "guarantee": "guarantee",
        "long_term": "longterm",
    }
    for long, short in abbrevs.items():
        s = s.replace(long, short)

    return s


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename DataFrame columns using a code→label mapping.

    Only renames columns that exist in the mapping. Handles duplicate
    target names by appending a suffix.
    """
    rename_map = {}
    used_names: set[str] = set(df.columns) - set(mapping.keys())

    for old_name, new_name in mapping.items():
        if old_name not in df.columns:
            continue
        final = new_name
        counter = 2
        while final in used_names:
            final = f"{new_name}_{counter}"
            counter += 1
        rename_map[old_name] = final
        used_names.add(final)

    return df.rename(columns=rename_map)
