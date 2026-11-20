# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .config import ModelSpec


def get_best_device() -> torch.device:
    """Auto-detect the best available compute device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_device_info() -> str:
    """Return a human-readable string describing the active compute device."""
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        return f"{name} (CUDA)"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        import platform
        chip = platform.processor() or "Apple Silicon"
        return f"{chip} (MPS)"
    import platform
    proc = platform.processor() or "unknown"
    return f"CPU ({proc})"


@dataclass
class ChoiceTensors:
    X: torch.Tensor
    y: torch.Tensor
    panel_idx: torch.Tensor
    n_individuals: int
    n_obs: int
    n_alts: int
    feature_names: list[str]
    id_values: np.ndarray


def load_long_csv(path: str | Path) -> pd.DataFrame:
    """Read a long-format CSV file."""
    return pd.read_csv(path)


def _validate_choice_layout(
    df: pd.DataFrame, id_col: str, task_col: str, alt_col: str, choice_col: str,
) -> tuple[int, int]:
    key_cols = [id_col, task_col, alt_col]
    required_cols = [*key_cols, choice_col]
    if len(set(required_cols)) != len(required_cols):
        raise ValueError("Respondent, task, alternative and choice must use distinct columns.")
    if not df.columns.is_unique:
        raise ValueError("Input data must have unique column names.")
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    empty_cols = df[required_cols].isna().any()
    if empty_cols.any():
        raise ValueError(f"Missing values in required columns: {empty_cols[empty_cols].index.tolist()}")

    if df.duplicated(subset=key_cols).any():
        dup_rows = int(df.duplicated(subset=key_cols).sum())
        raise ValueError(
            f"Found {dup_rows} duplicated (id, task, alt) rows. "
            "Each alternative in each task should appear once."
        )

    group_sizes = df.groupby([id_col, task_col], observed=True).size()
    if group_sizes.empty:
        raise ValueError("Input dataframe is empty after grouping by id and task.")
    if (group_sizes < 2).any():
        raise ValueError("Each (id, task) must have at least two alternatives.")
    if group_sizes.nunique() != 1:
        raise ValueError(
            "Each (id, task) must have the same number of alternatives. "
            "Variable-size choice sets are not supported in this baseline."
        )
    return len(group_sizes), int(group_sizes.iloc[0])


def validate_long_format(df: pd.DataFrame, spec: ModelSpec) -> None:
    """Validate structural columns and the selected utility attribute names."""
    _validate_choice_layout(df, spec.id_col, spec.task_col, spec.alt_col, spec.choice_col)
    _validate_variables(df, spec)


def _validate_variables(df: pd.DataFrame, spec: ModelSpec) -> None:
    columns = [v.column for v in spec.variables]
    if any(c in {spec.id_col, spec.task_col, spec.alt_col, spec.choice_col} for c in columns):
        raise ValueError("Utility attributes cannot use respondent, task, alternative or choice columns.")
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _choice_indices(choice_matrix: np.ndarray, alt_matrix: np.ndarray) -> np.ndarray:
    """Convert either one-hot choices or chosen-alt labels to index targets."""
    if pd.isna(choice_matrix).any() or pd.isna(alt_matrix).any():
        raise ValueError("Choice and alternative columns must not contain missing values.")
    choice_matrix = choice_matrix.astype(object, copy=False)
    alt_matrix = alt_matrix.astype(object, copy=False)
    if np.isin(choice_matrix, [0, 1]).all():
        indicators = choice_matrix == 1
        if (indicators.sum(axis=1) == 1).all():
            return np.argmax(indicators, axis=1)

    # Binary alternative codes can also be repeated labels.
    row_constant = np.all(choice_matrix == choice_matrix[:, [0]], axis=1)
    if not row_constant.all():
        raise ValueError(
            "Choice column is neither one-hot nor a repeated chosen-alt label per task."
        )

    chosen_codes = choice_matrix[:, 0]
    matches = alt_matrix == chosen_codes[:, None]
    valid = matches.sum(axis=1) == 1
    if not valid.all():
        bad = int(np.where(~valid)[0][0])
        raise ValueError(
            "Could not map choice code to exactly one alternative in each task. "
            f"First invalid task index: {bad}."
        )
    return np.argmax(matches, axis=1)


def chosen_row_mask(
    df: pd.DataFrame, id_col: str, task_col: str, alt_col: str, choice_col: str,
) -> np.ndarray:
    """Return a Boolean mask in original row order, true for each chosen alternative."""
    shape = _validate_choice_layout(df, id_col, task_col, alt_col, choice_col)
    ordered = df.reset_index(drop=True).sort_values([id_col, task_col, alt_col])
    choices = ordered[choice_col].to_numpy().reshape(shape)
    alternatives = ordered[alt_col].to_numpy().reshape(shape)
    selected = _choice_indices(choices, alternatives)
    mask = np.zeros(len(df), dtype=bool)
    mask[ordered.index.to_numpy().reshape(shape)[np.arange(len(selected)), selected]] = True
    return mask


def prepare_choice_tensors(
    df: pd.DataFrame,
    spec: ModelSpec,
    device: torch.device | None = None,
) -> ChoiceTensors:
    """
    Convert long-format dataframe into tensors used by estimators.

    Expected format: one row per (id, task, alternative), with choice as either:
    - one-hot indicator (0/1), or
    - chosen alternative label repeated across alternatives in the task.
    """
    n_obs, n_alts = _validate_choice_layout(
        df, spec.id_col, spec.task_col, spec.alt_col, spec.choice_col,
    )
    _validate_variables(df, spec)

    if device is None:
        device = get_best_device()

    sort_cols = [spec.id_col, spec.task_col, spec.alt_col]
    work = df.sort_values(sort_cols).reset_index(drop=True)

    group_cols = [spec.id_col, spec.task_col]
    n_vars = len(spec.variables)

    feature_cols = [v.column for v in spec.variables]
    try:
        with np.errstate(over="ignore", invalid="ignore"):
            X_flat = work[feature_cols].to_numpy(dtype=np.float32, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError("Selected utility attributes must be numeric. Encode categories first.") from exc
    nonfinite = ~np.isfinite(X_flat).all(axis=0)
    if nonfinite.any():
        invalid_cols = [col for col, bad in zip(feature_cols, nonfinite) if bad]
        raise ValueError(f"Utility attributes must be finite and non-missing: {invalid_cols}")
    X = X_flat.reshape(n_obs, n_alts, n_vars)

    choice_mat = work[spec.choice_col].to_numpy().reshape(n_obs, n_alts)
    alt_mat = work.loc[:, spec.alt_col].to_numpy().reshape(n_obs, n_alts)
    y = _choice_indices(choice_mat, alt_mat)

    task_table = work.loc[:, group_cols].drop_duplicates()
    panel_idx, unique_ids = pd.factorize(task_table[spec.id_col].to_numpy(), sort=True)

    return ChoiceTensors(
        X=torch.tensor(X, dtype=torch.float32, device=device),
        y=torch.tensor(y, dtype=torch.long, device=device),
        panel_idx=torch.tensor(panel_idx, dtype=torch.long, device=device),
        n_individuals=len(unique_ids),
        n_obs=n_obs,
        n_alts=n_alts,
        feature_names=[v.name for v in spec.variables],
        id_values=unique_ids,
    )
