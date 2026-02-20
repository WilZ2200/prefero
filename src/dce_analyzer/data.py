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


def validate_long_format(df: pd.DataFrame, spec: ModelSpec) -> None:
    """Validate core long-format assumptions."""
    required_cols = {
        spec.id_col,
        spec.task_col,
        spec.alt_col,
        spec.choice_col,
        *[v.column for v in spec.variables],
    }
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    key_cols = [spec.id_col, spec.task_col, spec.alt_col]
    if df.duplicated(subset=key_cols).any():
        dup_rows = int(df.duplicated(subset=key_cols).sum())
        raise ValueError(
            f"Found {dup_rows} duplicated (id, task, alt) rows. "
            "Each alternative in each task should appear once."
        )

    group_sizes = df.groupby([spec.id_col, spec.task_col]).size()
    if group_sizes.empty:
        raise ValueError("Input dataframe is empty after grouping by id and task.")
    if (group_sizes < 2).any():
        raise ValueError("Each (id, task) must have at least two alternatives.")
    if group_sizes.nunique() != 1:
        raise ValueError(
            "Each (id, task) must have the same number of alternatives. "
            "Variable-size choice sets are not supported in this baseline."
        )


def _choice_indices(choice_matrix: np.ndarray, alt_matrix: np.ndarray) -> np.ndarray:
    """Convert either one-hot choices or chosen-alt labels to index targets."""
    unique_vals = np.unique(choice_matrix)
    # One-hot / binary indicator format.
    if np.isin(unique_vals, [0, 1]).all():
        row_sums = choice_matrix.sum(axis=1)
        if not np.allclose(row_sums, 1.0):
            bad = int(np.where(~np.isclose(row_sums, 1.0))[0][0])
            raise ValueError(
                f"Choice indicator rows must sum to 1. Row {bad} sums to {row_sums[bad]}."
            )
        return np.argmax(choice_matrix, axis=1).astype(np.int64)

    # Label format: each row in a task repeats the same chosen alternative code.
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
    return np.argmax(matches, axis=1).astype(np.int64)


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
    validate_long_format(df, spec)

    if device is None:
        device = get_best_device()

    sort_cols = [spec.id_col, spec.task_col, spec.alt_col]
    work = df.sort_values(sort_cols).reset_index(drop=True)

    group_cols = [spec.id_col, spec.task_col]
    n_obs = int(work.groupby(group_cols).ngroups)
    n_alts = int(work.groupby(group_cols).size().iloc[0])
    n_vars = len(spec.variables)

    feature_cols = [v.column for v in spec.variables]
    X_flat = work.loc[:, feature_cols].astype(float).to_numpy(dtype=np.float32)
    X = X_flat.reshape(n_obs, n_alts, n_vars)

    choice_mat = (
        work.loc[:, spec.choice_col]
        .to_numpy(dtype=work.loc[:, spec.choice_col].dtype)
        .reshape(n_obs, n_alts)
    )
    alt_mat = work.loc[:, spec.alt_col].to_numpy().reshape(n_obs, n_alts)
    y = _choice_indices(choice_mat, alt_mat)

    task_table = work.loc[:, group_cols].drop_duplicates()
    obs_ids = task_table.loc[:, spec.id_col].to_numpy()
    unique_ids, panel_idx = np.unique(obs_ids, return_inverse=True)

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
