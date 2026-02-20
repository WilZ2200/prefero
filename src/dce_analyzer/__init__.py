# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under dual license: AGPL-3.0 (open-source) or commercial. See LICENSE.

"""Choice analysis toolkit."""

from .apollo import APOLLO_DATASETS, load_apollo_long
from .config import ModelSpec, VariableSpec
from .format_converter import (
    ColumnInference,
    detect_format,
    infer_structure,
    normalize_choice_column,
    wide_to_long,
)
from .latent_class import LatentClassEstimator, LatentClassResult
from .pipeline import PipelineResult, estimate_dataframe
from .simulate import generate_simulated_dce
from .wtp import compute_wtp

__all__ = [
    "APOLLO_DATASETS",
    "ColumnInference",
    "LatentClassEstimator",
    "LatentClassResult",
    "ModelSpec",
    "PipelineResult",
    "VariableSpec",
    "compute_wtp",
    "detect_format",
    "estimate_dataframe",
    "generate_simulated_dce",
    "infer_structure",
    "load_apollo_long",
    "normalize_choice_column",
    "wide_to_long",
]
