from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


DistributionType = Literal["fixed", "normal", "lognormal"]


@dataclass(frozen=True)
class VariableSpec:
    """One variable used in the utility function."""

    name: str
    column: str
    distribution: DistributionType = "fixed"

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("VariableSpec.name cannot be empty.")
        if not self.column:
            raise ValueError("VariableSpec.column cannot be empty.")
        if self.distribution not in {"fixed", "normal", "lognormal"}:
            raise ValueError(
                f"Unsupported distribution '{self.distribution}'. "
                "Use one of: fixed, normal, lognormal."
            )


@dataclass(frozen=True)
class ModelSpec:
    """Data layout and variable config for a model run."""

    id_col: str
    task_col: str
    alt_col: str
    choice_col: str
    variables: list[VariableSpec]
    n_draws: int = 200
    n_classes: int = 2
    membership_cols: list[str] | None = None

    def __post_init__(self) -> None:
        core_cols = [self.id_col, self.task_col, self.alt_col, self.choice_col]
        if any(not c for c in core_cols):
            raise ValueError("id_col, task_col, alt_col, and choice_col must all be set.")
        if len(self.variables) == 0:
            raise ValueError("At least one variable is required in ModelSpec.variables.")
        if self.n_draws < 1:
            raise ValueError("n_draws must be >= 1.")
        if self.n_classes < 1:
            raise ValueError("n_classes must be >= 1.")


@dataclass(frozen=True)
class DummyCoding:
    """Dummy-coding specification for a single attribute."""

    column: str          # original column name in the data
    ref_level: object    # reference level (omitted baseline)

    def expand(self, df) -> tuple[list[str], dict]:
        """Return (list of dummy column names, {dummy_name: level}) for this column.

        Does NOT mutate *df*.
        """
        import pandas as pd

        unique_vals = sorted(df[self.column].dropna().unique())
        non_ref = [v for v in unique_vals if v != self.ref_level]
        names: list[str] = []
        mapping: dict[str, object] = {}
        for level in non_ref:
            name = f"{self.column}_L{level}"
            names.append(name)
            mapping[name] = level
        return names, mapping


@dataclass(frozen=True)
class HeterogeneityInteraction:
    """An attribute x demographic interaction term (legacy, kept for backward compat)."""

    attribute: str       # name of the attribute variable
    demographic_col: str  # name of the demographic column in the data


@dataclass(frozen=True)
class InteractionTerm:
    """An arbitrary N-way interaction term: product of specified columns."""

    columns: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.columns) < 2:
            raise ValueError("InteractionTerm requires at least 2 columns.")

    @property
    def name(self) -> str:
        return "_x_".join(self.columns)


@dataclass
class FullModelSpec:
    """Complete model specification -- one object captures everything."""

    # Data layout
    id_col: str
    task_col: str
    alt_col: str
    choice_col: str

    # Variable specifications
    variables: list[VariableSpec]

    # Model type
    model_type: str = "mixed"  # "conditional", "mixed", "gmnl", "latent_class"

    # Dummy coding: backend expands these columns into dummy variables
    dummy_codings: list[DummyCoding] = field(default_factory=list)

    # Interaction terms (N-way, any columns) -- works for ALL model types
    interactions: list[InteractionTerm] = field(default_factory=list)

    # Correlation structure
    correlated: bool = False
    correlation_groups: list[list[int]] | None = None

    # BWS
    bws_worst_col: str | None = None
    estimate_lambda_w: bool = True

    # Latent class
    n_classes: int = 2
    membership_cols: list[str] | None = None
    lc_method: str = "em"  # "em" or "direct"

    # Estimation settings
    n_draws: int = 200
    maxiter: int = 300
    seed: int = 123
    n_starts: int = 10
    custom_start: list[float] | None = None  # user-supplied initial theta vector

    def __post_init__(self) -> None:
        valid_types = {"conditional", "mixed", "gmnl", "latent_class"}
        if self.model_type not in valid_types:
            raise ValueError(
                f"model_type must be one of {valid_types}, got '{self.model_type}'."
            )
        core_cols = [self.id_col, self.task_col, self.alt_col, self.choice_col]
        if any(not c for c in core_cols):
            raise ValueError("id_col, task_col, alt_col, and choice_col must all be set.")
        if len(self.variables) == 0:
            raise ValueError("At least one variable is required.")
        if self.n_draws < 1:
            raise ValueError("n_draws must be >= 1.")

    def to_model_spec(self) -> ModelSpec:
        """Convert to the legacy ModelSpec for backward compatibility."""
        return ModelSpec(
            id_col=self.id_col,
            task_col=self.task_col,
            alt_col=self.alt_col,
            choice_col=self.choice_col,
            variables=list(self.variables),
            n_draws=self.n_draws,
            n_classes=self.n_classes,
            membership_cols=self.membership_cols,
        )
