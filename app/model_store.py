"""Per-user model persistence via HF Dataset.

Each user's saved models are stored as ``models/{username}.json`` in the
private HF Dataset repo ``Wil2200/prefero-data``.  Models survive container
restarts **and** redeployments.

Public API
----------
- save_model(username, model_entry) -> bool
- load_models(username) -> list[dict]
- delete_saved_model(username, index) -> bool
- serialize_model_entry(entry) -> dict
- deserialize_model_entry(data) -> dict
"""

from __future__ import annotations

import json
import logging
import math
import os
import tempfile
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _sanitize_float(v):
    """Convert NaN / Inf to None so JSON stays spec-compliant."""
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    return v


def _sanitize_list(lst):
    """Recursively sanitize a (possibly nested) list of floats."""
    if lst is None:
        return None
    out = []
    for item in lst:
        if isinstance(item, list):
            out.append(_sanitize_list(item))
        elif isinstance(item, float):
            out.append(_sanitize_float(item))
        else:
            out.append(item)
    return out

_HF_DATASET_REPO = "Wil2200/prefero-data"
_MAX_MODELS_PER_USER = 10
_lock = threading.Lock()


# ---------------------------------------------------------------------------
# HF helpers (mirrors community_db.py pattern)
# ---------------------------------------------------------------------------

def _hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def _load_user_models_from_hf(username: str) -> list[dict]:
    """Download ``models/{username}.json`` from the HF Dataset repo."""
    token = _hf_token()
    if not token:
        logger.debug("No HF token -- skipping model load for %s", username)
        return []
    try:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(
            repo_id=_HF_DATASET_REPO,
            filename=f"models/{username}.json",
            repo_type="dataset",
            token=token,
        )
        with open(path) as f:
            data = json.load(f)
        return data.get("models", [])
    except Exception as exc:
        logger.debug("Failed to load models for %s from HF: %s", username, exc)
        return []


def _save_user_models_to_hf(username: str, models: list[dict]) -> None:
    """Upload ``models/{username}.json`` to the HF Dataset repo."""
    token = _hf_token()
    if not token:
        return
    try:
        from huggingface_hub import HfApi

        data = {"models": models}
        tmp = os.path.join(tempfile.gettempdir(), f"prefero_models_{username}.json")
        with open(tmp, "w") as f:
            json.dump(data, f)

        api = HfApi(token=token)
        api.upload_file(
            path_or_fileobj=tmp,
            path_in_repo=f"models/{username}.json",
            repo_id=_HF_DATASET_REPO,
            repo_type="dataset",
        )
        logger.info("Saved %d models for user %s to HF", len(models), username)
    except Exception as exc:
        logger.warning("Failed to save models for %s to HF: %s", username, exc)


# ---------------------------------------------------------------------------
# Serialization helpers
# ---------------------------------------------------------------------------

def _serialize_dataframe(df: pd.DataFrame | None) -> dict | None:
    if df is None:
        return None
    data = {}
    for col in df.columns:
        vals = df[col].tolist()
        data[col] = [_sanitize_float(v) if isinstance(v, float) else v for v in vals]
    return {
        "columns": list(df.columns),
        "data": data,
    }


def _deserialize_dataframe(d: dict | None) -> pd.DataFrame | None:
    if d is None:
        return None
    return pd.DataFrame(d["data"], columns=d["columns"])


def _serialize_ndarray(arr: np.ndarray | None) -> list | None:
    if arr is None:
        return None
    return _sanitize_list(arr.tolist())


def _deserialize_ndarray(lst: list | None) -> np.ndarray | None:
    if lst is None:
        return None
    return np.array(lst)


def _serialize_variable_spec(vs) -> dict:
    return {"name": vs.name, "column": vs.column, "distribution": vs.distribution}


def _serialize_dummy_coding(dc) -> dict:
    ref = dc.ref_level
    # ref_level can be int/str/float -- store as-is (JSON-safe for primitives)
    return {"column": dc.column, "ref_level": ref}


def _serialize_interaction_term(it) -> dict:
    return {"columns": list(it.columns)}


def _serialize_model_spec(spec) -> dict | None:
    if spec is None:
        return None
    from dce_analyzer.config import ModelSpec
    return {
        "id_col": spec.id_col,
        "task_col": spec.task_col,
        "alt_col": spec.alt_col,
        "choice_col": spec.choice_col,
        "variables": [_serialize_variable_spec(v) for v in spec.variables],
        "n_draws": spec.n_draws,
        "n_classes": getattr(spec, "n_classes", 2),
        "membership_cols": getattr(spec, "membership_cols", None),
    }


def _serialize_full_model_spec(spec) -> dict | None:
    if spec is None:
        return None
    return {
        "id_col": spec.id_col,
        "task_col": spec.task_col,
        "alt_col": spec.alt_col,
        "choice_col": spec.choice_col,
        "variables": [_serialize_variable_spec(v) for v in spec.variables],
        "model_type": spec.model_type,
        "dummy_codings": [_serialize_dummy_coding(dc) for dc in spec.dummy_codings],
        "interactions": [_serialize_interaction_term(it) for it in spec.interactions],
        "correlated": spec.correlated,
        "correlation_groups": spec.correlation_groups,
        "bws_worst_col": spec.bws_worst_col,
        "estimate_lambda_w": spec.estimate_lambda_w,
        "gmnl_variant": spec.gmnl_variant,
        "n_classes": spec.n_classes,
        "membership_cols": spec.membership_cols,
        "lc_method": spec.lc_method,
        "n_draws": spec.n_draws,
        "maxiter": spec.maxiter,
        "seed": spec.seed,
        "n_starts": spec.n_starts,
        "custom_start": spec.custom_start,
    }


def _deserialize_variable_spec(d: dict):
    from dce_analyzer.config import VariableSpec
    return VariableSpec(name=d["name"], column=d["column"], distribution=d.get("distribution", "fixed"))


def _deserialize_model_spec(d: dict | None):
    if d is None:
        return None
    from dce_analyzer.config import ModelSpec
    return ModelSpec(
        id_col=d["id_col"],
        task_col=d["task_col"],
        alt_col=d["alt_col"],
        choice_col=d["choice_col"],
        variables=[_deserialize_variable_spec(v) for v in d["variables"]],
        n_draws=d.get("n_draws", 200),
        n_classes=d.get("n_classes", 2),
        membership_cols=d.get("membership_cols"),
    )


def _deserialize_full_model_spec(d: dict | None):
    if d is None:
        return None
    from dce_analyzer.config import FullModelSpec, DummyCoding, InteractionTerm
    return FullModelSpec(
        id_col=d["id_col"],
        task_col=d["task_col"],
        alt_col=d["alt_col"],
        choice_col=d["choice_col"],
        variables=[_deserialize_variable_spec(v) for v in d["variables"]],
        model_type=d.get("model_type", "mixed"),
        dummy_codings=[DummyCoding(column=dc["column"], ref_level=dc["ref_level"]) for dc in d.get("dummy_codings", [])],
        interactions=[InteractionTerm(columns=tuple(it["columns"])) for it in d.get("interactions", [])],
        correlated=d.get("correlated", False),
        correlation_groups=d.get("correlation_groups"),
        bws_worst_col=d.get("bws_worst_col"),
        estimate_lambda_w=d.get("estimate_lambda_w", True),
        gmnl_variant=d.get("gmnl_variant", "general"),
        n_classes=d.get("n_classes", 2),
        membership_cols=d.get("membership_cols"),
        lc_method=d.get("lc_method", "em"),
        n_draws=d.get("n_draws", 200),
        maxiter=d.get("maxiter", 300),
        seed=d.get("seed", 123),
        n_starts=d.get("n_starts", 10),
        custom_start=d.get("custom_start"),
    )


def _serialize_estimation_result(est) -> dict:
    """Serialize an EstimationResult to a JSON-safe dict."""
    return {
        "_type": "EstimationResult",
        "success": est.success,
        "message": est.message,
        "log_likelihood": _sanitize_float(est.log_likelihood),
        "aic": _sanitize_float(est.aic),
        "bic": _sanitize_float(est.bic),
        "n_parameters": est.n_parameters,
        "n_observations": est.n_observations,
        "n_individuals": est.n_individuals,
        "optimizer_iterations": est.optimizer_iterations,
        "runtime_seconds": _sanitize_float(est.runtime_seconds),
        "estimates": _serialize_dataframe(est.estimates),
        "vcov_matrix": _serialize_ndarray(est.vcov_matrix),
        "covariance_matrix": _serialize_ndarray(est.covariance_matrix),
        "correlation_matrix": _serialize_ndarray(est.correlation_matrix),
        "random_param_names": est.random_param_names,
        "covariance_se": _serialize_ndarray(est.covariance_se),
        "correlation_se": _serialize_ndarray(est.correlation_se),
        "correlation_test": _serialize_dataframe(est.correlation_test),
        "raw_theta": _serialize_ndarray(est.raw_theta),
    }


def _serialize_latent_class_result(est) -> dict:
    """Serialize a LatentClassResult to a JSON-safe dict.

    ``posterior_probs`` is skipped (too large) and restored as an empty
    DataFrame on deserialization.
    """
    return {
        "_type": "LatentClassResult",
        "success": est.success,
        "message": est.message,
        "log_likelihood": _sanitize_float(est.log_likelihood),
        "aic": _sanitize_float(est.aic),
        "bic": _sanitize_float(est.bic),
        "n_parameters": est.n_parameters,
        "n_observations": est.n_observations,
        "n_individuals": est.n_individuals,
        "optimizer_iterations": est.optimizer_iterations,
        "runtime_seconds": _sanitize_float(est.runtime_seconds),
        "estimates": _serialize_dataframe(est.estimates),
        "n_classes": est.n_classes,
        "class_probabilities": _sanitize_list(list(est.class_probabilities)),
        "class_estimates": _serialize_dataframe(est.class_estimates),
        # posterior_probs intentionally skipped
        "vcov_matrix": _serialize_ndarray(est.vcov_matrix),
        "membership_estimates": _serialize_dataframe(est.membership_estimates),
        "n_starts_attempted": est.n_starts_attempted,
        "n_starts_succeeded": est.n_starts_succeeded,
        "all_start_lls": _sanitize_list(list(est.all_start_lls)),
        "best_start_index": est.best_start_index,
        "optimizer_method": est.optimizer_method,
        "em_iterations": est.em_iterations,
        "em_ll_history": _sanitize_list(list(est.em_ll_history)),
        "em_converged": est.em_converged,
        "raw_theta": _serialize_ndarray(est.raw_theta),
    }


def _deserialize_estimation_result(d: dict):
    from dce_analyzer.model import EstimationResult
    return EstimationResult(
        success=d["success"],
        message=d["message"],
        log_likelihood=d["log_likelihood"],
        aic=d["aic"],
        bic=d["bic"],
        n_parameters=d["n_parameters"],
        n_observations=d["n_observations"],
        n_individuals=d["n_individuals"],
        optimizer_iterations=d["optimizer_iterations"],
        runtime_seconds=d["runtime_seconds"],
        estimates=_deserialize_dataframe(d["estimates"]),
        vcov_matrix=_deserialize_ndarray(d.get("vcov_matrix")),
        covariance_matrix=_deserialize_ndarray(d.get("covariance_matrix")),
        correlation_matrix=_deserialize_ndarray(d.get("correlation_matrix")),
        random_param_names=d.get("random_param_names"),
        covariance_se=_deserialize_ndarray(d.get("covariance_se")),
        correlation_se=_deserialize_ndarray(d.get("correlation_se")),
        correlation_test=_deserialize_dataframe(d.get("correlation_test")),
        raw_theta=_deserialize_ndarray(d.get("raw_theta")),
    )


def _deserialize_latent_class_result(d: dict):
    from dce_analyzer.latent_class import LatentClassResult
    return LatentClassResult(
        success=d["success"],
        message=d["message"],
        log_likelihood=d["log_likelihood"],
        aic=d["aic"],
        bic=d["bic"],
        n_parameters=d["n_parameters"],
        n_observations=d["n_observations"],
        n_individuals=d["n_individuals"],
        optimizer_iterations=d["optimizer_iterations"],
        runtime_seconds=d["runtime_seconds"],
        estimates=_deserialize_dataframe(d["estimates"]),
        n_classes=d["n_classes"],
        class_probabilities=d["class_probabilities"],
        class_estimates=_deserialize_dataframe(d["class_estimates"]),
        posterior_probs=pd.DataFrame(),  # skipped on serialize
        vcov_matrix=_deserialize_ndarray(d.get("vcov_matrix")),
        membership_estimates=_deserialize_dataframe(d.get("membership_estimates")),
        n_starts_attempted=d.get("n_starts_attempted", 0),
        n_starts_succeeded=d.get("n_starts_succeeded", 0),
        all_start_lls=d.get("all_start_lls", []),
        best_start_index=d.get("best_start_index", -1),
        optimizer_method=d.get("optimizer_method", "L-BFGS-B"),
        em_iterations=d.get("em_iterations", 0),
        em_ll_history=d.get("em_ll_history", []),
        em_converged=d.get("em_converged", False),
        raw_theta=_deserialize_ndarray(d.get("raw_theta")),
    )


def _serialize_estimation(est) -> dict:
    """Route to the correct serializer based on type."""
    from dce_analyzer.latent_class import LatentClassResult
    if isinstance(est, LatentClassResult):
        return _serialize_latent_class_result(est)
    return _serialize_estimation_result(est)


def _deserialize_estimation(d: dict):
    """Route to the correct deserializer based on ``_type`` tag."""
    if d.get("_type") == "LatentClassResult":
        return _deserialize_latent_class_result(d)
    return _deserialize_estimation_result(d)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def serialize_model_entry(entry: dict) -> dict:
    """Convert a session_state model_history entry to a JSON-serializable dict.

    Expected keys in *entry*: label, model_type, spec, full_spec, estimation.
    """
    est = entry["estimation"]
    return {
        "label": entry.get("label", "unnamed"),
        "model_type": entry.get("model_type", "mixed"),
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "stats": {
            "log_likelihood": _sanitize_float(est.log_likelihood),
            "aic": _sanitize_float(est.aic),
            "bic": _sanitize_float(est.bic),
            "n_parameters": est.n_parameters,
            "n_observations": est.n_observations,
            "n_individuals": est.n_individuals,
            "runtime_seconds": _sanitize_float(est.runtime_seconds),
        },
        "spec": _serialize_model_spec(entry.get("spec")),
        "full_spec": _serialize_full_model_spec(entry.get("full_spec")),
        "estimation_data": _serialize_estimation(est),
    }


def deserialize_model_entry(data: dict) -> dict:
    """Reconstruct a model_history-compatible dict from stored JSON.

    Returns a dict with keys: label, model_type, spec, full_spec, estimation.
    """
    return {
        "label": data.get("label", "unnamed"),
        "model_type": data.get("model_type", "mixed"),
        "saved_at": data.get("saved_at"),
        "spec": _deserialize_model_spec(data.get("spec")),
        "full_spec": _deserialize_full_model_spec(data.get("full_spec")),
        "estimation": _deserialize_estimation(data["estimation_data"]),
    }


def save_model(username: str, model_entry: dict) -> bool:
    """Save a model to the user's profile.

    Enforces a maximum of ``_MAX_MODELS_PER_USER`` models.
    Returns ``True`` on success.
    """
    with _lock:
        models = _load_user_models_from_hf(username)
        if len(models) >= _MAX_MODELS_PER_USER:
            logger.warning(
                "User %s already has %d models (limit %d)",
                username, len(models), _MAX_MODELS_PER_USER,
            )
            return False
        serialized = serialize_model_entry(model_entry)
        models.append(serialized)
        _save_user_models_to_hf(username, models)
        return True


def load_models(username: str) -> list[dict]:
    """Load all saved models for a user.

    Returns a list of deserialized model_history-compatible dicts.
    """
    with _lock:
        raw_models = _load_user_models_from_hf(username)
    result = []
    for m in raw_models:
        try:
            result.append(deserialize_model_entry(m))
        except Exception as exc:
            logger.warning("Skipping corrupt model for %s: %s", username, exc)
    return result


def delete_saved_model(username: str, index: int) -> bool:
    """Delete a saved model by its index (0-based).

    Returns ``True`` on success.
    """
    with _lock:
        models = _load_user_models_from_hf(username)
        if index < 0 or index >= len(models):
            logger.warning("Invalid index %d for user %s (has %d models)", index, username, len(models))
            return False
        models.pop(index)
        _save_user_models_to_hf(username, models)
        return True
