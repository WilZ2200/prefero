# Copyright (C) 2026 Hengzhe Zhao. All rights reserved.
# Licensed under AGPL-3.0 or a commercial license. See LICENSE.
"""Tensor-native optimization, shared by CL, MXL and GMNL.

The parameter layout and likelihood belong to the calling model. Scaling by
the number of choice tasks keeps the stopping tolerance independent of N*T.
Callers can supply an objective in standardised parameter units.
"""
from dataclasses import dataclass
from typing import Callable
import time

import numpy as np
import torch


@dataclass
class OptimizationResult:
    x: np.ndarray
    fun: float
    success: bool
    message: str
    nit: int
    runtime_seconds: float


def synchronize(device: torch.device) -> None:
    if device.type == 'mps':
        torch.mps.synchronize()
    elif device.type == 'cuda':
        torch.cuda.synchronize(device)


def rescale_parameters(
    theta: np.ndarray,
    multipliers: np.ndarray,
    offsets: np.ndarray,
    positive: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map parameter units and return the diagonal of the transformation Jacobian.

    Multipliers act on standard deviations after softplus and on other
    coefficients directly. Offsets apply to the location of lognormal tastes.
    """
    output = theta * multipliers + offsets
    jacobian = multipliers.copy()
    raw_sd = theta[positive]
    scale = multipliers[positive]
    sd = np.maximum(np.logaddexp(0.0, raw_sd) * scale, np.finfo(float).tiny)
    transformed = sd + np.log(-np.expm1(-sd))
    output[positive] = transformed
    log_gradient = np.log(scale) - np.logaddexp(0.0, -raw_sd) + np.logaddexp(0.0, -transformed)
    jacobian[positive] = np.exp(log_gradient)
    return output, jacobian


def fit_lbfgs(objective: Callable, initial, *, device, maxiter: int,
              loss_scale: int = 1) -> OptimizationResult:
    if maxiter < 1 or loss_scale < 1:
        raise ValueError('maxiter and loss_scale must be positive.')
    device = torch.device(device)
    theta = torch.tensor(initial, dtype=torch.float32, device=device, requires_grad=True)
    if theta.ndim != 1 or not torch.isfinite(theta).all():
        raise ValueError('Initial parameters must be a finite one-dimensional vector.')
    # A per-task tolerance compatible with FP32 objective resolution.
    tolerance = 1e-4
    optimizer = torch.optim.LBFGS(
        [theta], max_iter=maxiter, history_size=100, line_search_fn='strong_wolfe',
        tolerance_grad=tolerance, tolerance_change=1e-9,
    )

    def closure():
        optimizer.zero_grad()
        loss = objective(theta) / loss_scale
        loss.backward()
        if not torch.isfinite(loss) or not torch.isfinite(theta.grad).all():
            raise FloatingPointError('Non-finite likelihood or gradient during estimation.')
        return loss

    synchronize(device)
    start = time.perf_counter()
    error = None
    try:
        optimizer.step(closure)
    except FloatingPointError as exc:
        error = str(exc)
    synchronize(device)
    runtime = time.perf_counter() - start
    # Evaluate the returned point, not the last line-search trial.
    optimizer.zero_grad()
    loss = objective(theta)
    (loss / loss_scale).backward()
    grad = float(theta.grad.detach().abs().max().cpu())
    value = float(loss.detach().cpu())
    state = optimizer.state[theta]
    iterations = int(state.get('n_iter', 0))
    valid = np.isfinite(value) and np.isfinite(grad)
    success = bool(valid and error is None and grad <= tolerance)
    if error or not valid:
        message = error or 'Non-finite likelihood or gradient at the returned point.'
    elif success:
        message = f'L-BFGS gradient tolerance reached (per-task gradient {grad:.2g}).'
    elif iterations >= maxiter:
        message = f'Iteration limit reached; per-task gradient {grad:.2g} exceeds {tolerance:g}.'
    elif state.get('func_evals', 0) >= optimizer.param_groups[0]['max_eval']:
        message = f'Evaluation limit reached; per-task gradient {grad:.2g}.'
    else:
        message = f'L-BFGS stopped with small steps or objective changes; per-task gradient {grad:.2g} exceeds {tolerance:g}.'
    return OptimizationResult(theta.detach().cpu().numpy().astype(np.float64),
                              value, success, message, iterations, runtime)
