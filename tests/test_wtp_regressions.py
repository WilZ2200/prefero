import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from dce_analyzer.model import EstimationResult
from dce_analyzer.wtp import compute_wtp


@pytest.mark.parametrize("correlated", [False, True])
def test_lognormal_wtp_uses_utility_means_and_raw_theta_delta_method(correlated):
    theta = np.array([-1.6, 0.1, -0.5, 0.2])
    vcov = np.eye(4) * 0.0001 + 0.00001
    sd = np.logaddexp(0.0, theta[2])
    rows = [
        ("beta_cost", "fixed", theta[0], 0),
        ("mu_attribute", "lognormal", theta[1], 1),
        ("sd_attribute", "lognormal", np.hypot(sd, theta[3]) if correlated else sd,
         -1 if correlated else 2),
    ]
    if correlated:
        rows += [
            ("chol_attribute_attribute", "cholesky", sd, 2),
            ("chol_attribute_other", "cholesky", theta[3], 3),
        ]
    estimates = pd.DataFrame(rows, columns=["parameter", "distribution", "estimate", "theta_index"])
    result = EstimationResult(
        True, "", 0, 0, 0, 4, 10, 10, 0, 0, estimates.sample(frac=1, random_state=17),
        vcov_matrix=vcov, random_param_names=["other", "attribute"] if correlated else None,
    )

    def wtp(params):
        variance = np.logaddexp(0.0, params[2])**2
        if correlated:
            variance += params[3]**2
        return -np.exp(params[1] + variance / 2) / params[0]

    gradient = approx_fprime(theta, wtp, 1e-6)
    row = compute_wtp(result, "cost").iloc[0]
    assert row.wtp_estimate == pytest.approx(wtp(theta))
    assert row.wtp_std_error == pytest.approx(np.sqrt(gradient @ vcov @ gradient), rel=1e-5)

    result.success = False
    with pytest.raises(ValueError, match="converged"):
        compute_wtp(result, "cost")
