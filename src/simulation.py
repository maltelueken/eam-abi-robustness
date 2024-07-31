import numpy as np


def rdm_experiment_simple(
    theta, num_obs, rng
):
    """Simulates data from a single subject in a multi-alternative response times experiment."""
    num_accumulators = 2
    v_mean = theta[0]
    v_diff = theta[1]
    s_true = theta[2]
    s_false = 1.0
    a = theta[3]
    t0 = theta[4]

    v = np.hstack([v_mean + 0.5*v_diff, v_mean - 0.5*v_diff])
    s = np.hstack([s_true, s_false])

    mu = a / v
    lam = (a / s) ** 2

    # First passage time
    fpt = np.zeros((num_accumulators, num_obs))
    
    for i in range(num_accumulators):
        fpt[i, :] = rng.wald(mu[i], lam[i], size=num_obs)

    resp = fpt.argmin(axis=0)
    rt = fpt.min(axis=0) + t0

    return np.c_[rt, resp]
