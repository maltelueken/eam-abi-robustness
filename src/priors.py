import logging
import os
import pickle

import numpy as np
from scipy import special, stats

logger = logging.getLogger(__name__)


def truncated_normal_rvs(
    loc: float,
    scale: float,
    lower: float = 0.0,
    size: int = 1,
    random_state: int = None,
) -> np.ndarray:
    quantile_l = stats.norm.cdf(lower, loc=loc, scale=scale)

    if random_state is not None:
        probs = random_state.uniform(quantile_l, 1.0, size=size)
    else:
        probs = np.random.default_rng().uniform(quantile_l, 1.0, size=size)

    return stats.norm.ppf(
        probs,
        loc=loc,
        scale=scale,
    )


def rdm_prior_simple(
    drift_intercept_loc,
    drift_intercept_scale,
    drift_slope_loc,
    drift_slope_scale,
    sd_true_shape,
    sd_true_scale,
    threshold_shape,
    threshold_scale,
    t0_loc,
    t0_scale,
    t0_lower,
    rng,
):
    drift_intercept = truncated_normal_rvs(
        drift_intercept_loc, drift_intercept_scale, random_state=rng
    )
    drift_slope = truncated_normal_rvs(
        drift_slope_loc, drift_slope_scale, random_state=rng
    )
    sd_true = rng.gamma(
        shape=sd_true_shape, scale=sd_true_scale
    )
    threshold = rng.gamma(
        shape=threshold_shape, scale=threshold_scale
    )
    t0 = truncated_normal_rvs(t0_loc, t0_scale, lower=t0_lower, random_state=rng)

    return {"v_intercept": drift_intercept, "v_slope": drift_slope, "s_true": sd_true, "b": threshold, "t0": t0}
