import logging
import os
import pickle

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def truncated_t_rvs(
    df: float,
    loc: float,
    scale: float,
    size: int = 1,
    random_state: int = None,
) -> np.ndarray:
    quantile_l = stats.t.cdf(0, df=df, loc=loc, scale=scale)

    if random_state is not None:
        probs = random_state.uniform(quantile_l, 1.0, size=size)
    else:
        probs = np.random.default_rng().uniform(quantile_l, 1.0, size=size)

    return stats.t.ppf(
        probs,
        df=df,
        loc=loc,
        scale=scale,
    )


def truncated_normal_rvs(
    loc: float,
    scale: float,
    size: int = 1,
    random_state: int = None,
) -> np.ndarray:
    quantile_l = stats.norm.cdf(0, loc=loc, scale=scale)

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
    drift_mean_loc,
    drift_mean_scale,
    drift_diff_loc,
    drift_diff_scale,
    sd_true_loc,
    sd_true_scale,
    threshold_shape,
    threshold_scale,
    t0_loc,
    t0_scale,
    rng=None,
):
    drift_mean = truncated_normal_rvs(
        drift_mean_loc, drift_mean_scale, random_state=rng
    )
    drift_diff = truncated_normal_rvs(
        drift_diff_loc, drift_diff_scale, random_state=rng
    )
    sd_true = truncated_normal_rvs(
        sd_true_loc, sd_true_scale, random_state=rng
    )
    threshold = rng.gamma(
        shape=threshold_shape, scale=threshold_scale
    )
    t0 = truncated_normal_rvs(t0_loc, t0_scale, random_state=rng)

    return np.hstack((drift_mean, drift_diff, sd_true, threshold, t0))
