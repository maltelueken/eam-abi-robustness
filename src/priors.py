import logging
import os
import pickle

import numpy as np
from scipy import special, stats

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


def truncated_normal_moments(loc: float, scale: float, lower: float = 0.0, moment: str = "m"):
    a = (lower - loc) / scale

    result = stats.truncnorm.stats(a, np.inf, loc=loc, scale=scale, moments=moment)

    if moment == "v":
        return np.sqrt(result)

    return result


def log_truncated_normal_moments(loc: float, scale: float, lower: float = 0.0, moment: str = "m", size=1000000):
    a = (lower - loc) / scale

    x = np.log(stats.truncnorm.rvs(a, np.inf, loc=loc, scale=scale, size=size))

    if moment == "v":
        return np.std(x)

    return x.mean()


def gamma_moments(shape: float, scale: float, moment: str):
    result = stats.gamma.stats(a=shape, scale=scale, moments=moment)

    if moment == "v":
        return np.sqrt(result)

    return result


def log_gamma_mean(shape, scale):
    return special.digamma(shape) + np.log(scale)


def log_gamma_sd(shape):
    return np.sqrt(special.polygamma(1, shape))


def log_gamma_moments(shape: float, scale: float, moment: str):
    if moment == "v":
        return log_gamma_sd(shape)

    return log_gamma_mean(shape, scale)


def rdm_prior_simple(
    batch_shape,
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
        drift_intercept_loc, drift_intercept_scale, size=batch_shape, random_state=rng
    )
    drift_slope = truncated_normal_rvs(
        drift_slope_loc, drift_slope_scale, size=batch_shape, random_state=rng
    )
    sd_true = rng.gamma(
        shape=sd_true_shape, scale=sd_true_scale, size=batch_shape
    )
    threshold = rng.gamma(
        shape=threshold_shape, scale=threshold_scale, size=batch_shape
    )
    t0 = truncated_normal_rvs(t0_loc, t0_scale, lower=t0_lower, size=batch_shape, random_state=rng)

    return {"v_intercept": drift_intercept, "v_slope": drift_slope, "s_true": sd_true, "b": threshold, "t0": t0}


def rdm_prior_multivariate(
    means,
    stds,
    corr_mat,
    size=1,
    rng=None,
):
    cov_mat = np.diag(stds).dot(corr_mat).dot(np.diag(stds))

    return np.exp(stats.multivariate_normal(means, cov_mat).rvs(random_state=rng, size=size))
