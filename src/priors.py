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


def probit_beta_moments(shape: float, scale: float, moment: str, size=1000000):
    x = stats.norm.ppf(stats.beta.rvs(shape, scale, size=size))

    if moment == "v":
        return np.std(x)

    return x.mean()


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


def rdmc_prior_simple(
    batch_shape,
    drift_c_intercept_loc,
    drift_c_intercept_scale,
    drift_c_slope_loc,
    drift_c_slope_scale,
    amp_shape,
    amp_scale,
    tau_shape,
    tau_scale,
    sd_true_shape,
    sd_true_scale,
    threshold_shape,
    threshold_scale,
    t0_loc,
    t0_scale,
    rng,
):
    drift_c_intercept = truncated_normal_rvs(drift_c_intercept_loc, drift_c_intercept_scale, size=batch_shape, random_state=rng)
    drift_c_slope = truncated_normal_rvs(drift_c_slope_loc, drift_c_slope_scale, size=batch_shape, random_state=rng)
    amp = rng.gamma(shape=amp_shape, scale=amp_scale, size=batch_shape)
    tau = rng.gamma(shape=tau_shape, scale=tau_scale, size=batch_shape)
    s_true = rng.gamma(shape=sd_true_shape, scale=sd_true_scale, size=batch_shape)
    b = rng.gamma(shape=threshold_shape, scale=threshold_scale, size=batch_shape)
    t0 = truncated_normal_rvs(t0_loc, t0_scale, size=batch_shape, random_state=rng)
    s_false = 1

    scale = 1000

    s_true = s_true / np.sqrt(scale)
    s_false = s_false / np.sqrt(scale)
    drift_c_intercept = drift_c_intercept / scale
    drift_c_slope = drift_c_slope / scale
    tau = tau * scale

    return {"v_c_intercept": drift_c_intercept, "v_c_slope": drift_c_slope, "amp": amp, "tau": tau, "s_true": s_true, "b": b, "t0": t0, "s_false": s_false}


def rrdmc_prior_simple(
    batch_shape,
    drift_c_intercept_loc,
    drift_c_intercept_scale,
    drift_c_slope_loc,
    drift_c_slope_scale,
    drift_a_intercept_loc,
    drift_a_intercept_scale,
    drift_a_slope_loc,
    drift_a_slope_scale,
    initial_shape,
    initial_scale,
    decay_shape,
    decay_scale,
    sd_true_shape,
    sd_true_scale,
    threshold_shape,
    threshold_scale,
    t0_loc,
    t0_scale,
    rng,
):
    drift_c_intercept = truncated_normal_rvs(drift_c_intercept_loc, drift_c_intercept_scale, size=batch_shape, random_state=rng)
    drift_c_slope = truncated_normal_rvs(drift_c_slope_loc, drift_c_slope_scale, size=batch_shape, random_state=rng)
    drift_a_intercept = truncated_normal_rvs(drift_a_intercept_loc, drift_a_intercept_scale, size=batch_shape, random_state=rng)
    drift_a_slope = truncated_normal_rvs(drift_a_slope_loc, drift_a_slope_scale, size=batch_shape, random_state=rng)
    A0 = rng.beta(initial_shape, initial_scale, size=batch_shape)
    k = rng.gamma(shape=decay_shape, scale=decay_scale, size=batch_shape)
    s_true = rng.gamma(shape=sd_true_shape, scale=sd_true_scale, size=batch_shape)
    b = rng.gamma(shape=threshold_shape, scale=threshold_scale, size=batch_shape)
    t0 = truncated_normal_rvs(t0_loc, t0_scale, size=batch_shape, random_state=rng)

    s_false = 1

    scale = 1000

    s_true = s_true / np.sqrt(scale)
    s_false = s_false / np.sqrt(scale)
    drift_c_intercept = drift_c_intercept / scale
    drift_c_slope = drift_c_slope / scale
    drift_a_intercept = drift_a_intercept / scale
    drift_a_slope = drift_a_slope / scale
    k = k / scale

    return {"v_c_intercept": drift_c_intercept, "v_c_slope": drift_c_slope, "v_a_intercept": drift_a_intercept, "v_a_slope": drift_a_slope, "A0": A0, "k": k, "s_true": s_true, "b": b, "t0": t0, "s_false": s_false}
