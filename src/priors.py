"""Prior utilities."""

import numpy as np
from scipy import stats


def truncated_normal_rvs(
    loc: float,
    scale: float,
    lower: float = 0.0,
    size: int = 1,
    random_state: int = None,
) -> np.ndarray:
    """Sample from a truncated normal distribution with a lower bound."""
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
    """Sample from a custom prior for the racing diffusion model with two accumulators."""
    drift_intercept = truncated_normal_rvs(
        drift_intercept_loc, drift_intercept_scale, random_state=rng,
    )
    drift_slope = truncated_normal_rvs(
        drift_slope_loc, drift_slope_scale, random_state=rng,
    )
    sd_true = rng.gamma(shape=sd_true_shape, scale=sd_true_scale)
    threshold = rng.gamma(shape=threshold_shape, scale=threshold_scale)
    t0 = truncated_normal_rvs(t0_loc, t0_scale, lower=t0_lower, random_state=rng)

    return {
        "v_intercept": drift_intercept,
        "v_slope": drift_slope,
        "s_true": sd_true,
        "b": threshold,
        "t0": t0,
    }


def lba_prior_simple(
    drift_intercept_loc,
    drift_intercept_scale,
    drift_slope_loc,
    drift_slope_scale,
    sd_true_shape,
    sd_true_scale,
    sp_max_shape,
    sp_max_scale,
    threshold_shape,
    threshold_scale,
    t0_loc,
    t0_scale,
    t0_lower,
    rng,
):
    """Sample from a custom prior for the linear ballistic accumulator with two accumulators.

    Mirrors `rdm_prior_simple`, but with two start-point parameters in place of the RDM's single
    threshold: the start-point range `A`, and the threshold *gap* `B = b - A` (Heathcote & Love
    2012). Sampling the gap rather than the threshold makes `b > A` true by construction -- the
    LBA is degenerate otherwise, since a start point above threshold would mean an instant
    response -- and decorrelates the pair, which a diagonal MCMC mass matrix handles far better.
    `threshold_shape`/`threshold_scale` therefore parameterize `B`, not `b`; the hyperparameter
    names are kept so `experiment_*/fit_mcmc_gpu.py` needs no changes.

    Returned keys are ordered to match `conf/approximator/adapter/lba_simple.yaml` and the
    unconstrained position vector built by `lba_jax.make_lba_simple_logdensity`.
    """
    drift_intercept = truncated_normal_rvs(
        drift_intercept_loc, drift_intercept_scale, random_state=rng,
    )
    drift_slope = truncated_normal_rvs(
        drift_slope_loc, drift_slope_scale, random_state=rng,
    )
    sd_true = rng.gamma(shape=sd_true_shape, scale=sd_true_scale)
    sp_max = rng.gamma(shape=sp_max_shape, scale=sp_max_scale)
    sp_gap = rng.gamma(shape=threshold_shape, scale=threshold_scale)
    t0 = truncated_normal_rvs(t0_loc, t0_scale, lower=t0_lower, random_state=rng)

    return {
        "v_intercept": drift_intercept,
        "v_slope": drift_slope,
        "s_true": sd_true,
        "A": sp_max,
        "B": sp_gap,
        "t0": t0,
    }


def rdm_prior_sat(
    drift_intercept_loc,
    drift_intercept_scale,
    drift_slope_loc,
    drift_slope_scale,
    sd_true_shape,
    sd_true_scale,
    threshold_shape,
    threshold_scale,
    threshold_diff_shape,
    threshold_diff_scale,
    t0_loc,
    t0_scale,
    t0_lower,
    rng,
):
    """Prior for the racing diffusion model with a speed-vs-accuracy manipulation.

    `rdm_prior_simple` plus `b_diff`, the amount by which the threshold is raised under the
    accuracy instruction. Sampling the *difference* rather than a second threshold keeps
    `b_accuracy > b_speed` true by construction and leaves every parameter positive, so the
    same log-transform unconstrains the whole vector (cf. the `A`/`B` parameterization in
    `lba_prior_simple`).

    `threshold_diff_scale` is the hyperparameter study 3 shifts between training and test:
    it sets the expected size of the speed-accuracy effect, `threshold_diff_shape *
    threshold_diff_scale`.

    Returned keys are ordered to match `conf/approximator/adapter/rdm_sat.yaml` and the
    unconstrained position vector built by `rdm_jax.make_rdm_sat_logdensity`.
    """
    drift_intercept = truncated_normal_rvs(
        drift_intercept_loc, drift_intercept_scale, random_state=rng,
    )
    drift_slope = truncated_normal_rvs(
        drift_slope_loc, drift_slope_scale, random_state=rng,
    )
    sd_true = rng.gamma(shape=sd_true_shape, scale=sd_true_scale)
    threshold = rng.gamma(shape=threshold_shape, scale=threshold_scale)
    threshold_diff = rng.gamma(shape=threshold_diff_shape, scale=threshold_diff_scale)
    t0 = truncated_normal_rvs(t0_loc, t0_scale, lower=t0_lower, random_state=rng)

    return {
        "v_intercept": drift_intercept,
        "v_slope": drift_slope,
        "s_true": sd_true,
        "b": threshold,
        "b_diff": threshold_diff,
        "t0": t0,
    }
