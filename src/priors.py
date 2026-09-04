"""Prior utilities.

Every prior here is a *batched* sampler (`is_batched: true` in
`conf/simulator/prior_simulator/*.yaml`): one call draws the whole batch, returning an array of
shape `batch_shape` per parameter. `simulation.CustomSimulator` then lifts those to `(batch, 1)`,
the same shape BayesFlow's per-element path used to produce.

Hyperparameters may arrive as a scalar from the yaml, as a `(batch,)` array from a hierarchical
model's meta simulator, or as a 0-d array pinned by a test case; numpy broadcasts all three
against `size=batch_shape` without special handling.
"""

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
    """Sample from a custom prior for the racing diffusion model with two accumulators."""
    drift_intercept = truncated_normal_rvs(
        drift_intercept_loc, drift_intercept_scale, size=batch_shape, random_state=rng,
    )
    drift_slope = truncated_normal_rvs(
        drift_slope_loc, drift_slope_scale, size=batch_shape, random_state=rng,
    )
    sd_true = rng.gamma(shape=sd_true_shape, scale=sd_true_scale, size=batch_shape)
    threshold = rng.gamma(shape=threshold_shape, scale=threshold_scale, size=batch_shape)
    t0 = truncated_normal_rvs(t0_loc, t0_scale, lower=t0_lower, size=batch_shape, random_state=rng)

    return {
        "v_intercept": drift_intercept,
        "v_slope": drift_slope,
        "s_true": sd_true,
        "b": threshold,
        "t0": t0,
    }


def lba_prior_simple(
    batch_shape,
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
        drift_intercept_loc, drift_intercept_scale, size=batch_shape, random_state=rng,
    )
    drift_slope = truncated_normal_rvs(
        drift_slope_loc, drift_slope_scale, size=batch_shape, random_state=rng,
    )
    sd_true = rng.gamma(shape=sd_true_shape, scale=sd_true_scale, size=batch_shape)
    sp_max = rng.gamma(shape=sp_max_shape, scale=sp_max_scale, size=batch_shape)
    sp_gap = rng.gamma(shape=threshold_shape, scale=threshold_scale, size=batch_shape)
    t0 = truncated_normal_rvs(t0_loc, t0_scale, lower=t0_lower, size=batch_shape, random_state=rng)

    return {
        "v_intercept": drift_intercept,
        "v_slope": drift_slope,
        "s_true": sd_true,
        "A": sp_max,
        "B": sp_gap,
        "t0": t0,
    }


def rdm_prior_sat(
    batch_shape,
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
        drift_intercept_loc, drift_intercept_scale, size=batch_shape, random_state=rng,
    )
    drift_slope = truncated_normal_rvs(
        drift_slope_loc, drift_slope_scale, size=batch_shape, random_state=rng,
    )
    sd_true = rng.gamma(shape=sd_true_shape, scale=sd_true_scale, size=batch_shape)
    threshold = rng.gamma(shape=threshold_shape, scale=threshold_scale, size=batch_shape)
    threshold_diff = rng.gamma(shape=threshold_diff_shape, scale=threshold_diff_scale, size=batch_shape)
    t0 = truncated_normal_rvs(t0_loc, t0_scale, lower=t0_lower, size=batch_shape, random_state=rng)

    return {
        "v_intercept": drift_intercept,
        "v_slope": drift_slope,
        "s_true": sd_true,
        "b": threshold,
        "b_diff": threshold_diff,
        "t0": t0,
    }


def lba_prior_sat(
    batch_shape,
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
    threshold_diff_shape,
    threshold_diff_scale,
    t0_loc,
    t0_scale,
    t0_lower,
    rng,
):
    """Prior for the linear ballistic accumulator with a speed-vs-accuracy manipulation.

    `lba_prior_simple` plus `B_diff`, the amount by which the accuracy instruction raises the
    threshold *gap* -- so the threshold is `A + B` under speed and `A + B + B_diff` under
    accuracy, and `b > A` still holds structurally in both conditions. The RDM counterpart is
    `rdm_prior_sat`; `threshold_diff_scale` is the hyperparameter study 3 shifts between
    training and test in both models.

    Returned keys are ordered to match `conf/approximator/adapter/lba_sat.yaml` and the
    unconstrained position vector built by `lba_jax.make_lba_sat_logdensity`.
    """
    drift_intercept = truncated_normal_rvs(
        drift_intercept_loc, drift_intercept_scale, size=batch_shape, random_state=rng,
    )
    drift_slope = truncated_normal_rvs(
        drift_slope_loc, drift_slope_scale, size=batch_shape, random_state=rng,
    )
    sd_true = rng.gamma(shape=sd_true_shape, scale=sd_true_scale, size=batch_shape)
    sp_max = rng.gamma(shape=sp_max_shape, scale=sp_max_scale, size=batch_shape)
    sp_gap = rng.gamma(shape=threshold_shape, scale=threshold_scale, size=batch_shape)
    sp_gap_diff = rng.gamma(shape=threshold_diff_shape, scale=threshold_diff_scale, size=batch_shape)
    t0 = truncated_normal_rvs(t0_loc, t0_scale, lower=t0_lower, size=batch_shape, random_state=rng)

    return {
        "v_intercept": drift_intercept,
        "v_slope": drift_slope,
        "s_true": sd_true,
        "A": sp_max,
        "B": sp_gap,
        "B_diff": sp_gap_diff,
        "t0": t0,
    }
