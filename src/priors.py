"""Prior utilities.

Every prior here is a *batched* sampler (`is_batched: true` in
`conf/simulator/prior_simulator/*.yaml`): one call draws the whole batch, returning an array of
shape `batch_shape` per parameter. `simulation.CustomSimulator` then lifts those to `(batch, 1)`,
the same shape BayesFlow's per-element path used to produce.

The draws are made with `jax.random`, from the `rdm_jax.SplittableKey` passed as `rng` -- the
same stateful key the experiment simulators use, on a stream of its own (see the prior
configs) -- and handed back as numpy arrays, since everything downstream of the simulator
(rejection sampling, the adapter, `data.save_dataset`) indexes and concatenates them as such.
They come out in JAX's default float width: 64-bit wherever `mcmc` has been imported, 32-bit
in a training process, which is the width the experiment simulator then runs in anyway.

Hyperparameters may arrive as a scalar from the yaml, as a `(batch,)` array from a hierarchical
model's meta simulator, or as a 0-d array pinned by a test case; `jax.random` broadcasts all
three against `shape=batch_shape` without special handling.
"""

import jax
import jax.numpy as jnp
import numpy as np


def _float(value):
    return jnp.asarray(value, dtype=jnp.result_type(float))


def truncated_normal_rvs(key, loc, scale, lower=0.0, size=()):
    """Sample from a normal distribution truncated below at `lower`.

    `jax.random.truncated_normal` draws on the standardized scale by inverting the CDF between
    the two bounds, the same construction the numpy version used. Its precision degrades only
    when `lower` sits far into the *upper* tail of the untruncated normal, which no prior here
    comes near: the bound is at least 1.4 sd below the location everywhere.
    """
    loc, scale = _float(loc), _float(scale)
    standard = jax.random.truncated_normal(
        key, (_float(lower) - loc) / scale, jnp.inf, shape=size, dtype=loc.dtype,
    )
    return loc + scale * standard


def gamma_rvs(key, shape, scale, size=()):
    """Sample a Gamma by shape and scale, the parameterization the prior configs use."""
    return _float(scale) * jax.random.gamma(key, _float(shape), shape=size, dtype=_float(shape).dtype)


def gamma_mean_sd_rvs(key, loc, sd, size=()):
    """Sample a Gamma parameterized by its *mean* and *standard deviation*.

    Study 3 sweeps how large a speed-accuracy effect the network was trained to expect. Sweeping
    a Gamma's `scale` at fixed `shape` -- what this used to do -- moves its mean and its sd
    together (`sd = mean / sqrt(shape)`), which is the confound study 2 dropped its second axis
    to avoid, and it leaves the upper arm of the sweep barely distinguishable from the training
    prior: the test prior *widens* as it shifts, so it keeps covering the trained-on one instead
    of separating from it. Holding `sd` fixed and moving the mean makes the sweep a pure location
    shift, and doing it inside the Gamma family rather than with a truncated normal keeps the
    support on (0, inf) -- so `b_diff` stays positive and log-transformable, and no mass is
    truncated at zero to shrink the realized sd at the bottom of the grid.

    `shape = (loc / sd)**2` drops below 1 -- a J-shaped density piling up at zero -- once
    `loc < sd`. `conf/test_case/threshold_diff_grid.yaml` keeps its lowest point well clear of
    that; at the trained-on `loc` of 0.6 with `sd` 0.245 the shape is ~6.
    """
    loc, sd = _float(loc), _float(sd)
    return gamma_rvs(key, (loc / sd) ** 2, sd**2 / loc, size=size)


def _to_numpy(draws):
    return {name: np.asarray(value) for name, value in draws.items()}


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
    keys = rng.next(5)
    drift_intercept = truncated_normal_rvs(
        keys[0], drift_intercept_loc, drift_intercept_scale, size=batch_shape,
    )
    drift_slope = truncated_normal_rvs(keys[1], drift_slope_loc, drift_slope_scale, size=batch_shape)
    sd_true = gamma_rvs(keys[2], sd_true_shape, sd_true_scale, size=batch_shape)
    threshold = gamma_rvs(keys[3], threshold_shape, threshold_scale, size=batch_shape)
    t0 = truncated_normal_rvs(keys[-1], t0_loc, t0_scale, lower=t0_lower, size=batch_shape)

    return _to_numpy({
        "v_intercept": drift_intercept,
        "v_slope": drift_slope,
        "s_true": sd_true,
        "b": threshold,
        "t0": t0,
    })


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
    names are kept so `scripts/fit_mcmc_cpu.py` needs no changes.

    Returned keys are ordered to match `conf/approximator/adapter/lba_simple.yaml` and the
    unconstrained position vector built by `lba_jax.make_lba_simple_logdensity`.
    """
    keys = rng.next(6)
    drift_intercept = truncated_normal_rvs(
        keys[0], drift_intercept_loc, drift_intercept_scale, size=batch_shape,
    )
    drift_slope = truncated_normal_rvs(keys[1], drift_slope_loc, drift_slope_scale, size=batch_shape)
    sd_true = gamma_rvs(keys[2], sd_true_shape, sd_true_scale, size=batch_shape)
    sp_max = gamma_rvs(keys[3], sp_max_shape, sp_max_scale, size=batch_shape)
    sp_gap = gamma_rvs(keys[4], threshold_shape, threshold_scale, size=batch_shape)
    t0 = truncated_normal_rvs(keys[-1], t0_loc, t0_scale, lower=t0_lower, size=batch_shape)

    return _to_numpy({
        "v_intercept": drift_intercept,
        "v_slope": drift_slope,
        "s_true": sd_true,
        "A": sp_max,
        "B": sp_gap,
        "t0": t0,
    })


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
    threshold_diff_loc,
    threshold_diff_sd,
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

    `threshold_diff_loc` is the hyperparameter study 3 shifts between training and test: it
    *is* the expected size of the speed-accuracy effect, with `threshold_diff_sd` holding the
    prior's spread fixed across the sweep (see `gamma_mean_sd_rvs`).

    Returned keys are ordered to match `conf/approximator/adapter/rdm_sat.yaml` and the
    unconstrained position vector built by `rdm_jax.make_rdm_sat_logdensity`.
    """
    keys = rng.next(6)
    drift_intercept = truncated_normal_rvs(
        keys[0], drift_intercept_loc, drift_intercept_scale, size=batch_shape,
    )
    drift_slope = truncated_normal_rvs(keys[1], drift_slope_loc, drift_slope_scale, size=batch_shape)
    sd_true = gamma_rvs(keys[2], sd_true_shape, sd_true_scale, size=batch_shape)
    threshold = gamma_rvs(keys[3], threshold_shape, threshold_scale, size=batch_shape)
    threshold_diff = gamma_mean_sd_rvs(keys[4], threshold_diff_loc, threshold_diff_sd, size=batch_shape)
    t0 = truncated_normal_rvs(keys[-1], t0_loc, t0_scale, lower=t0_lower, size=batch_shape)

    return _to_numpy({
        "v_intercept": drift_intercept,
        "v_slope": drift_slope,
        "s_true": sd_true,
        "b": threshold,
        "b_diff": threshold_diff,
        "t0": t0,
    })


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
    threshold_diff_loc,
    threshold_diff_sd,
    t0_loc,
    t0_scale,
    t0_lower,
    rng,
):
    """Prior for the linear ballistic accumulator with a speed-vs-accuracy manipulation.

    `lba_prior_simple` plus `B_diff`, the amount by which the accuracy instruction raises the
    threshold *gap* -- so the threshold is `A + B` under speed and `A + B + B_diff` under
    accuracy, and `b > A` still holds structurally in both conditions. The RDM counterpart is
    `rdm_prior_sat`; `threshold_diff_loc` is the hyperparameter study 3 shifts between training
    and test in both models.

    Returned keys are ordered to match `conf/approximator/adapter/lba_sat.yaml` and the
    unconstrained position vector built by `lba_jax.make_lba_sat_logdensity`.
    """
    keys = rng.next(7)
    drift_intercept = truncated_normal_rvs(
        keys[0], drift_intercept_loc, drift_intercept_scale, size=batch_shape,
    )
    drift_slope = truncated_normal_rvs(keys[1], drift_slope_loc, drift_slope_scale, size=batch_shape)
    sd_true = gamma_rvs(keys[2], sd_true_shape, sd_true_scale, size=batch_shape)
    sp_max = gamma_rvs(keys[3], sp_max_shape, sp_max_scale, size=batch_shape)
    sp_gap = gamma_rvs(keys[4], threshold_shape, threshold_scale, size=batch_shape)
    sp_gap_diff = gamma_mean_sd_rvs(keys[5], threshold_diff_loc, threshold_diff_sd, size=batch_shape)
    t0 = truncated_normal_rvs(keys[-1], t0_loc, t0_scale, lower=t0_lower, size=batch_shape)

    return _to_numpy({
        "v_intercept": drift_intercept,
        "v_slope": drift_slope,
        "s_true": sd_true,
        "A": sp_max,
        "B": sp_gap,
        "B_diff": sp_gap_diff,
        "t0": t0,
    })
