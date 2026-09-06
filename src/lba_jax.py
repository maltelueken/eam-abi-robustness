"""The linear ballistic accumulator, as this study parameterizes it, over `eamax`.

Companion to `rdm_jax.py`: same pipeline, same contracts, a different evidence-accumulation
model. Where the racing diffusion model races two noisy Wald accumulators, the LBA (Brown &
Heathcote 2008) races two *ballistic* (noise-free within a trial) accumulators whose randomness
lives entirely between trials -- a uniform start point `k ~ U(0, A)` and a normal drift rate
truncated at zero, `d ~ N(v, s)[0, inf)`. The first passage time of one accumulator is therefore
the deterministic `(b - k) / d`. Both the density and the sampler come from
`eamax.accumulators.LBA`; the split between what moved and what stayed is the one `rdm_jax`
documents.

Parameterization mirrors the RDM's exactly, plus the start-point range: the false accumulator
drifts at `v_intercept`, the true one at `v_intercept + v_slope`; the drift SD is `s_false`
(fixed to 1 for scale, bound in `conf/simulator/experiment_simulator/lba_simple.yaml`) and
`s_true` (free); `t0` is the non-decision time.

The threshold is carried as the *gap* `B = b - A` (Heathcote & Love 2012), not as `b` itself,
and the actual threshold is reconstructed as `b = A + B` wherever it is needed. Two reasons:

1. The LBA is degenerate unless `b > A` -- a start point above threshold would mean an instant
   response, and the closed-form density then integrates untruncated drift mass below zero and
   can come out negative. With `B > 0` sampled independently, `b > A` holds *structurally*, so
   there is no infeasible region to guard, penalize, or clamp.
2. `A` and `b` are almost perfectly correlated a posteriori (~0.99 at n=1500), and BlackJAX's
   window adaptation uses a diagonal mass matrix, which cannot absorb that. `A` and `B` are far
   better conditioned, so NUTS needs materially fewer leapfrog steps per effective sample.

`B` is the standard quantity in LBA fitting software (rtdists, DMC, EMC2) for exactly these
reasons. Recovering `b` for reporting is a one-line post-hoc `A + B` on the samples. In
`eamax.design.lba_intercept_slope_spec` the reconstruction is a `transform` on the threshold
quantity (`lambda q: q["A"] + q["b"]`), so it happens once, inside the parameterization, rather
than in every function that needs a threshold.

The unconstrained parameter vector is therefore length-6, ordered
`[v_intercept, v_slope, s_true, A, B, t0]`, with `t0` last -- which `mcmc.t0_support` reads back
out *by name* from the spec rather than assuming.

Naming: `A` and `B` are the literature's names, and are what appear as prior dict keys,
`inference_variables` entries and config keys. Internal helpers spell them `sp_max` and
`sp_gap` (bare capitals trip ruff's pep8-naming rules), but the two simulator entry points must
keep the literal names, because `simulation.CustomSimulator` splats the prior's dict in as
keyword arguments and the names have to line up. `eamax`'s speed/accuracy spec names the
threshold increment `b_diff` where this study's config calls it `B_diff`; the spec is only ever
indexed positionally here, and `mcmc_param_names` stays authoritative for what is written to
disk.

Note `rdm_jax` enables float64 globally on import; this module relies on that, and on its
`_as_scalar` / `batched_experiment` / `simulation_design` helpers, rather than duplicating them.
"""

import functools

import jax.numpy as jnp
from eamax.accumulators import LBA
from eamax.design import build_params_fn, lba_intercept_slope_spec, lba_sat_spec
from eamax.inference.transforms import BlockTransform
from eamax.simulate import simulate_race
from tensorflow_probability.substrates import jax as tfp

from rdm_jax import _log_prior_factory
from rdm_jax import _log_prior_from_dists
from rdm_jax import _meta_prior_sampler
from rdm_jax import _prior_sampler
from rdm_jax import _pack_dataset
from rdm_jax import _theta_from_natural
from rdm_jax import batched_experiment
from rdm_jax import race_design
from rdm_jax import race_log_likelihood
from rdm_jax import sat_conditions
from rdm_jax import simulation_design
from rdm_jax import static_noise_scale
from rdm_jax import trial_design

tfd = tfp.distributions

_LBA = LBA()


@functools.lru_cache(maxsize=None)
def lba_spec(sat=False, noise_scale=1.0):
    """The LBA parameterization, cached so one object serves the likelihood and the simulator."""
    return lba_sat_spec(noise_scale) if sat else lba_intercept_slope_spec(noise_scale)


@functools.lru_cache(maxsize=None)
def lba_params_fn(sat=False, noise_scale=1.0):
    """`lba_spec` bound to the LBA accumulator. Cached for the reason `rdm_jax.rdm_params_fn` is."""
    return build_params_fn(lba_spec(sat, noise_scale), _LBA)


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _lba_simple_simulator(noise_scale):
    """The single-dataset simple-LBA simulator, cached per noise identification."""
    params_fn = lba_params_fn(sat=False, noise_scale=noise_scale)

    def sample(key, v_intercept, v_slope, s_true, sp_max, sp_gap, t0, num_obs):
        design = simulation_design(int(num_obs))
        theta = _theta_from_natural((v_intercept, v_slope, s_true, sp_max, sp_gap, t0))
        rt, response = simulate_race(key, params_fn, theta, design, _LBA)
        return _pack_dataset(rt, response)

    return sample


def lba_experiment_simple_jax(key, v_intercept, v_slope, s_true, s_false, A, B, t0, num_obs):  # noqa: N803
    """Simulate `num_obs` trials from the two-accumulator linear ballistic accumulator.

    Structural twin of `rdm_jax.rdm_experiment_simple_jax` -- same `num_obs`-must-be-static
    rule, same `{"x": (num_obs, 2)}` return contract with column 0 the RT (non-decision time
    already added) and column 1 the choice (0 = false accumulator, 1 = true accumulator).

    `A` (start-point range) and `B` (threshold gap, so the threshold is `A + B`) are spelled as
    the bare literature names rather than `sp_max`/`sp_gap`: `simulation.CustomSimulator` splats
    the prior's output dict straight in as keyword arguments, so these names have to match the
    keys returned by `priors.lba_prior_simple` exactly.
    """
    return _lba_simple_simulator(static_noise_scale(s_false))(
        key, v_intercept, v_slope, s_true, A, B, t0, num_obs,
    )


def lba_experiment_simple_jax_stateful(v_intercept, v_slope, s_true, s_false, A, B, t0, num_obs, rng):  # noqa: N803
    """`lba_experiment_simple_jax`, adapted to the `sample_fn(..., rng)` calling convention
    BayesFlow's `LambdaSimulator` uses (see `conf/simulator/experiment_simulator/lba_simple.yaml`),
    with `rng` a `rdm_jax.SplittableKey` instead of a `numpy.random.Generator`.
    """
    return lba_experiment_simple_jax(rng.next(), v_intercept, v_slope, s_true, s_false, A, B, t0, num_obs)


def lba_experiment_simple_jax_batched(batch_shape, v_intercept, v_slope, s_true, s_false, A, B, t0, num_obs, rng):  # noqa: N803
    """`lba_experiment_simple_jax` for a whole batch at once, via `rdm_jax.batched_experiment`."""
    return batched_experiment(
        _lba_simple_simulator(static_noise_scale(s_false)), batch_shape, num_obs, rng,
        (v_intercept, v_slope, s_true, A, B, t0),
    )


# ---------------------------------------------------------------------------
# Priors and log-densities.
# ---------------------------------------------------------------------------


def _lba_simple_prior_dists(drift_slope_loc, threshold_scale):
    """One TFP distribution per coefficient, in `mcmc_param_names` order.

    The single definition of the simple LBA's prior; the RDM counterpart,
    `rdm_jax._rdm_simple_prior_dists`, explains why it is one object rather than a
    log-density and a sampler written separately. Every term is an ordinary density on a
    positive parameter -- because the threshold enters as the gap `B = b - A`, there is no
    `b > A` constraint left to enforce here.

    The drift intercept is centred on 2.0 rather than the RDM's 1.0. The LBA's first-passage
    time is `(b - k) / d`, so a drift rate near zero produces an arbitrarily large RT -- the
    distribution has a `t^-2` tail and no finite mean, and the weight of that tail is set by
    `Phi(-v/s)`. Measured over the prior predictive, centring at 1.0 gives P(RT > 5 s) = 3.6e-3
    with excursions past 190 s, which would swamp the summary network's statistics; centring at
    2.0 brings that to 4.0e-4, in line with the RDM's 2.8e-4, at no cost in accuracy (0.81
    either way) and with a median RT closer to the RDM's. The RDM needs no such adjustment
    because a Wald first-passage time cannot blow up the same way.
    """
    return (
        tfd.TruncatedNormal(2.0, 0.5, 0.0, jnp.inf),                # v_intercept
        tfd.TruncatedNormal(drift_slope_loc, 0.5, 0.0, jnp.inf),    # v_slope
        tfd.Gamma(12.0, 10.0),                                      # s_true
        tfd.Gamma(6.0, 10.0),                                       # A
        tfd.Gamma(8.0, 1.0 / threshold_scale),                      # B
        tfd.TruncatedNormal(0.3, 0.2, 0.0, jnp.inf),                # t0
    )


def _lba_simple_log_prior(
    v_intercept, v_slope, s_true, sp_max, sp_gap, t0, drift_slope_loc, threshold_scale,
):
    """Log prior, matching the hyperparameters sampled by `priors.lba_prior_simple`."""
    return _log_prior_from_dists(
        _lba_simple_prior_dists(drift_slope_loc, threshold_scale),
        (v_intercept, v_slope, s_true, sp_max, sp_gap, t0),
    )


def make_lba_simple_prior_sample(drift_slope_loc, threshold_scale):
    """Draw the simple LBA's prior on the natural scale, in `mcmc_param_names` order."""
    return _prior_sampler(_lba_simple_prior_dists(drift_slope_loc, threshold_scale))


def make_lba_meta_prior_sample(drift_slope_loc_lower, drift_slope_loc_upper, threshold_scale):
    """Draw the hierarchical LBA's joint prior; see `rdm_jax.make_rdm_meta_prior_sample`."""
    return _meta_prior_sampler(
        tfd.Uniform(drift_slope_loc_lower, drift_slope_loc_upper),
        lambda drift_slope_loc: _lba_simple_prior_dists(drift_slope_loc, threshold_scale),
    )


def _lba_simple_log_likelihood(rt, is_true, v_intercept, v_slope, s_true, sp_max, sp_gap, t0):
    """One dataset's total race log-likelihood, from natural-scale parameters.

    The LBA counterpart of `rdm_jax._rdm_simple_log_likelihood`; see it for why the
    natural-scale form is kept.
    """
    return race_log_likelihood(
        lba_params_fn(), _LBA,
        _theta_from_natural((v_intercept, v_slope, s_true, sp_max, sp_gap, t0)),
        race_design(rt, is_true),
    )


def make_lba_simple_logdensity(data_x, drift_slope_loc, threshold_scale):
    """Build a BlackJAX-ready log-density function for the simple (non-hierarchical) LBA.

    `position` passed to the returned function is a length-6 array of *unconstrained*
    (log-space) values in the order [v_intercept, v_slope, s_true, A, B, t0], where `B` is the
    threshold gap `b - A` -- matching `mcmc_param_names` in `conf/mcmc/lba.yaml` and
    `eamax.design.lba_intercept_slope_spec`.
    """
    spec, params_fn = lba_spec(), lba_params_fn()
    design = trial_design(data_x)

    def logdensity_fn(position):
        position = jnp.asarray(position)
        v_intercept, v_slope, s_true, sp_max, sp_gap, t0 = spec.constrain(position)

        log_prior = _lba_simple_log_prior(
            v_intercept, v_slope, s_true, sp_max, sp_gap, t0, drift_slope_loc, threshold_scale,
        )
        log_lik = race_log_likelihood(params_fn, _LBA, position, design)

        return log_prior + spec.log_det_jacobian(position) + log_lik

    return logdensity_fn


def make_lba_meta_logdensity(
    data_x, drift_slope_loc_lower, drift_slope_loc_upper, threshold_scale,
):
    """Build a BlackJAX-ready log-density function for the hierarchical (meta) LBA.

    `position` passed to the returned function is a length-7 array of *unconstrained*
    values in the order [drift_slope_loc, v_intercept, v_slope, s_true, A, B, t0], where `B` is
    the threshold gap `b - A`: the leading hyperparameter is Uniform on its support and so uses
    a Sigmoid bijector, the rest are positive and use Exp. See
    `eamax.inference.transforms.BlockTransform` for the matching forward transform.

    One randomized hyperparameter, not the two study 2 used to cross. The threshold prior's
    scale is now fixed and arrives interpolated from the training prior, exactly as it does for
    the non-hierarchical model.
    """
    params_fn = lba_params_fn()
    design = trial_design(data_x)
    transform = BlockTransform([drift_slope_loc_lower], [drift_slope_loc_upper])

    def logdensity_fn(position):
        position = jnp.asarray(position)
        drift_slope_loc, *subject = transform.forward(position)
        v_intercept, v_slope, s_true, sp_max, sp_gap, t0 = subject

        log_prior = tfd.Uniform(drift_slope_loc_lower, drift_slope_loc_upper).log_prob(
            drift_slope_loc,
        ) + _lba_simple_log_prior(
            v_intercept, v_slope, s_true, sp_max, sp_gap, t0, drift_slope_loc, threshold_scale,
        )
        log_lik = race_log_likelihood(params_fn, _LBA, position[1:], design)

        return log_prior + transform.log_det_jacobian(position) + log_lik

    return logdensity_fn


# ---------------------------------------------------------------------------
# Speed-vs-accuracy variant (study 3).
#
# Twin of `rdm_jax.rdm_experiment_sat_jax`: two blocks of trials per dataset,
# speed-instructed and accuracy-instructed, differing only in response
# threshold, with the condition carried as a third channel of `x`.
#
# The accuracy instruction raises the threshold *gap* -- `B_accuracy = B +
# B_diff`, so `b = A + B + B_diff` -- rather than the threshold itself. That
# keeps the module's central invariant (`b > A`, structurally, for both
# conditions) and leaves every parameter strictly positive, so the whole vector
# still log-transforms. `eamax.design.lba_sat_spec` composes exactly that: the
# increment rides `condition(0)` inside the gap, and the same `A + b` transform
# reconstructs the absolute threshold.
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _lba_sat_simulator(noise_scale):
    """The single-dataset speed/accuracy-LBA simulator, cached per noise identification."""
    params_fn = lba_params_fn(sat=True, noise_scale=noise_scale)

    def sample(key, v_intercept, v_slope, s_true, sp_max, sp_gap, sp_gap_diff, t0, num_obs):
        num_obs = int(num_obs)
        design = simulation_design(num_obs, has_condition=True)
        theta = _theta_from_natural((v_intercept, v_slope, s_true, sp_max, sp_gap, sp_gap_diff, t0))
        rt, response = simulate_race(key, params_fn, theta, design, _LBA)
        return _pack_dataset(rt, response, sat_conditions(num_obs))

    return sample


def lba_experiment_sat_jax(key, v_intercept, v_slope, s_true, s_false, A, B, B_diff, t0, num_obs):  # noqa: N803
    """Simulate `num_obs` LBA trials, half under a speed and half under an accuracy instruction.

    Returns `x` with three channels: response time, response (0 = false, 1 = true) and
    condition (0 = speed, 1 = accuracy), matching `rdm_jax.rdm_experiment_sat_jax`.
    """
    return _lba_sat_simulator(static_noise_scale(s_false))(
        key, v_intercept, v_slope, s_true, A, B, B_diff, t0, num_obs,
    )


def lba_experiment_sat_jax_stateful(v_intercept, v_slope, s_true, s_false, A, B, B_diff, t0, num_obs, rng):  # noqa: N803
    """`lba_experiment_sat_jax` under the `sample_fn(..., rng)` convention, with `rng` a `SplittableKey`."""
    return lba_experiment_sat_jax(rng.next(), v_intercept, v_slope, s_true, s_false, A, B, B_diff, t0, num_obs)


def lba_experiment_sat_jax_batched(batch_shape, v_intercept, v_slope, s_true, s_false, A, B, B_diff, t0, num_obs, rng):  # noqa: N803
    """`lba_experiment_sat_jax` for a whole batch at once."""
    return batched_experiment(
        _lba_sat_simulator(static_noise_scale(s_false)), batch_shape, num_obs, rng,
        (v_intercept, v_slope, s_true, A, B, B_diff, t0),
    )


def _lba_sat_prior_dists(
    drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
):
    """`_lba_simple_prior_dists` with the threshold-gap increment spliced in ahead of `t0`.

    The order is `mcmc_param_names`' -- `[..., A, B, B_diff, t0]` -- since `t0` stays last.
    `threshold_diff_shape`/`threshold_diff_scale` parameterize `Gamma(shape, scale)` exactly as
    `priors.lba_prior_sat` draws it; both are interpolated from the same prior config in
    `conf/mcmc/lba_sat.yaml`.
    """
    v_intercept, v_slope, s_true, sp_max, sp_gap, t0 = _lba_simple_prior_dists(
        drift_slope_loc, threshold_scale,
    )

    return (
        v_intercept, v_slope, s_true, sp_max, sp_gap,
        tfd.Gamma(threshold_diff_shape, 1.0 / threshold_diff_scale),  # B_diff
        t0,
    )


def _lba_sat_log_prior(
    v_intercept, v_slope, s_true, sp_max, sp_gap, sp_gap_diff, t0,
    drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
):
    """Log prior for the speed-accuracy LBA: the simple model's, plus the threshold difference."""
    return _log_prior_from_dists(
        _lba_sat_prior_dists(
            drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
        ),
        (v_intercept, v_slope, s_true, sp_max, sp_gap, sp_gap_diff, t0),
    )


def make_lba_sat_prior_sample(
    drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
):
    """Draw the speed-accuracy LBA's prior on the natural scale, in `mcmc_param_names` order."""
    return _prior_sampler(
        _lba_sat_prior_dists(
            drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
        ),
    )


def make_lba_sat_meta_prior_sample(
    drift_slope_loc, threshold_scale, threshold_diff_shape,
    threshold_diff_scale_lower, threshold_diff_scale_upper,
):
    """Draw the hierarchical speed-accuracy LBA's joint prior; see `rdm_jax.make_rdm_meta_prior_sample`."""
    return _meta_prior_sampler(
        tfd.Uniform(threshold_diff_scale_lower, threshold_diff_scale_upper),
        lambda threshold_diff_scale: _lba_sat_prior_dists(
            drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
        ),
    )


def _lba_sat_log_likelihood(
    rt, is_true, is_accuracy, v_intercept, v_slope, s_true, sp_max, sp_gap, sp_gap_diff, t0,
):
    """As `_lba_simple_log_likelihood`, but with a per-trial threshold gap."""
    return race_log_likelihood(
        lba_params_fn(sat=True), _LBA,
        _theta_from_natural((v_intercept, v_slope, s_true, sp_max, sp_gap, sp_gap_diff, t0)),
        race_design(rt, is_true, is_accuracy),
    )


def make_lba_sat_logdensity(data_x, drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale):
    """Build a BlackJAX-ready log-density for the speed-accuracy LBA.

    `position` is a length-7 array of *unconstrained* (log-space) values in the order
    [v_intercept, v_slope, s_true, A, B, B_diff, t0] -- matching `mcmc_param_names` in
    `conf/mcmc/lba_sat.yaml`, with `t0` last.
    """
    spec, params_fn = lba_spec(sat=True), lba_params_fn(sat=True)
    design = trial_design(data_x, has_condition=True)

    def logdensity_fn(position):
        position = jnp.asarray(position)
        v_intercept, v_slope, s_true, sp_max, sp_gap, sp_gap_diff, t0 = spec.constrain(position)

        log_prior = _lba_sat_log_prior(
            v_intercept, v_slope, s_true, sp_max, sp_gap, sp_gap_diff, t0,
            drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
        )
        log_lik = race_log_likelihood(params_fn, _LBA, position, design)

        return log_prior + spec.log_det_jacobian(position) + log_lik

    return logdensity_fn


def make_lba_sat_meta_logdensity(
    data_x, drift_slope_loc, threshold_scale, threshold_diff_shape,
    threshold_diff_scale_lower, threshold_diff_scale_upper,
):
    """Build a BlackJAX-ready log-density for the hierarchical speed-accuracy LBA.

    `position` is a length-8 array of *unconstrained* values in the order
    [threshold_diff_scale, v_intercept, v_slope, s_true, A, B, B_diff, t0]: the leading
    hyperparameter is Uniform on its support and uses a Sigmoid bijector, the rest use Exp.
    See `eamax.inference.transforms.BlockTransform` for the matching forward transform.
    """
    params_fn = lba_params_fn(sat=True)
    design = trial_design(data_x, has_condition=True)
    transform = BlockTransform([threshold_diff_scale_lower], [threshold_diff_scale_upper])

    def logdensity_fn(position):
        position = jnp.asarray(position)
        threshold_diff_scale, *subject = transform.forward(position)
        v_intercept, v_slope, s_true, sp_max, sp_gap, sp_gap_diff, t0 = subject

        log_prior = tfd.Uniform(threshold_diff_scale_lower, threshold_diff_scale_upper).log_prob(
            threshold_diff_scale,
        ) + _lba_sat_log_prior(
            v_intercept, v_slope, s_true, sp_max, sp_gap, sp_gap_diff, t0,
            drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
        )
        log_lik = race_log_likelihood(params_fn, _LBA, position[1:], design)

        return log_prior + transform.log_det_jacobian(position) + log_lik

    return logdensity_fn


def make_lba_simple_log_prior(drift_slope_loc, threshold_scale):
    """The simple LBA's prior density over `mcmc_param_names`, as a function of the draws."""
    return _log_prior_factory(
        _lba_simple_log_prior, 6,
        {"drift_slope_loc": drift_slope_loc, "threshold_scale": threshold_scale},
    )


def make_lba_sat_log_prior(drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale):
    """The speed-accuracy LBA's prior density over `mcmc_param_names`."""
    return _log_prior_factory(
        _lba_sat_log_prior, 7,
        {
            "drift_slope_loc": drift_slope_loc,
            "threshold_scale": threshold_scale,
            "threshold_diff_shape": threshold_diff_shape,
            "threshold_diff_scale": threshold_diff_scale,
        },
    )
