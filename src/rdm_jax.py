"""The racing diffusion model, as this study parameterizes it, over `eamax`.

The maths -- the Wald first-passage density and survival, the race likelihood, the
parameterization engine that turns a flat parameter vector into per-accumulator drifts,
thresholds and noise, and the sampler driven by that same parameterization -- lives in
`eamax`, shared with the other repositories that grew their own copy of it. What stays here
is what this study owns: its priors, the `x` layout its adapters and artifacts are built on,
and the BayesFlow calling conventions its simulators are reached through.

Three things about the seam are worth knowing.

**One `params_fn` feeds both directions.** `eamax.design.rdm_intercept_slope_spec` /
`rdm_sat_spec` reproduce this study's `[v_intercept, v_slope, s_true, b, (b_diff,) t0]`
layout exactly, and `eamax.simulate.simulate_race` and `eamax.race.race_loglik` both go
through the object `build_params_fn` returns. The model simulated from and the model fitted
can no longer drift apart by editing one and not the other.

**Accumulators are numbered, not named.** `eamax` labels them `1..N` and asks the design
which one is the *target* on each trial; this study always has two, with accumulator 2 the
"true" one. So `x`'s response channel (0 = false, 1 = true) is the accumulator index minus
one, and `trial_design` is the single place that conversion happens.

**The speed/accuracy condition flips sign at the boundary.** `x`'s third channel is
`is_accuracy`; `eamax` codes `condition == 1` as the *reference* level, which for this design
is speed. `trial_design` passes `1 - is_accuracy`.

One behaviour changed with the move, deliberately. The old `_finalize_race_logp` answered a
trial with `rt <= t0` with a sloped penalty, so that a chain stranded in the infeasible
region kept a gradient pushing `t0` back down. `eamax` has no slope: such a trial lands on
the same flat floor as any other hopeless one, and the region is kept out of at
*initialisation* instead, by `eamax.inference.init.T0Support` (see `mcmc.t0_support`). A
support constraint prevents what the penalty could only try to repair.
"""

import functools
import logging
import threading

import eamax
import jax
import jax.numpy as jnp
import numpy as np
from eamax.accumulators import Wald
from eamax.design import TrialDesign, build_params_fn, rdm_intercept_slope_spec, rdm_sat_spec
from eamax.inference.transforms import BlockTransform
from eamax.race import race_loglik
from eamax.simulate import simulate_race
from tensorflow_probability.substrates import jax as tfp

# NUTS window adaptation and the race log-density (small squared terms, exp/log transforms of
# sub-unit values like t0) are precision-sensitive. `eamax` never touches JAX's global config
# on import -- a library that did would change the numerics of unrelated code in the same
# process -- so the entry points that need float64 turn it on themselves. This module is one:
# `lba_jax` and `mcmc` both rely on the side effect of importing it.
eamax.enable_x64()

tfd = tfp.distributions

logger = logging.getLogger(__name__)

#: `eamax` numbers accumulators from this index; the two here are 1 (false) and 2 (true).
FIRST_RESPONSE = 1

#: The accumulator that is correct on every trial. This study's designs have no stimulus
#: variation -- accuracy is which accumulator won, not which one matched the stimulus -- so
#: the target is a constant rather than a column of `x`.
TARGET_ACCUMULATOR = 2

_WALD = Wald()


# ---------------------------------------------------------------------------
# The seam: this study's `x` layout and design, as `eamax` wants them.
# ---------------------------------------------------------------------------


def race_design(rt, is_true, is_accuracy=None):
    """Build the `TrialDesign` for one dataset, from this study's columns.

    Everything that translates between those columns and `eamax`'s conventions happens here:
    the response coding is shifted onto accumulator indices, the target is pinned to the
    "true" accumulator, and the condition is flipped so that speed is the reference level.
    """
    rt = jnp.asarray(rt)

    return TrialDesign(
        rt=rt,
        response=jnp.asarray(is_true).astype(int) + FIRST_RESPONSE,
        target=jnp.full(rt.shape, TARGET_ACCUMULATOR),
        condition=None if is_accuracy is None else 1.0 - jnp.asarray(is_accuracy).astype(rt.dtype),
        first_response=FIRST_RESPONSE,
    )


def trial_design(data_x, *, has_condition=False):
    """`race_design` for one dataset's packed `x`.

    `x` is `(num_obs, 2)` -- response time, then 1 when the true accumulator won -- or
    `(num_obs, 3)` for the speed/accuracy models, whose third channel is `is_accuracy`.
    """
    data_x = jnp.asarray(data_x)

    return race_design(
        data_x[:, 0],
        data_x[:, 1] == 1,
        data_x[:, 2] == 1 if has_condition else None,
    )


def simulation_design(num_obs, *, has_condition=False):
    """The design a simulator draws against: covariates only, with `rt` a placeholder.

    `simulate_race` reads every field but `rt` and `response`, which are what it generates;
    `rt`'s *length* is still what tells the parameterization engine how many trials there are.
    """
    condition = jnp.where(sat_conditions(num_obs), 0.0, 1.0) if has_condition else None

    return TrialDesign(
        rt=jnp.zeros(num_obs),
        target=jnp.full((num_obs,), TARGET_ACCUMULATOR),
        condition=condition,
        first_response=FIRST_RESPONSE,
    )


def race_log_likelihood(params_fn, accumulator, theta, design):
    """One dataset's total log-likelihood, from an unconstrained parameter vector.

    `eamax.race.race_loglik` returns one value per trial and deliberately does not reduce --
    a hierarchical model sums over a different axis than this one does -- so the sum is here.
    """
    params, t0 = params_fn(theta, design)

    per_trial = race_loglik(
        design.rt,
        design.response,
        t0,
        lambda t: accumulator.log_pdf_sf(t, params),
        mask=design.mask,
        first_response=design.first_response,
    )

    return jnp.sum(per_trial)


@functools.lru_cache(maxsize=None)
def rdm_spec(sat=False, noise_scale=1.0):
    """The RDM parameterization, cached so one object serves the likelihood and the simulator."""
    return rdm_sat_spec(noise_scale) if sat else rdm_intercept_slope_spec(noise_scale)


@functools.lru_cache(maxsize=None)
def rdm_params_fn(sat=False, noise_scale=1.0):
    """`rdm_spec` bound to the Wald accumulator.

    Cached on the same key, and identity matters twice over: `_compiled_experiment` keys its
    XLA cache on the simulator function this ends up closed over, and rebuilding the pair per
    call would retrace every batch.
    """
    return build_params_fn(rdm_spec(sat, noise_scale), _WALD)


def static_noise_scale(s_false):
    """Normalize the non-target accumulator's fixed noise to the Python float a spec needs.

    Unlike every other parameter, this one is *structural*: `eamax` expresses the "mismatch"
    noise identification as a constant coefficient in the parameterization, fixed before
    tracing, rather than as a value carried per draw. Every producer already satisfies that --
    it is the literal `s_false: 1` in `conf/simulator/experiment_simulator/*.yaml` -- so this
    is a guard with a message rather than a conversion that ever does real work.
    """
    array = np.asarray(s_false)

    if array.size != 1:
        msg = (
            f"s_false must be a single value shared by the whole batch, got shape {array.shape}. "
            "It identifies the model's scale and is fixed in the parameterization, not drawn."
        )
        raise ValueError(msg)

    return float(array.reshape(()))


# ---------------------------------------------------------------------------
# BayesFlow calling conventions. None of this is model mathematics; it is the
# adapter between `eamax`'s pure functions and how BayesFlow calls a simulator.
# ---------------------------------------------------------------------------


def _as_scalar(x):
    """Collapse a Python float, 0-d array, or shape-(1,) array to a true 0-d scalar.

    BayesFlow's `batched_call` (see `batched_experiment`) indexes batched kwargs down to
    shape-(1,) arrays per call rather than plain scalars (unlike `numpy.hstack`, `jnp.stack`
    doesn't silently collapse those), so inputs need normalizing before use.
    """
    return jnp.reshape(jnp.asarray(x), ())


def _theta_from_natural(values):
    """Stack natural-scale scalars into the unconstrained vector a `params_fn` reads.

    Every coefficient in these specs is on the log link, so unconstraining is `log`. Going
    through the parameter vector rather than handing the accumulator its quantities directly
    is what makes the simulator use the *same* parameterization the likelihood scores with.
    """
    float_dtype = jnp.result_type(float)

    return jnp.log(jnp.stack([_as_scalar(value) for value in values]).astype(float_dtype))


class SplittableKey:
    """Stateful `jax.random.PRNGKey` wrapper, for use as the experiment simulators' `rng`.

    BayesFlow's `LambdaSimulator` passes the *same* `rng` object reference on every call --
    exactly how `numpy.random.default_rng()` is used elsewhere in this pipeline
    (`conf/simulator/*/*.yaml`), which works because a numpy `Generator` mutates its own state
    on every draw. A bare `jax.random.PRNGKey` is immutable/stateless, so this wraps one to give
    the same call-by-reference-then-advance behaviour: `.next()` splits off fresh subkeys, then
    updates its own internal state, so repeated calls (e.g. across training batches) never
    repeat the same randomness.

    The lock is load-bearing. Keras runs `OnlineDataset.__getitem__` on up to `cpu_count()`
    worker *threads* sharing this one object, so without it two threads can read `self._key`
    before either writes, derive the same subkey, and hand the trainer two identical batches.
    """

    def __init__(self, seed):
        self._key = jax.random.PRNGKey(seed)
        self._lock = threading.Lock()

    def next(self, num=None):
        """Return one fresh subkey, or `num` of them, advancing the internal state exactly once."""
        with self._lock:
            keys = jax.random.split(self._key, (num or 1) + 1)
            self._key = keys[0]

        return keys[1] if num is None else keys[1:]


def static_num_obs(num_obs):
    """Normalize `num_obs` to the single Python int that jit/vmap needs.

    `num_obs` sets the number of trials the design is built for, so it has to be static and
    shared by the whole batch. Every producer already satisfies that -- the design simulator
    draws one value per batch, and a `Case` pins one -- so this is a guard with a message rather
    than a conversion that ever does real work.
    """
    array = np.asarray(num_obs)

    if array.size != 1:
        msg = (
            f"num_obs must be a single value shared by the whole batch, got shape {array.shape}. "
            "The design simulator draws one num_obs per batch; per-dataset trial counts are not "
            "supported."
        )
        raise ValueError(msg)

    return int(array.reshape(()))


@functools.lru_cache(maxsize=None)
def _compiled_experiment(single_fn):
    """`jax.jit(jax.vmap(single_fn))`, cached so each simulator is traced once per shape."""

    def call(keys, params, num_obs):
        return jax.vmap(lambda key, args: single_fn(key, *args, num_obs))(keys, params)

    return jax.jit(call, static_argnums=2)


def batched_experiment(single_fn, batch_shape, num_obs, rng, params):
    """Run a single-dataset simulator once per batch element, under one `vmap`.

    BayesFlow's default path calls `sample_fn` once per batch element in a Python for-loop,
    which costs ~7 ms per dataset *regardless of `num_obs`* -- pure dispatch overhead, and the
    binding constraint on training throughput. Mapping the same pure function over the batch
    instead costs ~40 us per dataset, which is what makes study 5's rejection sampling (up to
    17 draws per accepted dataset) affordable.

    `single_fn` is the untouched `(key, *scalar_params, num_obs)` simulator, so the batched and
    per-dataset paths are the same function by construction rather than by agreement.
    """
    # `LambdaSimulator` passes the tuple through from `allow_batch_size`, but accept a bare int
    # too, so calling one of these directly (as `metrics.calc_posterior_predictive` does) works.
    batch_shape = (batch_shape,) if isinstance(batch_shape, int) else tuple(batch_shape)
    size = int(np.prod(batch_shape))
    num_obs = static_num_obs(num_obs)

    # Priors hand over `(batch, 1)` and a `Case` may pin a 0-d value; flattening and
    # broadcasting here means `single_fn` needs no such handling. The explicit dtype matters
    # for the LBA, whose TFP distributions reject mixed int/float.
    float_dtype = jnp.result_type(float)
    params = tuple(
        jnp.broadcast_to(jnp.reshape(jnp.asarray(value, dtype=float_dtype), (-1,)), (size,))
        for value in params
    )

    out = _compiled_experiment(single_fn)(rng.next(size), params, num_obs)

    return {key: np.asarray(value).reshape(*batch_shape, *value.shape[1:]) for key, value in out.items()}


def _pack_dataset(rt, response, condition=None):
    """Assemble one dataset's `x` in this study's channel order."""
    channels = [rt, (response - FIRST_RESPONSE).astype(rt.dtype)]

    if condition is not None:
        channels.append(jnp.asarray(condition).astype(rt.dtype))

    return {"x": jnp.stack(channels, axis=-1)}


# ---------------------------------------------------------------------------
# Simulators
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _rdm_simple_simulator(noise_scale):
    """The single-dataset simple-RDM simulator, cached per noise identification."""
    params_fn = rdm_params_fn(sat=False, noise_scale=noise_scale)

    def sample(key, v_intercept, v_slope, s_true, b, t0, num_obs):
        design = simulation_design(int(num_obs))
        theta = _theta_from_natural((v_intercept, v_slope, s_true, b, t0))
        rt, response = simulate_race(key, params_fn, theta, design, _WALD)
        return _pack_dataset(rt, response)

    return sample


def rdm_experiment_simple_jax(key, v_intercept, v_slope, s_true, s_false, b, t0, num_obs):
    """Simulate `num_obs` trials from the two-accumulator racing diffusion model.

    Returns `x` with two channels: response time (non-decision time already added) and
    response (0 = false accumulator, 1 = true accumulator). `num_obs` must be static (it sets
    the design's length) so this is jit/vmap-able over `key` and the scalar parameters.
    """
    return _rdm_simple_simulator(static_noise_scale(s_false))(
        key, v_intercept, v_slope, s_true, b, t0, num_obs,
    )


def rdm_experiment_simple_jax_stateful(v_intercept, v_slope, s_true, s_false, b, t0, num_obs, rng):
    """`rdm_experiment_simple_jax`, adapted to the `sample_fn(..., rng)` calling convention
    BayesFlow's `LambdaSimulator` uses (see `conf/simulator/experiment_simulator/rdm_simple.yaml`),
    with `rng` a `SplittableKey` instead of a `numpy.random.Generator`.
    """
    return rdm_experiment_simple_jax(rng.next(), v_intercept, v_slope, s_true, s_false, b, t0, num_obs)


def rdm_experiment_simple_jax_batched(batch_shape, v_intercept, v_slope, s_true, s_false, b, t0, num_obs, rng):
    """`rdm_experiment_simple_jax` for a whole batch at once.

    This is what `conf/simulator/experiment_simulator/rdm_simple.yaml` points at
    (`is_batched: true`); the per-dataset `_stateful` wrapper above is kept as the reference the
    equivalence test in `tests/test_batched_simulators.py` compares against.

    `s_false` is resolved to a Python float *outside* the vmap, because it is a constant of the
    parameterization rather than a per-draw parameter -- see `static_noise_scale`.
    """
    return batched_experiment(
        _rdm_simple_simulator(static_noise_scale(s_false)), batch_shape, num_obs, rng,
        (v_intercept, v_slope, s_true, b, t0),
    )


# ---------------------------------------------------------------------------
# Priors and log-densities. `eamax` stops at the likelihood boundary: the prior
# is this study's scientific content and stays here.
# ---------------------------------------------------------------------------


def _rdm_simple_prior_dists(drift_slope_loc, threshold_scale):
    """One TFP distribution per coefficient, in `mcmc_param_names` order.

    The single definition of the simple RDM's prior. `_rdm_simple_log_prior` scores draws
    against it, and `make_rdm_simple_prior_sample` draws from it for the chains' starting
    values -- so the density NUTS targets and the distribution its chains start from cannot
    drift apart. Hyperparameters may arrive batched (`_log_prior_factory` evaluates a whole
    quadrature at once), which every term broadcasts over.
    """
    return (
        tfd.TruncatedNormal(1.0, 0.5, 0.0, jnp.inf),                # v_intercept
        tfd.TruncatedNormal(drift_slope_loc, 0.5, 0.0, jnp.inf),    # v_slope
        tfd.Gamma(12.0, 10.0),                                      # s_true
        tfd.Gamma(8.0, 1.0 / threshold_scale),                      # b
        tfd.TruncatedNormal(0.3, 0.2, 0.0, jnp.inf),                # t0
    )


def _log_prior_from_dists(dists, values):
    """Sum the log-densities of independent prior factors, one value per distribution."""
    return sum(dist.log_prob(value) for dist, value in zip(dists, values, strict=True))


def _prior_sampler(dists):
    """`f(key) -> (P,)`: one independent natural-scale draw per coefficient, in vector order.

    The dtype is pinned rather than inherited: TFP infers `float32` from the Python-float
    hyperparameters written in `conf/mcmc/*.yaml`, and a starting position in a different
    dtype from the log-density's would trace the whole fit in mixed precision.
    """

    def sample(key):
        keys = jax.random.split(key, len(dists))
        draws = [dist.sample(seed=dist_key) for dist, dist_key in zip(dists, keys, strict=True)]
        return jnp.stack(draws).astype(jnp.result_type(float))

    return sample


def _meta_prior_sampler(hyper_dist, subject_dists_fn):
    """`f(key) -> (1 + P,)`: draw a prior hyperparameter, then the coefficients given it."""

    def sample(key):
        hyper_key, subject_key = jax.random.split(key)
        hyper = hyper_dist.sample(seed=hyper_key).astype(jnp.result_type(float))
        subject = _prior_sampler(subject_dists_fn(hyper))(subject_key)
        return jnp.concatenate([hyper[None], subject])

    return sample


def _rdm_simple_log_prior(v_intercept, v_slope, s_true, b, t0, drift_slope_loc, threshold_scale):
    """Log prior, matching the fixed hyperparameters in the former `rdm_model_simple`."""
    return _log_prior_from_dists(
        _rdm_simple_prior_dists(drift_slope_loc, threshold_scale),
        (v_intercept, v_slope, s_true, b, t0),
    )


def make_rdm_simple_prior_sample(drift_slope_loc, threshold_scale):
    """Draw the simple RDM's prior on the natural scale, in `mcmc_param_names` order.

    What `mcmc.make_init_positions_fn` starts each chain from -- unconstrained by the model's
    transform and rejected back into the `t0` support. Drawing the starts from the prior is
    what makes them both dispersed and on the right scale, which is what R-hat needs to be
    able to fail.
    """
    return _prior_sampler(_rdm_simple_prior_dists(drift_slope_loc, threshold_scale))


def make_rdm_meta_prior_sample(drift_slope_loc_lower, drift_slope_loc_upper, threshold_scale):
    """Draw the hierarchical RDM's *joint* prior: the hyperparameter, then the rest given it.

    The same factorization `make_rdm_meta_logdensity` scores -- Uniform on the swept
    hyperparameter, and the subject-level prior conditioned on the value drawn -- so a start
    is a draw from the density the chain is about to explore.
    """
    return _meta_prior_sampler(
        tfd.Uniform(drift_slope_loc_lower, drift_slope_loc_upper),
        lambda drift_slope_loc: _rdm_simple_prior_dists(drift_slope_loc, threshold_scale),
    )


def _rdm_simple_log_likelihood(rt, is_true, v_intercept, v_slope, s_true, b, t0):
    """One dataset's total race log-likelihood, from natural-scale parameters.

    The log-densities above reach the same code by a different door -- unconstrained
    parameters rather than natural ones. This form exists for `tests/test_rdm_sat.py`, which
    checks that the speed-accuracy likelihood collapses onto this one when no trial is
    accuracy-instructed: that is a statement about the *condition wiring* in `race_design`,
    which `eamax` cannot see.
    """
    return race_log_likelihood(
        rdm_params_fn(), _WALD,
        _theta_from_natural((v_intercept, v_slope, s_true, b, t0)),
        race_design(rt, is_true),
    )


def make_rdm_simple_logdensity(data_x, drift_slope_loc, threshold_scale):
    """Build a BlackJAX-ready log-density function for the simple (non-hierarchical) RDM.

    `position` passed to the returned function is a length-5 array of *unconstrained*
    (log-space) values in the order [v_intercept, v_slope, s_true, b, t0] -- matching
    `mcmc_param_names` in `conf/mcmc/rdm.yaml` and `eamax.design.rdm_intercept_slope_spec`.
    """
    spec, params_fn = rdm_spec(), rdm_params_fn()
    design = trial_design(data_x)

    def logdensity_fn(position):
        position = jnp.asarray(position)
        v_intercept, v_slope, s_true, b, t0 = spec.constrain(position)

        log_prior = _rdm_simple_log_prior(v_intercept, v_slope, s_true, b, t0, drift_slope_loc, threshold_scale)
        log_lik = race_log_likelihood(params_fn, _WALD, position, design)

        return log_prior + spec.log_det_jacobian(position) + log_lik

    return logdensity_fn


def make_rdm_meta_logdensity(
    data_x, drift_slope_loc_lower, drift_slope_loc_upper, threshold_scale,
):
    """Build a BlackJAX-ready log-density function for the hierarchical (meta) RDM.

    `position` passed to the returned function is a length-6 array of *unconstrained*
    values in the order [drift_slope_loc, v_intercept, v_slope, s_true, b, t0]: the leading
    hyperparameter is Uniform on its support and so uses a Sigmoid bijector, the rest are
    positive and use Exp. `eamax.inference.transforms.BlockTransform` owns exactly that
    layout, and `conf/mcmc/rdm_meta.yaml` builds the matching one for the sampler.

    One randomized hyperparameter, not the two study 2 used to cross. The threshold prior's
    scale is now fixed and arrives interpolated from the training prior, exactly as it does for
    the non-hierarchical model.
    """
    params_fn = rdm_params_fn()
    design = trial_design(data_x)
    transform = BlockTransform([drift_slope_loc_lower], [drift_slope_loc_upper])

    def logdensity_fn(position):
        position = jnp.asarray(position)
        drift_slope_loc, *subject = transform.forward(position)
        v_intercept, v_slope, s_true, b, t0 = subject

        log_prior = tfd.Uniform(drift_slope_loc_lower, drift_slope_loc_upper).log_prob(
            drift_slope_loc,
        ) + _rdm_simple_log_prior(
            v_intercept, v_slope, s_true, b, t0, drift_slope_loc, threshold_scale,
        )
        log_lik = race_log_likelihood(params_fn, _WALD, position[1:], design)

        return log_prior + transform.log_det_jacobian(position) + log_lik

    return logdensity_fn


# ---------------------------------------------------------------------------
# Speed-vs-accuracy variant (study 3).
#
# Each dataset contains two blocks of trials -- a speed-instructed and an
# accuracy-instructed one -- differing in the response threshold and nothing
# else, the standard threshold account of the speed-accuracy tradeoff. The
# threshold *difference* is a parameter (`b_diff`) rather than a second
# threshold, so that `b_accuracy > b_speed` holds by construction and both
# parameters stay strictly positive, i.e. log-transformable like every other
# parameter here (the same trick `priors.lba_prior_simple` uses for `A`/`B`).
# In `eamax.design.rdm_sat_spec` that is one contrast: `b_diff` rides
# `condition(0)`, the non-reference level.
#
# The condition is carried as a third channel of `x`, because the adapter treats
# `x` as a set: anything the likelihood needs per trial has to travel with that
# trial rather than in a separate array.
# ---------------------------------------------------------------------------


def sat_conditions(num_obs):
    """Condition indicator per trial: the first half speed (0), the rest accuracy (1).

    A fixed split rather than a random assignment -- the number of trials per condition is
    then a design constant instead of a source of between-dataset variance, and `x` is
    permutation-invariant downstream anyway (`simulation.create_data_adapter` marks it
    `as_set`), so the ordering carries no information.

    This is the channel-2 coding, not `eamax`'s: `trial_design` and `simulation_design` flip it
    to `condition = 1 - is_accuracy`, since `eamax` reserves level 1 for the reference.
    """
    return jnp.arange(num_obs) >= num_obs // 2


@functools.lru_cache(maxsize=None)
def _rdm_sat_simulator(noise_scale):
    """The single-dataset speed/accuracy-RDM simulator, cached per noise identification."""
    params_fn = rdm_params_fn(sat=True, noise_scale=noise_scale)

    def sample(key, v_intercept, v_slope, s_true, b, b_diff, t0, num_obs):
        num_obs = int(num_obs)
        design = simulation_design(num_obs, has_condition=True)
        theta = _theta_from_natural((v_intercept, v_slope, s_true, b, b_diff, t0))
        rt, response = simulate_race(key, params_fn, theta, design, _WALD)
        return _pack_dataset(rt, response, sat_conditions(num_obs))

    return sample


def rdm_experiment_sat_jax(key, v_intercept, v_slope, s_true, s_false, b, b_diff, t0, num_obs):
    """Simulate `num_obs` trials, half under a speed and half under an accuracy instruction.

    Returns `x` with three channels: response time, response (0 = false, 1 = true) and
    condition (0 = speed, 1 = accuracy). `num_obs` must be static, as in
    `rdm_experiment_simple_jax`.
    """
    return _rdm_sat_simulator(static_noise_scale(s_false))(
        key, v_intercept, v_slope, s_true, b, b_diff, t0, num_obs,
    )


def rdm_experiment_sat_jax_stateful(v_intercept, v_slope, s_true, s_false, b, b_diff, t0, num_obs, rng):
    """`rdm_experiment_sat_jax` under the `sample_fn(..., rng)` convention, with `rng` a `SplittableKey`."""
    return rdm_experiment_sat_jax(rng.next(), v_intercept, v_slope, s_true, s_false, b, b_diff, t0, num_obs)


def rdm_experiment_sat_jax_batched(batch_shape, v_intercept, v_slope, s_true, s_false, b, b_diff, t0, num_obs, rng):
    """`rdm_experiment_sat_jax` for a whole batch at once."""
    return batched_experiment(
        _rdm_sat_simulator(static_noise_scale(s_false)), batch_shape, num_obs, rng,
        (v_intercept, v_slope, s_true, b, b_diff, t0),
    )


def _rdm_sat_prior_dists(
    drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
):
    """`_rdm_simple_prior_dists` with the threshold difference spliced in ahead of `t0`.

    The order is `mcmc_param_names`' -- `[..., b, b_diff, t0]` -- not "the simple model's plus
    one at the end", since `t0` stays last.

    `threshold_diff_shape`/`threshold_diff_scale` parameterize `Gamma(shape, scale)` exactly as
    `priors.rdm_prior_sat` draws it; both are interpolated from the same prior config in
    `conf/mcmc/rdm_sat.yaml`, so the sampled prior and the fitted prior cannot drift apart.
    """
    v_intercept, v_slope, s_true, b, t0 = _rdm_simple_prior_dists(drift_slope_loc, threshold_scale)

    return (
        v_intercept, v_slope, s_true, b,
        tfd.Gamma(threshold_diff_shape, 1.0 / threshold_diff_scale),  # b_diff
        t0,
    )


def _rdm_sat_log_prior(
    v_intercept, v_slope, s_true, b, b_diff, t0,
    drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
):
    """Log prior for the speed-accuracy model: the simple model's, plus the threshold difference."""
    return _log_prior_from_dists(
        _rdm_sat_prior_dists(
            drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
        ),
        (v_intercept, v_slope, s_true, b, b_diff, t0),
    )


def make_rdm_sat_prior_sample(
    drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
):
    """Draw the speed-accuracy RDM's prior on the natural scale, in `mcmc_param_names` order."""
    return _prior_sampler(
        _rdm_sat_prior_dists(
            drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
        ),
    )


def make_rdm_sat_meta_prior_sample(
    drift_slope_loc, threshold_scale, threshold_diff_shape,
    threshold_diff_scale_lower, threshold_diff_scale_upper,
):
    """Draw the hierarchical speed-accuracy RDM's joint prior; see `make_rdm_meta_prior_sample`."""
    return _meta_prior_sampler(
        tfd.Uniform(threshold_diff_scale_lower, threshold_diff_scale_upper),
        lambda threshold_diff_scale: _rdm_sat_prior_dists(
            drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
        ),
    )


def _rdm_sat_log_likelihood(rt, is_true, is_accuracy, v_intercept, v_slope, s_true, b, b_diff, t0):
    """As `_rdm_simple_log_likelihood`, but with a per-trial threshold.

    With no accuracy-instructed trials this must agree exactly with the simple likelihood for
    any `b_diff`, which is what `tests/test_rdm_sat.py` pins.
    """
    return race_log_likelihood(
        rdm_params_fn(sat=True), _WALD,
        _theta_from_natural((v_intercept, v_slope, s_true, b, b_diff, t0)),
        race_design(rt, is_true, is_accuracy),
    )


def make_rdm_sat_logdensity(data_x, drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale):
    """Build a BlackJAX-ready log-density for the speed-accuracy RDM.

    `position` is a length-6 array of *unconstrained* (log-space) values in the order
    [v_intercept, v_slope, s_true, b, b_diff, t0] -- matching `mcmc_param_names` in
    `conf/mcmc/rdm_sat.yaml`. `t0` stays last, which is where `eamax.design.rdm_sat_spec`
    puts it and what `mcmc.t0_support` reads back out by name.
    """
    spec, params_fn = rdm_spec(sat=True), rdm_params_fn(sat=True)
    design = trial_design(data_x, has_condition=True)

    def logdensity_fn(position):
        position = jnp.asarray(position)
        v_intercept, v_slope, s_true, b, b_diff, t0 = spec.constrain(position)

        log_prior = _rdm_sat_log_prior(
            v_intercept, v_slope, s_true, b, b_diff, t0,
            drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
        )
        log_lik = race_log_likelihood(params_fn, _WALD, position, design)

        return log_prior + spec.log_det_jacobian(position) + log_lik

    return logdensity_fn


def make_rdm_sat_meta_logdensity(
    data_x, drift_slope_loc, threshold_scale, threshold_diff_shape,
    threshold_diff_scale_lower, threshold_diff_scale_upper,
):
    """Build a BlackJAX-ready log-density for the hierarchical speed-accuracy RDM.

    `position` is a length-7 array of *unconstrained* values in the order
    [threshold_diff_scale, v_intercept, v_slope, s_true, b, b_diff, t0]: the leading
    hyperparameter is Uniform on its support and so uses a Sigmoid bijector, the rest are
    positive and use Exp. See `eamax.inference.transforms.BlockTransform` for the matching
    forward transform.
    """
    params_fn = rdm_params_fn(sat=True)
    design = trial_design(data_x, has_condition=True)
    transform = BlockTransform([threshold_diff_scale_lower], [threshold_diff_scale_upper])

    def logdensity_fn(position):
        position = jnp.asarray(position)
        threshold_diff_scale, *subject = transform.forward(position)
        v_intercept, v_slope, s_true, b, b_diff, t0 = subject

        log_prior = tfd.Uniform(threshold_diff_scale_lower, threshold_diff_scale_upper).log_prob(
            threshold_diff_scale,
        ) + _rdm_sat_log_prior(
            v_intercept, v_slope, s_true, b, b_diff, t0,
            drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
        )
        log_lik = race_log_likelihood(params_fn, _WALD, position[1:], design)

        return log_prior + transform.log_det_jacobian(position) + log_lik

    return logdensity_fn


def _log_prior_factory(term_fn, num_params, fixed):
    """Turn one of this module's log-prior terms into a callable over a matrix of draws.

    `scripts/prior_distance.py` needs the prior's *density*, not the posterior log-density the
    samplers are built from, and it needs it at hyperparameter values other than the one a model
    trains on: a hierarchical model's training prior is the mixture over everything its meta
    simulator randomizes, which is a quadrature over that hyperparameter. So the returned callable
    takes the fixed hyperparameters from the config and lets a caller override any of them, the
    same way a test case's kwargs override the `_partial_` in a prior simulator's yaml.

    `params` is `(..., num_params)` in the order of `mcmc_param_names`, and every term broadcasts,
    so passing `params[:, None, :]` against a `(grid,)` hyperparameter evaluates the whole
    quadrature at once.
    """

    def log_prior(params, **overrides):
        params = jnp.asarray(params)
        return term_fn(*(params[..., index] for index in range(num_params)), **{**fixed, **overrides})

    return log_prior


def make_rdm_simple_log_prior(drift_slope_loc, threshold_scale):
    """The simple RDM's prior density over `mcmc_param_names`, as a function of the draws."""
    return _log_prior_factory(
        _rdm_simple_log_prior, 5,
        {"drift_slope_loc": drift_slope_loc, "threshold_scale": threshold_scale},
    )


def make_rdm_sat_log_prior(drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale):
    """The speed-accuracy RDM's prior density over `mcmc_param_names`."""
    return _log_prior_factory(
        _rdm_sat_log_prior, 6,
        {
            "drift_slope_loc": drift_slope_loc,
            "threshold_scale": threshold_scale,
            "threshold_diff_shape": threshold_diff_shape,
            "threshold_diff_scale": threshold_diff_scale,
        },
    )
