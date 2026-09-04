"""JAX/TFP implementation of the racing diffusion model and its BlackJAX likelihood.

Replaces the PyMC + `pymc.sampling.jax.get_jaxified_logp` path (former `pymc_models.py`)
with a pure JAX implementation built on `tensorflow_probability.substrates.jax`. The
log-density functions built here are ordinary JAX functions of an unconstrained
parameter array and are passed straight to BlackJAX (e.g. `blackjax.nuts`) -- no
PyMC model object or jaxification step involved.

Both accumulators' first-passage times are Wald/inverse-Gaussian with mean `mu = b/v`
and shape `lam = (b/s)**2`.
"""

import functools
import logging
import threading

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy import stats as jstats
from tensorflow_probability.substrates import jax as tfp

# NUTS window adaptation and the race log-density (small squared terms, exp/log
# transforms of sub-unit values like t0) are precision-sensitive; PyMC's jax
# backend ran these in float64 by default, so match that here.
jax.config.update("jax_enable_x64", True)

tfd = tfp.distributions
tfb = tfp.bijectors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Numerically stable inverse-Gaussian (Wald) log density and log survival.
#
# tfd.InverseGaussian.log_survival_function underflows to -inf for moderately
# large x (verified against scipy.stats.invgauss.logsf), which would zero out
# gradients for the race likelihood's "loser hasn't finished yet" term.
# Follows Giner & Smyth 2016 (https://journal.r-project.org/archive/2016-1/
# giner-smyth.pdf), matching confrdm_jax.likelihoods.rdm's implementation:
# `jax.scipy.stats.norm.logcdf` is used directly rather than a hand-rolled
# erfcx-based version (verified numerically stable -- including finite
# gradients and correct values -- across x in [-50, 10] against scipy, so the
# earlier hand-rolled version's extra care against a confirmed jax.scipy.
# special.erfcx bug is unnecessary here). Callers are expected to keep `t`
# strictly positive (see `rdm_race_logpdf`'s upstream floor).
# ---------------------------------------------------------------------------


def inv_gauss_logpdf(t, mu, lam):
    """Log density of the Wald/inverse-Gaussian distribution with mean `mu`, shape `lam`."""
    e = -(lam / (2 * t)) * (t**2 / mu**2 - 2 * t / mu + 1)
    return e + 0.5 * jnp.log(lam) - 0.5 * jnp.log(2 * t**3 * jnp.pi)


def inv_gauss_logsf(t, mu, lam):
    """Log survival function of the Wald/inverse-Gaussian distribution (`t > 0`)."""
    mu = mu / lam
    t = t / lam
    r = 1.0 / jnp.sqrt(t)
    a = jstats.norm.logcdf(-r * (t / mu - 1.0))
    b = 2.0 / mu + jstats.norm.logcdf(-r * (t + mu) / mu)
    # b <= a always holds mathematically, but floating-point arithmetic can violate
    # it; clamp b - a <= 0 so the log1p argument never drops below -1 (-> NaN).
    return a + jnp.log1p(-jnp.exp(jnp.minimum(b - a, 0.0)))


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


def _as_scalar(x):
    """Collapse a Python float, 0-d array, or shape-(1,) array to a true 0-d scalar.

    BayesFlow's `batched_call` (see `rdm_experiment_simple_jax_stateful`) indexes
    batched kwargs down to shape-(1,) arrays per call rather than plain scalars
    (unlike `numpy.hstack`, `jnp.stack` doesn't silently collapse those), so inputs
    need normalizing before use.
    """
    return jnp.reshape(jnp.asarray(x), ())


def rdm_experiment_simple_jax(key, v_intercept, v_slope, s_true, s_false, b, t0, num_obs):
    """Simulate `num_obs` trials from the two-accumulator racing diffusion model.

    Draws first-passage times from `tfd.InverseGaussian(mu, lam)`.
    `num_obs` must be static (it sets the sample shape) so this is jit/vmap-able over
    `key` and the scalar parameters.
    """
    v_intercept, v_slope, s_true, s_false, b, t0 = (
        _as_scalar(x) for x in (v_intercept, v_slope, s_true, s_false, b, t0)
    )
    num_obs = int(num_obs)

    v_false, v_true = v_intercept, v_intercept + v_slope
    s_arr = jnp.stack([s_false, s_true])
    v_arr = jnp.stack([v_false, v_true])

    mu = b / v_arr
    lam = (b / s_arr) ** 2

    key_false, key_true = jax.random.split(key)
    fpt_false = tfd.InverseGaussian(mu[0], lam[0]).sample(num_obs, seed=key_false)
    fpt_true = tfd.InverseGaussian(mu[1], lam[1]).sample(num_obs, seed=key_true)

    fpt = jnp.stack([fpt_false, fpt_true], axis=0)
    resp = jnp.argmin(fpt, axis=0)
    rt = jnp.min(fpt, axis=0) + t0

    return {"x": jnp.stack([rt, resp.astype(rt.dtype)], axis=-1)}


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

    `num_obs` sets the sample shape of the first-passage-time draws, so it has to be static and
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

    # Priors hand over `(batch, 1)`, a `Case` may pin a 0-d value, and `s_false` is a bare int
    # from the yaml; flattening and broadcasting here means `single_fn` needs no such handling.
    # The explicit dtype matters for the LBA, whose TFP distributions reject mixed int/float.
    float_dtype = jnp.result_type(float)
    params = tuple(
        jnp.broadcast_to(jnp.reshape(jnp.asarray(value, dtype=float_dtype), (-1,)), (size,))
        for value in params
    )

    out = _compiled_experiment(single_fn)(rng.next(size), params, num_obs)

    return {key: np.asarray(value).reshape(*batch_shape, *value.shape[1:]) for key, value in out.items()}


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
    """
    return batched_experiment(
        rdm_experiment_simple_jax, batch_shape, num_obs, rng,
        (v_intercept, v_slope, s_true, s_false, b, t0),
    )


# ---------------------------------------------------------------------------
# Race log-likelihood: pdf(winner) * survival(loser), as in the former
# PyMC `RdmSimple.logp`, but as a plain JAX function.
# ---------------------------------------------------------------------------


def _finalize_race_logp(rt_shifted, log_pdf, log_sf, min_p):
    """Combine a winner's log-density with a loser's log-survival into one trial's log-likelihood.

    Two things happen here, and the order between them is the whole point:

    1. Feasible trials are floored at `log(min_p)`. This matches EMC2's `log_likelihood_race`,
       which floors each trial at `min_ll = log(1e-10)` -- the same constant -- so the two
       implementations agree even where the density underflows.
    2. Trials with `rt - t0 <= 0` are impossible under the model, and get a steep linear penalty
       instead. A flat floor would give *zero* gradient there, which can strand NUTS in the
       infeasible region; the slope keeps a gradient that pushes `t0` back below the minimum
       observed RT.

    Applying the floor after the penalty (which is what the previous `_penalize_invalid_rt` plus
    a trailing `jnp.maximum` in each race function did) clamps the slope straight back off and
    leaves an exactly-zero gradient -- precisely the failure the penalty exists to prevent. So
    the floor goes first, and the penalty is substituted in afterwards.
    """
    log_floor = jnp.log(min_p)
    logp = jnp.maximum(jnp.nan_to_num(log_pdf + log_sf, nan=log_floor), log_floor)
    penalty = log_floor + 1e3 * jnp.minimum(rt_shifted - min_p, 0.0)
    return jnp.where(rt_shifted > min_p, logp, penalty)


def rdm_race_logpdf(rt, drift_winner, drift_loser, s_winner, s_loser, threshold, ndt, min_p=1e-10):
    """Log density of one race outcome: winner finished at `rt`, loser hadn't yet."""
    rt_shifted = rt - ndt
    rt_safe = jnp.maximum(rt_shifted, min_p)

    # Priors keep these positive via a log-transform (see mcmc.simple_to_unconstrained),
    # so this floor only guards against exp() underflowing to exactly 0.0, rather
    # than gating validity -- unlike a hard `-inf` rejection, it keeps a finite
    # gradient even if the sampler briefly explores a near-zero parameter.
    drift_winner = jnp.maximum(drift_winner, min_p)
    drift_loser = jnp.maximum(drift_loser, min_p)
    s_winner = jnp.maximum(s_winner, min_p)
    s_loser = jnp.maximum(s_loser, min_p)
    threshold = jnp.maximum(threshold, min_p)

    mu_winner = threshold / drift_winner
    mu_loser = threshold / drift_loser
    lam_winner = (threshold / s_winner) ** 2
    lam_loser = (threshold / s_loser) ** 2

    log_pdf = inv_gauss_logpdf(rt_safe, mu_winner, lam_winner)
    log_sf = inv_gauss_logsf(rt_safe, mu_loser, lam_loser)

    return _finalize_race_logp(rt_shifted, log_pdf, log_sf, min_p)


def _rdm_simple_log_prior(v_intercept, v_slope, s_true, b, t0, drift_slope_loc, threshold_scale):
    """Log prior, matching the fixed hyperparameters in the former `rdm_model_simple`."""
    lp = tfd.TruncatedNormal(1.0, 0.5, 0.0, jnp.inf).log_prob(v_intercept)
    lp += tfd.TruncatedNormal(drift_slope_loc, 0.5, 0.0, jnp.inf).log_prob(v_slope)
    lp += tfd.Gamma(12.0, 10.0).log_prob(s_true)
    lp += tfd.Gamma(8.0, 1.0 / threshold_scale).log_prob(b)
    lp += tfd.TruncatedNormal(0.3, 0.2, 0.0, jnp.inf).log_prob(t0)
    return lp


def _rdm_simple_log_likelihood(rt, is_true, v_intercept, v_slope, s_true, b, t0):
    v_true_drift = v_intercept + v_slope
    v_false_drift = v_intercept

    ll_true = rdm_race_logpdf(rt, v_true_drift, v_false_drift, s_true, 1.0, b, t0)
    ll_false = rdm_race_logpdf(rt, v_false_drift, v_true_drift, 1.0, s_true, b, t0)

    # Masking (rather than boolean-indexing rt/resp into separate arrays, as the
    # PyMC version did) keeps this jit/grad-able: array shapes must stay static.
    return jnp.sum(jnp.where(is_true, ll_true, 0.0)) + jnp.sum(jnp.where(~is_true, ll_false, 0.0))


_EXP = tfb.Exp()


def make_rdm_simple_logdensity(data_x, drift_slope_loc, threshold_scale):
    """Build a BlackJAX-ready log-density function for the simple (non-hierarchical) RDM.

    `position` passed to the returned function is a length-5 array of *unconstrained*
    (log-space) values in the order [v_intercept, v_slope, s_true, b, t0] -- matching
    `mcmc_sampling_fun.init_position` in conf/experiment/*.yaml (via `mcmc.simple_to_unconstrained`).
    """
    data_x = jnp.asarray(data_x)
    rt = data_x[:, 0]
    is_true = data_x[:, 1] == 1

    def logdensity_fn(position):
        position = jnp.asarray(position)
        constrained = _EXP.forward(position)
        v_intercept, v_slope, s_true, b, t0 = constrained
        jacobian = jnp.sum(_EXP.forward_log_det_jacobian(position, event_ndims=0))

        log_prior = _rdm_simple_log_prior(v_intercept, v_slope, s_true, b, t0, drift_slope_loc, threshold_scale)
        log_lik = _rdm_simple_log_likelihood(rt, is_true, v_intercept, v_slope, s_true, b, t0)

        return log_prior + jacobian + log_lik

    return logdensity_fn


def make_rdm_meta_logdensity(
    data_x, drift_slope_loc_lower, drift_slope_loc_upper, threshold_scale_lower, threshold_scale_upper,
):
    """Build a BlackJAX-ready log-density function for the hierarchical (meta) RDM.

    `position` passed to the returned function is a length-7 array of *unconstrained*
    values in the order [drift_slope_loc, threshold_scale, v_intercept, v_slope, s_true,
    b, t0] (see `mcmc.make_meta_to_unconstrained` for the matching forward transform).
    """
    data_x = jnp.asarray(data_x)
    rt = data_x[:, 0]
    is_true = data_x[:, 1] == 1

    slope_bij = tfb.Sigmoid(low=drift_slope_loc_lower, high=drift_slope_loc_upper)
    scale_bij = tfb.Sigmoid(low=threshold_scale_lower, high=threshold_scale_upper)

    def logdensity_fn(position):
        position = jnp.asarray(position)
        y_slope, y_scale, y_rest = position[0], position[1], position[2:]

        drift_slope_loc = slope_bij.forward(y_slope)
        threshold_scale = scale_bij.forward(y_scale)
        constrained_rest = _EXP.forward(y_rest)
        v_intercept, v_slope, s_true, b, t0 = constrained_rest

        jacobian = (
            slope_bij.forward_log_det_jacobian(y_slope, event_ndims=0)
            + scale_bij.forward_log_det_jacobian(y_scale, event_ndims=0)
            + jnp.sum(_EXP.forward_log_det_jacobian(y_rest, event_ndims=0))
        )

        log_prior_hyper = tfd.Uniform(drift_slope_loc_lower, drift_slope_loc_upper).log_prob(
            drift_slope_loc,
        ) + tfd.Uniform(threshold_scale_lower, threshold_scale_upper).log_prob(threshold_scale)
        log_prior = log_prior_hyper + _rdm_simple_log_prior(
            v_intercept, v_slope, s_true, b, t0, drift_slope_loc, threshold_scale,
        )
        log_lik = _rdm_simple_log_likelihood(rt, is_true, v_intercept, v_slope, s_true, b, t0)

        return log_prior + jacobian + log_lik

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
#
# The condition is carried as a third channel of `x`, alongside RT and response,
# because the adapter treats `x` as a set: anything the likelihood needs per
# trial has to travel with that trial rather than in a separate array.
# ---------------------------------------------------------------------------


def sat_conditions(num_obs):
    """Condition indicator per trial: the first half speed (0), the rest accuracy (1).

    A fixed split rather than a random assignment -- the number of trials per condition is
    then a design constant instead of a source of between-dataset variance, and `x` is
    permutation-invariant downstream anyway (`simulation.create_data_adapter` marks it
    `as_set`), so the ordering carries no information.
    """
    return jnp.arange(num_obs) >= num_obs // 2


def rdm_experiment_sat_jax(key, v_intercept, v_slope, s_true, s_false, b, b_diff, t0, num_obs):
    """Simulate `num_obs` trials, half under a speed and half under an accuracy instruction.

    Returns `x` with three channels: response time, response (0 = false, 1 = true) and
    condition (0 = speed, 1 = accuracy). `num_obs` must be static, as in
    `rdm_experiment_simple_jax`.
    """
    v_intercept, v_slope, s_true, s_false, b, b_diff, t0 = (
        _as_scalar(x) for x in (v_intercept, v_slope, s_true, s_false, b, b_diff, t0)
    )
    num_obs = int(num_obs)

    is_accuracy = sat_conditions(num_obs)
    threshold = jnp.where(is_accuracy, b + b_diff, b)

    v_arr = jnp.stack([v_intercept, v_intercept + v_slope])
    s_arr = jnp.stack([s_false, s_true])

    # (num_obs, 2): one row per trial, one column per accumulator, so the per-trial
    # threshold broadcasts into both accumulators of that trial.
    mu = threshold[:, None] / v_arr[None, :]
    lam = (threshold[:, None] / s_arr[None, :]) ** 2

    fpt = tfd.InverseGaussian(mu, lam).sample(seed=key)

    resp = jnp.argmin(fpt, axis=-1)
    rt = jnp.min(fpt, axis=-1) + t0

    return {"x": jnp.stack([rt, resp.astype(rt.dtype), is_accuracy.astype(rt.dtype)], axis=-1)}


def rdm_experiment_sat_jax_stateful(v_intercept, v_slope, s_true, s_false, b, b_diff, t0, num_obs, rng):
    """`rdm_experiment_sat_jax` under the `sample_fn(..., rng)` convention, with `rng` a `SplittableKey`."""
    return rdm_experiment_sat_jax(rng.next(), v_intercept, v_slope, s_true, s_false, b, b_diff, t0, num_obs)


def rdm_experiment_sat_jax_batched(batch_shape, v_intercept, v_slope, s_true, s_false, b, b_diff, t0, num_obs, rng):
    """`rdm_experiment_sat_jax` for a whole batch at once."""
    return batched_experiment(
        rdm_experiment_sat_jax, batch_shape, num_obs, rng,
        (v_intercept, v_slope, s_true, s_false, b, b_diff, t0),
    )


def _rdm_sat_log_prior(
    v_intercept, v_slope, s_true, b, b_diff, t0,
    drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
):
    """Log prior for the speed-accuracy model: the simple model's, plus the threshold difference.

    `threshold_diff_shape`/`threshold_diff_scale` parameterize `Gamma(shape, scale)` exactly as
    `priors.rdm_prior_sat` draws it -- both are interpolated from the same prior config in
    `conf/mcmc/rdm_sat.yaml`, so the sampled prior and the fitted prior cannot drift apart.
    """
    lp = _rdm_simple_log_prior(v_intercept, v_slope, s_true, b, t0, drift_slope_loc, threshold_scale)
    return lp + tfd.Gamma(threshold_diff_shape, 1.0 / threshold_diff_scale).log_prob(b_diff)


def _rdm_sat_log_likelihood(rt, is_true, is_accuracy, v_intercept, v_slope, s_true, b, b_diff, t0):
    """As `_rdm_simple_log_likelihood`, but with a per-trial threshold."""
    threshold = jnp.where(is_accuracy, b + b_diff, b)

    v_true_drift = v_intercept + v_slope
    v_false_drift = v_intercept

    ll_true = rdm_race_logpdf(rt, v_true_drift, v_false_drift, s_true, 1.0, threshold, t0)
    ll_false = rdm_race_logpdf(rt, v_false_drift, v_true_drift, 1.0, s_true, threshold, t0)

    return jnp.sum(jnp.where(is_true, ll_true, 0.0)) + jnp.sum(jnp.where(~is_true, ll_false, 0.0))


def make_rdm_sat_logdensity(data_x, drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale):
    """Build a BlackJAX-ready log-density for the speed-accuracy RDM.

    `position` is a length-6 array of *unconstrained* (log-space) values in the order
    [v_intercept, v_slope, s_true, b, b_diff, t0] -- matching `mcmc_param_names` in
    `conf/mcmc/rdm_sat.yaml`. `t0` stays last: `mcmc.fit_mcmc_gpu` initializes the final
    entry from the smallest observed RT.
    """
    data_x = jnp.asarray(data_x)
    rt = data_x[:, 0]
    is_true = data_x[:, 1] == 1
    is_accuracy = data_x[:, 2] == 1

    def logdensity_fn(position):
        position = jnp.asarray(position)
        v_intercept, v_slope, s_true, b, b_diff, t0 = _EXP.forward(position)
        jacobian = jnp.sum(_EXP.forward_log_det_jacobian(position, event_ndims=0))

        log_prior = _rdm_sat_log_prior(
            v_intercept, v_slope, s_true, b, b_diff, t0,
            drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
        )
        log_lik = _rdm_sat_log_likelihood(
            rt, is_true, is_accuracy, v_intercept, v_slope, s_true, b, b_diff, t0,
        )

        return log_prior + jacobian + log_lik

    return logdensity_fn


def make_rdm_sat_meta_logdensity(
    data_x, drift_slope_loc, threshold_scale, threshold_diff_shape,
    threshold_diff_scale_lower, threshold_diff_scale_upper,
):
    """Build a BlackJAX-ready log-density for the hierarchical speed-accuracy RDM.

    `position` is a length-7 array of *unconstrained* values in the order
    [threshold_diff_scale, v_intercept, v_slope, s_true, b, b_diff, t0]: the leading
    hyperparameter is Uniform on its support and so uses a Sigmoid bijector, the rest are
    positive and use Exp. See `mcmc.make_bounded_to_unconstrained` for the matching forward
    transform.
    """
    data_x = jnp.asarray(data_x)
    rt = data_x[:, 0]
    is_true = data_x[:, 1] == 1
    is_accuracy = data_x[:, 2] == 1

    scale_bij = tfb.Sigmoid(low=threshold_diff_scale_lower, high=threshold_diff_scale_upper)

    def logdensity_fn(position):
        position = jnp.asarray(position)
        y_scale, y_rest = position[0], position[1:]

        threshold_diff_scale = scale_bij.forward(y_scale)
        v_intercept, v_slope, s_true, b, b_diff, t0 = _EXP.forward(y_rest)

        jacobian = scale_bij.forward_log_det_jacobian(y_scale, event_ndims=0) + jnp.sum(
            _EXP.forward_log_det_jacobian(y_rest, event_ndims=0),
        )

        log_prior = tfd.Uniform(threshold_diff_scale_lower, threshold_diff_scale_upper).log_prob(
            threshold_diff_scale,
        ) + _rdm_sat_log_prior(
            v_intercept, v_slope, s_true, b, b_diff, t0,
            drift_slope_loc, threshold_scale, threshold_diff_shape, threshold_diff_scale,
        )
        log_lik = _rdm_sat_log_likelihood(
            rt, is_true, is_accuracy, v_intercept, v_slope, s_true, b, b_diff, t0,
        )

        return log_prior + jacobian + log_lik

    return logdensity_fn
