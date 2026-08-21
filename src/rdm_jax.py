"""JAX/TFP implementation of the racing diffusion model and its BlackJAX likelihood.

Replaces the PyMC + `pymc.sampling.jax.get_jaxified_logp` path (former `pymc_models.py`)
with a pure JAX implementation built on `tensorflow_probability.substrates.jax`. The
log-density functions built here are ordinary JAX functions of an unconstrained
parameter array and are passed straight to BlackJAX (e.g. `blackjax.nuts`) -- no
PyMC model object or jaxification step involved.

Both accumulators' first-passage times are Wald/inverse-Gaussian with mean `mu = b/v`
and shape `lam = (b/s)**2`.
"""

import logging
import jax
import jax.numpy as jnp
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
    """Stateful `jax.random.PRNGKey` wrapper, for use as `rdm_experiment_simple_jax_stateful`'s `rng`.

    `bayesflow.utils.batched_call` calls `sample_fn` once per batch element via a plain
    Python for-loop, passing the *same* `rng` object reference on every call -- exactly
    how `numpy.random.default_rng()` is used elsewhere in this pipeline (`conf/simulator/
    */*.yaml`), which works because a numpy `Generator` mutates its own state on every
    draw. A bare `jax.random.PRNGKey` is immutable/stateless, so this wraps one to give
    the same call-by-reference-then-advance behaviour: `.next()` splits off and returns a
    fresh subkey, then updates its own internal state, so repeated calls (e.g. across
    training batches) never repeat the same randomness.
    """

    def __init__(self, seed):
        self._key = jax.random.PRNGKey(seed)

    def next(self):
        self._key, subkey = jax.random.split(self._key)
        return subkey


def rdm_experiment_simple_jax_stateful(v_intercept, v_slope, s_true, s_false, b, t0, num_obs, rng):
    """`rdm_experiment_simple_jax`, adapted to the `sample_fn(..., rng)` calling convention
    BayesFlow's `LambdaSimulator` uses (see `conf/simulator/experiment_simulator/rdm_simple.yaml`),
    with `rng` a `SplittableKey` instead of a `numpy.random.Generator`.
    """
    return rdm_experiment_simple_jax(rng.next(), v_intercept, v_slope, s_true, s_false, b, t0, num_obs)


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
