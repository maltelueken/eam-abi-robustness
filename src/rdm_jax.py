"""JAX/TFP implementation of the racing diffusion model and its BlackJAX likelihood.

Replaces the PyMC + `pymc.sampling.jax.get_jaxified_logp` path (former `pymc_models.py`)
with a pure JAX implementation built on `tensorflow_probability.substrates.jax`. The
log-density functions built here are ordinary JAX functions of an unconstrained
parameter array and are passed straight to BlackJAX (e.g. `blackjax.nuts`) -- no
PyMC model object or jaxification step involved.

Both accumulators' first-passage times are Wald/inverse-Gaussian with mean `mu = b/v`
and shape `lam = (b/s)**2`, exactly as in `simulation.rdm_experiment_simple`.
"""

import logging
from datetime import date
import blackjax
import jax
import jax.numpy as jnp
from jax.scipy import special as jspecial
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
# gradients for the race likelihood's "loser hasn't finished yet" term. This
# stable version is a JAX port of the PyTensor implementation formerly in
# pymc_models.py, following Giner & Smyth 2016
# (https://journal.r-project.org/archive/2016-1/giner-smyth.pdf).
# ---------------------------------------------------------------------------


def inv_gauss_logpdf(t, mu, lam):
    """Log density of the Wald/inverse-Gaussian distribution with mean `mu`, shape `lam`."""
    e = -(lam / (2 * t)) * (t**2 / mu**2 - 2 * t / mu + 1)
    return e + 0.5 * jnp.log(lam) - 0.5 * jnp.log(2 * t**3 * jnp.pi)


_ERFCX_CUTOFF = 8.0


def _erfcx_stable(x):
    """`erfcx(x)` for `x >= 0`, avoiding `jax.scipy.special.erfcx` directly.

    That primitive has a confirmed, isolated bug at this JAX version (0.11.0):
    `erfcx(26.627...)` silently returns `0.0` instead of `~0.0212` (verified
    against scipy), with correct neighbouring values -- almost certainly a
    rational-approximation edge case in XLA's implementation. Below the cutoff
    this uses the safe direct identity `exp(x**2) * erfc(x)` (no overflow for
    `x <= 8`: `exp(64)` is far under float64's range, and `erfc(8)` doesn't
    underflow); above it, the standard asymptotic series, whose terms shrink
    with `x`, so it's most accurate exactly where it's used. Matches
    `scipy.special.erfcx` to a relative error < 3e-9 for `x` in [0, 200]
    (verified numerically).
    """
    is_small = x <= _ERFCX_CUTOFF
    x_small = jnp.where(is_small, x, 1.0)
    small = jnp.exp(x_small**2) * jspecial.erfc(x_small)

    x_large = jnp.where(is_small, 1.0, x)
    inv_x2 = 1.0 / (x_large * x_large)
    series = 1.0 - inv_x2 * (0.5 - inv_x2 * (0.75 - inv_x2 * (1.875 - inv_x2 * (6.5625 - inv_x2 * 29.53125))))
    large = series / (x_large * jnp.sqrt(jnp.pi))

    return jnp.where(is_small, small, large)


def _standard_normal_logcdf(t):
    """Numerically stable log CDF of the standard normal distribution.

    `jnp.where` still runs autodiff through *both* branches (it only masks the
    cotangent afterwards), so if either branch has a genuine singularity at the
    other branch's typical values -- and it does here: `log1p(-erfc(x)/2)`
    saturates to exactly `log1p(-1) = -inf` once `x` is a few units past 0, an
    infinite/NaN gradient -- the unselected branch still poisons the total via
    `0 * inf = nan`. Feeding each branch a safe substitute value whenever it is
    not the active one keeps both branches' local gradients finite.
    """
    is_left_tail = t < -1.0
    t_left = jnp.where(is_left_tail, t, -1.0)
    t_right = jnp.where(is_left_tail, -1.0, t)

    left = jnp.log(_erfcx_stable(-t_left / jnp.sqrt(2.0)) / 2.0) - t_left**2 / 2.0
    right = jnp.log1p(-jspecial.erfc(t_right / jnp.sqrt(2.0)) / 2.0)

    return jnp.where(is_left_tail, left, right)


def inv_gauss_logsf(t, mu, lam):
    """Log survival function of the Wald/inverse-Gaussian distribution.

    See the `_standard_normal_logcdf` docstring for why `t` is clamped away
    from 0 (where `1/sqrt(t)` diverges) even in the branch where it's unused.
    """
    mu = mu / lam
    t = t / lam

    is_pos = t > 0.0
    t_safe = jnp.where(is_pos, t, 1.0)

    r = 1.0 / jnp.sqrt(t_safe)
    a = _standard_normal_logcdf(-r * (t_safe / mu - 1.0))
    b = 2.0 / mu + _standard_normal_logcdf(-r * (t_safe + mu) / mu)
    logsf = a + jnp.log1p(-jnp.exp(b - a))

    logsf = jnp.where(is_pos, logsf, 0.0)
    return jnp.where(jnp.isposinf(t), -jnp.inf, logsf)


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

    JAX/TFP equivalent of `simulation.rdm_experiment_simple`: draws first-passage
    times from `tfd.InverseGaussian(mu, lam)` instead of `numpy.random.Generator.wald`.
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


def rdm_race_logpdf(rt, drift_winner, drift_loser, s_winner, s_loser, threshold, ndt, min_p=1e-10):
    """Log density of one race outcome: winner finished at `rt`, loser hadn't yet."""
    t = rt - ndt
    is_pos = t > 0.0
    t_safe = jnp.where(is_pos, t, 1.0)  # keep 1/t (in inv_gauss_logpdf) finite when unused

    mu_winner = threshold / drift_winner
    mu_loser = threshold / drift_loser
    lam_winner = (threshold / s_winner) ** 2
    lam_loser = (threshold / s_loser) ** 2

    logp = jnp.where(
        is_pos,
        inv_gauss_logpdf(t_safe, mu_winner, lam_winner) + inv_gauss_logsf(t_safe, mu_loser, lam_loser),
        jnp.log(min_p),
    )
    logp = jnp.where(jnp.isnan(logp) | jnp.isinf(logp), jnp.log(min_p), logp)
    logp = jnp.maximum(logp, jnp.log(min_p))

    valid = (
        (drift_winner > 0)
        & (drift_loser > 0)
        & (s_winner > 0)
        & (s_loser > 0)
        & (threshold > 0)
        & (ndt > 0)
    )
    return jnp.where(valid, logp, -jnp.inf)


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


def simple_to_unconstrained(position):
    """[v_intercept, v_slope, s_true, b, t0] natural scale -> unconstrained log-space."""
    return jnp.log(position)


def make_rdm_simple_logdensity(data_x, drift_slope_loc, threshold_scale):
    """Build a BlackJAX-ready log-density function for the simple (non-hierarchical) RDM.

    `position` passed to the returned function is a length-5 array of *unconstrained*
    (log-space) values in the order [v_intercept, v_slope, s_true, b, t0] -- matching
    `mcmc_sampling_fun.init_position` in conf/experiment/*.yaml (via `simple_to_unconstrained`).
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


def make_meta_to_unconstrained(drift_slope_loc_lower, drift_slope_loc_upper, threshold_scale_lower, threshold_scale_upper):
    """Build the natural-scale -> unconstrained transform for the meta RDM.

    position = [drift_slope_loc, threshold_scale, v_intercept, v_slope, s_true, b, t0].
    The two hyperparameters are Uniform-distributed (bounded both sides), so they use a
    Sigmoid bijector rather than the Exp/log-transform used for the other five.
    """
    slope_bij = tfb.Sigmoid(low=drift_slope_loc_lower, high=drift_slope_loc_upper)
    scale_bij = tfb.Sigmoid(low=threshold_scale_lower, high=threshold_scale_upper)

    def to_unconstrained(position):
        position = jnp.asarray(position)
        y_slope = slope_bij.inverse(position[0])
        y_scale = scale_bij.inverse(position[1])
        y_rest = jnp.log(position[2:])
        return jnp.concatenate([jnp.stack([y_slope, y_scale]), y_rest])

    return to_unconstrained


def make_rdm_meta_logdensity(
    data_x, drift_slope_loc_lower, drift_slope_loc_upper, threshold_scale_lower, threshold_scale_upper,
):
    """Build a BlackJAX-ready log-density function for the hierarchical (meta) RDM.

    `position` passed to the returned function is a length-7 array of *unconstrained*
    values in the order [drift_slope_loc, threshold_scale, v_intercept, v_slope, s_true,
    b, t0] (see `make_meta_to_unconstrained` for the matching forward transform).
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
# BlackJAX runner (unchanged in spirit from the former pymc_models.py: it only
# ever consumed a plain logdensity_fun, so it never actually depended on PyMC).
# ---------------------------------------------------------------------------


def inference_loop(rng_key, kernel, initial_state, num_samples):
    """BlackJAX MCMC inference loop."""

    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, infos


def warmup(sampler_fun, logdensity_fun, init_position, num_steps, rng_key, **kwargs):
    """BlackJAX window adaptation warmup."""
    adapt = blackjax.window_adaptation(sampler_fun, logdensity_fun, **kwargs)
    (last_state, parameters), _ = adapt.run(rng_key, init_position, num_steps=num_steps)
    kernel = sampler_fun(logdensity_fun, **parameters).step
    return kernel, last_state, parameters


def run_mcmc(
    logdensity_fun,
    sampler_fun,
    init_position,
    num_chains,
    num_steps_warmup,
    num_steps_sampling,
    min_rt=None,
    rng_key=None,
    to_unconstrained=simple_to_unconstrained,
    **kwargs,
):
    """Perform MCMC inference directly against a JAX log-density function."""
    if rng_key is None:
        rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))

    rng_key, warmup_key = jax.random.split(rng_key)

    init_position = jnp.asarray(init_position)
    if min_rt is not None:
        init_position = init_position.at[-1].set(0.5 * min_rt)

    kernel, last_state, _ = warmup(
        sampler_fun,
        logdensity_fun,
        to_unconstrained(init_position),
        num_steps_warmup,
        warmup_key,
        **kwargs,
    )

    last_states = jax.vmap(lambda x: last_state)(jnp.arange(num_chains))

    sample_keys = jax.random.split(rng_key, num_chains)

    inference_loop_multiple_chains = jax.pmap(
        inference_loop, in_axes=(0, None, 0, None), static_broadcasted_argnums=(1, 3),
    )

    trace = inference_loop_multiple_chains(sample_keys, kernel, last_states, num_steps_sampling)

    return trace
