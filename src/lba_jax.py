"""JAX/TFP implementation of the linear ballistic accumulator and its BlackJAX likelihood.

Companion to `rdm_jax.py`: same pipeline, same contracts, a different evidence-accumulation
model. Where the racing diffusion model races two noisy Wald accumulators, the LBA (Brown &
Heathcote 2008) races two *ballistic* (noise-free within a trial) accumulators whose randomness
lives entirely between trials -- a uniform start point `k ~ U(0, A)` and a normal drift rate
truncated at zero, `d ~ N(v, s)[0, inf)`. The first passage time of one accumulator is therefore
the deterministic `(b - k) / d`.

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
reasons. Recovering `b` for reporting is a one-line post-hoc `A + B` on the samples.

The unconstrained parameter vector is therefore length-6, ordered
`[v_intercept, v_slope, s_true, A, B, t0]` -- `t0` stays last because `rdm_jax.fit_mcmc_gpu`
initializes it from the data as `init_position.at[-1]`, and every entry is strictly positive
because the NPE adapter log-transforms all inference variables.

Naming: `A` and `B` are the literature's names, and are what appear as prior dict keys,
`inference_variables` entries and config keys. Internal helpers spell them `sp_max` and
`sp_gap` (bare capitals trip ruff's pep8-naming rules), but the two simulator entry points must
keep the literal names, because `simulation.CustomSimulator` splats the prior's dict in as
keyword arguments and the names have to line up.

Note `rdm_jax` enables float64 globally on import; this module relies on that, and on its
`_as_scalar` / `_finalize_race_logp` helpers, rather than duplicating them.
"""

import jax
import jax.numpy as jnp
from jax.scipy import stats as jstats
from tensorflow_probability.substrates import jax as tfp
from rdm_jax import _as_scalar
from rdm_jax import _finalize_race_logp

tfd = tfp.distributions
tfb = tfp.bijectors


# ---------------------------------------------------------------------------
# LBA first-passage log density and log survival.
#
# For decision time `t`, drift mean `v`, drift SD `s`, start-point range `A`
# and threshold `b`, with z1 = (b - A - t*v)/(t*s) and z2 = (b - t*v)/(t*s):
#
#     f(t) = (1/A) [ -v Phi(z1) + s phi(z1) + v Phi(z2) - s phi(z2) ]
#     F(t) = 1 + ((b-A-t v)/A) Phi(z1) - ((b-t v)/A) Phi(z2)
#              + (t s/A) phi(z1) - (t s/A) phi(z2)
#
# both divided by Phi(v/s), the probability that the truncated drift is
# positive (i.e. that the accumulator terminates at all).
#
# Unlike the RDM -- whose Wald survival underflows exponentially and needed the
# hand-rolled `inv_gauss_logsf` -- the LBA's truncated-normal drift gives a
# heavy first-passage tail, so `1 - F(t)` stays well conditioned in float64 far
# out (~5.8e-6 at t = 1e4). Linear-space evaluation plus a floor is therefore
# enough here; no special function is required.
#
# The genuine hazard is the *small*-t end, where f(t) is a difference of nearly
# equal terms and underflows to exactly 0. A signed-logsumexp reformulation was
# tried and buys only about one decade (down to t ~ 0.05) before the
# cancellation eats all precision, so it is not worth the complexity: floor the
# density at `min_p` (the same constant EMC2 floors its race likelihood at) and
# let `_finalize_race_logp` supply the sloped gradient that keeps NUTS from
# stranding when t0 crowds the fastest observed RT.
# ---------------------------------------------------------------------------


def _lba_guard(t, s, sp_max, b, min_p):
    """Clamp inputs into the domain where the LBA pdf/cdf formulae are valid.

    `lba_logpdf`/`lba_logsf` take the raw threshold `b`, so they are callable with any `b`,
    including the degenerate `b < A` where the closed form integrates untruncated drift mass
    below zero and can return a negative density. Capping `sp_max` just below `b` keeps them
    inside their support for arbitrary inputs. Everything upstream of `lba_race_logpdf` passes
    `b = A + B` with `B > 0`, so this cap is inert on the model's own code path.
    """
    t = jnp.maximum(t, min_p)
    s = jnp.maximum(s, min_p)
    b = jnp.maximum(b, min_p)
    sp_max = jnp.maximum(jnp.minimum(sp_max, b - min_p), min_p)
    return t, s, sp_max, b


def _lba_log_truncation(v, s):
    """Log Phi(v/s): the log probability that a truncated drift rate is positive."""
    # `norm.logcdf` rather than `log(norm.cdf(...))` so this stays finite when v/s is
    # very negative (an accumulator that almost never terminates).
    return jstats.norm.logcdf(v / s)


def lba_logpdf(t, v, s, sp_max, b, min_p=1e-10):
    """Log density of one LBA accumulator's first passage time at decision time `t`."""
    t, s, sp_max, b = _lba_guard(t, s, sp_max, b, min_p)

    z1 = (b - sp_max - t * v) / (t * s)
    z2 = (b - t * v) / (t * s)

    bracket = -v * jstats.norm.cdf(z1) + s * jstats.norm.pdf(z1) + v * jstats.norm.cdf(z2) - s * jstats.norm.pdf(z2)

    log_d = jnp.log(jnp.maximum(bracket, min_p)) - jnp.log(sp_max) - _lba_log_truncation(v, s)

    # What gets floored has to be the *density*, not the bracket. Clamping only the bracket (as
    # this did originally) leaves `-log(sp_max) - log Phi(v/s)` to be applied afterwards, and
    # those two terms are usually positive, so they lift the clamped value back *above* the
    # floor: at A=0.6, b=1.8, v=3.5, s=1.2, t=0.05 that returned -22.51 where the true density
    # is e^-146. So substitute the floor outright wherever the bracket has underflowed, and
    # keep a plain lower bound for the (rarer, sp_max > 1) case where the normalization pushes
    # a non-underflowed value below it instead. `min_p` is the same constant as EMC2's
    # `min_ll`, so the floored value is exactly what EMC2's log_likelihood_race records here.
    log_floor = jnp.log(min_p)

    return jnp.where(bracket > min_p, jnp.maximum(log_d, log_floor), log_floor)


def lba_logsf(t, v, s, sp_max, b, min_p=1e-10):
    """Log survival function of one LBA accumulator: it has *not* finished by time `t`."""
    t, s, sp_max, b = _lba_guard(t, s, sp_max, b, min_p)

    z1 = (b - sp_max - t * v) / (t * s)
    z2 = (b - t * v) / (t * s)

    cdf = (
        1.0
        + ((b - sp_max - t * v) / sp_max) * jstats.norm.cdf(z1)
        - ((b - t * v) / sp_max) * jstats.norm.cdf(z2)
        + (t * s / sp_max) * jstats.norm.pdf(z1)
        - (t * s / sp_max) * jstats.norm.pdf(z2)
    )
    cdf = cdf / jnp.maximum(jnp.exp(_lba_log_truncation(v, s)), min_p)

    return jnp.log(jnp.clip(1.0 - cdf, min_p, 1.0))


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------


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
    # Cast to a common float dtype as well as to 0-d: `s_false` arrives as the integer 1 from
    # `conf/simulator/experiment_simulator/lba_simple.yaml`, and tfd's distributions reject a
    # mix of int and float arguments (the RDM never noticed because `jnp.stack` promoted for it).
    float_dtype = jnp.result_type(float)
    v_intercept, v_slope, s_true, s_false, sp_max, sp_gap, t0 = (
        _as_scalar(x).astype(float_dtype) for x in (v_intercept, v_slope, s_true, s_false, A, B, t0)
    )
    num_obs = int(num_obs)

    b = sp_max + sp_gap

    v_false, v_true = v_intercept, v_intercept + v_slope

    # One key per (accumulator, source of between-trial variability). Sampling each
    # accumulator from its own scalar-parameter distribution keeps every draw shaped
    # (num_obs,), so the stack below fixes index 0 = false / 1 = true exactly as the RDM
    # does -- batching both accumulators into one distribution would return (num_obs, 2)
    # and need a transpose, an easy place to silently swap the two.
    keys = jax.random.split(key, 4)

    start_false = tfd.Uniform(0.0, sp_max).sample(num_obs, seed=keys[0])
    drift_false = tfd.TruncatedNormal(v_false, s_false, 0.0, jnp.inf).sample(num_obs, seed=keys[1])
    start_true = tfd.Uniform(0.0, sp_max).sample(num_obs, seed=keys[2])
    drift_true = tfd.TruncatedNormal(v_true, s_true, 0.0, jnp.inf).sample(num_obs, seed=keys[3])

    fpt = jnp.stack([(b - start_false) / drift_false, (b - start_true) / drift_true], axis=0)
    resp = jnp.argmin(fpt, axis=0)
    rt = jnp.min(fpt, axis=0) + t0

    return {"x": jnp.stack([rt, resp.astype(rt.dtype)], axis=-1)}


def lba_experiment_simple_jax_stateful(v_intercept, v_slope, s_true, s_false, A, B, t0, num_obs, rng):  # noqa: N803
    """`lba_experiment_simple_jax`, adapted to the `sample_fn(..., rng)` calling convention
    BayesFlow's `LambdaSimulator` uses (see `conf/simulator/experiment_simulator/lba_simple.yaml`),
    with `rng` a `rdm_jax.SplittableKey` instead of a `numpy.random.Generator`.
    """
    return lba_experiment_simple_jax(rng.next(), v_intercept, v_slope, s_true, s_false, A, B, t0, num_obs)


# ---------------------------------------------------------------------------
# Race log-likelihood: pdf(winner) * survival(loser), mirroring rdm_race_logpdf.
# ---------------------------------------------------------------------------


def lba_race_logpdf(rt, drift_winner, drift_loser, s_winner, s_loser, sp_max, sp_gap, ndt, min_p=1e-10):
    """Log density of one race outcome: winner finished at `rt`, loser hadn't yet.

    Takes the threshold *gap* `sp_gap = b - A` rather than the threshold itself, so the
    reconstructed `threshold` below always exceeds `sp_max` and the density can never be
    evaluated outside its support (see the module docstring).
    """
    rt_shifted = rt - ndt
    rt_safe = jnp.maximum(rt_shifted, min_p)

    threshold = sp_max + jnp.maximum(sp_gap, min_p)

    log_pdf = lba_logpdf(rt_safe, drift_winner, s_winner, sp_max, threshold, min_p=min_p)
    log_sf = lba_logsf(rt_safe, drift_loser, s_loser, sp_max, threshold, min_p=min_p)

    return _finalize_race_logp(rt_shifted, log_pdf, log_sf, min_p)


def _lba_simple_log_prior(
    v_intercept, v_slope, s_true, sp_max, sp_gap, t0, drift_slope_loc, threshold_scale,
):
    """Log prior, matching the hyperparameters sampled by `priors.lba_prior_simple`.

    Every term is an ordinary density on a positive parameter -- because the threshold enters
    as the gap `B = b - A`, there is no `b > A` constraint left to enforce here.

    The drift intercept is centred on 2.0 rather than the RDM's 1.0. The LBA's first-passage
    time is `(b - k) / d`, so a drift rate near zero produces an arbitrarily large RT -- the
    distribution has a `t^-2` tail and no finite mean, and the weight of that tail is set by
    `Phi(-v/s)`. Measured over the prior predictive, centring at 1.0 gives P(RT > 5 s) = 3.6e-3
    with excursions past 190 s, which would swamp the summary network's statistics; centring at
    2.0 brings that to 4.0e-4, in line with the RDM's 2.8e-4, at no cost in accuracy (0.81
    either way) and with a median RT closer to the RDM's. The RDM needs no such adjustment
    because a Wald first-passage time cannot blow up the same way.
    """
    lp = tfd.TruncatedNormal(2.0, 0.5, 0.0, jnp.inf).log_prob(v_intercept)
    lp += tfd.TruncatedNormal(drift_slope_loc, 0.5, 0.0, jnp.inf).log_prob(v_slope)
    lp += tfd.Gamma(12.0, 10.0).log_prob(s_true)
    lp += tfd.Gamma(6.0, 10.0).log_prob(sp_max)
    lp += tfd.Gamma(8.0, 1.0 / threshold_scale).log_prob(sp_gap)
    lp += tfd.TruncatedNormal(0.3, 0.2, 0.0, jnp.inf).log_prob(t0)
    return lp


def _lba_simple_log_likelihood(rt, is_true, v_intercept, v_slope, s_true, sp_max, sp_gap, t0):
    v_true_drift = v_intercept + v_slope
    v_false_drift = v_intercept

    ll_true = lba_race_logpdf(rt, v_true_drift, v_false_drift, s_true, 1.0, sp_max, sp_gap, t0)
    ll_false = lba_race_logpdf(rt, v_false_drift, v_true_drift, 1.0, s_true, sp_max, sp_gap, t0)

    # Masking rather than boolean-indexing keeps array shapes static, hence jit/grad-able.
    return jnp.sum(jnp.where(is_true, ll_true, 0.0)) + jnp.sum(jnp.where(~is_true, ll_false, 0.0))


_EXP = tfb.Exp()


def make_lba_simple_logdensity(data_x, drift_slope_loc, threshold_scale):
    """Build a BlackJAX-ready log-density function for the simple (non-hierarchical) LBA.

    `position` passed to the returned function is a length-6 array of *unconstrained*
    (log-space) values in the order [v_intercept, v_slope, s_true, A, B, t0], where `B` is the
    threshold gap `b - A` -- matching `mcmc_sampling_fun.init_position` in
    conf/model/lba_simple.yaml (via `mcmc.simple_to_unconstrained`, which is just `jnp.log`
    and so is length-agnostic).
    """
    data_x = jnp.asarray(data_x)
    rt = data_x[:, 0]
    is_true = data_x[:, 1] == 1

    def logdensity_fn(position):
        position = jnp.asarray(position)
        constrained = _EXP.forward(position)
        v_intercept, v_slope, s_true, sp_max, sp_gap, t0 = constrained
        jacobian = jnp.sum(_EXP.forward_log_det_jacobian(position, event_ndims=0))

        log_prior = _lba_simple_log_prior(
            v_intercept, v_slope, s_true, sp_max, sp_gap, t0, drift_slope_loc, threshold_scale,
        )
        log_lik = _lba_simple_log_likelihood(rt, is_true, v_intercept, v_slope, s_true, sp_max, sp_gap, t0)

        return log_prior + jacobian + log_lik

    return logdensity_fn


def make_lba_meta_logdensity(
    data_x, drift_slope_loc_lower, drift_slope_loc_upper, threshold_scale_lower, threshold_scale_upper,
):
    """Build a BlackJAX-ready log-density function for the hierarchical (meta) LBA.

    `position` passed to the returned function is a length-8 array of *unconstrained*
    values in the order [drift_slope_loc, threshold_scale, v_intercept, v_slope, s_true,
    A, B, t0], where `B` is the threshold gap `b - A`.
    `mcmc.make_meta_to_unconstrained` is the matching forward transform -- it
    Sigmoid-transforms the two leading hyperparameters and log-transforms `position[2:]`, so it
    applies unchanged to the LBA's six subject-level parameters.
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
        v_intercept, v_slope, s_true, sp_max, sp_gap, t0 = constrained_rest

        jacobian = (
            slope_bij.forward_log_det_jacobian(y_slope, event_ndims=0)
            + scale_bij.forward_log_det_jacobian(y_scale, event_ndims=0)
            + jnp.sum(_EXP.forward_log_det_jacobian(y_rest, event_ndims=0))
        )

        log_prior_hyper = tfd.Uniform(drift_slope_loc_lower, drift_slope_loc_upper).log_prob(
            drift_slope_loc,
        ) + tfd.Uniform(threshold_scale_lower, threshold_scale_upper).log_prob(threshold_scale)
        log_prior = log_prior_hyper + _lba_simple_log_prior(
            v_intercept, v_slope, s_true, sp_max, sp_gap, t0, drift_slope_loc, threshold_scale,
        )
        log_lik = _lba_simple_log_likelihood(rt, is_true, v_intercept, v_slope, s_true, sp_max, sp_gap, t0)

        return log_prior + jacobian + log_lik

    return logdensity_fn
