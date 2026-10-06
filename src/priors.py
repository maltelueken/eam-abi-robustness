"""The priors: one definition per model, shared by the simulator and the MCMC reference.

Each model's prior is written once, as a `*_prior_dists` function that takes *every*
hyperparameter -- under the names `conf/simulator/prior_simulator/*.yaml` binds -- and returns
one TFP distribution per parameter, keyed by name, in `mcmc_param_names` order. Everything that
needs the prior is built from that one function:

* the prior simulators (`rdm_prior_simple` and friends) draw the training and test data from it;
* `rdm_jax`/`lba_jax` score MCMC positions against it, draw the chains' starting values from it,
  and expose its density to `scripts/prior_distance.py`.

So the prior the data were generated under and the prior the reference posterior is fitted
under are the same object, not two transcriptions of the same numbers. `conf/mcmc/*.yaml` reads
the hyperparameters straight out of the prior simulator's `_partial_` (`hyperparameters`), so
there is nothing left to keep in step by hand either.

The simulators are *batched* (`is_batched: true`): one call draws the whole batch, an array of
shape `batch_shape` per parameter, which `simulation.CustomSimulator` lifts to `(batch, 1)`.
Hyperparameters may arrive as a scalar from the yaml, as a `(batch,)` array from a hierarchical
model's meta simulator, or as a 0-d array pinned by a test case; each is broadcast to
`batch_shape` before the distributions are built, so a draw is always one value per dataset.
The draws come from the `rdm_jax.SplittableKey` passed as `rng`, on a stream of their own (see
the prior configs), and are handed back as numpy, since everything downstream -- rejection
sampling, the adapter, `data.save_dataset` -- indexes and concatenates them as such. They are in
JAX's default float width: 64-bit wherever `mcmc` has been imported, 32-bit in a training
process, which is the width the experiment simulator then runs in anyway.
"""

import inspect

import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions


def _float(value):
    """As an array of JAX's default float width.

    TFP infers `float32` from Python floats, so without this every prior built from the yaml's
    hyperparameters would score and sample in single precision even under 64-bit JAX.
    """
    return jnp.asarray(value, dtype=jnp.result_type(float))


def truncated_normal(loc, scale, lower=0.0):
    """A normal distribution truncated below at `lower`."""
    return tfd.TruncatedNormal(_float(loc), _float(scale), _float(lower), jnp.inf)


def gamma(shape, scale):
    """A Gamma by shape and *scale*, the parameterization the prior configs use.

    `tfd.Gamma` takes a rate, so the conversion lives here and nowhere else.
    """
    return tfd.Gamma(_float(shape), 1.0 / _float(scale))


def gamma_mean_sd(loc, sd):
    """A Gamma parameterized by its *mean* and *standard deviation*.

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
    return tfd.Gamma((loc / sd) ** 2, loc / sd**2)


def _before_t0(dists, name, dist):
    """`dists` with `name` spliced in ahead of `t0`, which every parameter vector keeps last."""
    *head, (last, t0) = dists.items()
    assert last == "t0", last
    return {**dict(head), name: dist, "t0": t0}


def rdm_simple_prior_dists(
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
):
    """The racing diffusion model with two accumulators."""
    return {
        "v_intercept": truncated_normal(drift_intercept_loc, drift_intercept_scale),
        "v_slope": truncated_normal(drift_slope_loc, drift_slope_scale),
        "s_true": gamma(sd_true_shape, sd_true_scale),
        "b": gamma(threshold_shape, threshold_scale),
        "t0": truncated_normal(t0_loc, t0_scale, t0_lower),
    }


def lba_simple_prior_dists(
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
):
    """The linear ballistic accumulator with two accumulators.

    Mirrors `rdm_simple_prior_dists`, but with two start-point parameters in place of the RDM's
    single threshold: the start-point range `A`, and the threshold *gap* `B = b - A` (Heathcote &
    Love 2012). Sampling the gap rather than the threshold makes `b > A` true by construction --
    the LBA is degenerate otherwise, since a start point above threshold would mean an instant
    response -- and decorrelates the pair, which a diagonal MCMC mass matrix handles far better.
    `threshold_shape`/`threshold_scale` therefore parameterize `B`, not `b`.

    The LBA configs centre the drift intercept on 2.0 rather than the RDM's 1.0. The LBA's
    first-passage time is `(b - k) / d`, so a drift rate near zero produces an arbitrarily large
    RT -- the distribution has a `t^-2` tail and no finite mean, and the weight of that tail is
    set by `Phi(-v/s)`. Measured over the prior predictive, centring at 1.0 gives
    P(RT > 5 s) = 3.6e-3 with excursions past 190 s, which would swamp the summary network's
    statistics; centring at 2.0 brings that to 4.0e-4, in line with the RDM's 2.8e-4, at no cost
    in accuracy (0.81 either way) and with a median RT closer to the RDM's. The RDM needs no such
    adjustment because a Wald first-passage time cannot blow up the same way.
    """
    return {
        "v_intercept": truncated_normal(drift_intercept_loc, drift_intercept_scale),
        "v_slope": truncated_normal(drift_slope_loc, drift_slope_scale),
        "s_true": gamma(sd_true_shape, sd_true_scale),
        "A": gamma(sp_max_shape, sp_max_scale),
        "B": gamma(threshold_shape, threshold_scale),
        "t0": truncated_normal(t0_loc, t0_scale, t0_lower),
    }


def rdm_sat_prior_dists(
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
):
    """The racing diffusion model with a speed-vs-accuracy manipulation.

    `rdm_simple_prior_dists` plus `b_diff`, the amount by which the threshold is raised under the
    accuracy instruction. Sampling the *difference* rather than a second threshold keeps
    `b_accuracy > b_speed` true by construction and leaves every parameter positive, so the same
    log-transform unconstrains the whole vector (cf. `A`/`B` in `lba_simple_prior_dists`).

    `threshold_diff_loc` is the hyperparameter study 3 shifts between training and test: it *is*
    the expected size of the speed-accuracy effect, with `threshold_diff_sd` holding the prior's
    spread fixed across the sweep (see `gamma_mean_sd`).
    """
    return _before_t0(
        rdm_simple_prior_dists(
            drift_intercept_loc, drift_intercept_scale, drift_slope_loc, drift_slope_scale,
            sd_true_shape, sd_true_scale, threshold_shape, threshold_scale, t0_loc, t0_scale, t0_lower,
        ),
        "b_diff",
        gamma_mean_sd(threshold_diff_loc, threshold_diff_sd),
    )


def lba_sat_prior_dists(
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
):
    """The linear ballistic accumulator with a speed-vs-accuracy manipulation.

    `lba_simple_prior_dists` plus `B_diff`, the amount by which the accuracy instruction raises
    the threshold *gap* -- so the threshold is `A + B` under speed and `A + B + B_diff` under
    accuracy, and `b > A` still holds structurally in both conditions. `threshold_diff_loc` is
    the hyperparameter study 3 shifts, as in `rdm_sat_prior_dists`.
    """
    return _before_t0(
        lba_simple_prior_dists(
            drift_intercept_loc, drift_intercept_scale, drift_slope_loc, drift_slope_scale,
            sd_true_shape, sd_true_scale, sp_max_shape, sp_max_scale, threshold_shape, threshold_scale,
            t0_loc, t0_scale, t0_lower,
        ),
        "B_diff",
        gamma_mean_sd(threshold_diff_loc, threshold_diff_sd),
    )


def _draw(dists, keys):
    """One draw per distribution, each from its own key, in the distributions' order."""
    return tuple(dist.sample(seed=key) for dist, key in zip(dists.values(), keys, strict=True))


def _prior_simulator(dists_fn, name):
    """The batched prior simulator for `dists_fn`: `f(batch_shape, <hyperparameters>, rng)`.

    The signature is spelled out rather than left as `**hyperparameters`, because
    `LambdaSimulator` filters the keyword arguments it forwards against it: a test case's kwargs
    reach the prior this way, and anything that is not one of its hyperparameters (`num_obs`,
    another model's hyperparameter) has to be dropped rather than passed through.
    """
    params = inspect.signature(dists_fn).parameters
    names = list(dists_fn(**dict.fromkeys(params, 1.0)))

    # Compiled, because TFP's samplers run eagerly at ~370 ms a call -- hours over a training
    # run's ~50,000 batches. The hyperparameters are broadcast to the batch before the call, so
    # there is one compilation per batch size, and `DataBandSimulator` already quantizes its
    # round sizes to keep that set small for the experiment simulators. The draws come back as
    # a tuple rather than a dict because `jit` returns dicts with their keys sorted, and the
    # parameter order is the one `mcmc_param_names` uses.
    draw = jax.jit(lambda keys, kwargs: _draw(dists_fn(**kwargs), keys))

    def sample(batch_shape, rng, **kwargs):
        batch_shape = tuple(int(size) for size in np.atleast_1d(batch_shape))
        kwargs = {name: jnp.broadcast_to(_float(value), batch_shape) for name, value in kwargs.items()}
        draws = draw(rng.next(len(names)), kwargs)
        return {name: np.asarray(value) for name, value in zip(names, draws, strict=True)}

    sample.__signature__ = inspect.Signature([
        inspect.Parameter("batch_shape", inspect.Parameter.POSITIONAL_OR_KEYWORD),
        *(p.replace(kind=inspect.Parameter.KEYWORD_ONLY) for p in params.values()),
        inspect.Parameter("rng", inspect.Parameter.KEYWORD_ONLY),
    ])
    sample.__name__ = sample.__qualname__ = name
    sample.__doc__ = f"Draw a batch from `{dists_fn.__name__}`; see the module docstring."

    return sample


rdm_prior_simple = _prior_simulator(rdm_simple_prior_dists, "rdm_prior_simple")
lba_prior_simple = _prior_simulator(lba_simple_prior_dists, "lba_prior_simple")
rdm_prior_sat = _prior_simulator(rdm_sat_prior_dists, "rdm_prior_sat")
lba_prior_sat = _prior_simulator(lba_sat_prior_dists, "lba_prior_sat")


def hyperparameters(sample_fn):
    """The hyperparameters bound into a prior simulator's `_partial_`, as a plain dict.

    `conf/mcmc/*.yaml` hands the simulator's `sample_fn` node to this, which Hydra instantiates
    into the `functools.partial` the simulator itself calls -- so the MCMC side reads the very
    values the training data were drawn under, all of them, rather than an interpolated subset.
    """
    return {name: value for name, value in sample_fn.keywords.items() if name != "rng"}
