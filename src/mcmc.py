"""Model-agnostic MCMC plumbing: unconstraining transforms, BlackJAX NUTS, and posterior IO.

Nothing here is specific to the racing diffusion model -- both `rdm_jax` and `lba_jax`
supply log-densities that this module samples, and both use the same two unconstraining
transforms (all-positive parameters via `Exp`; the hierarchical models' two bounded
hyperparameters via `Sigmoid`). It previously lived in `rdm_jax`, which meant the LBA
scripts had to import their sampler and their transforms from the RDM module.

Posterior samples are written as NetCDF with dims `(chain, draw, dataset, param)` -- see
`data.save_posterior`. Naming the `param` axis is what lets `load_mcmc_posterior` select
the subject-level parameters by name: the hierarchical models sample two hyperparameters
ahead of them, which the NPE never sees.
"""

import logging

import arviz_stats as azs
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from tensorflow_probability.substrates import jax as tfp

from data import load_posterior, save_posterior

# NUTS window adaptation and the race log-densities (small squared terms, exp/log
# transforms of sub-unit values like t0) are precision-sensitive. `rdm_jax` sets this
# too, but this module is the one doing the numerics and is importable without it --
# without this line the transforms silently return float32.
jax.config.update("jax_enable_x64", True)

tfb = tfp.bijectors

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Unconstraining transforms. `to_unconstrained` maps natural-scale parameters to
# the space NUTS explores; `to_constrained` is its inverse, used when reading the
# samples back. They must stay in step -- applying `exp` to a Sigmoid-transformed
# hyperparameter silently produces plausible-looking nonsense.
# ---------------------------------------------------------------------------


def simple_to_unconstrained(position):
    """Natural scale -> unconstrained, for an all-positive parameter vector."""
    return jnp.log(position)


def simple_to_constrained(position):
    """Unconstrained -> natural scale, for an all-positive parameter vector."""
    return jnp.exp(position)


def make_bounded_to_unconstrained(lower, upper):
    """Build the natural-scale -> unconstrained transform for a partly bounded parameter vector.

    `position` is `[*bounded_params, *positive_params]`, with one entry per element of
    `lower`/`upper` in the bounded block. The hierarchical models' prior hyperparameters are
    Uniform (bounded both sides), so they need a Sigmoid bijector rather than the Exp/log
    transform the strictly positive subject-level parameters use. Studies 2 and 3 each have one
    such hyperparameter; the list form is what let study 2 drop from two to one without a
    second transform pair.
    """
    bijectors = [tfb.Sigmoid(low=low, high=high) for low, high in zip(list(lower), list(upper), strict=True)]

    def to_unconstrained(position):
        position = jnp.asarray(position)
        y_bounded = jnp.stack(
            [bij.inverse(position[..., i]) for i, bij in enumerate(bijectors)], axis=-1,
        )
        y_rest = jnp.log(position[..., len(bijectors) :])
        return jnp.concatenate([y_bounded, y_rest], axis=-1)

    return to_unconstrained


def make_bounded_to_constrained(lower, upper):
    """Build the unconstrained -> natural-scale transform. Exact inverse of `make_bounded_to_unconstrained`."""
    bijectors = [tfb.Sigmoid(low=low, high=high) for low, high in zip(list(lower), list(upper), strict=True)]

    def to_constrained(position):
        position = jnp.asarray(position)
        x_bounded = jnp.stack(
            [bij.forward(position[..., i]) for i, bij in enumerate(bijectors)], axis=-1,
        )
        x_rest = jnp.exp(position[..., len(bijectors) :])
        return jnp.concatenate([x_bounded, x_rest], axis=-1)

    return to_constrained

# ---------------------------------------------------------------------------
# GPU-parallelized batch fitting: vmap over chains *and* over datasets, in the
# style of racing-diffusion-conflict/scripts/parameter_recovery.py. All three
# experiments' MCMC fitting goes through this -- one GPU job vmaps every
# dataset (and every chain within each) at once, instead of one SLURM job per
# dataset with chains parallelized via `jax.pmap` on faked CPU devices.
# ---------------------------------------------------------------------------


def warmup(sampler_fun, logdensity_fun, init_position, num_steps, rng_key, **kwargs):
    """BlackJAX window adaptation warmup."""
    adapt = blackjax.window_adaptation(sampler_fun, logdensity_fun, **kwargs)
    (last_state, parameters), _ = adapt.run(rng_key, init_position, num_steps=num_steps)
    kernel = sampler_fun(logdensity_fun, **parameters).step
    return kernel, last_state, parameters


def inference_loop_multiple_chains(rng_key, kernel, initial_state, num_samples, num_chains):
    """Run `num_chains` BlackJAX chains via `vmap`.

    A single XLA computation that runs efficiently on one GPU and composes with an outer
    `vmap` over datasets (`fit_mcmc_gpu_batch`), unlike `jax.pmap` (which needs one
    physical device per chain).
    """

    @jax.jit
    def one_step(states, rng_key):
        keys = jax.random.split(rng_key, num_chains)
        states, infos = jax.vmap(kernel)(keys, states)
        return states, (states.position, infos)

    keys = jax.random.split(rng_key, num_samples)
    _, (positions, infos) = jax.lax.scan(one_step, initial_state, keys)

    return positions, infos


def fit_mcmc_gpu(
    rng_key,
    data,
    make_logdensity_fn,
    init_position,
    num_chains,
    num_steps_warmup,
    num_steps_sampling,
    to_unconstrained=simple_to_unconstrained,
):
    """Fit one dataset's posterior with vmap-parallelized chains.

    `make_logdensity_fn(data)` builds the logdensity function for *this* dataset --
    called in here, not by the caller, so it composes correctly with an outer
    `jax.vmap` over datasets (`fit_mcmc_gpu_batch`): under `vmap`, `data` is a
    per-dataset traced slice, so the closure is (re)built per dataset rather than
    one fixed dataset getting baked in before vmapping.
    """
    init_position = jnp.asarray(init_position).at[-1].set(data[:, 0].min() / 2)

    logdensity_fn = make_logdensity_fn(data)

    rng_key, warmup_key = jax.random.split(rng_key)
    kernel, last_state, _ = warmup(
        blackjax.nuts, logdensity_fn, to_unconstrained(init_position), num_steps_warmup, warmup_key,
    )

    last_states = jax.vmap(lambda _: last_state)(jnp.arange(num_chains))

    return inference_loop_multiple_chains(rng_key, kernel, last_states, num_steps_sampling, num_chains)


def fit_mcmc_gpu_batch(
    rng_key,
    data,
    make_logdensity_fn,
    init_position,
    num_chains,
    num_steps_warmup,
    num_steps_sampling,
    to_unconstrained=simple_to_unconstrained,
):
    """Fit MCMC posteriors for a whole batch of datasets in one call.

    Vmaps over `data`'s leading axis -- e.g. every simulated dataset for one `num_obs`
    value in one GPU job, instead of one SLURM job per dataset.

    Returns `(positions, infos)` with `positions` of shape `(num_datasets, num_samples,
    num_chains, num_params)` and NUTS diagnostics (e.g. `infos.is_divergent`) with a
    matching leading `num_datasets` axis.
    """
    keys = jax.random.split(rng_key, data.shape[0])

    def fit_one(key, dataset):
        return fit_mcmc_gpu(
            key,
            dataset,
            make_logdensity_fn,
            init_position,
            num_chains,
            num_steps_warmup,
            num_steps_sampling,
            to_unconstrained=to_unconstrained,
        )

    return jax.vmap(fit_one, in_axes=(0, 0))(keys, data)


# ---------------------------------------------------------------------------
# Posterior IO. These two functions replace a ~12-line block that was copy-pasted
# into all ten check_*.py scripts, where the PSRF threshold was hardcoded, the
# thinning stride was the magic constant 4, and the back-transform was always
# `np.exp` regardless of which bijector the model was actually fitted with.
# ---------------------------------------------------------------------------


def save_mcmc_posterior(filename, positions, param_names):
    """Save `fit_mcmc_gpu_batch` output to NetCDF.

    `positions` arrives as `(dataset, draw, chain, param)`; it is stored as
    `(chain, draw, dataset, param)`, the layout ArviZ expects.
    """
    samples = np.transpose(np.asarray(positions), (2, 1, 0, 3))

    save_posterior(filename, samples, param_names)


def load_mcmc_posterior(filename, *, to_constrained, param_names, psrf_threshold, num_target_samples):
    """Load MCMC samples, drop unconverged datasets, back-transform, and thin.

    Args:
        filename: NetCDF file written by `save_mcmc_posterior`.
        to_constrained: inverse of the transform used to fit; applied over the full
            parameter vector before any parameters are selected.
        param_names: parameters to keep, selected by name. The hierarchical models store
            two hyperparameters ahead of the subject-level ones, so this is how the MCMC
            posterior is aligned with the NPE posterior rather than by position.
        psrf_threshold: a dataset is converged when every parameter's R-hat is below this.
        num_target_samples: thin `draw` down to (at most) this many samples, so that MCMC
            and NPE posteriors are compared at equal sample size.

    Returns:
        `(posterior, is_converged)` where `posterior` has shape
        `(num_converged_datasets, num_target_samples, len(param_names))`.
    """
    posterior = load_posterior(filename)

    rhat = azs.rhat(posterior, var_names=["theta"])
    is_converged = (rhat["theta"] < psrf_threshold).all("param").to_numpy()

    if not is_converged.all():
        logger.info(
            "%.3f of MCMC fits did not converge: %s",
            1.0 - is_converged.mean(),
            np.where(~is_converged)[0].tolist(),
        )

    stored_names = [str(name) for name in posterior.coords["param"].to_numpy()]
    missing = [name for name in param_names if name not in stored_names]
    if missing:
        msg = f"{filename}: requested parameters {missing} are not among the stored {stored_names}."
        raise ValueError(msg)

    # Pool the chains -> (dataset, sample, param), then drop the unconverged datasets.
    # Masking after the stack, rather than before, keeps this working when nothing
    # converged: xarray cannot stack a zero-length dimension.
    theta = (
        posterior["theta"]
        .stack(sample=("chain", "draw"))
        .transpose("dataset", "sample", "param")
        .to_numpy()[is_converged]
    )

    # Back-transform over the *full* parameter vector, then select by name: the
    # hierarchical models' first two entries use a different bijector from the rest,
    # so the transform cannot be applied to a subset.
    theta = np.asarray(to_constrained(theta))
    theta = theta[..., [stored_names.index(name) for name in param_names]]

    stride = max(theta.shape[1] // num_target_samples, 1)

    return theta[:, ::stride, :][:, :num_target_samples, :], is_converged


def bounded_init_position(lower, upper, subject_init):
    """Build a hierarchical model's initial position, starting the hyperparameters mid-support.

    The hyperparameters are Sigmoid-transformed, so `to_unconstrained` returns NaN for a
    starting value outside `[lower, upper]` -- and the `_lower`/`_upper` model variants narrow
    those bounds without touching the initial position. Deriving the midpoint from the same
    bounds the bijector uses keeps the two in step by construction.
    """
    midpoints = [0.5 * (low + high) for low, high in zip(list(lower), list(upper), strict=True)]

    return np.array([*midpoints, *subject_init])

