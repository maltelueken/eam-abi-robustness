"""Model-agnostic MCMC plumbing: starting values, the CPU driver, and posterior read-back.

Nothing here is specific to the racing diffusion model -- both `rdm_jax` and `lba_jax` supply
log-densities that this module samples, and both describe their parameter vector with an
`eamax.design.Parameterization` and unconstrain it with an
`eamax.inference.transforms.BlockTransform` (an empty bounded block for the all-positive
models; one `Sigmoid` entry for the hierarchical ones' prior hyperparameter).

The sampler itself is `eamax.inference`: window adaptation, the vmapped inference loop, the
NUTS driver, R-hat, pooling and thinning. What is left in here is this study's three
decisions on top of it.

**Chains run on separate CPU cores.** `fit_mcmc_cpu_batch` `pmap`s one chain per device --
over the host devices `cpu_devices.configure_cpu_devices` splits the CPU into -- and `vmap`s
every dataset of the case inside each. That is the same total work as the old single-GPU
`vmap`-over-both, laid out so the four chains proceed in parallel rather than in lockstep on
one accelerator.

**Every chain adapts its own tuning, from its own starting point.** `eamax` offers no shared
warm-up, and the reason is the point of this study: replicating one warmed state across chains
starts between-chain variance at zero, so R-hat cannot fail, and the converged fraction *is*
the comparison set for the NPE-vs-MCMC result. Per-chain adaptation costs `num_chains` times
the warm-up work -- which is exactly what running the chains on `num_chains` cores gives back.

**Every chain starts from its own draw from the prior, rejected into the `t0` support.** The
prior supplies the dispersion R-hat needs, and supplies it already on the right scale -- unlike
the single configured vector every chain used to be handed, or an arbitrary jitter width around
it. `mcmc_prior_sample_fun` is built from the same distributions `mcmc_log_prior_fun` scores
against, so a chain cannot start from one prior and target another. And the support constraint
is not optional: a chain that starts with `t0` above the fastest observed response time sits on
the race likelihood's flat floor, where the gradient carries no information and step-size
adaptation cannot recover. `eamax` has no sloped penalty to walk it back out (see `rdm_jax`'s
module docstring), so the region is kept out of here instead, by `t0_support`. Rejection rather
than clipping, because a clip is a point mass in the one coordinate the constraint exists to
protect.

Posterior samples are written as NetCDF with dims `(chain, draw, dataset, param)`. Naming the
`param` axis is what lets `load_mcmc_posterior` select the subject-level parameters by name:
the hierarchical models sample a hyperparameter ahead of them, which the NPE never sees.
"""

import dataclasses
import logging

import jax
import jax.numpy as jnp
import numpy as np
from eamax.inference.diagnostics import rhat
from eamax.inference.init import (
    DEFAULT_MAX_T0_FRACTION,
    T0Support,
    init_positions_from_prior,
    min_valid_rt,
)
from eamax.inference.mcmc import fit_nuts_batch
from eamax.inference.posterior import pool_chains, thin
from eamax.inference.transforms import (  # noqa: F401  (re-exported; the tests reach for them here)
    BlockTransform,
    simple_to_constrained,
    simple_to_unconstrained,
)
from eamax.io import load_dataset_posterior, save_dataset_posterior

from cpu_devices import ENV_VAR

# `rdm_jax` turns on JAX's 64-bit mode; the transforms and the race log-densities are
# precision-sensitive and this module is importable without it, so pull it in for the side
# effect rather than relying on the caller's import order.
import rdm_jax  # noqa: F401

logger = logging.getLogger(__name__)

#: Highest fraction of a dataset's fastest response time that a starting `t0` may take --
#: `eamax`'s own default, restated here because it is a modelling decision rather than a
#: detail. Close to 1 on purpose: the true `t0` often sits just below `min(rt)`, so a low
#: ceiling would put every chain below the truth and leave window adaptation to walk them
#: all up together.
T0_MAX_FRACTION = DEFAULT_MAX_T0_FRACTION


def t0_support(spec, transform, data_x):
    """The `t0 < min(rt)` constraint for this model's *full* parameter vector.

    `eamax.inference.init.T0Support.from_spec` finds `t0` by name in the spec and checks it is
    on the log link, rather than assuming it is the last entry -- but the spec describes only
    the subject-level block. A hierarchical model stores its prior hyperparameter ahead of
    that, so the index is shifted by the transform's bounded block, which is the one place
    that length is stated.
    """
    support = T0Support.from_spec(
        spec, min_valid_rt(jnp.asarray(data_x)[:, 0]), max_fraction=T0_MAX_FRACTION,
    )

    return dataclasses.replace(support, index=support.index + transform.num_bounded)


def make_init_positions_fn(prior_sample_fn, spec, transform, num_starts):
    """Build the `f(key, data) -> (num_starts, P)` starting-value factory `fit_nuts` calls.

    Called *inside* the driver, once per dataset under the `vmap`, which is what lets the
    starts be per-dataset (the `t0` cap comes from that dataset's fastest response time),
    per-chain (an independent prior draw each) and support-aware at once.

    `prior_sample_fn` draws on the natural scale, in `mcmc_param_names` order; the model's
    transform is what carries it into the unconstrained coordinates the sampler explores, so
    the forward map appears exactly once. `num_starts` is chains *per call*, which
    `fit_mcmc_cpu_batch` sets to 1: each device runs a single chain, and the four differ
    because they are given four different keys.
    """

    def make_init_positions(key, data):
        # `num_exhausted` is dropped rather than reported: this runs under `vmap`, so it is a
        # traced scalar with nowhere to go. It counts draws that hit `max_attempts` and fell
        # back to a clipped `t0`. The `t0` prior is TruncatedNormal(0.3, 0.2), so the cap
        # rejects between 9% (min_rt 0.6) and 56% (min_rt 0.3) of draws across the range this
        # study sees -- exhausting 2000 independent attempts at any of those is unreachable.
        positions, _ = init_positions_from_prior(
            lambda draw_key: transform.inverse(prior_sample_fn(draw_key)),
            num_starts,
            key,
            support=t0_support(spec, transform, data),
        )

        return positions

    return make_init_positions


def chain_devices(num_chains):
    """The devices `fit_mcmc_cpu_batch` puts one chain on each of.

    Raises with the fix rather than silently falling back to fewer chains or to a single
    device: `pmap` needs one device per chain, and the device count is fixed before JAX
    initialises its backend (see `cpu_devices.configure_cpu_devices`), so by the time this
    runs it cannot be changed.
    """
    devices = jax.local_devices()

    if len(devices) < num_chains:
        msg = (
            f"{num_chains} chains need {num_chains} JAX devices, but only {len(devices)} "
            f"are visible ({[d.device_kind for d in devices]}). Run this stage through "
            f"scripts/fit_mcmc_cpu.py, and set {ENV_VAR}={num_chains} in the environment if "
            "you have changed mcmc_sampling_fun.num_chains -- the device count is fixed "
            "before JAX starts and cannot be raised from here."
        )
        raise RuntimeError(msg)

    return devices[:num_chains]


def fit_mcmc_cpu_batch(
    rng_key,
    data,
    make_logdensity_fn,
    *,
    prior_sample_fn,
    spec,
    transform,
    num_chains,
    num_steps_warmup,
    num_steps_sampling,
):
    """Fit MCMC posteriors for a whole batch of datasets, one chain per CPU core.

    Two nested maps, and which one is which is the whole design. `pmap` takes the **chain**
    axis, so the four chains of every dataset advance on four cores at once; `vmap` inside
    each takes the **dataset** axis, so a case's datasets still cost one XLA computation
    rather than one job each. Adaptation is per chain, so the warm-up work that per-chain
    adaptation multiplies by `num_chains` is precisely what the `num_chains` cores absorb.

    `make_logdensity_fn(data)` and the starting-value factory are both called *inside*
    `eamax.inference.mcmc.fit_nuts`, under the dataset `vmap`, so each dataset gets its own
    closure and its own support-aware starts rather than one dataset's being baked in. Each
    device's chain draws its own starting position from the prior, adapts its own step size and
    mass matrix, and samples with that tuning unmodified -- there is no repair step, because a
    collapsed adaptation is a finding to report rather than something to overwrite.

    Everything after `make_logdensity_fn` is keyword-only. The tail is six same-shaped
    arguments that a caller reorders silently -- an array landing in `prior_sample_fn` surfaces
    as "not callable" three frames inside a vmapped `while_loop` -- and every real call site
    (the Hydra `_partial_`, `scripts/fit_mcmc_cpu.py`) already names them.

    Args:
        rng_key: split once per chain, then per dataset inside each.
        data: `(num_datasets, num_obs, num_channels)`. Every dataset must have the same
            number of trials -- `pipeline.num_obs_groups` is what splits a ragged case.
        make_logdensity_fn: `f(data_x) -> logdensity_fn(position)`, from `mcmc_model_fun`.
        prior_sample_fn: `f(key) -> (P,)` natural-scale prior draw, from `mcmc_prior_sample_fun`.
        spec: the model's `eamax.design.Parameterization`, from `mcmc_spec`.
        transform: the model's `BlockTransform`, from `mcmc_transform`.
        num_chains: chains per dataset, and CPU devices.
        num_steps_warmup: window-adaptation steps, per chain.
        num_steps_sampling: draws recorded per chain.

    Returns:
        `(positions, infos)` with `positions` of shape `(num_datasets, num_steps_sampling,
        num_chains, num_params)` -- the layout `save_mcmc_posterior` expects -- and NUTS
        diagnostics (e.g. `infos.is_divergent`) with a matching leading `num_datasets` axis.
    """
    devices = chain_devices(num_chains)
    data = jnp.asarray(data)

    make_init_positions = make_init_positions_fn(prior_sample_fn, spec, transform, num_starts=1)

    def one_chain(chain_key, chain_data):
        # One chain per device, so `fit_nuts` adapts and runs a single chain here; the four
        # devices between them are the four independently adapted chains.
        return fit_nuts_batch(
            chain_key, chain_data, make_logdensity_fn, make_init_positions,
            num_chains=1,
            num_steps_warmup=num_steps_warmup,
            num_steps_sampling=num_steps_sampling,
        )

    keys = jax.random.split(rng_key, num_chains)

    positions, infos = jax.pmap(one_chain, in_axes=(0, None), devices=devices)(keys, data)

    # (chain, dataset, draw, 1, ...) -> (dataset, draw, chain, ...): drop the within-device
    # chain axis of length 1 and move the pmapped one into its place. The same two moves are
    # right for every diagnostic leaf too, whatever trailing axes it carries.
    positions = jnp.moveaxis(jnp.squeeze(positions, axis=3), 0, 2)
    infos = jax.tree.map(lambda leaf: jnp.moveaxis(jnp.squeeze(leaf, axis=3), 0, 2), infos)

    return positions, infos


# ---------------------------------------------------------------------------
# Posterior IO. `eamax.io` reads and writes and does nothing else -- it applies
# no diagnostic and no threshold -- so the convergence filter, the pooling and
# the thinning are composed here, where they are visible and can be varied.
# ---------------------------------------------------------------------------


def save_mcmc_posterior(filename, positions, param_names):
    """Save `fit_mcmc_cpu_batch` output to NetCDF.

    `positions` arrives as `(dataset, draw, chain, param)`; it is stored as
    `(chain, draw, dataset, param)`, the layout ArviZ expects.
    """
    save_dataset_posterior(filename, np.asarray(positions), param_names, layout="sampler")


def load_mcmc_posterior(filename, *, to_constrained, param_names, psrf_threshold, num_target_samples):
    """Load MCMC samples, drop unconverged datasets, back-transform, and thin.

    Four steps in an order that does not commute. `eamax.io.load_dataset_posterior` does the
    first -- back-transform the *full* vector, then select by name -- and hands back every
    draw of every chain. The rest are here: diagnose while the chain axis still exists, mask
    the pooled array (not the `Dataset`, so total non-convergence comes back as a length-zero
    axis rather than raising), then thin.

    R-hat is `eamax.inference.diagnostics.rhat`, the rank-normalized split statistic; running
    it on the back-transformed values rather than the stored unconstrained ones changes
    nothing, since it is invariant under any monotone link.

    Args:
        filename: NetCDF file written by `save_mcmc_posterior`.
        to_constrained: unconstrained -> natural scale, applied over the full parameter vector
            before any parameters are selected -- `BlockTransform.forward`, from
            `mcmc_transform`.
        param_names: parameters to keep, selected by name. The hierarchical models store a
            hyperparameter ahead of the subject-level ones, so this is how the MCMC posterior
            is aligned with the NPE posterior rather than by position.
        psrf_threshold: a dataset is converged when every parameter's R-hat is below this.
        num_target_samples: thin `draw` down to (at most) this many samples, so that MCMC
            and NPE posteriors are compared at equal sample size.

    Returns:
        `(posterior, is_converged)` where `posterior` has shape
        `(num_converged_datasets, num_target_samples, len(param_names))`.
    """
    theta = load_dataset_posterior(filename, to_constrained=to_constrained, param_names=param_names)

    is_converged = np.all(
        np.asarray(rhat(theta, chain_axis=0, sample_axis=1)) < psrf_threshold, axis=-1,
    )

    if not is_converged.all():
        logger.info(
            "%.3f of MCMC fits did not converge: %s",
            1.0 - is_converged.mean(),
            np.where(~is_converged)[0].tolist(),
        )

    return thin(pool_chains(theta)[is_converged], num_target_samples, axis=1), is_converged
