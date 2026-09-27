"""How far a test case's prior is from the one the approximator was trained on.

The figures plot posterior mismatch against the *nominal* value of the swept hyperparameter, and
that is not a uniform axis. Study 2's `drift_slope_loc` is a truncated-normal location, so the
lower bound at zero compresses the bottom of the sweep: a step of 0.4 there moves the prior far
less than the same step at the top, and the two ends look symmetric on the axis while differing
by an order of magnitude in divergence. These functions measure the move itself.

The KL is computed from the prior's *analytic* density, not estimated from samples. Both densities
are available in closed form -- `conf/mcmc/*.yaml` exposes each model's as `mcmc_log_prior_fun` --
so the only approximation left is the Monte Carlo average of a log ratio that is known exactly at
every draw. That matters at the far end of a sweep: a nearest-neighbour estimator reads the
distance to the closest training draw, and where the training prior's tail is exponentially thin
it understates the divergence badly and converges only slowly. Averaging an exact integrand has
none of that behaviour -- its error is the ordinary `O(1/sqrt(n))`, and it is unbiased.

A hierarchical model has no single training prior: it is amortized over a *range* of the
hyperparameter, so what a case must be compared against is the mixture over that range, which
`mixture_log_density` integrates by quadrature.

Studies 5 and 6 get no entry here, and that is not an omission: they hold the prior fixed and
restrict the *data*, so there is no prior distance to report. What moves for them is the
distribution of `x`, which `scripts/prior_distance.py` measures in the summary network's space.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.special import logsumexp

# `bayesflow.metrics.functional.kernels.default_scales`, i.e. `logspace(-6, 6, 11)`: the mixture
# of bandwidths `mmd_members` must share with BayesFlow's MMD to report the same number.
MMD_SCALES = np.logspace(-6, 6, 11)


def kl_from_log_densities(log_density_test, log_density_trained):
    """`KL(test || trained)`, averaged over draws from the test prior.

    Both arguments are the analytic log densities evaluated at the *same* draws, which must come
    from the test prior -- KL is an expectation under the first argument, and using the wrong
    sample silently computes something else.

    The direction is the one the studies are about: how much probability the test prior puts where
    the training prior put little, which is the gap the amortized network is asked to cover.
    """
    log_density_test, log_density_trained = (
        np.reshape(np.asarray(value, dtype=float), -1) for value in (log_density_test, log_density_trained)
    )

    if log_density_test.shape != log_density_trained.shape:
        msg = (
            "densities must be evaluated at the same draws, got "
            f"{log_density_test.shape} and {log_density_trained.shape}"
        )
        raise ValueError(msg)

    usable = np.isfinite(log_density_test) & np.isfinite(log_density_trained)

    if not usable.any():
        return float("nan")

    return float(np.mean(log_density_test[usable] - log_density_trained[usable]))


def mixture_log_density(log_prior, params, name, lower, upper, num_nodes=513):
    """Log density of a hierarchical model's training prior at each draw.

    That prior is `p(theta) = 1/(upper - lower) * integral p(theta | h) dh` over the
    hyperparameter `name` the meta simulator randomizes -- there is no closed form, but it is one
    smooth integral in one variable, so a midpoint rule on a fine grid resolves it to far better
    than the Monte Carlo error of the average it feeds. `log_prior` is broadcast over the grid in
    a single call rather than looped.
    """
    if upper <= lower:
        msg = f"the hyperparameter range must be non-empty, got [{lower}, {upper}]"
        raise ValueError(msg)

    nodes = lower + (upper - lower) * (np.arange(num_nodes) + 0.5) / num_nodes
    per_node = np.asarray(log_prior(np.asarray(params)[:, None, :], **{name: nodes}), dtype=float)

    # Uniform weights, so the mixture is a mean over nodes: logsumexp minus log(num_nodes).
    return logsumexp(per_node, axis=1) - np.log(num_nodes)


def standardize(reference, *samples):
    """Put every sample on the scale of `reference`, column by column.

    The summary network's outputs have no natural scale and no reason for their coordinates to
    share one, and a kernel bandwidth chosen over raw coordinates is dominated by whichever
    happens to be largest. Standardizing by the *training* sample keeps the transform identical
    across cases, so the MMDs stay comparable down the sweep.
    """
    reference = np.asarray(reference, dtype=float)
    centre = np.mean(reference, axis=0)
    spread = np.std(reference, axis=0)
    spread = np.where(spread > 0, spread, 1.0)

    return tuple((np.asarray(sample, dtype=float) - centre) / spread for sample in samples)


def _inverse_multiquadratic_mean(x, y):
    """Mean of BayesFlow's inverse-multiquadratic kernel mixture over all pairs of `x` and `y`.

    Squared distances are taken from the differences, as BayesFlow does, not from the cheaper
    `|x|^2 + |y|^2 - 2 x.y`. In float32 that expansion leaves a residue around 1e-6 where two draws
    coincide -- every diagonal entry of a self-term, and any repeated MCMC draw -- and 1e-6 is the
    smallest bandwidth, so those pairs lose half their kernel value: 1% error on 300 draws.
    Differences give exactly zero there, and under `jit` they cost little more.
    """
    sq_dist = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
    scales = jnp.asarray(MMD_SCALES, dtype=x.dtype)

    return jnp.mean(jnp.sum(scales / (sq_dist[..., None] + scales), axis=-1))


@jax.jit
def _mmd_members(reference, members):
    reference_term = _inverse_multiquadratic_mean(reference, reference)

    def one(member):
        return (
            reference_term
            + _inverse_multiquadratic_mean(member, member)
            - 2.0 * _inverse_multiquadratic_mean(reference, member)
        )

    return jax.vmap(one)(members)


def mmd_members(reference, members):
    """MMD between `reference` and each of `members`, as `bf.metrics.functional.maximum_mean_discrepancy`.

    The same biased estimator and inverse-multiquadratic mixture BayesFlow uses by default, for
    `reference` of shape `(num_draws, num_features)` against every slice of `members`,
    `(num_members, num_draws, num_features)`, in one compiled call. The `reference` self-term is
    shared by all members and computed once. That and compilation make it
    several times faster than calling BayesFlow once per member, which evaluates op by op under
    the JAX backend. Returns a NumPy array of shape `(num_members,)`.

    It computes in float32 whatever JAX's x64 setting, which costs about half as much as float64
    and agrees with it to around 1e-4 relative at 2000 draws -- far below the estimator's own
    Monte Carlo noise.
    """
    return np.asarray(
        _mmd_members(jnp.asarray(reference, dtype=jnp.float32), jnp.asarray(members, dtype=jnp.float32)),
        dtype=float,
    )
