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

import numpy as np
from scipy.special import logsumexp


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
