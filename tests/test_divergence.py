"""Tests for the prior-distance measurements.

The KL is computed from the priors' analytic densities, so what has to be pinned is that those
densities are the ones the *simulator* actually draws from -- a mismatch there would be silent and
would make every number in the table wrong in the same direction. Study 2's non-hierarchical prior
has a closed-form KL (two truncated normals sharing a scale and a truncation point), derived below
and checked against over the study's real grid; the hierarchical mixture has none, so it is
checked against a direct Monte Carlo estimate instead.
"""

import numpy as np
import pytest
from scipy import stats

from divergence import kl_from_log_densities, mixture_log_density, standardize
from rdm_jax import make_rdm_simple_log_prior

SCALE = 0.5  # `drift_slope_scale` in conf/simulator/prior_simulator/rdm_simple.yaml
TRAINED_ON = 1.5  # `drift_slope_loc` there
THRESHOLD_SCALE = 0.15
GRID = [0.7, 1.1, 1.5, 1.9, 2.3, 2.7, 3.1, 3.5, 3.9]  # conf/test_case/drift_slope_grid.yaml
META = (0.7, 3.9)  # the range random_prior_meta_continuous_multivariate.yaml randomizes


def truncated_normal(loc):
    """`v_slope` as `priors.rdm_prior_simple` draws it, at a given `drift_slope_loc`."""
    return stats.truncnorm(a=(0.0 - loc) / SCALE, b=np.inf, loc=loc, scale=SCALE)


def kl_closed_form(loc, other=TRAINED_ON):
    """`KL(p_loc || p_other)` for two lower-truncated normals sharing a scale and a bound.

    Both densities are the same Gaussian kernel over the same support, so the quadratic terms
    collapse to a linear function of the mean and what is left is the ratio of the two
    normalizing constants.
    """
    normalizer, other_normalizer = (stats.norm.cdf(value / SCALE) for value in (loc, other))
    shift = (loc - other) * (2 * truncated_normal(loc).mean() - other - loc) / (2 * SCALE**2)

    return shift + np.log(other_normalizer / normalizer)


def draw(loc, size=40_000, seed=0):
    """A matrix of prior draws in `mcmc_param_names` order, at a given `drift_slope_loc`."""
    rng = np.random.default_rng(seed)

    return np.column_stack([
        stats.truncnorm.rvs(a=-2.0, b=np.inf, loc=1.0, scale=0.5, size=size, random_state=rng),
        truncated_normal(loc).rvs(size, random_state=rng),
        rng.gamma(12.0, 0.1, size),
        rng.gamma(8.0, THRESHOLD_SCALE, size),
        stats.truncnorm.rvs(a=-1.5, b=np.inf, loc=0.3, scale=0.2, size=size, random_state=rng),
    ])


@pytest.fixture
def log_prior():
    return make_rdm_simple_log_prior(drift_slope_loc=TRAINED_ON, threshold_scale=THRESHOLD_SCALE)


@pytest.mark.parametrize("loc", GRID)
def test_kl_matches_the_closed_form_across_the_whole_sweep(loc, log_prior):
    """Including the far end, which is the point of not estimating this from samples."""
    draws = draw(loc)

    estimated = kl_from_log_densities(log_prior(draws, drift_slope_loc=loc), log_prior(draws))

    assert estimated == pytest.approx(kl_closed_form(loc), rel=0.02, abs=0.01), loc


def test_kl_is_zero_at_the_cell_the_model_trained_on(log_prior):
    draws = draw(TRAINED_ON)

    assert kl_from_log_densities(
        log_prior(draws, drift_slope_loc=TRAINED_ON), log_prior(draws),
    ) == pytest.approx(0.0, abs=1e-9)


def test_kl_stays_accurate_where_a_sample_based_estimator_would_not(log_prior):
    """The far end is where the two priors barely overlap.

    A nearest-neighbour estimator reads the distance to the closest training draw and understates
    the divergence badly out here -- around 24% low at the top of this sweep, converging only
    slowly in the sample size. Averaging an exact log ratio has no such regime, which is the
    reason `mcmc_log_prior_fun` exists.
    """
    far = 3.9
    exact = kl_closed_form(far)

    assert exact > 11.0  # the regime in question, not a mild extrapolation

    coarse = kl_from_log_densities(
        log_prior(draw(far, size=2_000), drift_slope_loc=far), log_prior(draw(far, size=2_000)),
    )
    fine = kl_from_log_densities(
        log_prior(draw(far, size=40_000), drift_slope_loc=far), log_prior(draw(far, size=40_000)),
    )

    # Accurate already at 2000 draws, rather than approaching from far below as the sample grows.
    assert coarse == pytest.approx(exact, rel=0.05)
    assert fine == pytest.approx(exact, rel=0.02)


def test_kl_rejects_densities_from_different_draws(log_prior):
    with pytest.raises(ValueError, match="same draws"):
        kl_from_log_densities(np.zeros(10), np.zeros(11))


@pytest.mark.parametrize("loc", GRID)
def test_the_mixture_matches_a_direct_monte_carlo_of_the_hierarchical_training_prior(loc, log_prior):
    """A hierarchical model's training prior is the mixture over the range it is amortized over.

    There is no closed form, so the quadrature is checked against sampling the hyperparameter and
    then the prior -- the thing the meta simulator literally does.
    """
    draws = draw(loc, size=4_000)

    by_quadrature = mixture_log_density(
        log_prior, draws, name="drift_slope_loc", lower=META[0], upper=META[1],
    )

    rng = np.random.default_rng(1)
    sampled_hyper = rng.uniform(*META, 4_000)
    by_sampling = np.log(np.mean(
        np.exp(np.asarray(log_prior(draws[:, None, :], drift_slope_loc=sampled_hyper))), axis=1,
    ))

    assert np.mean(np.abs(by_quadrature - by_sampling)) < 0.02, loc


def test_the_mixture_is_flatter_across_the_sweep_than_a_point_prior(log_prior):
    """What amortization buys, and the reason both are worth reporting.

    Against a fixed prior the sweep spans zero to more than eleven nats; against the mixture the
    hierarchical models train on it stays inside a narrow band, with its minimum in the middle of
    the randomized range rather than at the value any single model trained on.
    """
    against_mixture = [
        kl_from_log_densities(
            log_prior(draw(loc), drift_slope_loc=loc),
            mixture_log_density(log_prior, draw(loc), name="drift_slope_loc",
                                lower=META[0], upper=META[1]),
        )
        for loc in GRID
    ]

    assert max(against_mixture) < 2.0
    assert min(against_mixture) > 0.0  # a point prior is never the mixture
    assert np.argmin(against_mixture) == GRID.index(2.3)  # the midpoint of [0.7, 3.9]

    against_point = [kl_closed_form(loc) for loc in GRID]
    assert max(against_point) > 5 * max(against_mixture)


def test_the_mixture_rejects_an_empty_range(log_prior):
    with pytest.raises(ValueError, match="non-empty"):
        mixture_log_density(log_prior, draw(1.5, size=10), name="drift_slope_loc", lower=2.0, upper=2.0)


def test_standardize_uses_the_reference_sample_for_every_argument():
    reference = np.array([[0.0, 10.0], [2.0, 30.0]])
    other = np.array([[1.0, 20.0]])

    scaled_reference, scaled_other = standardize(reference, reference, other)

    assert np.mean(scaled_reference, axis=0) == pytest.approx([0.0, 0.0])
    assert np.std(scaled_reference, axis=0) == pytest.approx([1.0, 1.0])
    # The second sample is moved by the *first* sample's centre and spread, not its own.
    assert scaled_other.ravel() == pytest.approx([0.0, 0.0])


def test_standardize_leaves_a_constant_column_alone_rather_than_dividing_by_zero():
    reference = np.array([[1.0, 5.0], [3.0, 5.0]])

    scaled, = standardize(reference, reference)

    assert np.all(np.isfinite(scaled))
    assert scaled[:, 1] == pytest.approx([0.0, 0.0])
