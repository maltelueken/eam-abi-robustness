"""Tests for reading MCMC posteriors back: convergence masking, thinning, back-transform.

These guard the block that used to be copy-pasted into all ten check_*.py scripts, where
the R-hat threshold was hardcoded, the thinning stride was a magic 4, and the
back-transform was always `np.exp` -- which is wrong for the hierarchical models, whose
two leading hyperparameters go through a Sigmoid bijector instead.
"""

import numpy as np
import pytest
from data import save_posterior
from mcmc import (
    load_mcmc_posterior,
    make_bounded_to_constrained,
    make_bounded_to_unconstrained,
    simple_to_constrained,
    simple_to_unconstrained,
)

SUBJECT_PARAMS = ["v_intercept", "v_slope", "s_true", "b", "t0"]
META_PARAMS = ["drift_slope_loc", *SUBJECT_PARAMS]

LOWER, UPPER = [0.7], [3.9]


def write_posterior(path, num_datasets, param_names, num_chains=4, num_draws=500, scale=0.01, seed=0):
    """Write a posterior whose chains agree closely, so every dataset converges."""
    rng = np.random.default_rng(seed)
    centre = rng.normal(size=(1, 1, num_datasets, len(param_names)))
    samples = centre + scale * rng.normal(size=(num_chains, num_draws, num_datasets, len(param_names)))
    save_posterior(str(path), samples, param_names)
    return samples


def test_simple_transform_pair_are_inverses():
    x = np.array([1.0, 2.0, 1.0, 1.5, 0.2])
    assert np.allclose(np.asarray(simple_to_constrained(simple_to_unconstrained(x))), x)


def test_meta_transform_pair_are_inverses_and_respect_the_bounds():
    to_unconstrained = make_bounded_to_unconstrained(LOWER, UPPER)
    to_constrained = make_bounded_to_constrained(LOWER, UPPER)

    x = np.array([1.5, 1.0, 2.0, 1.0, 1.5, 0.2])
    assert np.allclose(np.asarray(to_constrained(to_unconstrained(x))), x)

    # Whatever the unconstrained value, the hyperparameter lands inside its support -- to within
    # the float32 resolution the Sigmoid saturates at. A bound that is not exactly representable
    # in binary (neither 0.7 nor 3.9 is) can come back an ulp outside itself, which only happens
    # once the Sigmoid has fully saturated, far outside anything a warmed-up chain visits.
    tol = 1e-6
    for y in (-40.0, 40.0):
        extreme = to_constrained(np.array([y, 0.0, 0.0, 0.0, 0.0, 0.0]))
        assert LOWER[0] - tol <= float(extreme[0]) <= UPPER[0] + tol


def test_it_back_transforms_and_thins_to_the_requested_sample_count(tmp_path):
    path = tmp_path / "simple.nc"
    samples = write_posterior(path, num_datasets=3, param_names=SUBJECT_PARAMS)

    posterior, is_converged = load_mcmc_posterior(
        str(path),
        to_constrained=simple_to_constrained,
        param_names=SUBJECT_PARAMS,
        psrf_threshold=1.01,
        num_target_samples=200,
    )

    assert is_converged.all()
    assert posterior.shape == (3, 200, len(SUBJECT_PARAMS))
    # Values are on the natural scale, i.e. exp() of what was stored.
    assert posterior.min() > 0
    assert np.isclose(posterior[0, 0, 0], np.exp(samples[0, 0, 0, 0]))


def test_meta_posteriors_are_reduced_to_the_parameters_the_npe_infers(tmp_path):
    """A hierarchical fit stores six parameters; the NPE only ever sees five."""
    path = tmp_path / "meta.nc"
    write_posterior(path, num_datasets=3, param_names=META_PARAMS)

    posterior, _ = load_mcmc_posterior(
        str(path),
        to_constrained=make_bounded_to_constrained(LOWER, UPPER),
        param_names=SUBJECT_PARAMS,
        psrf_threshold=1.01,
        num_target_samples=100,
    )

    # Five, not six -- and selected by name, so the hyperparameter is dropped
    # rather than a positional slice silently misaligning the comparison.
    assert posterior.shape == (3, 100, len(SUBJECT_PARAMS))


def test_unconverged_datasets_are_dropped(tmp_path):
    rng = np.random.default_rng(3)
    num_datasets = 4
    samples = 0.01 * rng.normal(size=(4, 400, num_datasets, len(SUBJECT_PARAMS)))
    # Push one dataset's chains far apart so its R-hat is large.
    samples[:, :, 2, :] += np.arange(4).reshape(4, 1, 1) * 5.0

    path = tmp_path / "mixed.nc"
    save_posterior(str(path), samples, SUBJECT_PARAMS)

    posterior, is_converged = load_mcmc_posterior(
        str(path),
        to_constrained=simple_to_constrained,
        param_names=SUBJECT_PARAMS,
        psrf_threshold=1.01,
        num_target_samples=100,
    )

    assert not is_converged[2]
    assert is_converged.sum() == num_datasets - 1
    assert posterior.shape[0] == num_datasets - 1


def test_all_unconverged_returns_empty_rather_than_raising(tmp_path):
    """xarray cannot stack a zero-length dimension, so this path is easy to get wrong."""
    rng = np.random.default_rng(4)
    samples = rng.normal(size=(4, 300, 2, len(SUBJECT_PARAMS))) * 5.0
    samples += np.arange(4).reshape(4, 1, 1, 1) * 20.0

    path = tmp_path / "none.nc"
    save_posterior(str(path), samples, SUBJECT_PARAMS)

    posterior, is_converged = load_mcmc_posterior(
        str(path),
        to_constrained=simple_to_constrained,
        param_names=SUBJECT_PARAMS,
        psrf_threshold=1.01,
        num_target_samples=100,
    )

    assert not is_converged.any()
    assert posterior.shape[0] == 0


def test_requesting_an_unknown_parameter_is_an_error(tmp_path):
    path = tmp_path / "simple.nc"
    write_posterior(path, num_datasets=2, param_names=SUBJECT_PARAMS)

    with pytest.raises(ValueError, match="not among the stored"):
        load_mcmc_posterior(
            str(path),
            to_constrained=simple_to_constrained,
            param_names=["nonexistent"],
            psrf_threshold=1.01,
            num_target_samples=10,
        )
