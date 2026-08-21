"""Tests for study 3's speed-vs-accuracy racing diffusion model.

The simulator and the log-density are written twice over -- once in JAX for sampling, once
in JAX for the likelihood -- so the tests that matter are the ones tying the two together:
that data simulated under a known threshold difference are recovered by NUTS, and that the
likelihood reduces to the already-validated simple one when the manipulation is switched off.
"""

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from mcmc import (
    bounded_init_position,
    inference_loop_multiple_chains,
    make_bounded_to_constrained,
    make_bounded_to_unconstrained,
    make_meta_to_unconstrained,
    simple_to_unconstrained,
    warmup,
)
from rdm_jax import (
    _rdm_sat_log_likelihood,
    _rdm_simple_log_likelihood,
    make_rdm_sat_logdensity,
    make_rdm_sat_meta_logdensity,
    rdm_experiment_sat_jax,
    rdm_experiment_sat_jax_stateful,
    rdm_experiment_simple_jax,
    sat_conditions,
)

TRUE = {"v_intercept": 1.0, "v_slope": 1.5, "s_true": 0.3, "b": 1.0, "b_diff": 0.6, "t0": 0.3}


def simulate(key, num_obs, **overrides):
    params = TRUE | overrides
    return np.array(rdm_experiment_sat_jax(key, s_false=1.0, num_obs=num_obs, **params)["x"])


def test_conditions_are_split_evenly_and_labelled_zero_one():
    conditions = np.array(sat_conditions(500))

    assert conditions.sum() == 250
    assert set(np.unique(conditions.astype(int))) == {0, 1}


def test_simulated_data_carry_rt_response_and_condition():
    data_x = simulate(jax.random.PRNGKey(0), 400)

    assert data_x.shape == (400, 3)
    assert set(np.unique(data_x[:, 1])) <= {0.0, 1.0}
    assert data_x[:, 2].sum() == 200
    assert np.all(data_x[:, 0] > TRUE["t0"])


def test_the_accuracy_block_is_slower_and_more_accurate():
    # The whole point of the manipulation: raising the threshold buys accuracy with time.
    # A large `b_diff` so the contrast is unmistakable at a testable sample size.
    data_x = simulate(jax.random.PRNGKey(1), 4000, b_diff=1.0)
    is_accuracy = data_x[:, 2] == 1

    assert data_x[is_accuracy, 0].mean() > data_x[~is_accuracy, 0].mean()
    assert data_x[is_accuracy, 1].mean() > data_x[~is_accuracy, 1].mean()


def test_stateful_wrapper_matches_the_bayesflow_calling_convention():
    class FixedKey:
        def next(self):
            return jax.random.PRNGKey(7)

    expected = rdm_experiment_sat_jax(jax.random.PRNGKey(7), s_false=1.0, num_obs=50, **TRUE)
    actual = rdm_experiment_sat_jax_stateful(s_false=1.0, num_obs=50, rng=FixedKey(), **TRUE)

    assert np.allclose(np.array(actual["x"]), np.array(expected["x"]))


def test_likelihood_reduces_to_the_simple_one_when_every_trial_is_speed():
    # With no accuracy-instructed trials the threshold is `b` everywhere, so the
    # speed-accuracy likelihood must agree exactly with the already-validated simple one --
    # whatever `b_diff` happens to be.
    simple_x = np.array(
        rdm_experiment_simple_jax(
            jax.random.PRNGKey(2), TRUE["v_intercept"], TRUE["v_slope"], TRUE["s_true"], 1.0, TRUE["b"], TRUE["t0"], 300,
        )["x"],
    )
    rt, is_true = jnp.asarray(simple_x[:, 0]), jnp.asarray(simple_x[:, 1] == 1)
    is_accuracy = jnp.zeros_like(is_true)

    args = (TRUE["v_intercept"], TRUE["v_slope"], TRUE["s_true"], TRUE["b"], TRUE["t0"])
    expected = _rdm_simple_log_likelihood(rt, is_true, *args)
    actual = _rdm_sat_log_likelihood(rt, is_true, is_accuracy, *args[:4], 0.9, args[4])

    assert np.isclose(float(actual), float(expected))


def test_likelihood_is_sensitive_to_the_threshold_difference():
    # The mirror image of the test above: with both conditions present, the wrong `b_diff`
    # must cost likelihood. Otherwise `b_diff` would be unidentified and the recovery test
    # below could pass on the prior alone.
    data_x = simulate(jax.random.PRNGKey(3), 2000)
    logdensity_fn = make_rdm_sat_logdensity(
        data_x, drift_slope_loc=TRUE["v_slope"], threshold_scale=1.0,
        threshold_diff_shape=6.0, threshold_diff_scale=1.0,
    )

    at_truth = logdensity_fn(simple_to_unconstrained(jnp.array(list(TRUE.values()))))
    at_wrong = logdensity_fn(simple_to_unconstrained(jnp.array(list((TRUE | {"b_diff": 0.05}).values()))))

    assert at_truth > at_wrong


def test_logdensity_recovers_the_parameters_with_blackjax_nuts():
    key = jax.random.PRNGKey(4)
    key, sim_key = jax.random.split(key)
    data_x = simulate(sim_key, 1600)

    logdensity_fn = make_rdm_sat_logdensity(
        data_x, drift_slope_loc=TRUE["v_slope"], threshold_scale=1.0,
        threshold_diff_shape=6.0, threshold_diff_scale=1.0,
    )
    init_position = simple_to_unconstrained(jnp.array([1.0, 1.0, 0.5, 1.0, 0.5, 0.2]))

    key, warmup_key, sample_key = jax.random.split(key, 3)
    kernel, last_state, _ = warmup(blackjax.nuts, logdensity_fn, init_position, 500, warmup_key)

    last_states = jax.vmap(lambda _: last_state)(jnp.arange(1))
    positions, infos = inference_loop_multiple_chains(sample_key, kernel, last_states, 1000, num_chains=1)

    post_mean = jnp.mean(jnp.exp(positions[300:, 0]), axis=0)
    truth = jnp.array(list(TRUE.values()))

    assert jnp.max(jnp.abs(post_mean - truth) / truth) < 0.4
    assert jnp.mean(infos.is_divergent) < 0.05


def test_meta_logdensity_is_finite_at_its_initial_position():
    data_x = simulate(jax.random.PRNGKey(5), 200)

    logdensity_fn = make_rdm_sat_meta_logdensity(
        data_x, drift_slope_loc=1.5, threshold_scale=0.15, threshold_diff_shape=6.0,
        threshold_diff_scale_lower=0.02, threshold_diff_scale_upper=0.18,
    )
    to_unconstrained = make_bounded_to_unconstrained([0.02], [0.18])
    position = to_unconstrained(jnp.array([0.1, 1.0, 1.5, 0.3, 1.0, 0.6, 0.2]))

    assert jnp.isfinite(logdensity_fn(position))
    assert jnp.all(jnp.isfinite(jax.grad(logdensity_fn)(position)))


def test_meta_init_position_starts_inside_the_narrowed_bounds():
    # The failure this guards against: a fixed initial value outside the Sigmoid's support
    # unconstrains to NaN, so NUTS starts from NaN and the whole GPU job is wasted.
    init = bounded_init_position([0.14], [0.18], [1, 2, 1, 1, 0.5, 0.2])
    unconstrained = make_bounded_to_unconstrained([0.14], [0.18])(init)

    assert np.all(np.isfinite(np.asarray(unconstrained)))
    assert 0.14 < init[0] < 0.18


@pytest.mark.parametrize("bounds", [([0.02], [0.18]), ([0.5, 0.05], [2.5, 0.25])])
def test_bounded_transform_pair_are_inverses_for_any_number_of_hyperparameters(bounds):
    lower, upper = bounds
    position = np.array([*(0.5 * (lo + hi) for lo, hi in zip(lower, upper)), 1.0, 2.0, 1.0, 0.2])

    round_tripped = make_bounded_to_constrained(lower, upper)(make_bounded_to_unconstrained(lower, upper)(position))

    assert np.allclose(np.asarray(round_tripped), position)


def test_bounded_transform_generalizes_the_two_hyperparameter_one():
    # `make_meta_to_unconstrained` is now a wrapper; study 2's models must be unaffected.
    position = np.array([1.5, 0.15, 1.0, 2.0, 1.0, 1.0, 0.2])

    general = make_bounded_to_unconstrained([0.5, 0.05], [2.5, 0.25])(position)
    specific = make_meta_to_unconstrained(0.5, 2.5, 0.05, 0.25)(position)

    assert np.allclose(np.asarray(general), np.asarray(specific))
