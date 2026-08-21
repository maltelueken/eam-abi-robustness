"""Tests for study 3's speed-vs-accuracy linear ballistic accumulator.

The LBA counterpart of `tests/test_rdm_sat.py`, and the same two tests carry the weight: that
NUTS recovers a known threshold-gap difference from simulated data, and that the likelihood
collapses onto the already-validated simple one when the manipulation is switched off.
"""

import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from lba_jax import (
    _lba_sat_log_likelihood,
    _lba_simple_log_likelihood,
    lba_experiment_sat_jax,
    lba_experiment_sat_jax_stateful,
    lba_experiment_simple_jax,
    make_lba_sat_logdensity,
    make_lba_sat_meta_logdensity,
)
from mcmc import (
    bounded_init_position,
    inference_loop_multiple_chains,
    make_bounded_to_unconstrained,
    simple_to_unconstrained,
    warmup,
)

# v_intercept, v_slope, s_true, A, B, B_diff, t0 -- the centre of the prior in
# conf/simulator/prior_simulator/lba_sat.yaml, with a threshold gap that grows by half again
# under the accuracy instruction.
TRUE = {"v_intercept": 2.0, "v_slope": 1.5, "s_true": 1.2, "A": 0.6, "B": 1.2, "B_diff": 0.6, "t0": 0.3}


def simulate(key, num_obs, **overrides):
    params = TRUE | overrides
    return np.array(lba_experiment_sat_jax(key, s_false=1.0, num_obs=num_obs, **params)["x"])


def test_simulated_data_carry_rt_response_and_condition():
    data_x = simulate(jax.random.PRNGKey(0), 400)

    assert data_x.shape == (400, 3)
    assert set(np.unique(data_x[:, 1])) <= {0.0, 1.0}
    assert data_x[:, 2].sum() == 200
    assert np.all(data_x[:, 0] > TRUE["t0"])


def test_the_accuracy_block_is_slower_and_more_accurate():
    # The manipulation raises the threshold *gap*, so the threshold is A + B under speed and
    # A + B + B_diff under accuracy: more evidence required, hence slower and more accurate.
    data_x = simulate(jax.random.PRNGKey(1), 4000, B_diff=1.5)
    is_accuracy = data_x[:, 2] == 1

    assert data_x[is_accuracy, 0].mean() > data_x[~is_accuracy, 0].mean()
    assert data_x[is_accuracy, 1].mean() > data_x[~is_accuracy, 1].mean()


def test_stateful_wrapper_matches_the_bayesflow_calling_convention():
    class FixedKey:
        def next(self):
            return jax.random.PRNGKey(7)

    expected = lba_experiment_sat_jax(jax.random.PRNGKey(7), s_false=1.0, num_obs=50, **TRUE)
    actual = lba_experiment_sat_jax_stateful(s_false=1.0, num_obs=50, rng=FixedKey(), **TRUE)

    assert np.allclose(np.array(actual["x"]), np.array(expected["x"]))


def test_likelihood_reduces_to_the_simple_one_when_every_trial_is_speed():
    # With no accuracy-instructed trials the gap is `B` everywhere, so this must agree exactly
    # with the simple LBA likelihood -- whatever `B_diff` happens to be.
    simple_x = np.array(
        lba_experiment_simple_jax(
            jax.random.PRNGKey(2), TRUE["v_intercept"], TRUE["v_slope"], TRUE["s_true"], 1.0,
            TRUE["A"], TRUE["B"], TRUE["t0"], 300,
        )["x"],
    )
    rt, is_true = jnp.asarray(simple_x[:, 0]), jnp.asarray(simple_x[:, 1] == 1)
    is_accuracy = jnp.zeros_like(is_true)

    args = (TRUE["v_intercept"], TRUE["v_slope"], TRUE["s_true"], TRUE["A"], TRUE["B"])
    expected = _lba_simple_log_likelihood(rt, is_true, *args, TRUE["t0"])
    actual = _lba_sat_log_likelihood(rt, is_true, is_accuracy, *args, 0.9, TRUE["t0"])

    assert np.isclose(float(actual), float(expected))


def test_likelihood_is_sensitive_to_the_threshold_difference():
    # Otherwise `B_diff` would be unidentified and the recovery test could pass on the prior alone.
    data_x = simulate(jax.random.PRNGKey(3), 2000)
    logdensity_fn = make_lba_sat_logdensity(
        data_x, drift_slope_loc=TRUE["v_slope"], threshold_scale=0.15,
        threshold_diff_shape=6.0, threshold_diff_scale=0.1,
    )

    at_truth = logdensity_fn(simple_to_unconstrained(jnp.array(list(TRUE.values()))))
    at_wrong = logdensity_fn(simple_to_unconstrained(jnp.array(list((TRUE | {"B_diff": 0.05}).values()))))

    assert at_truth > at_wrong


def test_logdensity_recovers_the_parameters_with_blackjax_nuts():
    key = jax.random.PRNGKey(4)
    key, sim_key = jax.random.split(key)
    data_x = simulate(sim_key, 1600)

    logdensity_fn = make_lba_sat_logdensity(
        data_x, drift_slope_loc=TRUE["v_slope"], threshold_scale=0.15,
        threshold_diff_shape=6.0, threshold_diff_scale=0.1,
    )
    init_position = simple_to_unconstrained(jnp.array([1.0, 1.0, 1.0, 0.5, 1.0, 0.5, 0.2]))

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

    logdensity_fn = make_lba_sat_meta_logdensity(
        data_x, drift_slope_loc=1.5, threshold_scale=0.15, threshold_diff_shape=6.0,
        threshold_diff_scale_lower=0.02, threshold_diff_scale_upper=0.18,
    )
    position = make_bounded_to_unconstrained([0.02], [0.18])(
        jnp.array([0.1, *TRUE.values()]),
    )

    assert jnp.isfinite(logdensity_fn(position))
    assert jnp.all(jnp.isfinite(jax.grad(logdensity_fn)(position)))


def test_meta_init_position_starts_inside_the_narrowed_bounds():
    init = bounded_init_position([0.14], [0.18], [1, 2, 1, 0.5, 1.0, 0.5, 0.2])
    unconstrained = make_bounded_to_unconstrained([0.14], [0.18])(init)

    assert np.all(np.isfinite(np.asarray(unconstrained)))
    assert 0.14 < init[0] < 0.18
    assert len(init) == 8
