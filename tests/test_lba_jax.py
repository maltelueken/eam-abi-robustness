"""What this repository owns on the LBA side: its simulator, its priors, and its sampler.

The LBA's defective density, its survival function and the race that assembles them are
`eamax`'s, and are tested there -- integrating to one, against EMC2, and as an assembled race
in `eamax/tests/{test_lba,test_emc2_reference,test_race}.py`. So is the `b > A` invariant the
threshold-gap parameterization exists to guarantee. None of that is repeated here.

What is left is the seam: the simulator's `x` layout and BayesFlow calling convention, the
log-densities that fold this study's priors into `eamax`'s likelihood, and the CPU batch
fitter.
"""

import functools
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from lba_jax import lba_experiment_simple_jax
from lba_jax import lba_experiment_simple_jax_stateful
from lba_jax import make_lba_meta_logdensity
from lba_jax import make_lba_simple_logdensity
from lba_jax import make_lba_simple_prior_sample
from eamax.inference.mcmc import inference_loop_multiple_chains
from eamax.inference.warmup import window_adaptation
from lba_jax import lba_spec
from rdm_jax import SplittableKey
from mcmc import BlockTransform
from mcmc import fit_mcmc_cpu_batch
from mcmc import simple_to_unconstrained

def test_lba_experiment_simple_jax_stateful_matches_bayesflow_calling_convention():
    # BayesFlow's LambdaSimulator/batched_call indexes batched prior draws down to
    # shape-(1,) arrays (not scalars) per call; `s_false` also arrives as a plain int
    # from the config, which must not trip tfd's dtype checks.
    out = lba_experiment_simple_jax_stateful(
        jnp.array([1.0]), jnp.array([1.5]), jnp.array([1.2]), 1, jnp.array([0.6]), jnp.array([1.2]),
        jnp.array([0.3]), 50, SplittableKey(0),
    )
    assert out["x"].shape == (50, 2)


def test_lba_simulator_respects_the_shift_and_start_point_bounds():
    t0, sp_max, sp_gap = 0.3, 0.6, 1.2
    out = lba_experiment_simple_jax(jax.random.PRNGKey(0), 1.0, 1.5, 1.2, 1.0, sp_max, sp_gap, t0, 20_000)
    rt, resp = np.array(out["x"][:, 0]), np.array(out["x"][:, 1])

    assert np.all(rt > t0)  # every RT is the non-decision time plus a positive decision time
    assert set(np.unique(resp)) <= {0.0, 1.0}
    # The true accumulator has the larger drift, so it must win more often than not.
    assert resp.mean() > 0.5
    # Decision time is (b - k)/d with k <= A, so it can never undercut (b - A)/max_drift; the
    # gap parameterization is what guarantees that bound is positive at all.
    assert np.all(rt - t0 > 0.0)


def test_make_lba_simple_logdensity_recovers_parameters_with_blackjax_nuts():
    truth = (2.0, 1.5, 1.2, 0.6, 1.2, 0.3)  # v_intercept, v_slope, s_true, A, B, t0
    key = jax.random.PRNGKey(0)
    key, sim_key = jax.random.split(key)

    out = lba_experiment_simple_jax(sim_key, *truth[:3], 1.0, *truth[3:], 800)
    data_x = np.array(out["x"])

    logdensity_fn = make_lba_simple_logdensity(data_x, drift_slope_loc=truth[1], threshold_scale=0.15)
    init_positions = simple_to_unconstrained(jnp.array([[1.0, 1.0, 1.0, 0.5, 1.0, 0.2]]))

    key, warmup_key, sample_key = jax.random.split(key, 3)
    last_states, parameters = window_adaptation(
        blackjax.nuts, logdensity_fn, init_positions, 500, warmup_key, num_chains=1,
    )

    step = blackjax.nuts.build_kernel()

    def kernel(chain_key, state, params):
        return step(chain_key, state, logdensity_fn, **params)

    positions, infos = inference_loop_multiple_chains(
        sample_key, kernel, last_states, 1000, num_chains=1, kernel_params=parameters,
    )
    post_mean = jnp.mean(jnp.exp(positions[300:, 0]), axis=0)

    assert jnp.max(jnp.abs(post_mean - jnp.array(truth)) / jnp.array(truth)) < 0.4
    assert jnp.mean(infos.is_divergent) < 0.05


def test_make_lba_meta_logdensity_finite_at_init():
    data_x = np.array([[0.5, 1.0], [0.7, 0.0], [1.2, 1.0]])
    logdensity_fn = make_lba_meta_logdensity(data_x, 0.7, 3.9, 0.15)
    # The transform is length-agnostic (it log-transforms everything after the bounded
    # hyperparameters), so the LBA reuses it unchanged for its seven-element position.
    transform = BlockTransform([0.7], [3.9])

    position = transform.inverse(jnp.array([1.5, 1.0, 1.0, 1.0, 0.5, 1.0, 0.2]))
    value = logdensity_fn(position)
    grad = jax.grad(logdensity_fn)(position)

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(grad))


def test_fit_mcmc_cpu_batch_recovers_lba_parameters_across_datasets():
    # The LBA reuses the model-agnostic CPU batch fitter unchanged; this pins that it works
    # for a six-parameter model, including the per-dataset `t0` initialization, which finds
    # `t0` in the spec by name rather than at a fixed slot.
    truth = (2.0, 1.5, 1.2, 0.6, 1.2, 0.3)  # v_intercept, v_slope, s_true, A, B, t0
    num_datasets, n_trials = 3, 500

    keys = jax.random.split(jax.random.PRNGKey(0), num_datasets)
    datasets = jnp.stack(
        [lba_experiment_simple_jax(k, *truth[:3], 1.0, *truth[3:], n_trials)["x"] for k in keys],
    )

    make_logdensity_fn = functools.partial(
        make_lba_simple_logdensity, drift_slope_loc=truth[1], threshold_scale=0.15,
    )
    positions, infos = fit_mcmc_cpu_batch(
        jax.random.PRNGKey(1), datasets, make_logdensity_fn,
        prior_sample_fn=make_lba_simple_prior_sample(
            drift_slope_loc=truth[1], threshold_scale=0.15,
        ),
        spec=lba_spec(), transform=BlockTransform(),
        num_chains=1, num_steps_warmup=500, num_steps_sampling=500,
    )
    positions.block_until_ready()

    assert positions.shape == (num_datasets, 500, 1, 6)

    post_mean = jnp.mean(jnp.exp(positions[:, 200:]), axis=(1, 2))
    rel_err = jnp.abs(post_mean - jnp.array(truth)) / jnp.array(truth)
    # Mean rather than max across only 3 datasets, for the same reason as the RDM's
    # equivalent test: one poorly mixing dataset shouldn't fail the batch fitter.
    assert jnp.mean(rel_err) < 0.3
    assert jnp.mean(infos.is_divergent) < 0.1
