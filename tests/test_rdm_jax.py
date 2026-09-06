"""What this repository owns on the RDM side: its simulator, its priors, and its sampler.

The Wald density, its survival function and the race that assembles them are `eamax`'s, and
are tested there -- against scipy in `eamax/tests/test_wald.py`, against the R package EMC2 in
`eamax/tests/test_emc2_reference.py`, and as an assembled race (floors, masks, censoring) in
`eamax/tests/test_race.py`. Duplicating any of that here would pin the same numbers twice.

What is left is the seam: the simulator's `x` layout and BayesFlow calling convention, the
log-densities that fold this study's priors into `eamax`'s likelihood, and the CPU batch
fitter.
"""

import functools
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
from eamax.inference.mcmc import inference_loop_multiple_chains
from eamax.inference.warmup import window_adaptation
from rdm_jax import SplittableKey
from mcmc import BlockTransform
from rdm_jax import make_rdm_meta_logdensity
from rdm_jax import make_rdm_simple_logdensity
from rdm_jax import make_rdm_simple_prior_sample
from rdm_jax import rdm_experiment_simple_jax
from rdm_jax import rdm_experiment_simple_jax_stateful
from rdm_jax import rdm_spec
from mcmc import fit_mcmc_cpu_batch
from mcmc import simple_to_unconstrained


def test_rdm_experiment_simple_jax_matches_numpy_reference():
    v_intercept, v_slope, s_true, s_false, b, t0 = 1.0, 1.5, 0.3, 1.0, 1.2, 0.3
    num_obs = 100_000

    out = rdm_experiment_simple_jax(jax.random.PRNGKey(0), v_intercept, v_slope, s_true, s_false, b, t0, num_obs)
    rt, resp = np.array(out["x"][:, 0]), np.array(out["x"][:, 1])

    rng = np.random.default_rng(0)
    v = np.hstack([v_intercept, v_intercept + v_slope])
    s = np.hstack([s_false, s_true])
    mu, lam = b / v, (b / s) ** 2
    fpt = np.stack([rng.wald(mu[i], lam[i], size=num_obs) for i in range(2)])
    resp_ref = fpt.argmin(axis=0)
    rt_ref = fpt.min(axis=0) + t0

    assert abs(resp.mean() - resp_ref.mean()) < 0.01
    qs = [0.1, 0.3, 0.5, 0.7, 0.9]
    assert np.max(np.abs(np.quantile(rt, qs) - np.quantile(rt_ref, qs))) < 0.02


def test_rdm_experiment_simple_jax_stateful_matches_bayesflow_calling_convention():
    # BayesFlow's LambdaSimulator/batched_call indexes batched prior draws down to
    # shape-(1,) arrays (not scalars) per call, unlike numpy.hstack which collapses
    # them -- this must not crash and must produce the expected (num_obs, 2) shape.
    out = rdm_experiment_simple_jax_stateful(
        jnp.array([1.0]), jnp.array([1.5]), jnp.array([0.3]), 1.0, jnp.array([1.2]), jnp.array([0.3]), 50, SplittableKey(0),
    )
    assert out["x"].shape == (50, 2)


def test_splittable_key_advances_across_calls():
    # LambdaSimulator passes the *same* rng object reference on every per-sample
    # call, relying on it being stateful (like numpy.random.Generator) to avoid
    # repeating the same randomness across a batch.
    rng = SplittableKey(0)
    args = (jnp.array([1.0]), jnp.array([1.5]), jnp.array([0.3]), 1.0, jnp.array([1.2]), jnp.array([0.3]), 50)
    out1 = rdm_experiment_simple_jax_stateful(*args, rng)
    out2 = rdm_experiment_simple_jax_stateful(*args, rng)
    assert not np.array_equal(np.array(out1["x"]), np.array(out2["x"]))


def test_make_rdm_simple_logdensity_recovers_parameters_with_blackjax_nuts():
    true_v_intercept, true_v_slope, true_s_true, true_b, true_t0 = 1.0, 1.5, 0.3, 1.2, 0.3
    key = jax.random.PRNGKey(0)
    key, sim_key = jax.random.split(key)

    out = rdm_experiment_simple_jax(sim_key, true_v_intercept, true_v_slope, true_s_true, 1.0, true_b, true_t0, 800)
    data_x = np.array(out["x"])

    logdensity_fn = make_rdm_simple_logdensity(data_x, drift_slope_loc=true_v_slope, threshold_scale=1.0)
    init_positions = simple_to_unconstrained(jnp.array([[1.0, 1.0, 0.5, 1.0, 0.2]]))

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

    truth = jnp.array([true_v_intercept, true_v_slope, true_s_true, true_b, true_t0])
    assert jnp.max(jnp.abs(post_mean - truth) / truth) < 0.4
    assert jnp.mean(infos.is_divergent) < 0.05


def test_make_rdm_meta_logdensity_finite_at_init():
    data_x = np.array([[0.5, 1.0], [0.7, 0.0], [1.2, 1.0]])
    logdensity_fn = make_rdm_meta_logdensity(data_x, 0.7, 3.9, 0.15)
    transform = BlockTransform([0.7], [3.9])

    position = transform.inverse(jnp.array([1.5, 1.0, 1.0, 0.5, 1.0, 0.2]))
    value = logdensity_fn(position)
    grad = jax.grad(logdensity_fn)(position)

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(grad))


def test_fit_mcmc_cpu_batch_recovers_parameters_across_datasets():
    # scripts/fit_mcmc_cpu.py's core: pmap BlackJAX NUTS over chains -- one per CPU core --
    # and vmap over every dataset of the case inside each. Under pytest the host has not been
    # split into devices, so this runs the four chains on the one device it has; what it pins
    # is the shape contract and the recovery, not the parallelism.
    true_v_intercept, true_v_slope, true_s_true, true_b, true_t0 = 1.0, 1.5, 0.3, 1.2, 0.3
    drift_slope_loc, threshold_scale = 1.5, 1.0
    num_datasets, n_trials = 3, 500

    key = jax.random.PRNGKey(0)
    keys = jax.random.split(key, num_datasets)
    datasets = jnp.stack(
        [
            rdm_experiment_simple_jax(k, true_v_intercept, true_v_slope, true_s_true, 1.0, true_b, true_t0, n_trials)["x"]
            for k in keys
        ],
    )

    make_logdensity_fn = functools.partial(
        make_rdm_simple_logdensity, drift_slope_loc=drift_slope_loc, threshold_scale=threshold_scale,
    )
    positions, infos = fit_mcmc_cpu_batch(
        jax.random.PRNGKey(1), datasets, make_logdensity_fn,
        prior_sample_fn=make_rdm_simple_prior_sample(
            drift_slope_loc=drift_slope_loc, threshold_scale=threshold_scale,
        ),
        spec=rdm_spec(), transform=BlockTransform(),
        num_chains=1, num_steps_warmup=500, num_steps_sampling=500,
    )
    positions.block_until_ready()

    assert positions.shape == (num_datasets, 500, 1, 5)

    post_mean = jnp.mean(jnp.exp(positions[:, 200:]), axis=(1, 2))
    truth = jnp.array([true_v_intercept, true_v_slope, true_s_true, true_b, true_t0])
    rel_err = jnp.abs(post_mean - truth) / truth
    # Mean rather than max across only 3 datasets: with a small sample/chain budget
    # any single dataset can mix poorly and swing its own error up, without that
    # reflecting a problem with fit_mcmc_cpu_batch itself.
    assert jnp.mean(rel_err) < 0.3
    assert jnp.mean(infos.is_divergent) < 0.1
