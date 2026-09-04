import functools
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import stats
from rdm_jax import SplittableKey
from mcmc import fit_mcmc_gpu_batch
from mcmc import inference_loop_multiple_chains
from rdm_jax import inv_gauss_logpdf
from rdm_jax import inv_gauss_logsf
from mcmc import make_meta_to_unconstrained
from rdm_jax import make_rdm_meta_logdensity
from rdm_jax import make_rdm_simple_logdensity
from rdm_jax import rdm_experiment_simple_jax
from rdm_jax import rdm_experiment_simple_jax_stateful
from rdm_jax import rdm_race_logpdf
from mcmc import simple_to_unconstrained
from mcmc import warmup


@pytest.fixture
def x():
    return jnp.array([0.01, 0.5, 1.0, 2.0, 10.0, 100.0, 500.0])


def test_inv_gauss_logpdf(x):
    mu = np.array([0.01, 1.0, 2.0, 10.0, 100.0])
    lam = np.array([0.01, 1.0, 2.0, 10.0, 100.0])

    for m in mu:
        for l in lam:
            ref = stats.invgauss.logpdf(np.array(x), mu=m / l, scale=l)
            res = np.array(inv_gauss_logpdf(x, m, l))
            assert np.all(pytest.approx(res, rel=1e-6) == ref)


def test_inv_gauss_logsf(x):
    mu = np.array([0.01, 1.0, 2.0, 10.0, 100.0])
    lam = np.array([0.01, 1.0, 2.0, 10.0, 100.0])

    for m in mu:
        for l in lam:
            ref = stats.invgauss.logsf(np.array(x), mu=m / l, scale=l)
            res = np.array(inv_gauss_logsf(x, m, l))
            relerr = np.abs(ref - res) / np.maximum(np.abs(ref), 1.0)
            assert np.nanmax(relerr) < 1e-6


@pytest.mark.parametrize(
    ("t", "mu", "lam"),
    [(0.01, 1.5, 2.0), (0.5, 0.3, 50.0), (5.0, 3.0, 1.0), (1e-6, 1.0, 100.0)],
)
def test_inv_gauss_logsf_gradient_is_finite(t, mu, lam):
    grad = jax.grad(lambda t: inv_gauss_logsf(t, mu, lam))(jnp.array(t))
    assert jnp.isfinite(grad)


def test_rdm_race_logpdf_gradient_finite_random_sweep():
    rng = np.random.default_rng(0)
    n = 2000
    args = [
        rng.uniform(0.0, 3.0, n),  # rt
        rng.uniform(0.1, 5.0, n),  # drift_winner
        rng.uniform(0.1, 5.0, n),  # drift_loser
        rng.uniform(0.01, 5.0, n),  # s_winner
        rng.uniform(0.01, 5.0, n),  # s_loser
        rng.uniform(0.1, 3.0, n),  # threshold
        rng.uniform(0.0, 1.0, n),  # ndt
    ]
    grads = jax.vmap(jax.grad(rdm_race_logpdf, argnums=tuple(range(7))))(*(jnp.array(a) for a in args))
    assert jnp.all(jnp.isfinite(jnp.stack(grads, axis=-1)))


def test_rdm_race_logpdf_penalty_slopes_when_rt_precedes_ndt():
    # `rt <= t0` is impossible under the model, and _finalize_race_logp answers with a sloped
    # penalty rather than a flat floor so NUTS keeps a gradient pushing t0 back below the
    # fastest observed RT. That slope used to be applied *before* the log(min_p) floor, which
    # clamped it straight back off and left this gradient at exactly -0.0.
    grad = jax.grad(rdm_race_logpdf, argnums=6)(jnp.array(0.20), 3.5, 2.0, 1.2, 1.0, 1.2, jnp.array(0.30))
    assert jnp.isfinite(grad)
    assert grad < -1.0

    # ... while a feasible RT is unaffected, and never drops below the floor.
    logp = rdm_race_logpdf(jnp.array(0.50), 3.5, 2.0, 1.2, 1.0, 1.2, 0.30)
    assert logp > jnp.log(1e-10)


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
    init_position = simple_to_unconstrained(jnp.array([1.0, 1.0, 0.5, 1.0, 0.2]))

    key, warmup_key, sample_key = jax.random.split(key, 3)
    kernel, last_state, _ = warmup(blackjax.nuts, logdensity_fn, init_position, 500, warmup_key)

    last_states = jax.vmap(lambda _: last_state)(jnp.arange(1))
    positions, infos = inference_loop_multiple_chains(sample_key, kernel, last_states, 1000, num_chains=1)
    post_mean = jnp.mean(jnp.exp(positions[300:, 0]), axis=0)

    truth = jnp.array([true_v_intercept, true_v_slope, true_s_true, true_b, true_t0])
    assert jnp.max(jnp.abs(post_mean - truth) / truth) < 0.4
    assert jnp.mean(infos.is_divergent) < 0.05


def test_make_rdm_meta_logdensity_finite_at_init():
    data_x = np.array([[0.5, 1.0], [0.7, 0.0], [1.2, 1.0]])
    logdensity_fn = make_rdm_meta_logdensity(data_x, 0.5, 2.5, 0.05, 0.25)
    to_unconstrained = make_meta_to_unconstrained(0.5, 2.5, 0.05, 0.25)

    position = to_unconstrained(jnp.array([1.5, 0.15, 1.0, 1.0, 0.5, 1.0, 0.2]))
    value = logdensity_fn(position)
    grad = jax.grad(logdensity_fn)(position)

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(grad))


def test_fit_mcmc_gpu_batch_recovers_parameters_across_datasets():
    # experiment_1/fit_mcmc_gpu.py's core: vmap BlackJAX NUTS over chains *and* over
    # every dataset in one call, instead of pmap + one SLURM job per dataset.
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
    init_position = jnp.array([1.0, 1.0, 0.5, 1.0, 0.2])

    positions, infos = fit_mcmc_gpu_batch(
        jax.random.PRNGKey(1), datasets, make_logdensity_fn, init_position,
        num_chains=4, num_steps_warmup=500, num_steps_sampling=500,
    )
    positions.block_until_ready()

    assert positions.shape == (num_datasets, 500, 4, 5)

    post_mean = jnp.mean(jnp.exp(positions[:, 200:]), axis=(1, 2))
    truth = jnp.array([true_v_intercept, true_v_slope, true_s_true, true_b, true_t0])
    rel_err = jnp.abs(post_mean - truth) / truth
    # Mean rather than max across only 3 datasets: with a small sample/chain budget
    # any single dataset can mix poorly and swing its own error up, without that
    # reflecting a problem with fit_mcmc_gpu_batch itself.
    assert jnp.mean(rel_err) < 0.3
    assert jnp.mean(infos.is_divergent) < 0.1
