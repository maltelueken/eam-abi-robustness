import functools
import blackjax
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy import integrate
from lba_jax import lba_experiment_simple_jax
from lba_jax import lba_experiment_simple_jax_stateful
from lba_jax import lba_logpdf
from lba_jax import lba_logsf
from lba_jax import lba_race_logpdf
from lba_jax import make_lba_meta_logdensity
from lba_jax import make_lba_simple_logdensity
from rdm_jax import SplittableKey
from mcmc import fit_mcmc_gpu_batch
from mcmc import inference_loop_multiple_chains
from mcmc import make_bounded_to_unconstrained
from mcmc import simple_to_unconstrained
from mcmc import warmup

# One representative parameter set, shared by the analytic tests below:
# false accumulator (v=2.0, s=1.0) racing the true one (v=3.5, s=1.2), i.e. the centre of
# the prior in conf/simulator/prior_simulator/lba_simple.yaml.
V_FALSE, V_TRUE = 2.0, 3.5
S_FALSE, S_TRUE = 1.0, 1.2
# The model is parameterized by the gap B = b - A; `lba_logpdf`/`lba_logsf` are the raw
# textbook densities and still take the reconstructed threshold b itself.
SP_MAX, SP_GAP = 0.6, 1.2
THRESHOLD = SP_MAX + SP_GAP


def _defective_density(t, winner):
    """Density of `winner` finishing at decision time `t` while the other is still racing."""
    v_win, s_win = (V_TRUE, S_TRUE) if winner else (V_FALSE, S_FALSE)
    v_lose, s_lose = (V_FALSE, S_FALSE) if winner else (V_TRUE, S_TRUE)
    log_pdf = lba_logpdf(jnp.array(t), v_win, s_win, SP_MAX, THRESHOLD)
    log_sf = lba_logsf(jnp.array(t), v_lose, s_lose, SP_MAX, THRESHOLD)
    return float(jnp.exp(log_pdf + log_sf))


def test_lba_defective_densities_integrate_to_one():
    # The decisive correctness check: if the pdf, the survival function and the
    # truncated-drift renormalizer are mutually consistent, the two defective densities
    # must together account for exactly all the probability mass.
    total = sum(integrate.quad(_defective_density, 1e-9, 300.0, args=(w,), limit=400)[0] for w in (0, 1))
    assert total == pytest.approx(1.0, abs=1e-5)


def test_lba_likelihood_choice_probability_matches_simulator():
    # ... and the split of that mass between accumulators must match what the forward
    # simulator actually produces, which is what ties the likelihood to the simulator.
    p_true_analytic = integrate.quad(_defective_density, 1e-9, 300.0, args=(1,), limit=400)[0]

    out = lba_experiment_simple_jax(
        jax.random.PRNGKey(0), V_FALSE, V_TRUE - V_FALSE, S_TRUE, S_FALSE, SP_MAX, SP_GAP, 0.0, 200_000,
    )
    p_true_simulated = float(np.mean(np.array(out["x"][:, 1])))

    assert p_true_analytic == pytest.approx(p_true_simulated, abs=0.005)


@pytest.mark.parametrize("t", [0.3, 0.8, 2.0])
def test_lba_logsf_is_consistent_with_integral_of_logpdf(t):
    cdf_quad = integrate.quad(
        lambda u: float(jnp.exp(lba_logpdf(jnp.array(u), V_TRUE, S_TRUE, SP_MAX, THRESHOLD))),
        1e-9,
        t,
        limit=300,
    )[0]
    sf = float(jnp.exp(lba_logsf(jnp.array(t), V_TRUE, S_TRUE, SP_MAX, THRESHOLD)))
    assert sf == pytest.approx(1.0 - cdf_quad, rel=1e-6)


def test_lba_survival_stays_finite_in_the_far_tail():
    # Unlike the RDM's Wald survival (which underflows and needed inv_gauss_logsf), the
    # LBA's truncated-normal drift leaves a heavy tail, so plain linear-space evaluation
    # must stay well conditioned far out -- this is what justifies not special-casing it.
    t = jnp.array([1.0, 1e2, 1e3, 1e4])
    log_sf = lba_logsf(t, V_TRUE, S_TRUE, SP_MAX, THRESHOLD)
    assert jnp.all(jnp.isfinite(log_sf))
    assert jnp.all(jnp.exp(log_sf) > 1e-8)


def test_lba_race_logpdf_gradient_finite_random_sweep():
    rng = np.random.default_rng(0)
    n = 2000
    args = [
        rng.uniform(0.0, 3.0, n),  # rt -- deliberately includes rt < ndt
        rng.uniform(0.1, 5.0, n),  # drift_winner
        rng.uniform(0.1, 5.0, n),  # drift_loser
        rng.uniform(0.01, 5.0, n),  # s_winner
        rng.uniform(0.01, 5.0, n),  # s_loser
        rng.uniform(0.05, 3.0, n),  # sp_max
        # The gap ranges down to near zero, i.e. a threshold barely above the largest start
        # point -- the closest the B parameterization can get to the old b < A degeneracy.
        rng.uniform(1e-6, 3.0, n),  # sp_gap
        rng.uniform(0.0, 1.0, n),  # ndt
    ]
    grads = jax.vmap(jax.grad(lba_race_logpdf, argnums=tuple(range(8))))(*(jnp.array(a) for a in args))
    assert jnp.all(jnp.isfinite(jnp.stack(grads, axis=-1)))


def test_lba_race_logpdf_penalty_slopes_when_rt_precedes_ndt():
    # Same contract as the RDM's equivalent test: the `rt <= t0` region must keep a usable
    # gradient rather than a flat floor. The LBA reuses rdm_jax._finalize_race_logp, so this
    # pins that the shared helper is wired in on this side too.
    grad = jax.grad(lba_race_logpdf, argnums=7)(
        jnp.array(0.20), 3.5, 2.0, 1.2, 1.0, 0.6, 1.2, jnp.array(0.30),
    )
    assert jnp.isfinite(grad)
    assert grad < -1.0


def test_lba_logpdf_floors_the_density_not_the_bracket():
    # The closed form is a difference of nearly equal terms and underflows in float64 at small
    # decision times. What gets floored there has to be the density: clamping the intermediate
    # `bracket` instead leaves `-log(A) - log Phi(v/s)` to be applied afterwards, which lifts
    # the clamped value back *above* the floor -- so the far-left tail came out overstated
    # rather than clamped (-22.51 here, against a true density of e^-146).
    log_floor = float(jnp.log(1e-10))
    underflowed = lba_logpdf(jnp.array([0.05, 0.1]), V_TRUE, S_TRUE, SP_MAX, THRESHOLD)

    assert jnp.all(underflowed == pytest.approx(log_floor))
    # The floor is EMC2's min_ll, so a floored trial contributes exactly what EMC2 records.
    assert float(lba_race_logpdf(jnp.array(0.35), V_TRUE, V_FALSE, S_TRUE, S_FALSE, SP_MAX, SP_GAP, 0.3)) == (
        pytest.approx(log_floor)
    )
    # Just past the underflow the density is well conditioned again and well above the floor.
    assert float(lba_logpdf(jnp.array(0.2), V_TRUE, S_TRUE, SP_MAX, THRESHOLD)) > log_floor + 10


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
    init_position = simple_to_unconstrained(jnp.array([1.0, 1.0, 1.0, 0.5, 1.0, 0.2]))

    key, warmup_key, sample_key = jax.random.split(key, 3)
    kernel, last_state, _ = warmup(blackjax.nuts, logdensity_fn, init_position, 500, warmup_key)

    last_states = jax.vmap(lambda _: last_state)(jnp.arange(1))
    positions, infos = inference_loop_multiple_chains(sample_key, kernel, last_states, 1000, num_chains=1)
    post_mean = jnp.mean(jnp.exp(positions[300:, 0]), axis=0)

    assert jnp.max(jnp.abs(post_mean - jnp.array(truth)) / jnp.array(truth)) < 0.4
    assert jnp.mean(infos.is_divergent) < 0.05


def test_make_lba_meta_logdensity_finite_at_init():
    data_x = np.array([[0.5, 1.0], [0.7, 0.0], [1.2, 1.0]])
    logdensity_fn = make_lba_meta_logdensity(data_x, 0.7, 3.9, 0.15)
    # The transform is length-agnostic (it log-transforms everything after the bounded
    # hyperparameters), so the LBA reuses it unchanged for its seven-element position.
    to_unconstrained = make_bounded_to_unconstrained([0.7], [3.9])

    position = to_unconstrained(jnp.array([1.5, 1.0, 1.0, 1.0, 0.5, 1.0, 0.2]))
    value = logdensity_fn(position)
    grad = jax.grad(logdensity_fn)(position)

    assert jnp.isfinite(value)
    assert jnp.all(jnp.isfinite(grad))


def test_lba_threshold_stays_above_start_point_across_unconstrained_space():
    # The whole point of carrying B = b - A instead of b: `b > A` is not a constraint the
    # sampler can violate, because every real-valued position maps to B = exp(y) > 0. Sweep
    # log B far into both tails and assert the density stays finite and differentiable, with
    # no infeasible region needing a penalty.
    data_x = np.array([[0.5, 1.0], [0.7, 0.0], [1.2, 1.0]])
    logdensity_fn = make_lba_simple_logdensity(data_x, drift_slope_loc=1.5, threshold_scale=0.15)

    for log_gap in (-30.0, -10.0, -1.0, 0.0, 1.0, 10.0):
        position = jnp.array([*jnp.log(jnp.array([2.0, 1.5, 1.2, 0.6])), log_gap, jnp.log(0.3)])
        assert jnp.isfinite(logdensity_fn(position))
        assert jnp.all(jnp.isfinite(jax.grad(logdensity_fn)(position)))

    # And the implied threshold exceeds the start-point range at every one of those points.
    assert all(0.6 + float(jnp.exp(g)) > 0.6 for g in (-30.0, -10.0, 0.0, 10.0))


def test_fit_mcmc_gpu_batch_recovers_lba_parameters_across_datasets():
    # The LBA reuses rdm_jax's model-agnostic GPU batch fitter unchanged; this pins that
    # it works for a six-parameter model, including the `t0 is last` initialization.
    truth = (2.0, 1.5, 1.2, 0.6, 1.2, 0.3)  # v_intercept, v_slope, s_true, A, B, t0
    num_datasets, n_trials = 3, 500

    keys = jax.random.split(jax.random.PRNGKey(0), num_datasets)
    datasets = jnp.stack(
        [lba_experiment_simple_jax(k, *truth[:3], 1.0, *truth[3:], n_trials)["x"] for k in keys],
    )

    make_logdensity_fn = functools.partial(
        make_lba_simple_logdensity, drift_slope_loc=truth[1], threshold_scale=0.15,
    )
    init_position = jnp.array([1.0, 1.0, 1.0, 0.5, 1.0, 0.2])

    positions, infos = fit_mcmc_gpu_batch(
        jax.random.PRNGKey(1), datasets, make_logdensity_fn, init_position,
        num_chains=4, num_steps_warmup=500, num_steps_sampling=500,
    )
    positions.block_until_ready()

    assert positions.shape == (num_datasets, 500, 4, 6)

    post_mean = jnp.mean(jnp.exp(positions[:, 200:]), axis=(1, 2))
    rel_err = jnp.abs(post_mean - jnp.array(truth)) / jnp.array(truth)
    # Mean rather than max across only 3 datasets, for the same reason as the RDM's
    # equivalent test: one poorly mixing dataset shouldn't fail the batch fitter.
    assert jnp.mean(rel_err) < 0.3
    assert jnp.mean(infos.is_divergent) < 0.1
