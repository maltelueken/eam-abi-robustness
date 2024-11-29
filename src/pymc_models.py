
import logging
import os

import blackjax
import jax
import jax.numpy as jnp
import keras
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from datetime import date
from functools import partial

from pytensor.tensor.random.op import RandomVariable
from pymc.distributions import transforms
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import DIST_PARAMETER_TYPES, Continuous
from pymc.distributions.shape_utils import rv_size_is_none
from pymc.distributions.transforms import _default_transform
from pymc.sampling_jax import get_jaxified_logp


logger = logging.getLogger(__name__)


def inv_gauss_logpdf(t, mu, lam):

    e = -(lam / (2 * t)) * (t**2 / mu**2 - 2 * t / mu  + 1)

    x = e + 0.5 * pt.log(lam) - 0.5 * pt.log(2 * t**3 * pt.pi)

    return x


def standard_normal_logcdf(t):
    return pt.switch(
        pt.lt(t, -1.0),
        pt.log(pt.erfcx(-t / pt.sqrt(2.0)) / 2.0) - pt.sqr(t) / 2.0,
        pt.log1p(-pt.erfc(t / pt.sqrt(2.0)) / 2.0),
    )


def inv_gauss_logsf(t, mu, lam):
    """https://journal.r-project.org/archive/2016-1/giner-smyth.pdf"""
    mu = mu / lam
    t = t / lam
    r = 1.0 / pt.sqrt(t)
    a = standard_normal_logcdf(-r * ((t / mu) - 1.0))
    b = 2.0 / mu + standard_normal_logcdf(-r * (t + mu) / mu)
    return pt.switch(pt.isposinf(t), -np.inf, pt.switch(pt.gt(t, 0.0), a + pt.log1p(-pt.exp(b - a)), 0.0))


class RdmSimpleRV(RandomVariable):
    name = "wald"
    signature = "(),(),(),(),(),()->(2)"
    dtype = "floatX"
    _print_name = ("Wald", "\\operatorname{Wald}")

    @classmethod
    def rng_fn(cls, rng, drift_winner, drift_loser, s_winner, s_loser, threshold, ndt, size) -> np.ndarray:
        v = np.hstack([drift_loser, drift_winner])
        s = np.hstack([s_loser, s_winner])

        mu = threshold / v
        lam = (threshold / s) ** 2

        # First passage time
        fpt_winner = rng.wald(mu[1], lam[1], size=size)
        fpt_loser = rng.wald(mu[0], lam[0], size=size)

        fpt = np.c_[fpt_loser, fpt_winner]

        resp = fpt.argmin(axis=1)
        rt = fpt.min(axis=1) + ndt

        return rt, resp


rdm = RdmSimpleRV()


class RdmSimple(Continuous):
    rv_op = rdm

    @classmethod
    def dist(
        cls,
        drift_winner: DIST_PARAMETER_TYPES | None = None,
        drift_loser: DIST_PARAMETER_TYPES | None = None,
        s_winner: DIST_PARAMETER_TYPES | None = None,
        s_loser: DIST_PARAMETER_TYPES | None = None,
        threshold: DIST_PARAMETER_TYPES | None = None,
        ndt: DIST_PARAMETER_TYPES | None = None,
        **kwargs,
    ):
        drift_winner = pt.as_tensor_variable(drift_winner)
        drift_loser = pt.as_tensor_variable(drift_loser)
        s_winner = pt.as_tensor_variable(s_winner)
        s_loser = pt.as_tensor_variable(s_loser)
        threshold = pt.as_tensor_variable(threshold)
        ndt = pt.as_tensor_variable(ndt)
        return super().dist([drift_winner, drift_loser, s_winner, s_loser, threshold, ndt], **kwargs)


    def support_point(rv, size, drift_winner, drift_loser, s_winner, s_loser, threshold, ndt):
        drift_winner, _, _, _, _, _ = pt.broadcast_arrays(drift_winner, drift_loser, s_winner, s_loser, threshold, ndt)
        if not rv_size_is_none(size):
            drift_winner = pt.full(size, drift_winner)
        return drift_winner


    def logp(value, drift_winner, drift_loser, s_winner, s_loser, threshold, ndt, min_p = 1e-10):
        value = value - ndt
        value = pt.maximum(0.0, value)

        mu_winner = threshold/drift_winner
        mu_loser = threshold/drift_loser
        lam_winner = (threshold/s_winner)**2
        lam_loser = (threshold/s_loser)**2

        logp = pt.switch(
            value > 0.0,
            # Log pdf of winner
            inv_gauss_logpdf(value, mu_winner, lam_winner) + 
            # Log survival of loser
            inv_gauss_logsf(value, mu_loser, lam_loser), pt.log(min_p)
        )

        logp = pt.maximum(pt.switch(pt.isinf(logp) | pt.isnan(logp), pt.log(min_p), logp), pt.log(min_p))

        return check_parameters(
            logp,
            drift_winner > 0,
            drift_loser > 0,
            s_winner > 0,
            s_loser > 0, 
            threshold > 0,
            ndt > 0,
            msg="drift_winner > 0, drift_loser > 0, threshold > 0, ndt > 0",
        )
    

@_default_transform.register(RdmSimple)
def pos_cont_transform(op, rv):
    return transforms.log


def rdm_model_simple(sim_data):
    rt_true = sim_data[sim_data[:, 1] == 1, 0]
    rt_false = sim_data[sim_data[:, 1] == 0, 0]

    with pm.Model() as model:
        v_intercept = pm.TruncatedNormal("v_intercept", mu=1.0, sigma=0.5, lower=0)
        v_slope = pm.TruncatedNormal("v_slope", mu=1.5, sigma=0.5, lower=0)
        s_true = pm.Gamma("s_true", alpha=12, beta=1.0 / 0.1)
        b = pm.Gamma("b", alpha=8, beta=1.0 / 0.15)
        t0 = pm.TruncatedNormal("t0", mu=0.3, sigma=0.3, lower=0)

        v_true = v_intercept+v_slope
        v_false = v_intercept

        RdmSimple("ll_true", drift_winner=v_true, drift_loser=v_false, s_winner=s_true, s_loser=1.0, threshold=b, ndt=t0, observed=rt_true)
        RdmSimple("ll_false", drift_winner=v_false, drift_loser=v_true, s_winner=1.0, s_loser=s_true, threshold=b, ndt=t0, observed=rt_false)

    return get_jaxified_logp(model)


def model_nle(sim_data, approximator, num_obs):
    sim_data = keras.tree.map_structure(keras.ops.convert_to_tensor, approximator.adapter(sim_data, strict=True, stage="inference"))

    print(sim_data.keys())
    print(sim_data["inference_conditions"].shape)

    @jax.jit
    def nle_logdensity_fun(x):
        sim_data["inference_conditions"] = jnp.tile(x, (1, num_obs, 1))

        ll = jnp.sum(approximator._log_prob(**sim_data))

        # jax.debug.print(ll)

        return ll

    return nle_logdensity_fun


def inference_loop(rng_key, kernel, initial_state, num_samples):
    @jax.jit
    def one_step(state, rng_key):
        state, info = kernel(rng_key, state)
        return state, (state, info)

    keys = jax.random.split(rng_key, num_samples)
    _, (states, infos) = jax.lax.scan(one_step, initial_state, keys)

    return states, infos


def warmup(sampler_fun, logdensity_fun, init_position, num_steps, rng_key, **kwargs):
    logger.debug("Warmup initial position: %s", np.exp(init_position))
    adapt = blackjax.window_adaptation(sampler_fun, logdensity_fun, **kwargs)
    (last_state, parameters), _ = adapt.run(rng_key, init_position, num_steps=num_steps)
    logger.debug("Warmup return state: %s", np.exp(last_state.position))
    # logger.debug("Warmup return parameters: %s", parameters)
    kernel = sampler_fun(logdensity_fun, **parameters).step
    return kernel, last_state, parameters


def run_mcmc(logdensity_fun, sampler_fun, init_position, num_chains, num_steps_warmup, num_steps_sampling, min_rt=None, rng_key=None, **kwargs):
    if rng_key is None:
        rng_key = jax.random.key(int(date.today().strftime("%Y%m%d")))
    
    rng_key, warmup_key = jax.random.split(rng_key)

    if min_rt is not None:
        init_position[-1] = 0.5 * min_rt

    kernel, last_state, _ = warmup(sampler_fun, logdensity_fun, jnp.log(init_position), num_steps_warmup, warmup_key, **kwargs)

    last_states = jax.vmap(lambda x: last_state)(np.arange(num_chains))

    sample_keys = jax.random.split(rng_key, num_chains)

    inference_loop_multiple_chains = jax.pmap(inference_loop, in_axes=(0, None, 0, None), static_broadcasted_argnums=(1, 3))

    trace = inference_loop_multiple_chains(
        sample_keys, kernel, last_states, num_steps_sampling
    )

    # _ = trace[0].position[0, 0, 0].block_unitl_ready()

    return trace


def run_mcmc_robust(logdensity_fun, sampler_fun, prior_fun, init_position, num_chains, num_steps_warmup, num_steps_sampling, min_rt=None, rng_key=None, **kwargs):
    
    is_converged = False
    iter = 0
    max_iter = 20

    if min_rt is not None:
        init_position[-1] = 0.5 * min_rt

    # init_position = np.log(init_position)

    while iter < max_iter and not is_converged:
        if iter > 0:
            logger.info("Generating new initial parameters and trying sampling again")
            logger.info("PSRF last sampling run: %s", psrf)
            new_init_position = prior_fun()
        else:
            new_init_position = init_position

        new_init_position = np.log(new_init_position)

        trace = run_mcmc(logdensity_fun, sampler_fun, new_init_position, num_chains, num_steps_warmup, num_steps_sampling, rng_key, **kwargs)

        psrf = blackjax.diagnostics.potential_scale_reduction(trace[0].position)

        is_converged = np.all(psrf < 1.01) and np.all(np.var(trace[0].position, axis=(0, 1)) > 10e-8)

        iter += 1
    
    if not is_converged:
        logger.info("Sampler not converged")

    return trace