
import os

import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt

from pytensor.tensor.random.op import RandomVariable
from pymc.distributions import transforms
from pymc.distributions.dist_math import check_parameters
from pymc.distributions.distribution import DIST_PARAMETER_TYPES, Continuous
from pymc.distributions.shape_utils import rv_size_is_none
from pymc.distributions.transforms import _default_transform


def inv_gauss_pdf(t, mu, lam):

    e = -(lam/(2*t)) * (t**2/mu**2 - 2*t/mu  + 1)

    x = pt.exp(e + .5*pt.log(lam) - .5*pt.log(2*t**3*pt.pi))
    x = pt.maximum(0, x)

    return x


def standard_normal_cdf(t):
    return 0.5 * (1 + pt.erf(t/pt.sqrt(2)))


def inv_gauss_cdf(t, mu, lam):

    e = pt.exp(pt.log(2*lam) - pt.log(mu))
    add = pt.sqrt(lam/t) * (1 + t/mu)
    sub = pt.sqrt(lam/t) * (1 - t/mu)

    p_1 = 1 - standard_normal_cdf(add)
    p_2 = 1 - standard_normal_cdf(sub)
    x = pt.exp(e + pt.log(p_1)) + p_2
    x = pt.maximum(0, x)
    return x


class RdmSimpleRV(RandomVariable):
    name = "wald"
    signature = "(),(),(),(),()->(2)"
    dtype = "floatX"
    _print_name = ("Wald", "\\operatorname{Wald}")

    @classmethod
    def rng_fn(cls, rng, drift_winner, drift_loser, s_winner, threshold, ndt, size) -> np.ndarray:
        v = np.hstack([drift_loser, drift_winner])

        mu = threshold / v
        lam = (threshold / s_winner) ** 2

        # First passage time
        fpt_winner = rng.wald(mu[1], lam, size=size)
        fpt_loser = rng.wald(mu[0], lam, size=size)

        fpt = np.c_[fpt_loser, fpt_winner]

        resp = fpt.argmin(axis=1)
        rt = fpt.min(axis=1) + ndt

        return rt, resp


wald = RdmSimpleRV()


class RdmSimple(Continuous):
    rv_op = wald

    @classmethod
    def dist(
        cls,
        drift_winner: DIST_PARAMETER_TYPES | None = None,
        drift_loser: DIST_PARAMETER_TYPES | None = None,
        threshold: DIST_PARAMETER_TYPES | None = None,
        ndt: DIST_PARAMETER_TYPES | None = None,
        **kwargs,
    ):
        drift_winner = pt.as_tensor_variable(drift_winner)
        drift_loser = pt.as_tensor_variable(drift_loser)
        threshold = pt.as_tensor_variable(threshold)
        ndt = pt.as_tensor_variable(ndt)
        return super().dist([drift_winner, drift_loser, threshold, ndt], **kwargs)


    def support_point(rv, size, drift_winner, drift_loser, threshold, ndt):
        drift_winner, _, _, _ = pt.broadcast_arrays(drift_winner, drift_loser, threshold, ndt)
        if not rv_size_is_none(size):
            drift_winner = pt.full(size, drift_winner)
        return drift_winner


    def logp(value, drift_winner, drift_loser, s_winner, s_loser, threshold, ndt, min_p = 1e-14):
        value = value - ndt
        value = pt.maximum(0.0, value)

        mu_winner = threshold/drift_winner
        mu_loser = threshold/drift_loser
        lam_winner = (threshold/s_winner)**2
        lam_loser = (threshold/s_loser)**2

        f = pt.log(inv_gauss_pdf(value, mu_winner, lam_winner))

        F = inv_gauss_cdf(value, mu_loser, lam_loser)

        S = pt.log(1. - F)

        logp = f + S

        logp = pt.maximum(pt.switch(pt.isinf(logp) | pt.isnan(logp), pt.log(min_p), logp), pt.log(min_p))

        return check_parameters(
            logp,
            drift_winner > 0,
            drift_loser > 0,
            threshold > 0,
            ndt > 0,
            msg="drift_winner > 0, drift_loser > 0, threshold > 0, ndt > 0",
        )
    

@_default_transform.register(RdmSimple)
def pos_cont_transform(op, rv):
    return transforms.log


def rdm_model_simple(sim_data, **kwargs):
    # home_dir = os.environ["HOME"]
    # # os.environ["PYTENSOR_FLAGS"] = f"compiledir_format=compiledir_mcmc_{index},base_compiledir={home_dir}/.pytensor"

    # pytensor.config.compile_dir = f"{home_dir}/.pytensor/compiledir_mcmc_{index}"

    rt_true = sim_data[sim_data[:, 1] == 1, 0]
    rt_false = sim_data[sim_data[:, 1] == 0, 0]

    with pm.Model() as model:
        v_intercept = pm.Truncated("v_intercept", pm.Normal.dist(3, 0.5), lower=0)
        v_slope = pm.Truncated("v_slope", pm.Normal.dist(2, 0.5), lower=0)
        s_true = pm.Truncated("s_true", pm.Normal.dist(1, 0.5), lower=0)
        b = pm.Gamma("b", 10, 0.2)
        t0 = pm.Truncated("t0", pm.Normal.dist(0.3, 0.2), lower=0)

        v_true = v_intercept+v_slope
        v_false = v_intercept

        RdmSimple("ll_true", drift_winner=v_true, drift_loser=v_false, s_winner=s_true, s_loser=1.0, threshold=b, ndt=t0, observed=rt_true)
        RdmSimple("ll_false", drift_winner=v_false, drift_loser=v_true, s_winner=1.0, s_loser=s_true, threshold=b, ndt=t0, observed=rt_false)

        trace = pm.sample(step=pm.step_methods.HamiltonianMC(), initvals={"v_intercept": 2, "v_slope": 1, "a": 2, "s_true": 1, "t0": 0.2}, **kwargs)

    return trace
