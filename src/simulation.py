from typing import Callable

import bayesflow as bf
import numpy as np

from bayesflow.utils import batched_call, tree_stack


from bayesflow.types import Shape


class RdmSimulator(bf.simulators.Simulator):
    def __init__(self, prior_fun: Callable, design_fun: Callable, simulator_fun: Callable):
        self.prior_fun = prior_fun
        self.design_fun = design_fun
        self.simulator_fun = simulator_fun


    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        prior_dict = self.prior_fun(batch_shape)

        design_dict = self.design_fun(batch_shape)

        design_dict.update(**kwargs)

        sims_dict = self.simulator_fun(batch_shape, **prior_dict, **design_dict)

        data = prior_dict | design_dict | sims_dict

        data = {
            key: np.expand_dims(value, axis=-1) if np.ndim(value) == 1 else value for key, value in data.items()
        }

        return data


def batch_simulator(batch_shape, simulator_fun, **kwargs):
    data = batched_call(simulator_fun, batch_shape, kwargs=kwargs, flatten=True)
    data = tree_stack(data, axis=0, numpy=True)
    return data


def rdm_experiment_simple(
    v_intercept,
    v_slope,
    s_true,
    s_false,
    b,
    t0,
    num_obs,
    rng
):
    """Simulates data from a single subject in a multi-alternative response times experiment."""
    num_accumulators = 2

    # Acc1 = false, Acc2 = true
    v = np.hstack([v_intercept, v_intercept + v_slope])
    s = np.hstack([s_false, s_true])

    mu = b / v
    lam = (b / s) ** 2

    # First passage time
    fpt = np.zeros((num_accumulators, num_obs))
    
    for i in range(num_accumulators):
        fpt[i, :] = rng.wald(mu[i], lam[i], size=num_obs)

    resp = fpt.argmin(axis=0)
    rt = fpt.min(axis=0) + t0

    return {"x": np.c_[rt, resp]}
