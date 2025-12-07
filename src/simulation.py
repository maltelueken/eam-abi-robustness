from typing import Callable

import bayesflow as bf
import numpy as np

from bayesflow.utils import batched_call, tree_stack
from bayesflow.utils.decorators import allow_batch_size
from bayesflow.types import Shape


class CustomSimulator(bf.simulators.Simulator):
    def __init__(self,
                 prior_simulator: Callable,
                 design_simulator: Callable,
                 experiment_simulator: Callable):
        self.prior_simulator = prior_simulator
        self.design_simulator = design_simulator
        self.experiment_simulator = experiment_simulator

    @allow_batch_size
    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        prior_dict = self.prior_simulator.sample(batch_shape)

        design_dict = self.design_simulator.sample(batch_shape)

        design_dict.update(**kwargs)

        sims_dict = self.experiment_simulator.sample(batch_shape, **prior_dict, **design_dict)

        data = prior_dict | design_dict | sims_dict

        data = {
            key: np.expand_dims(value, axis=-1) if np.ndim(value) == 1 else value for key, value in data.items()
        }

        return data


class CustomMetaSimulator(bf.simulators.Simulator):
    def __init__(self, meta_simulator: Callable,
                 prior_simulator: Callable,
                 design_simulator: Callable,
                 experiment_simulator: Callable):
        self.meta_simulator = meta_simulator
        self.prior_simulator = prior_simulator
        self.design_simulator = design_simulator
        self.experiment_simulator = experiment_simulator


    @staticmethod
    def update_existing(d: dict, **kwargs):
        """Update dictionary only with kwargs that match existing keys."""
        d.update((k, v) for k, v in kwargs.items() if k in d)

    @allow_batch_size
    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        meta_dict = self.meta_simulator.sample(batch_shape)

        self.update_existing(meta_dict, **kwargs)

        prior_dict = self.prior_simulator.sample(batch_shape, **meta_dict)

        design_dict = self.design_simulator.sample(batch_shape)

        self.update_existing(design_dict, **kwargs)

        sims_dict = self.experiment_simulator.sample(batch_shape, **prior_dict, **design_dict)

        data = meta_dict | prior_dict | design_dict | sims_dict

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
    # if np.any(np.array((v_intercept, v_slope, s_true, s_false, b, t0)) <= 0):
    #     raise ValueError("Model parameters must be positive")

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


def create_simulator(prior_fun, simulator_fun, design_fun):
    return bf.make_simulator([prior_fun, simulator_fun], meta_fn=design_fun)


def create_data_adapter(inference_variables, inference_conditions=None, summary_variables=None):
    adapter = (
        bf.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .broadcast("num_obs", to="x", exclude=(-2, -1), squeeze=-1)
        .broadcast(inference_conditions, to="num_obs") # Make sure that all inference conditions have same shape
        .as_set(summary_variables)
        .log(inference_variables)
        .sqrt("num_obs")
        .concatenate(inference_variables, into="inference_variables")
        .concatenate(summary_variables, into="summary_variables")
        .rename("num_obs", "inference_conditions")
    )

    adapter = adapter.keep(
        ["inference_variables", "inference_conditions", "summary_variables"]
    )

    return adapter
