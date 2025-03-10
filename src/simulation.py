from typing import Callable

import bayesflow as bf
import keras
import numpy as np
import scipy.stats as stats

from bayesflow.utils import batched_call, tree_stack
from bayesflow.utils.decorators import allow_batch_size
from bayesflow.types import Shape
from numba import njit, prange


class CustomSimulator(bf.simulators.Simulator):
    def __init__(self, prior_simulator: Callable, design_simulator: Callable, experiment_simulator: Callable):
        self.prior_simulator = prior_simulator
        self.design_simulator = design_simulator
        self.experiment_simulator = experiment_simulator


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
    def __init__(self, meta_fun: Callable, prior_fun: Callable, design_fun: Callable, simulator_fun: Callable):
        self.meta_fun = meta_fun
        self.prior_fun = prior_fun
        self.design_fun = design_fun
        self.simulator_fun = simulator_fun


    @staticmethod
    def update_existing(d: dict, **kwargs):
        """Update dictionary only with kwargs that match existing keys."""
        d.update((k, v) for k, v in kwargs.items() if k in d)

    @allow_batch_size
    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        meta_dict = self.meta_fun(batch_shape)

        self.update_existing(meta_dict, **kwargs)

        prior_dict = self.prior_fun(batch_shape, **meta_dict)

        design_dict = self.design_fun(batch_shape)

        self.update_existing(design_dict, **kwargs)

        sims_dict = self.simulator_fun(batch_shape, **prior_dict, **design_dict)

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


@njit(parallel=True)
def find_min_and_argmin(arr):
    rows, cols = arr.shape
    min_vals = np.empty(cols, dtype=arr.dtype)
    min_idxs = np.empty(cols, dtype=np.int64)
    
    for j in prange(cols):
        min_val = arr[0, j]
        min_idx = 0
        for i in range(1, rows):
            if arr[i, j] < min_val:
                min_val = arr[i, j]
                min_idx = i
        min_vals[j] = min_val
        min_idxs[j] = min_idx
    
    return min_vals, min_idxs


@njit
def set_seed_numba(value):
    np.random.seed(value)


@njit(parallel=True)
def rdmc_experiment_simple_numba(mu, b, s, t0, num_obs, t_max):
    num_accumulators = mu.shape[0]

    fpt = np.full((num_accumulators, num_obs), fill_value=t_max, dtype=np.float64)
    
    for n in prange(num_obs):
        for i in prange(num_accumulators):
            xt = 0.0
            for t in range(t_max):
                xt += mu[i, t] + s[i] * np.random.randn()
                if xt > b:
                    fpt[i, n] = t
                    break

    rt, resp = find_min_and_argmin(fpt)
    rt += t0

    return rt / 1000, resp # Convert back to s scale


def rdmc_experiment_simple(v_c_intercept, v_c_slope, amp, tau, s_true, s_false, b, t0, a_shape, num_obs, t_max, seed):
    if np.any(np.array((v_c_intercept, v_c_slope, amp, tau, s_true, s_false, b, t0, a_shape)) <= 0):
        raise ValueError("Model parameters must be positive")
    
    mu_c = np.hstack([v_c_intercept, v_c_intercept + v_c_slope])
    s = np.hstack([s_false, s_true])

    t = np.arange(1, t_max + 1, 1)

    eq4 = (
        amp
        * np.exp(-t / tau)
        * (np.exp(1) * t / (a_shape - 1) / tau) ** (a_shape - 1)
    ) * ((a_shape - 1) / t - 1 / tau)

    mu = np.tile(mu_c, (t_max, 1)).T
    mu[1, :] += eq4

    set_seed_numba(seed)

    rt, resp = rdmc_experiment_simple_numba(mu, b, s, float(t0), num_obs, t_max)

    return {"x": np.c_[rt, resp]}


def rrdmc_experiment_simple(v_c_intercept, v_c_slope, v_a_intercept, v_a_slope, A0, k, s_true, s_false, b, t0, num_obs, t_max, seed):
    if np.any(np.array((v_c_intercept, v_c_slope, v_a_intercept, v_a_slope, A0, k, s_true, s_false, b, t0)) <= 0):
        raise ValueError("Model parameters must be positive")
    
    mu_c = np.hstack([v_c_intercept, v_c_intercept + v_c_slope])
    mu_a = np.hstack([v_a_intercept, v_a_intercept + v_a_slope])
    s = np.hstack([s_false, s_true])
    
    t = np.arange(1, t_max + 1, 1)

    weight_a = A0 * np.exp(-k*t)
    weight_c = 1.0 - weight_a

    mu = weight_a[None, :] * mu_a[:, None] + weight_c[None, :] * mu_c[:, None]

    set_seed_numba(seed)

    rt, resp = rdmc_experiment_simple_numba(mu, b, s, float(t0), num_obs, t_max)

    return {"x": np.c_[rt, resp]}


def create_simulator(prior_fun, simulator_fun, design_fun):
    return bf.make_simulator([prior_fun, simulator_fun], meta_fn=design_fun)


def create_data_adapter(inference_variables, inference_conditions=None, summary_variables=None):
    adapter = (
        bf.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .broadcast("num_obs", to="x", exclude=[-2, -1], squeeze=2)
        .as_set(["x"])
        .apply(include=inference_variables, forward=log_transform, inverse=inverse_log_transform)
        .standardize(include=inference_variables)
        .apply(include="num_obs", forward=sqrt_transform, inverse=inverse_sqrt_transform)
        .concatenate(inference_variables, into="inference_variables")
        .concatenate(["x"], into="summary_variables")
        .rename("num_obs", "inference_conditions")
    )

    adapter = adapter.keep(
        ["inference_variables", "inference_conditions", "summary_variables"]
    )

    return adapter
    
@keras.saving.register_keras_serializable(package="src", name="log_transform")
def log_transform(x):
    return np.log(x)

@keras.saving.register_keras_serializable(package="src", name="inverse_log_transform")
def inverse_log_transform(x):
    return np.exp(x)

@keras.saving.register_keras_serializable(package="src", name="sqrt_transform")
def sqrt_transform(x):
    return np.sqrt(x)

@keras.saving.register_keras_serializable(package="src", name="inverse_sqrt_transform")
def inverse_sqrt_transform(x):
    return np.square(x)


def probit_transform():
    return stats.norm.ppf


def inverse_probit_transform():
    return stats.norm.cdf
