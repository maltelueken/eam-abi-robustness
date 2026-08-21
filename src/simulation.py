"""Utilities to use the racing diffusion model as a simulator in BayesFlow."""

from typing import Callable

import bayesflow as bf
import numpy as np

from bayesflow.utils.decorators import allow_batch_size
from bayesflow.types import Shape


class CustomSimulator(bf.simulators.Simulator):
    """Custom simulator for the racing diffusion model."""
    def __init__(
        self,
        prior_simulator: Callable,
        design_simulator: Callable,
        experiment_simulator: Callable,
    ):
        self.prior_simulator = prior_simulator
        self.design_simulator = design_simulator
        self.experiment_simulator = experiment_simulator

    @allow_batch_size
    def sample(self, batch_shape: Shape, **kwargs) -> dict[str, np.ndarray]:
        prior_dict = self.prior_simulator.sample(batch_shape)

        design_dict = self.design_simulator.sample(batch_shape)

        design_dict.update(**kwargs)

        sims_dict = self.experiment_simulator.sample(
            batch_shape, **prior_dict, **design_dict
        )

        data = prior_dict | design_dict | sims_dict

        data = {
            key: np.expand_dims(value, axis=-1) if np.ndim(value) == 1 else value
            for key, value in data.items()
        }

        return data


class CustomMetaSimulator(bf.simulators.Simulator):
    """Custom simulator for the racing diffusion model with a varying prior."""
    def __init__(
        self,
        meta_simulator: Callable,
        prior_simulator: Callable,
        design_simulator: Callable,
        experiment_simulator: Callable,
    ):
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

        sims_dict = self.experiment_simulator.sample(
            batch_shape, **prior_dict, **design_dict
        )

        data = meta_dict | prior_dict | design_dict | sims_dict

        data = {
            key: np.expand_dims(value, axis=-1) if np.ndim(value) == 1 else value
            for key, value in data.items()
        }

        return data


def create_data_adapter(
    inference_variables, inference_conditions=None, summary_variables=None
):
    """Create an adapter for the racing diffusion model simulator."""
    adapter = (
        bf.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .broadcast("num_obs", to="x", exclude=(-2, -1), squeeze=-1)
        .broadcast(
            inference_conditions, to="num_obs"
        )  # Make sure that all inference conditions have same shape
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


def sample_prior(simulator, batch_shape):
    """Draw from a simulator's prior without running the (expensive) forward model.

    `simulator.sample()` would simulate a full dataset per draw; the prior-range figures
    only need the parameters. For the hierarchical simulators the prior is conditional on
    the meta-parameters, so those are drawn first.
    """
    if isinstance(simulator, CustomMetaSimulator):
        meta_dict = simulator.meta_simulator.sample(batch_shape)
        return simulator.prior_simulator.sample(batch_shape, **meta_dict)

    return simulator.prior_simulator.sample(batch_shape)
