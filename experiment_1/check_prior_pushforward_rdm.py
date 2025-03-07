import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow.diagnostics.plots as bf_plots
import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils import convert_prior_samples, create_pushforward_plot_rdm

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def prior_pushforward(cfg: DictConfig):
    simulator = instantiate(cfg["simulator"], _convert_="partial")

    param_names = cfg["approximator"]["adapter"]["inference_variables"]
    
    forward_dict = simulator.sample(500, num_obs=np.array(500))

    logger.info("Number of observations: %s", forward_dict["num_obs"])

    for k, v in forward_dict.items():
        logger.info("Parameter %s has shape %s", k, v.shape)

    prior_samples = convert_prior_samples(forward_dict, param_names)

    fig = bf_plots.pairs_samples(forward_dict, variable_keys=param_names, variable_names=param_names)
    
    fig.savefig("prior2d.png")

    data = forward_dict[cfg["approximator"]["adapter"]["summary_variables"][0]]

    fig = create_pushforward_plot_rdm(data[:25, :, :], prior_samples, param_names)

    fig.savefig("prior_pushforward.png")


if __name__ == "__main__":
    prior_pushforward()
