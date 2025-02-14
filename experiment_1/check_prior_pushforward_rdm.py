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
    workflow = instantiate(cfg["workflow"])

    param_names = cfg["workflow"]["adapter"]["inference_variables"]
    
    forward_dict = workflow.simulate((500, ), num_obs=np.tile(np.array([500]), (500,)))

    prior_samples = convert_prior_samples(forward_dict, param_names)

    fig = bf_plots.pairs_samples(forward_dict, variable_names=param_names)
    
    fig.savefig("prior2d.png")

    data = forward_dict[cfg["workflow"]["adapter"]["summary_variables"][0]]

    fig = create_pushforward_plot_rdm(data[:25, :, :], prior_samples, param_names)

    fig.savefig("prior_pushforward.png")


if __name__ == "__main__":
    prior_pushforward()
