import logging

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils import convert_samples, create_prior_2d_plot, create_pushforward_plot

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="train_npe")
def prior_pushforward(cfg: DictConfig):
    simulator = instantiate(cfg["simulator"])
    data_adapter = instantiate(cfg["approximator"]["data_adapter"])

    param_names = data_adapter.keys["inference_variables"]

    forward_dict = simulator.sample((500,), num_obs=np.tile(np.array([500]), (500,)))

    prior_samples = convert_samples(forward_dict, param_names)

    fig = create_prior_2d_plot(prior_samples, param_names)
    fig.savefig("prior2d.png")

    data = forward_dict[data_adapter.keys["summary_variables"][0]]

    fig = create_pushforward_plot(data[:25, :, :], prior_samples, param_names)
    fig.savefig("prior_pushforward.png")


if __name__ == "__main__":
    prior_pushforward()
