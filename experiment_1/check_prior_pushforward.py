import logging

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils import convert_prior_samples, create_prior_2d_plot, create_pushforward_plot_rdm, create_pushforward_plot_rdmc

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="train_npe")
def prior_pushforward(cfg: DictConfig):
    simulator = instantiate(cfg["simulator"])
    data_adapter = instantiate(cfg["approximator"]["data_adapter"])

    param_names = cfg["approximator"]["data_adapter"]["inference_variables"]
    
    forward_dict = simulator.sample((500,), num_obs=np.tile(np.array([500]), (500,)))

    prior_samples = convert_prior_samples(forward_dict, param_names)

    fig = create_prior_2d_plot(prior_samples, param_names)
    fig.savefig("prior2d.png")

    data = forward_dict[cfg["approximator"]["data_adapter"]["summary_variables"][0]]

    if data.shape[-1] == 2:
        fig = create_pushforward_plot_rdm(data[:25, :, :], prior_samples, param_names)
    elif data.shape[-1] == 3:
        fig = create_pushforward_plot_rdmc(data[:25, :, :], prior_samples, param_names)
    else:
        raise RuntimeError("Data must have a third dimension of shape 2 or 3")
    fig.savefig("prior_pushforward.png")


if __name__ == "__main__":
    prior_pushforward()
