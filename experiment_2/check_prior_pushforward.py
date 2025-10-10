
import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf
import hydra
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils import convert_prior_samples, create_missing_dirs, create_pushforward_plot_rdm

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_npe_diagnostics(cfg: DictConfig):
    simulator = instantiate(cfg["simulator"])

    param_names = cfg["approximator"]["adapter"]["inference_variables"]

    create_missing_dirs(["prior", "push_forward"])

    forward_dict = simulator.sample(
        batch_shape=cfg["diag_batch_size"],
        num_obs=np.array(cfg["diag_num_obs"])
    )

    data = forward_dict["x"]

    print(data.shape)

    error_rates = 1.0 - np.mean(data[:, :, 1], axis=1)

    print(error_rates.shape)

    fig, ax = plt.subplots(1, 1)

    ax.hist(error_rates)
    ax.set_xlim((0, 0.5))
    ax.set_xlabel("Error rate")

    plt.savefig("error_rates.png")

    prior_samples = convert_prior_samples(forward_dict, param_names)

    fig = bf.diagnostics.pairs_samples({key: val for key, val in forward_dict.items() if key in param_names})

    fig.savefig(os.path.join("prior", f"prior2d.png"))

    data = forward_dict[cfg["approximator"]["adapter"]["summary_variables"][0]]

    fig = create_pushforward_plot_rdm(data[:25, :, :], prior_samples, param_names)

    fig.savefig(os.path.join("push_forward", f"prior_pushforward.png"))


if __name__ == "__main__":
    check_npe_diagnostics()
