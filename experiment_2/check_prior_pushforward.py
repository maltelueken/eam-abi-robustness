
import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import hydra
import keras
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.utils import instantiate
from omegaconf import DictConfig

from bayesflow_plots import *
from utils import convert_prior_samples, create_missing_dirs, create_prior_2d_plot, create_pushforward_plot_rdm

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_npe_diagnostics(cfg: DictConfig):
    simulator = instantiate(cfg["simulator"])

    param_names = cfg["approximator"]["adapter"]["inference_variables"]

    drift_slope_loc = instantiate(cfg["diag_drift_slope_loc"])
    threshold_scale = instantiate(cfg["diag_threshold_scale"])

    create_missing_dirs(["prior", "push_forward"])

    error_rates = np.zeros((len(drift_slope_loc), len(threshold_scale)))

    for i, p1 in enumerate(drift_slope_loc):
        for j, p2 in enumerate(threshold_scale):
            logger.info("Creating diag data for params %s - %s", p1, p2)

            forward_dict = simulator.sample(
                batch_shape=(cfg["diag_batch_size"],),
                drift_slope_loc=np.tile([p1], (cfg["diag_batch_size"],)),
                threshold_scale=np.tile([p2], (cfg["diag_batch_size"],)),
                num_obs=np.tile([cfg["diag_num_obs"]], (cfg["diag_batch_size"],))
            )

            data = forward_dict["x"]

            error_rates[i, j] = np.mean(np.mean(data[:, :, 1], axis=0))

            prior_samples = convert_prior_samples(forward_dict, param_names)

            fig = create_prior_2d_plot(prior_samples, param_names)
    
            fig.savefig(os.path.join("prior", f"prior2d_drift_slope_loc_{p1}_threshold_scale_{p2}.png"))

            data = forward_dict[cfg["approximator"]["adapter"]["summary_variables"][0]]

            fig = create_pushforward_plot_rdm(data[:25, :, :], prior_samples, param_names)
        
            fig.savefig(os.path.join("push_forward", f"prior_pushforward_drift_slope_loc_{p1}_threshold_scale_{p2}.png"))


    fig, ax = plt.subplots(1, 1)

    sns.heatmap(error_rates, cmap="viridis", annot=True, ax=ax)

    ax.set_yticklabels(drift_slope_loc)
    ax.set_xticklabels(threshold_scale)

    plt.savefig("error_rates.png")


if __name__ == "__main__":
    check_npe_diagnostics()
