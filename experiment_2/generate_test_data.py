import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from hydra.utils import instantiate
from omegaconf import DictConfig

from data import save_hdf5
from utils import create_missing_dirs

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def generate_test_data(cfg: DictConfig):
    simulator = instantiate(cfg["simulator"], _convert_="partial")

    meta_param_1 = instantiate(cfg["meta_param_1"])
    meta_param_2 = instantiate(cfg["meta_param_2"])

    meta_param_name_1 = cfg["meta_param_name_1"]
    meta_param_name_2 = cfg["meta_param_name_2"]

    create_missing_dirs([os.path.join(cfg["test_data_path"], "test_data")])

    error_rates = np.zeros((len(meta_param_1), len(meta_param_2)))

    for i, p1 in enumerate(meta_param_1):
        for j, p2 in enumerate(meta_param_2):
            logger.info("Creating test data set for params %s - %s", p1, p2)

            forward_dict = simulator.sample(
                batch_shape=cfg["test_batch_size"],
                **{
                    meta_param_name_1: np.array(p1),
                    meta_param_name_2: np.array(p2)
                },
                num_obs=np.array(cfg["test_num_obs"])
            )

            data = forward_dict["x"]

            error_rates[i, j] = np.mean(np.mean(data[:, :, 1], axis=0))

            test_data_path = os.path.join(cfg["test_data_path"], "test_data", f"test_data_{meta_param_name_1}_{p1}_{meta_param_name_2}_{p2}.hdf5")
            logger.info("Saving test data to %s", os.path.abspath(test_data_path))
            save_hdf5(test_data_path, forward_dict)
    
    fig, ax = plt.subplots(1, 1)

    sns.heatmap(error_rates, cmap="viridis", annot=True, ax=ax)

    ax.set_ylabel(cfg["meta_param_name_1"])
    ax.set_xlabel(cfg["meta_param_name_2"])
    ax.set_yticklabels(np.round(meta_param_1, 2))
    ax.set_xticklabels(np.round(meta_param_2, 2))

    plt.savefig("test_error_rates.png")


if __name__ == "__main__":
    generate_test_data()
