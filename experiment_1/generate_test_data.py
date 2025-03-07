import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import hydra
import keras
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from data import save_hdf5
from utils import create_missing_dirs

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def generate_test_data(cfg: DictConfig):
    simulator = instantiate(cfg["simulator"], _convert_="partial")

    sample_sizes = instantiate(cfg["test_num_obs"])

    create_missing_dirs([os.path.join(cfg["test_data_path"], "test_data")])

    for t in sample_sizes:
        logger.info("Creating test data set for sample size %s", t)

        forward_dict = simulator.sample(
            cfg["test_batch_size"], num_obs=np.array(t)
        )
        
        test_data_path = os.path.join(cfg["test_data_path"], "test_data", f"test_data_sample_size_{t}.hdf5")
        logger.info("Saving test data to %s", os.path.abspath(test_data_path))
        save_hdf5(test_data_path, forward_dict)


if __name__ == "__main__":
    generate_test_data()
