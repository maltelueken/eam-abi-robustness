import os
import logging

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

# os.environ["JAX_PLATFORMS"] = "cpu"

import hydra
import keras
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from data import load_hdf5, save_hdf5
from utils import create_missing_dirs, load_approximator

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def generate_test_data(cfg: DictConfig):
    approximator, _ = load_approximator(cfg)

    param_names = cfg["approximator"]["adapter"]["inference_variables"]

    sample_sizes = instantiate(cfg["test_num_obs"])

    create_missing_dirs(["npe_samples"])

    for t in sample_sizes:
        logger.info("Loading test data set for sample size %s", t)

        test_data_path = os.path.join(cfg["test_data_path"], "test_data", f"test_data_sample_size_{t}.hdf5")
        forward_dict = load_hdf5(test_data_path)
        sample_dict = {k: v for k, v in forward_dict.items() if k not in param_names}
        sample_dict["num_obs"] = sample_dict["num_obs"][:1]

        posterior_samples = approximator.sample(
            conditions=sample_dict,
            num_samples=cfg["test_num_posterior_samples"]
        )
        
        logger.info("Saving predictions to %s", os.path.abspath(test_data_path))
        save_hdf5(os.path.join("npe_samples", f"posterior_samples_sample_size_{t}.hdf5"), posterior_samples)


if __name__ == "__main__":
    generate_test_data()
