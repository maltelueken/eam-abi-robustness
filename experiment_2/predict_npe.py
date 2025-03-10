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

    meta_param_1 = instantiate(cfg["meta_param_1"])
    meta_param_2 = instantiate(cfg["meta_param_2"])

    meta_param_name_1 = cfg["meta_param_name_1"]
    meta_param_name_2 = cfg["meta_param_name_2"]

    create_missing_dirs(["npe_samples"])

    for p1 in meta_param_1:
        for p2 in meta_param_2:
            logger.info("Loading test data set for params %s - %s", p1, p2)

            test_data_path = os.path.join(cfg["test_data_path"], "test_data", f"test_data_{meta_param_name_1}_{p1}_{meta_param_name_2}_{p2}.hdf5")
            forward_dict = load_hdf5(test_data_path)
            sample_dict = {k: v for k, v in forward_dict.items() if k not in param_names}

            posterior_samples = approximator.sample(
                conditions=sample_dict,
                num_samples=cfg["test_num_posterior_samples"]
            )
            
            logger.info("Saving predictions to %s", os.path.abspath(test_data_path))
            save_hdf5(os.path.join("npe_samples", f"posterior_samples_{meta_param_name_1}_{p1}_{meta_param_name_2}_{p2}.hdf5"), posterior_samples)


if __name__ == "__main__":
    generate_test_data()
