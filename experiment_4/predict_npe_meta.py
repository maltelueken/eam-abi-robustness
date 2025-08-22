import os
import logging

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

# os.environ["JAX_PLATFORMS"] = "cpu"

import hydra
import numpy as np
import optuna
import polars as pl
from omegaconf import DictConfig

from data import save_hdf5
from utils import create_missing_dirs, load_approximator, read_data_from_txt

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def generate_test_data(cfg: DictConfig):
    approximator, _ = load_approximator(cfg)

    create_missing_dirs(["npe_samples"])

    meta_param_name_1 = cfg["meta_param_name_1"]
    meta_param_name_2 = cfg["meta_param_name_2"]


    for filename in os.listdir(os.path.join(cfg["test_data_path"], "test_data")):
        logger.info("Loading test data for file %s", filename)

        study = optuna.create_study(
            study_name=os.path.splitext(filename)[0],
            storage="sqlite:///check_summary_mmd.db",
            load_if_exists=True
        )

        p1 = study.best_params[meta_param_name_1]
        p2 = study.best_params[meta_param_name_2]

        logger.info("Predicting with parameters %s", (p1, p2))

        data = read_data_from_txt(os.path.join(cfg["test_data_path"], "test_data", filename))

        data[meta_param_name_1] = np.array(p1)
        data[meta_param_name_2] = np.array(p2)

        posterior_samples= approximator.sample(
            conditions=data,
            num_samples=cfg["test_num_posterior_samples"]
        )

        save_hdf5(os.path.join("npe_samples", f"posterior_samples_{os.path.splitext(filename)[0]}.hdf5"), posterior_samples)



if __name__ == "__main__":
    generate_test_data()
