import os
import logging

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

# os.environ["JAX_PLATFORMS"] = "cpu"

import hydra
import keras
import numpy as np
import optuna
import pandas as pd
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

    create_missing_dirs(["npe_samples", "summary_mmd"])

    meta_dict = {
        f"true_{meta_param_name_1}": [],
        f"true_{meta_param_name_2}": [],
        f"est_{meta_param_name_1}": [],
        f"est_{meta_param_name_2}": []
    }

    for p1 in meta_param_1:
        for p2 in meta_param_2:
            logger.info("Loading test data set for params %s - %s", p1, p2)

            test_data_path = os.path.join(cfg["test_data_path"], "test_data", f"test_data_{meta_param_name_1}_{p1}_{meta_param_name_2}_{p2}.hdf5")
            forward_dict = load_hdf5(test_data_path)

            basename = os.path.splitext(os.path.basename(test_data_path))[0]

            study = optuna.create_study(
                study_name=basename,
                storage="sqlite:///check_summary_mmd.db",
                load_if_exists=True
            )

            p1_ = study.best_params[meta_param_name_1]
            p2_ = study.best_params[meta_param_name_2]

            meta_dict[f"true_{meta_param_name_1}"].append(p1)
            meta_dict[f"true_{meta_param_name_2}"].append(p2)
            meta_dict[f"est_{meta_param_name_1}"].append(p1_)
            meta_dict[f"est_{meta_param_name_2}"].append(p2_)

            logger.info("Predicting with parameters %s", (p1_, p2_))

            forward_dict[meta_param_name_1] = np.array(p1_)
            forward_dict[meta_param_name_2] = np.array(p2_)

            posterior_samples = approximator.sample(
                conditions=forward_dict,
                num_samples=cfg["test_num_posterior_samples"]
            )
            
            logger.info("Saving predictions to %s", os.path.abspath(test_data_path))
            save_hdf5(os.path.join("npe_samples", f"posterior_samples_{meta_param_name_1}_{p1}_{meta_param_name_2}_{p2}.hdf5"), posterior_samples)

    pd.DataFrame(meta_dict).to_csv(os.path.join("summary_mmd", "summary_mmd_params.csv"))


if __name__ == "__main__":
    generate_test_data()
