
import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf
import hydra
import numpy as np
import optuna
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils import create_missing_dirs, load_approximator, read_data_from_txt, sample_mmd_null

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_summary_mmd(cfg: DictConfig):
    approximator, simulator = load_approximator(cfg)

    create_missing_dirs(["summary_mmd"])

    meta_param_name_1 = cfg["meta_param_name_1"]
    meta_param_name_2 = cfg["meta_param_name_2"]

    def objective(trial):
        p1 = trial.suggest_float(meta_param_name_1, 0.5, 4.0)
        p2 = trial.suggest_float(meta_param_name_2, 0.05, 0.5)

        data_real = read_data_from_txt(os.path.join(cfg["test_data_path"], "test_data", filename))
        summary_real = approximator.summary_network(data_real["x"])

        data_sim = simulator.sample(cfg["summary_mmd_num_reps"], **{
            "num_obs": np.array(data_real["num_obs"]),
            meta_param_name_1: np.array(p1),
            meta_param_name_2: np.array(p2)
        })
        summary_sim = approximator.summary_network(data_sim["x"])

        mmd_obs = bf.metrics.functional.maximum_mean_discrepancy(summary_sim, summary_real)

        return mmd_obs

    for filename in os.listdir(os.path.join(cfg["test_data_path"], "test_data")):
        logger.info("Loading test data for file %s", filename)

        basename = os.path.splitext(filename)[0]

        study = optuna.create_study(study_name=basename, direction="minimize", storage="sqlite:///check_summary_mmd.db")
        study.optimize(objective, n_trials=200)


if __name__ == "__main__":
    check_summary_mmd()
