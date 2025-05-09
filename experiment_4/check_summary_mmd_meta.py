
import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf
import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from utils import create_missing_dirs, load_approximator, read_data_from_txt, sample_mmd_null

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_summary_mmd(cfg: DictConfig):
    approximator, simulator = load_approximator(cfg)

    create_missing_dirs(["summary_mmd"])

    basenames = []
    dist = []
    meta_p1 = []
    meta_p2 = []
    mmd = []

    meta_param_1 = instantiate(cfg["meta_param_1"])
    meta_param_2 = instantiate(cfg["meta_param_2"])

    meta_param_name_1 = cfg["meta_param_name_1"]
    meta_param_name_2 = cfg["meta_param_name_2"]

    for filename in os.listdir(os.path.join(cfg["test_data_path"], "test_data")):
        for p1 in meta_param_1:
            for p2 in meta_param_2:
                logger.info("Loading test data for file %s with %s and %s", filename, p1, p2)

                basename = os.path.splitext(filename)[0]

                data_real = read_data_from_txt(os.path.join(cfg["test_data_path"], "test_data", filename))
                summary_real = approximator.summary_network(data_real["x"])

                mmd_null = sample_mmd_null(
                    simulator,
                    approximator,
                    cfg["summary_mmd_num_reps"],
                    cfg["summary_mmd_batch_size"],
                    data_real["x"].shape[0],
                    **{
                        "num_obs": np.array(data_real["num_obs"]),
                        meta_param_name_1: np.array(p1),
                        meta_param_name_2: np.array(p2)
                    }
                )

                basenames.append(basename)
                dist.append("null")
                meta_p1.append(p1)
                meta_p2.append(p2)
                mmd.append(mmd_null)

                data_sim = simulator.sample(cfg["summary_mmd_num_reps"], **{
                    "num_obs": np.array(data_real["num_obs"]),
                    meta_param_name_1: np.array(p1),
                    meta_param_name_2: np.array(p2)
                })
                summary_sim = approximator.summary_network(data_sim["x"])

                mmd_obs = bf.metrics.functional.maximum_mean_discrepancy(summary_sim, summary_real)

                basenames.append(basename)
                dist.append("obs")
                meta_p1.append(p1)
                meta_p2.append(p2)
                mmd.append([mmd_obs])

    pd.DataFrame({
        "name": basenames,
        "type": dist,
        meta_param_name_1: meta_p1,
        meta_param_name_2: meta_p2,
        "mmd": mmd
    }).explode("mmd").to_csv(os.path.join("summary_mmd", "summary_mmd.csv"))


if __name__ == "__main__":
    check_summary_mmd()
