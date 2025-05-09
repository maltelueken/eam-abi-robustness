
import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf
import bayesflow.diagnostics.plots as bf_plots
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from utils import create_missing_dirs, load_approximator, read_data_from_txt, sample_mmd_null

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_summary_mmd(cfg: DictConfig):
    approximator, simulator = load_approximator(cfg)

    create_missing_dirs(["summary_mmd"])

    basenames = []
    dist = []
    mmd = []

    for filename in os.listdir(os.path.join(cfg["test_data_path"], "test_data")):
        logger.info("Loading test data for file %s", filename)

        basename = os.path.splitext(filename)[0]

        data_real = read_data_from_txt(os.path.join(cfg["test_data_path"], "test_data", filename))
        summary_real = approximator.summary_network(data_real["x"])

        mmd_null = sample_mmd_null(
            simulator,
            approximator,
            cfg["summary_mmd_num_reps"],
            cfg["summary_mmd_batch_size"],
            data_real["x"].shape[0], 
            num_obs=data_real["num_obs"]
        )

        basenames.append(basename)
        dist.append("null")
        mmd.append(mmd_null)

        data_sim = simulator.sample(cfg["summary_mmd_num_reps"], num_obs=np.array(data_real["num_obs"]))
        summary_sim = approximator.summary_network(data_sim["x"])

        mmd_obs = bf.metrics.functional.maximum_mean_discrepancy(summary_sim, summary_real)

        f = bf_plots.mmd_hypothesis_test(mmd_null, mmd_obs)

        f.savefig(os.path.join("summary_mmd", f"hypothesis_test_{basename}.png"))

        basenames.append(basename)
        dist.append("obs")
        mmd.append([mmd_obs])

    pd.DataFrame({
        "name": basenames,
        "type": dist,
        "mmd": mmd
    }).explode("mmd").to_csv(os.path.join("summary_mmd", "summary_mmd.csv"))


if __name__ == "__main__":
    check_summary_mmd()
