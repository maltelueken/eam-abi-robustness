import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

os.environ["JAX_PLATFORMS"] = "cpu"

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig

from data import load_hdf5, save_hdf5
from utils import create_missing_dirs

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def fit_mcmc(cfg: DictConfig):
    t = cfg["slurm_num_obs"]

    create_missing_dirs([os.path.join(cfg["test_data_path"], "test_data")])

    logger.info("Collecting test data set for sample size %s", t)

    trace_dict = {}

    for i in range(cfg["test_batch_size"]):
        try:
            trace_dict = trace_dict | load_hdf5(os.path.join(cfg["test_data_path"], "mcmc_samples", str(t), f"samples_{i}.hdf5"))
        except:
            logger.info("No MCMC samples found for num samples %s and dataset %s", t, i)

    trace_data = dict(sorted(trace_dict.items()))

    # Save to hdf5 file
    save_hdf5(os.path.join(cfg["test_data_path"], "mcmc_samples", f"fit_mcmc_sample_size_{t}.hdf5"), {"samples": np.array(list(trace_data.values()))})


if __name__ == "__main__":
    fit_mcmc()
