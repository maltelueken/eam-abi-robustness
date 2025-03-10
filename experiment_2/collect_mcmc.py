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
    p1 = cfg["slurm_p1"]
    p2 = cfg["slurm_p2"]

    meta_param_name_1 = cfg["meta_param_name_1"]
    meta_param_name_2 = cfg["meta_param_name_2"]

    create_missing_dirs([os.path.join(cfg["test_data_path"], "test_data")])

    logger.info("Collecting test data set for for params %s - %s", p1, p2)

    trace_dict = {}

    for i in range(cfg["test_batch_size"]):
        try:
            trace_dict = trace_dict | load_hdf5(os.path.join(cfg["test_data_path"], "mcmc_samples", f"{p1}_{p2}", f"samples_{i}.hdf5"))
        except:
            logger.info("No MCMC samples found for params %s - %s and dataset %s", p1, p2, i)

    trace_dict = {int(k): v for k, v in trace_dict.items()}

    trace_data = dict(sorted(trace_dict.items()))

    # Save to hdf5 file
    save_hdf5(os.path.join(cfg["test_data_path"], "mcmc_samples", f"fit_mcmc_{meta_param_name_1}_{p1}_{meta_param_name_2}_{p2}.hdf5"), {"samples": np.array(list(trace_data.values()))})


if __name__ == "__main__":
    fit_mcmc()
