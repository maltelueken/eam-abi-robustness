import logging
import os

os.environ["JAX_PLATFORMS"] = "cpu"

import hydra
import numpy as np
from omegaconf import DictConfig

from data import load_hdf5, save_hdf5
from utils import create_missing_dirs, read_data_from_txt

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def fit_mcmc(cfg: DictConfig):
    filename = cfg["slurm_filename"]

    data = read_data_from_txt(os.path.join(cfg["test_data_path"], "test_data", filename))
    
    data_x = data["x"]

    basename = os.path.splitext(filename)[0]

    create_missing_dirs([os.path.join(cfg["test_data_path"], "test_data")])

    logger.info("Collecting MCMC samples for file %s", filename)

    trace_dict = {}

    for i in range(data_x.shape[0]):
        try:
            trace_dict = trace_dict | load_hdf5(os.path.join(cfg["test_data_path"], "mcmc_samples", basename, f"samples_{i}.hdf5"))
        except:
            logger.info("No MCMC samples found for filename %s and dataset %s", filename, i)

    trace_dict = {int(k): v for k, v in trace_dict.items()}

    trace_data = dict(sorted(trace_dict.items()))

    # Save to hdf5 file
    save_hdf5(os.path.join(cfg["test_data_path"], "mcmc_samples", f"fit_mcmc_{basename}.hdf5"), {"samples": np.array(list(trace_data.values()))})


if __name__ == "__main__":
    fit_mcmc()
