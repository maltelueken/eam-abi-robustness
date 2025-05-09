import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf
import blackjax
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from data import load_hdf5
from utils import create_missing_dirs, convert_posterior_samples

logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_robustness(cfg: DictConfig):
    param_names = cfg["approximator"]["adapter"]["inference_variables"]

    basenames = []
    mmd = []

    create_missing_dirs(["robustness"])

    for filename in os.listdir(os.path.join(cfg["test_data_path"], "test_data")):
        logger.info("Checking robustness for file %s", filename)

        basename = os.path.splitext(filename)[0]

        mcmc_data_path = os.path.join(cfg["test_data_path"], "mcmc_samples", f"fit_mcmc_{basename}.hdf5")
        logger.info("Loading MCMC samples from %s", os.path.abspath(mcmc_data_path))
        mcmc_samples = load_hdf5(mcmc_data_path)

        posterior_mcmc = mcmc_samples["samples"]
        is_converged = np.all(blackjax.diagnostics.potential_scale_reduction(posterior_mcmc, chain_axis=1, sample_axis=2) < 1.01, axis=1)

        logger.info("%s MCMC models did not converge: %s", 1.0-is_converged.mean(), np.where(~is_converged))
        posterior_mcmc = np.exp(np.reshape(posterior_mcmc, (posterior_mcmc.shape[0], -1, posterior_mcmc.shape[3])))[is_converged]
        posterior_mcmc = posterior_mcmc[:, ::4,:]

        npe_data_path = os.path.join("npe_samples", f"posterior_samples_{basename}.hdf5")
        logger.info("Loading NPE samples from %s", os.path.abspath(npe_data_path))
        npe_samples = load_hdf5(npe_data_path)
        posterior_npe = convert_posterior_samples(npe_samples, param_names)[is_converged]

        basenames.append(basename)
        mmd.append(np.array([bf.metrics.functional.maximum_mean_discrepancy(x, y) for x, y in zip(posterior_mcmc, posterior_npe)]))

    pd.DataFrame({
        "name": basenames,
        "mmd": mmd
    }).explode("mmd").to_csv(os.path.join("robustness", "mmd.csv"))

if __name__ == "__main__":
    check_robustness()
