import logging
import os

if "KERAS_BACKEND" not in os.environ:
    # set this to "torch", "tensorflow", or "jax"
    os.environ["KERAS_BACKEND"] = "jax"

import bayesflow as bf
import bayesflow.diagnostics.metrics as bf_metrics
import blackjax
import hydra
import numpy as np
import pandas as pd
from hydra.utils import instantiate
from omegaconf import DictConfig

from data import load_hdf5
from utils import create_missing_dirs, convert_prior_samples, convert_posterior_samples

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_metrics(cfg: DictConfig):
    param_names = cfg["approximator"]["adapter"]["inference_variables"]

    basenames = []
    params = []
    mean_mcmc = []
    median_mcmc = []
    upper_mcmc = []
    lower_mcmc = []
    mean_npe = []
    median_npe = []
    upper_npe = []
    lower_npe = []

    create_missing_dirs(["metrics"])

    for filename in os.listdir(os.path.join(cfg["test_data_path"], "test_data")):
        logger.info("Checking metrics for file %s", filename)

        basename = os.path.splitext(filename)[0]

        mcmc_data_path = os.path.join("mcmc_samples", f"fit_mcmc_{basename}.hdf5")
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
        params.append(np.tile(param_names, (posterior_mcmc.shape[0], 1)))

        mean_mcmc.append(posterior_mcmc.mean(axis=1))
        median_mcmc.append(np.median(posterior_mcmc, axis=1))
        upper_mcmc.append(np.quantile(posterior_mcmc, q=0.975, axis=1))
        lower_mcmc.append(np.quantile(posterior_mcmc, q=0.025, axis=1))

        mean_npe.append(posterior_npe.mean(axis=1))
        median_npe.append(np.median(posterior_npe, axis=1))
        upper_npe.append(np.quantile(posterior_npe, q=0.975, axis=1))
        lower_npe.append(np.quantile(posterior_npe, q=0.025, axis=1))
    
    explode_columns = ["param", "mcmc_mean", "mcmc_median", "mcmc_lower", "mcmc_upper",
                "npe_mean", "npe_median", "npe_lower", "npe_upper"]

    pd.DataFrame({
        "name": basenames,
        "param": params,
        "mcmc_mean": mean_mcmc,
        "mcmc_median": median_mcmc,
        "mcmc_lower": lower_mcmc,
        "mcmc_upper": upper_mcmc,
        "npe_mean": mean_npe,
        "npe_median": median_npe,
        "npe_lower": lower_npe,
        "npe_upper": upper_npe
    }).explode(explode_columns).explode(explode_columns).to_csv(os.path.join("metrics", "metrics.csv"))


if __name__ == "__main__":
    check_metrics()