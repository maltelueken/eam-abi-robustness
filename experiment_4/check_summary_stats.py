import logging
import os

import blackjax
import hydra
import numpy as np
import pandas as pd
from omegaconf import DictConfig

from data import load_hdf5
from utils import create_missing_dirs, convert_posterior_samples, read_data_from_txt

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_metrics(cfg: DictConfig):

    param_names = cfg["approximator"]["adapter"]["inference_variables"]

    create_missing_dirs(["metrics"])

    basenames = []
    param = []

    npe_posterior_median = []
    npe_posterior_lower = []
    npe_posterior_upper = []

    mcmc_posterior_median = []
    mcmc_posterior_lower = []
    mcmc_posterior_upper = []

    accuracy = []

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

        data = read_data_from_txt(os.path.join(cfg["test_data_path"], "test_data", filename))

        data_x = data["x"][is_converged]

        acc = data_x[:,:,1].mean(axis=1)

        basenames.append(basename)

        param.append(np.tile(param_names, posterior_npe.shape[0]).flatten())

        npe_posterior_median.append(np.median(posterior_npe, axis=1).flatten())
        npe_posterior_lower.append(np.quantile(posterior_npe, axis=1, q=0.025).flatten())
        npe_posterior_upper.append(np.quantile(posterior_npe, axis=1, q=0.975).flatten())

        mcmc_posterior_median.append(np.median(posterior_mcmc, axis=1).flatten())
        mcmc_posterior_lower.append(np.quantile(posterior_mcmc, axis=1, q=0.025).flatten())
        mcmc_posterior_upper.append(np.quantile(posterior_mcmc, axis=1, q=0.975).flatten())

        accuracy.append(np.repeat(acc, len(param_names)))

        
    pd.DataFrame({
        "name": basenames,
        "param": param,
        "npe_median": npe_posterior_median,
        "npe_lower": npe_posterior_lower,
        "npe_upper": npe_posterior_upper,
        "mcmc_median": mcmc_posterior_median,
        "mcmc_lower": mcmc_posterior_lower,
        "mcmc_upper": mcmc_posterior_upper,
        "acc": accuracy,
    }).explode(["param", "npe_median", "npe_lower", "npe_upper", "mcmc_median", "mcmc_lower", "mcmc_upper", "acc"]).to_csv(os.path.join("metrics", "summary_stats.csv"))


if __name__ == "__main__":
    check_metrics()