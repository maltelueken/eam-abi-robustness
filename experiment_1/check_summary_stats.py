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
from metrics import calibration_error
from utils import create_missing_dirs, convert_prior_samples, convert_posterior_samples

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_summary_stats(cfg: DictConfig):
    sample_sizes = instantiate(cfg["test_num_obs"])

    param_names = cfg["approximator"]["adapter"]["inference_variables"]

    create_missing_dirs(["metrics"])

    param = []
    true_param = []
    npe_posterior_median = []
    npe_posterior_lower = []
    npe_posterior_upper = []

    mcmc_posterior_median = []
    mcmc_posterior_lower = []
    mcmc_posterior_upper = []

    for t in sample_sizes:
        test_data_path = os.path.join(cfg["test_data_path"], "test_data", f"test_data_sample_size_{t}.hdf5")
        logger.info("Loading test data from %s", os.path.abspath(test_data_path))
        forward_dict = load_hdf5(test_data_path)

        mcmc_data_path = os.path.join(cfg["test_data_path"], "mcmc_samples", f"fit_mcmc_sample_size_{t}.hdf5")
        logger.info("Loading MCMC samples from %s", os.path.abspath(mcmc_data_path))
        mcmc_samples = load_hdf5(mcmc_data_path)

        posterior_mcmc = mcmc_samples["samples"]
        is_converged = np.all(blackjax.diagnostics.potential_scale_reduction(posterior_mcmc, chain_axis=1, sample_axis=2) < 1.01, axis=1)
        
        logger.info("%s MCMC models did not converge: %s", 1.0-is_converged.mean(), np.where(~is_converged))
        posterior_mcmc = np.exp(np.reshape(posterior_mcmc, (posterior_mcmc.shape[0], -1, posterior_mcmc.shape[3])))[is_converged]
        posterior_mcmc = posterior_mcmc[:, ::4,:]

        prior_samples = convert_prior_samples(forward_dict, param_names)[is_converged]

        npe_data_path = os.path.join("npe_samples", f"posterior_samples_sample_size_{t}.hdf5")
        logger.info("Loading NPE samples from %s", os.path.abspath(npe_data_path))
        npe_samples = load_hdf5(npe_data_path)
        posterior_npe = convert_posterior_samples(npe_samples, param_names)[is_converged]

        param.append(np.tile(param_names, posterior_npe.shape[0]).flatten())

        true_param.append(prior_samples.flatten())

        npe_posterior_median.append(np.median(posterior_npe, axis=1).flatten())
        npe_posterior_lower.append(np.quantile(posterior_npe, axis=1, q=0.025).flatten())
        npe_posterior_upper.append(np.quantile(posterior_npe, axis=1, q=0.975).flatten())

        mcmc_posterior_median.append(np.median(posterior_mcmc, axis=1).flatten())
        mcmc_posterior_lower.append(np.quantile(posterior_mcmc, axis=1, q=0.025).flatten())
        mcmc_posterior_upper.append(np.quantile(posterior_mcmc, axis=1, q=0.975).flatten())

    pd.DataFrame({
        "sample_size": sample_sizes,
        "param": param,
        "true": true_param,
        "npe_median": npe_posterior_median,
        "npe_lower": npe_posterior_lower,
        "npe_upper": npe_posterior_upper,
        "mcmc_median": mcmc_posterior_median,
        "mcmc_lower": mcmc_posterior_lower,
        "mcmc_upper": mcmc_posterior_upper,
    }).explode(["param", "true", "npe_median", "npe_lower", "npe_upper", "mcmc_median", "mcmc_lower", "mcmc_upper"]).to_csv(os.path.join("metrics", "summary_stats.csv"))


if __name__ == "__main__":
    check_summary_stats()