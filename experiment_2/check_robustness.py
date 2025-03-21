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
from hydra.utils import instantiate
from omegaconf import DictConfig

from data import load_hdf5
from utils import create_missing_dirs, create_robustness_2d_plot, convert_prior_samples, convert_posterior_samples

logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../conf", config_name="config")
def check_robustness(cfg: DictConfig):
    param_names = cfg["approximator"]["adapter"]["inference_variables"]

    meta_param_1 = instantiate(cfg["meta_param_1"])
    meta_param_2 = instantiate(cfg["meta_param_2"])

    meta_param_name_1 = cfg["meta_param_name_1"]
    meta_param_name_2 = cfg["meta_param_name_2"]

    create_missing_dirs(["robustness"])

    mmd = []

    for p1 in meta_param_1:
        for p2 in meta_param_2:
            test_data_path = os.path.join(cfg["test_data_path"], "test_data", f"test_data_{meta_param_name_1}_{p1}_{meta_param_name_2}_{p2}.hdf5")
            logger.info("Loading test data from %s", os.path.abspath(test_data_path))
            forward_dict = load_hdf5(test_data_path)

            mcmc_data_path = os.path.join(cfg["test_data_path"], "mcmc_samples", f"fit_mcmc_{meta_param_name_1}_{p1}_{meta_param_name_2}_{p2}.hdf5")
            logger.info("Loading MCMC samples from %s", os.path.abspath(mcmc_data_path))
            mcmc_samples = load_hdf5(mcmc_data_path)

            posterior_mcmc = mcmc_samples["samples"]
            is_converged = np.all(blackjax.diagnostics.potential_scale_reduction(posterior_mcmc, chain_axis=1, sample_axis=2) < 1.01, axis=1)
            
            logger.info("%s MCMC models did not converge: %s", 1.0-is_converged.mean(), np.where(~is_converged))
            posterior_mcmc = np.exp(np.reshape(posterior_mcmc, (posterior_mcmc.shape[0], -1, posterior_mcmc.shape[3])))[is_converged]
            posterior_mcmc = posterior_mcmc[:, ::4,:]

            prior_samples = convert_prior_samples(forward_dict, param_names)

            npe_data_path = os.path.join("npe_samples", f"posterior_samples_{meta_param_name_1}_{p1}_{meta_param_name_2}_{p2}.hdf5")
            logger.info("Loading NPE samples from %s", os.path.abspath(npe_data_path))
            npe_samples = load_hdf5(npe_data_path)
            posterior_npe = convert_posterior_samples(npe_samples, param_names)[is_converged]

            idx = 3

            fig = create_robustness_2d_plot(posterior_npe[idx,:,:], posterior_mcmc[idx,:,:], prior_samples, prior_samples[idx,:], param_names)
            fig.savefig(os.path.join("robustness", f"robustness_2d_{meta_param_name_1}_{p1}_{meta_param_name_2}_{p2}.png"))

            mmd.append(np.array([bf.metrics.functional.maximum_mean_discrepancy(x, y) for x, y in zip(posterior_mcmc, posterior_npe)]))

    pd.DataFrame({
        meta_param_name_1: meta_param_1,
        meta_param_name_2: meta_param_2,
        "mmd": mmd
    }).explode("mmd").to_csv(os.path.join("robustness", "mmd.csv"))


if __name__ == "__main__":
    check_robustness()